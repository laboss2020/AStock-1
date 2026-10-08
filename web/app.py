"""AStock Web UI — Streamlit wrapper around the five-stage pipeline.

Run:  streamlit run web/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from astock.config import LLM_PRESETS, load_config
from astock.llm import LLMClient, LLMError
from astock.pipeline import run_pipeline
from astock.report import save_report

st.set_page_config(page_title="A股大盘选股", page_icon="📈", layout="wide")


def _provider_options() -> list[str]:
    return list(LLM_PRESETS)


def _default_model(provider: str) -> str:
    return LLM_PRESETS.get(provider, {}).get("default_model", "")


# --- Sidebar: config ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 配置")
    provider = st.selectbox("AI Provider", _provider_options(), index=0)
    base_url = st.text_input("Base URL", LLM_PRESETS[provider]["base_url"])
    model = st.text_input("模型", _default_model(provider))
    api_key = st.text_input("API Key", type="password",
                            help="NVIDIA NIM: nvapi- 开头（build.nvidia.com 免费申请）")
    st.divider()
    max_picks = st.slider("最终选股数量上限", 3, 10, 10)
    candidates = st.slider("LLM 快筛候选数", 10, 60, 30)
    debug = st.checkbox("调试日志", value=False,
                        help="显示数据层 WARNING（东财封禁/fallback 触发可见）")
    st.divider()
    st.caption("仅供 AI 分析参考，不构成投资建议。A 股有风险，入市需谨慎。")

# --- Main --------------------------------------------------------------------
st.title("📈 A股大盘选股")
st.caption("大盘体检 → 量化初筛 → LLM 快筛 → 深度分析 → 买卖定价（五阶段，源自 tradingagents-astock 策略）")

run_clicked = st.button("🚀 开始选股", type="primary", use_container_width=True)

if run_clicked:
    if not api_key:
        st.error("请先在侧边栏填入 API Key（NVIDIA NIM 免费 key：https://build.nvidia.com）")
        st.stop()

    cfg = load_config(provider=provider, base_url=base_url or None,
                      model=model or None, api_key=api_key)
    llm = LLMClient(cfg)

    placeholder = st.empty()
    log_area = st.empty()
    lines: list[str] = []

    def _log(msg: str) -> None:
        lines.append(msg)
        log_area.code("\n".join(lines[-30:]), language=None)

    import logging

    class _StHandler(logging.Handler):
        def emit(self, record):
            _log(record.getMessage())

    logging.getLogger("astock").setLevel(logging.INFO if not debug else logging.DEBUG)
    logging.getLogger("astock").addHandler(_StHandler())

    with placeholder.status("运行中（约 5-10 分钟，全程串行防东财封禁）…", expanded=False):
        try:
            result = run_pipeline(cfg, llm)
        except LLMError as e:
            placeholder.update(label="失败", state="error")
            st.error(f"LLM 调用失败：{e}")
            st.stop()

    placeholder.update(label="完成", state="complete")
    _log(f"=== 完成：{len(result.picks)} 支最终选股，LLM {result.llm_calls} 次调用 ===")

    # --- 大盘体检 ---------------------------------------------------------
    st.header("🌤️ 大盘体检")
    rg = result.regime
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场环境", rg.regime or "—")
    c2.metric("温度", rg.temperature or "—")
    c3.metric("风格", rg.style or "—")
    c4.metric("风险偏好", rg.risk_appetite or "—")
    st.write(f"**关注板块**：{'、'.join(rg.sector_focus) if rg.sector_focus else '—'}")
    st.write(f"**回避板块**：{'、'.join(rg.avoid_sectors) if rg.avoid_sectors else '—'}")
    st.info(rg.rationale or "")

    # --- 筛选漏斗 ---------------------------------------------------------
    st.header("🔬 筛选漏斗")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("全A清单", result.universe_count)
    f2.metric("硬过滤后", result.filtered_count)
    f3.metric("LLM快筛", result.triaged_count)
    f4.metric("深度分析", result.triaged_count - result.rejected_count)
    f5.metric("最终选股", len(result.picks))

    # --- 最终选股 ---------------------------------------------------------
    st.header("🎯 最终选股")
    if not result.picks:
        st.warning("本轮没有选出值得买入的股票（防守环境下这是正常结果）。")
    else:
        rows = []
        for p in result.picks:
            pp = p.price_plan
            rows.append({
                "代码": p.code, "名称": p.name,
                "现价": p.bundle.get("行情快照", {}).get("price"),
                "买入区间": f"{pp.entry_low:.2f} ~ {pp.entry_high:.2f}" if pp.entry_low else "—",
                "止损": f"{pp.stop_loss:.2f}" if pp.stop_loss else "—",
                "卖出区间": f"{pp.target_low:.2f} ~ {pp.target_high:.2f}" if pp.target_low else "—",
                "仓位": pp.position_pct or "—",
                "理由": (pp.rationale or "")[:60],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📋 详细分析（5维评价 + 定价依据）", expanded=False):
            for p in result.picks:
                st.subheader(f"{p.code} {p.name} — {p.verdict.verdict}（{p.verdict.score} 分）")
                v = p.verdict
                if v.dimension_summary:
                    da, db, dc, dd, de = st.columns(5)
                    dims = list(v.dimension_summary.items())
                    for col, (dim, comment) in zip((da, db, dc, dd, de), dims):
                        col.metric(dim, (comment or "—")[:14], help=comment)
                st.write(f"**买入逻辑**：{v.buy_logic or '—'}")
                if v.catalysts:
                    st.write(f"**催化因素**：{'；'.join(v.catalysts)}")
                if v.risks:
                    st.write(f"**风险提示**：{'；'.join(v.risks)}")
                pp = p.price_plan
                if pp.entry_low:
                    st.write(
                        f"**买入** {pp.entry_low:.2f}~{pp.entry_high:.2f}　"
                        f"**止损** {pp.stop_loss:.2f}　"
                        f"**卖出** {pp.target_low:.2f}~{pp.target_high:.2f}　"
                        f"**持有** {pp.holding_period or '—'}"
                    )
                    if pp.rationale:
                        st.caption(pp.rationale)

    # --- 候选池 / 错误 -----------------------------------------------------
    if result.screened_out:
        with st.expander(f"📊 候选池 Top {len(result.screened_out)}（量化打分）", expanded=False):
            st.dataframe(pd.DataFrame(result.screened_out), use_container_width=True)

    if result.errors:
        with st.expander(f"⚠️ 错误记录（{len(result.errors)}）", expanded=True):
            for e in result.errors:
                st.write(f"- {e}")

    # --- 落盘 -------------------------------------------------------------
    out_dir = save_report(cfg, result)
    st.success(f"报告已保存：{out_dir / 'report.md'}")
    st.download_button(
        "⬇️ 下载 report.md",
        data=(out_dir / "report.md").read_text(encoding="utf-8"),
        file_name="astock_report.md", mime="text/markdown",
        use_container_width=True,
    )
    picks_path = out_dir / "picks.json"
    if picks_path.exists():
        st.download_button(
            "⬇️ 下载 picks.json",
            data=picks_path.read_text(encoding="utf-8"),
            file_name="picks.json", mime="application/json",
            use_container_width=True,
        )
