"""Report generation: Markdown per run.

One directory per run under ``output_dir``: ``report.md`` (human-readable
full report), ``picks.json`` (machine-readable final picks with price
plans), ``run_meta.json`` (stage counters and errors).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import Config
from .pipeline import RunResult

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "> ⚠️ **免责声明**：本报告全部内容由 AI 自动生成，仅供学习研究与技术演示，"
    "**不构成任何投资建议**。所列价格、评级、仓位均为算法输出，可能存在错误或偏差。"
    "投资决策请咨询持有中国证监会颁发资质的专业机构。股市有风险，投资需谨慎。"
)


def render_report(result: RunResult) -> str:
    """Render the run's full Markdown report."""
    parts = [
        f"# A股大盘选股报告 — {result.trade_date}",
        "",
        _DISCLAIMER,
        "",
    ]

    # --- 大盘体检 ---
    r = result.regime
    parts += [
        "## 一、大盘体检",
        "",
        f"- **市场环境**：{r.regime}（温度：{r.temperature}）",
        f"- **风格偏好**：{r.style or '无明确偏好'}",
        f"- **风险偏好**：{r.risk_appetite}",
        f"- **建议关注板块**：{'、'.join(r.sector_focus) if r.sector_focus else '无'}",
        f"- **建议回避板块**：{'、'.join(r.avoid_sectors) if r.avoid_sectors else '无'}",
        f"- **判断依据**：{r.rationale or '（无）'}",
        "",
    ]

    # --- 筛选漏斗 ---
    parts += [
        "## 二、筛选漏斗",
        "",
        f"- 全市场股票：**{result.universe_count}** 只",
        f"- 硬性过滤后（剔 ST/次新/停牌/低价/低流动性/追高）：**{result.filtered_count}** 只",
        f"- 量化多因子初筛候选：**{result.candidate_count}** 只",
        f"- LLM 快筛通过：**{result.triaged_count}** 只（放弃 {result.rejected_count} 只）",
        f"- 深度分析 + 定价 → **最终选股 {len(result.picks)}** 支",
        f"- LLM 调用次数：{result.llm_calls}",
        f"- 策略版本：{'v' + str(result.strategy_version_used) if result.strategy_version_used else '默认（无复盘版本）'}",
        "",
    ]

    # --- 最终选股 ---
    parts += ["## 三、最终选股（按综合评分排序）", ""]
    if result.picks:
        parts += [
            "| # | 代码 | 名称 | 现价 | 买入区间 | 止损价 | 卖出目标 | 持有周期 | 评分 | 结论 |",
            "|---|------|------|------|----------|--------|----------|----------|------|------|",
        ]
        for i, p in enumerate(result.picks, 1):
            pl = p.price_plan
            parts.append(
                f"| {i} | {p.code} | {p.name} | {pl.anchors.get('close', '-')} "
                f"| {pl.entry_low}~{pl.entry_high} | {pl.stop_loss} "
                f"| {pl.target_low}~{pl.target_high} "
                f"| {pl.holding_period} | {p.verdict.score} | {p.verdict.verdict} |"
            )
        parts.append("")
        parts += ["---", ""]
        for i, p in enumerate(result.picks, 1):
            parts += _render_pick_detail(i, p)
    else:
        parts += ["今日无符合策略的选股（市场环境或个股证据不足）。", ""]

    # --- 候选池参考 ---
    if result.screened_out:
        parts += [
            "## 四、量化初筛候选池（前 20，供参考）",
            "",
            "| 代码 | 名称 | 现价 | 涨跌幅 | 换手率 | 量化得分 |",
            "|------|------|------|--------|--------|----------|",
        ]
        for c in result.screened_out[:20]:
            parts.append(
                f"| {c.get('code','')} | {c.get('name','')} | {c.get('price','')} "
                f"| {c.get('change_pct','')}% | {c.get('turnover_pct','')}% "
                f"| {round(c.get('score', 0) or 0, 3)} |"
            )
        parts.append("")

    # --- 复盘（选股之后，复盘上一次 run → 新策略版本）--------------------
    if result.review:
        rv = result.review
        parts += ["## 五、复盘（上一次 run）", ""]
        parts.append(f"- **复盘对象**：{rv.get('based_on', '（无）')}")
        parts.append(f"- **新策略版本**：v{rv.get('new_version', '?')}（下次运行默认使用）")
        if rv.get("summary"):
            parts.append(f"- **复盘结论**：{rv['summary']}")
        for n in rv.get("notes", []):
            parts.append(f"- 改进点：{n}")

        rows = rv.get("rows") or []
        if rows:
            # 股价前后对比表（当时收盘 vs 现价 vs 当时挂的价位）
            parts += ["", "### 复盘明细：上一次选股的前后对比", ""]
            parts += [
                "| 代码 | 名称 | 当时收盘 | 现价 | 累计涨跌 | 当时买入区间 | 当时止损 | 当时卖出目标 | 当前状态 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in rows:
                chg = r.get("累计涨跌%")
                parts.append(
                    f"| {r.get('code','')} | {r.get('name','')} "
                    f"| {r.get('当时收盘价') or '-'} | {r.get('当前价') or '-'} "
                    f"| {'+' + str(chg) if isinstance(chg, (int, float)) and chg > 0 else (chg if chg is not None else '-')}% "
                    f"| {r.get('当时买入区间')} | {r.get('当时止损')} | {r.get('当时卖出目标')} "
                    f"| {r.get('当前表现')} |"
                )
            # 逐只：当时选股原因 + 价格变化原因
            parts += ["", "### 逐只复盘：当时选股原因与价格变化原因", ""]
            for r in rows:
                parts.append(
                    f"- **{r.get('code','')} {r.get('name','')}**（{r.get('当时结论','')}，"
                    f"{r.get('当时评分','')} 分）"
                )
                parts.append(f"  - 当时选股原因：{r.get('当时选股原因') or '（无）'}")
                parts.append(f"  - 价格变化原因：{r.get('变化原因') or '（未给出）'}")
            parts.append("")
    elif result.strategy_version_used == 0:
        parts += ["## 五、复盘", "", "- 无上一次 run（首次运行），未生成策略版本。", ""]

    # --- 错误与缺失 ---
    if result.errors:
        parts += [
            "## 六、本次运行的错误与缺失",
            "",
        ]
        for e in result.errors:
            parts.append(f"- {e}")
        parts.append("")

    return "\n".join(parts)


def _render_pick_detail(i: int, p) -> list[str]:
    pl = p.price_plan
    v = p.verdict
    parts = [
        f"### {i}. {p.code} {p.name} — {v.verdict}（{v.score} 分）",
        "",
        "**定价建议**：",
        "",
        f"- 建议买入区间：**{pl.entry_low} ~ {pl.entry_high}**",
        f"- 止损价：**{pl.stop_loss}**",
        f"- 卖出目标区间：**{pl.target_low} ~ {pl.target_high}**",
        f"- 建议持有周期：{pl.holding_period or '未给出'}",
        f"- 建议仓位：{pl.position_pct or '未给出'}",
        f"- 定价依据：{pl.rationale or '（无）'}" + ("（⚠️ 数值关系不合法，已回退规则兜底）" if pl.fallback else ""),
        "",
        "**技术锚点**（代码计算）：",
        "",
    ]
    anchors = pl.anchors or {}
    if anchors:
        parts.append(
            f"MA5={anchors.get('ma5')} MA10={anchors.get('ma10')} "
            f"MA20={anchors.get('ma20')} MA60={anchors.get('ma60')} | "
            f"BOLL上轨={anchors.get('boll_ub')} 下轨={anchors.get('boll_lb')} | "
            f"ATR14={anchors.get('atr14')} | 20日高/低={anchors.get('recent_high_20d')}"
            f"/{anchors.get('recent_low_20d')} | 60日高/低={anchors.get('recent_high_60d')}"
            f"/{anchors.get('recent_low_60d')}"
        )
    parts += [
        "",
        "**买入逻辑**：" + (v.buy_logic or "（无）"),
        "",
    ]
    if v.dimension_summary:
        parts += ["**分维度评价**：", ""]
        for dim, comment in v.dimension_summary.items():
            parts.append(f"- {dim}：{comment}")
        parts.append("")
    if v.catalysts:
        parts += ["**催化因素**：" + "；".join(v.catalysts), ""]
    if v.risks:
        parts += ["**风险提示**：" + "；".join(v.risks), ""]
    parts += ["---", ""]
    return parts


def save_report(cfg: Config, result: RunResult) -> Path:
    """Save report.md + picks.json + run_meta.json under a per-run directory."""
    run_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "report.md").write_text(render_report(result), encoding="utf-8")

    picks_payload = {
        "trade_date": result.trade_date,
        "regime": {
            "regime": result.regime.regime,
            "temperature": result.regime.temperature,
            "style": result.regime.style,
            "risk_appetite": result.regime.risk_appetite,
            "sector_focus": result.regime.sector_focus,
            "avoid_sectors": result.regime.avoid_sectors,
        },
        "picks": [
            {
                "code": p.code,
                "name": p.name,
                "score": p.verdict.score,
                "verdict": p.verdict.verdict,
                "buy_logic": p.verdict.buy_logic,
                "risks": p.verdict.risks,
                "catalysts": p.verdict.catalysts,
                "dimension_summary": p.verdict.dimension_summary,
                "price_plan": {
                    "entry_low": p.price_plan.entry_low,
                    "entry_high": p.price_plan.entry_high,
                    "stop_loss": p.price_plan.stop_loss,
                    "target_low": p.price_plan.target_low,
                    "target_high": p.price_plan.target_high,
                    "holding_period": p.price_plan.holding_period,
                    "position_pct": p.price_plan.position_pct,
                    "rationale": p.price_plan.rationale,
                    "fallback": p.price_plan.fallback,
                    "anchors": p.price_plan.anchors,
                },
            }
            for p in result.picks
        ],
    }
    (run_dir / "picks.json").write_text(
        json.dumps(picks_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    meta = {
        "trade_date": result.trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_count": result.universe_count,
        "filtered_count": result.filtered_count,
        "candidate_count": result.candidate_count,
        "triaged_count": result.triaged_count,
        "rejected_count": result.rejected_count,
        "pick_count": len(result.picks),
        "llm_calls": result.llm_calls,
        "errors": result.errors,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("报告已保存: %s", run_dir)
    return run_dir
