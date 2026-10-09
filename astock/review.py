"""Post-run review: compare the previous run's picks against how they
actually performed, and distill the comparison into a new strategy version.

    1. locate the previous run (latest report dir)
    2. read its picks.json + pull current quotes for those codes (1 batch call)
    3. code-computed outcome per pick (已达目标/跌破止损/回落到买入区间…)
    4. one LLM call: review summary → strategy improvements (structured JSON)
    5. save strategies/vN.json (version = latest + 1)

Failures degrade per-item and are logged — the review never blocks a
finished run. 首次运行（无上一次 run）不生成策略版本。

⚠️ 复盘由 AI 自动生成，仅供研究参考，不构成任何投资建议。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .data import tencent_quotes
from .strategy import StrategyVersion, latest_version, save_strategy

logger = logging.getLogger(__name__)


def _latest_run(reports_root: str) -> Path | None:
    """Latest report dir under ``reports_root`` (name = YYYY-MM-DD_HHMMSS,
    so name order is time order). None when there is none."""
    root = Path(reports_root)
    if not root.is_dir():
        return None
    runs = sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "picks.json").exists())
    return runs[-1] if runs else None


def _outcome(pick: dict, quote: dict | None) -> str:
    """Code-computed verdict on one pick's performance vs its price plan:
    current price vs 止损/目标/买入区间. Missing data is labeled, never
    guessed."""
    if not quote or not float(quote.get("price", 0)):
        return "[数据缺失]"
    price = float(quote["price"])
    parts = [f"现价 {price}", f"当日涨跌 {quote.get('change_pct', 0)}%"]
    pp = pick.get("price_plan") or {}
    stop = float(pp.get("stop_loss") or 0)
    target_low = float(pp.get("target_low") or 0)
    entry_low = float(pp.get("entry_low") or 0)
    entry_high = float(pp.get("entry_high") or 0)
    if stop and price <= stop:
        parts.append("已跌破止损位")
    elif target_low and price >= target_low:
        parts.append("已进入卖出目标区间")
    elif entry_low and price < entry_low:
        parts.append("回落至买入区间下方")
    elif entry_high and price <= entry_high:
        parts.append("处于建议买入区间内")
    elif entry_high:
        parts.append("高于买入区间上沿（未追入）")
    return "，".join(parts)


def run_review(cfg, llm, reports_root: str) -> StrategyVersion | None:
    """Review the previous run → new strategy version. Returns None when
    there is nothing to review (no previous run) or the review fails."""
    prev = _latest_run(reports_root)
    if prev is None:
        logger.info("无上一次 run（首次运行），跳过复盘——不生成策略版本")
        return None

    try:
        raw = json.loads((prev / "picks.json").read_text(encoding="utf-8"))
        prev_picks = raw.get("picks") or []
    except Exception as e:
        logger.warning("读取上次报告失败 %s: %s，跳过复盘", prev, e)
        return None
    if not prev_picks:
        logger.info("上次 run（%s）无选股，跳过复盘", prev.name)
        return None

    # --- 当前表现（1 次腾讯批量调用；失败标注 [数据缺失]）----------------
    codes = [str(p.get("code", "")) for p in prev_picks if p.get("code")]
    try:
        quotes = tencent_quotes(codes) if codes else {}
    except Exception as e:
        logger.warning("复盘拉取行情失败: %s（按数据缺失处理）", e)
        quotes = {}

    rows = []
    for p in prev_picks:
        code = str(p.get("code", ""))
        quote = quotes.get(code)
        pp = p.get("price_plan") or {}
        then_close = float((pp.get("anchors") or {}).get("close") or 0)
        now_price = float((quote or {}).get("price") or 0)
        chg_since = (round((now_price / then_close - 1) * 100, 2)
                     if then_close and now_price else None)
        reason = str(p.get("buy_logic", "") or "")
        cats = p.get("catalysts") or []
        if cats:
            reason += "；催化剂：" + "；".join(str(c) for c in cats)
        rows.append({
            "code": code,
            "name": p.get("name", ""),
            "当时评分": p.get("score", ""),
            "当时结论": p.get("verdict", ""),
            "当时收盘价": then_close or "",
            "当前价": now_price or "",
            "累计涨跌%": chg_since,
            "当时买入区间": _fmt_zone(p),
            "当时止损": pp.get("stop_loss", ""),
            "当时卖出目标": _fmt_target(p),
            "当前表现": _outcome(p, quote),
            "当时选股原因": reason[:200],
        })

    system = (
        "你是一位 A 股策略复盘分析师。现在复盘上一次选股 run 的结果：对比当时的选股判断"
        "与这些股票的当前实际表现，总结策略哪里准、哪里偏，并把改进落成新的策略参数。\n\n"
        "复盘要点：\n"
        "1. **判断准确度**：当时评为值得买入/可关注的股票现在表现如何（已达目标=判断准，"
        "跌破止损=判断错，未入买入区间=入场时机判断偏早）\n"
        "2. **共性问题**：表现差的选股有哪些共同特征（如追高、题材热度权重过高、资金流向"
        "判断不准、大盘环境判断偏乐观/悲观）\n"
        "3. **改进落点**：把教训落成可执行的策略参数——因子权重调整（乘数 0.5~2.0）与"
        "各阶段提示词校准\n\n"
        "输出**只包含一个 JSON 对象**：\n"
        '{"review_summary": "复盘结论（150字内：上次选股实际表现、判断准确度、共性问题）", '
        '"regime_calibration": "大盘体检的校准提示（下次体检时参考，100字内；没有偏差就写 保持现有判断框架）", '
        '"triage_criteria": "快筛评分标准的补充要求（100字内；如 更重视资金流向真实性，题材热度降权）", '
        '"deep_focus": "深度分析的补充关注点（100字内；如 重点核查解禁压力与主力资金背离）", '
        '"pricing_discipline": "定价纪律的补充要求（80字内；没有就写空字符串）", '
        '"stock_reviews": [{"code": "股票代码", "change_reason": "该股从当时收盘价到当前价的变化原因（60字内；结合当时选股逻辑与实际走势，如题材发酵/利好兑现/大盘拖累/资金撤离）"}], '
        '"factor_multipliers": {"momentum": 0.5-2.0, "volume_price": 0.5-2.0, "capital": 0.5-2.0, "heat": 0.5-2.0, "valuation": 0.5-2.0}, '
        '"notes": ["策略改进点1", "改进点2（每条40字内）"]}'
    )
    user = f"# 上次选股 run（{prev.name}）\n# 各股当前表现\n"
    for r in rows:
        user += json.dumps(r, ensure_ascii=False) + "\n"
    user += f"\n# 当前日期\n{datetime.now():%Y-%m-%d}\n"

    raw = llm.chat_json(system, user, max_tokens=3000, temperature=0.2)
    if isinstance(raw, list):  # 某些模型把对象包进数组——取第一个 dict
        raw = next((x for x in raw if isinstance(x, dict)), {})
    if not raw:
        logger.warning("复盘 LLM 输出为空，跳过策略版本生成")
        return None

    # 逐只价格变化原因（LLM 结合当时选股逻辑与实际走势解释）
    change_reasons = {
        str(r.get("code", "")): str(r.get("change_reason", ""))
        for r in raw.get("stock_reviews", []) or [] if isinstance(r, dict)
    }
    for r in rows:
        r["变化原因"] = change_reasons.get(r["code"], "（未给出）")

    sv = StrategyVersion(
        version=latest_version() + 1,
        created=f"{datetime.now():%Y-%m-%d}",
        based_on=prev.name,
        review_summary=str(raw.get("review_summary", "")),
        regime_calibration=str(raw.get("regime_calibration", "")),
        triage_criteria=str(raw.get("triage_criteria", "")),
        deep_focus=str(raw.get("deep_focus", "")),
        pricing_discipline=str(raw.get("pricing_discipline", "")),
        factor_multipliers=dict(raw.get("factor_multipliers", {}) or {}),
        notes=[str(n) for n in raw.get("notes", []) if n],
        review_rows=rows,
    )
    save_strategy(sv)
    logger.info("复盘完成: v%d（based_on=%s）", sv.version, sv.based_on)
    return sv


def _fmt_zone(pick: dict) -> str:
    pp = pick.get("price_plan") or {}
    lo, hi = pp.get("entry_low"), pp.get("entry_high")
    if lo and hi:
        return f"{lo}~{hi}"
    return "[数据缺失]"


def _fmt_target(pick: dict) -> str:
    pp = pick.get("price_plan") or {}
    lo, hi = pp.get("target_low"), pp.get("target_high")
    if lo and hi:
        return f"{lo}~{hi}"
    return "[数据缺失]"
