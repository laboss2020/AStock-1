"""Pricing: suggested buy/sell price levels (Stage 5).

Code-computed technical anchors (support/resistance from MAs, Bollinger
bands, ATR, swing highs/lows) + one LLM call that reasons over them, then a
deterministic validation pass that fixes violated number relationships with
rule-based fallbacks — the schema alone cannot stop a model from producing
stop_loss > entry_low, so code checks it.

⚠️ 本模块产出的价位由 AI 自动生成，仅供研究参考，不构成任何投资建议。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .analysis import ASTOCK_RULES
from .config import Config
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class PricePlan:
    """Suggested price levels for one stock. All prices are raw CNY."""
    entry_low: float = 0.0      # 建议买入区间下沿
    entry_high: float = 0.0     # 建议买入区间上沿
    stop_loss: float = 0.0      # 止损价
    target_low: float = 0.0     # 卖出目标区间下沿
    target_high: float = 0.0    # 卖出目标区间上沿
    holding_period: str = ""    # 建议持有周期
    position_pct: str = ""      # 建议仓位（占单只个股总资金）
    rationale: str = ""
    anchors: dict = field(default_factory=dict)  # 代码计算的技术锚点
    fallback: bool = False      # True = LLM 数值关系不合法，走了规则兜底


def compute_anchors(df) -> dict:
    """Technical anchors from the K-line frame — the LLM reasons over these.

    Anchors: MAs (5/10/20/60), Bollinger bands, ATR14, recent swing high/low
    (20d & 60d), and the latest close. All rounded to 2 decimals (tick size).
    """
    from .data.indicators import compute_indicators

    if df is None or df.empty:
        return {}
    ind = compute_indicators(df)
    close = df["Close"]
    last = float(close.iloc[-1])

    def _g(col: str) -> float | None:
        s = ind[col].dropna() if col in ind.columns else None
        return round(float(s.iloc[-1]), 2) if s is not None and len(s) else None

    hi20 = round(float(close.tail(20).max()), 2)
    lo20 = round(float(close.tail(20).min()), 2)
    hi60 = round(float(close.tail(60).max()), 2) if len(close) >= 60 else hi20
    lo60 = round(float(close.tail(60).min()), 2) if len(close) >= 60 else lo20

    return {
        "close": round(last, 2),
        "ma5": _g("close_5_sma"),
        "ma10": _g("close_10_sma"),
        "ma20": _g("close_20_sma"),
        "ma60": _g("close_60_sma"),
        "boll_ub": _g("boll_ub"),
        "boll_lb": _g("boll_lb"),
        "atr14": _g("atr"),
        "recent_high_20d": hi20,
        "recent_low_20d": lo20,
        "recent_high_60d": hi60,
        "recent_low_60d": lo60,
    }


def _rule_fallback(anchors: dict) -> PricePlan:
    """Deterministic fallback when the LLM violates number relationships.

    Pullback entry at MA20, stop 2×ATR below entry, target 1.15-1.25× entry —
    a conservative default anchored in the same technical levels.
    """
    ma20 = anchors.get("ma20") or anchors.get("close", 0)
    atr = anchors.get("atr14") or (ma20 * 0.03 if ma20 else 0)
    entry_low = round(ma20 * 0.98, 2)
    entry_high = round(ma20 * 1.02, 2)
    stop = round(entry_low - 2 * atr, 2)
    target_low = round(entry_high * 1.10, 2)
    target_high = round(entry_high * 1.25, 2)
    return PricePlan(
        entry_low=entry_low, entry_high=entry_high, stop_loss=stop,
        target_low=target_low, target_high=target_high,
        holding_period="2-4周（规则兜底）",
        position_pct="≤5%（规则兜底）",
        rationale="LLM 定价数值关系不合法，已回退到规则兜底：MA20 回调买入 + 2×ATR 止损 + 10-25% 目标。",
        anchors=anchors,
        fallback=True,
    )


def _validate(plan: PricePlan, anchors: dict) -> bool:
    """Number-relationship check: stop < entry_low ≤ entry_high < target_low
    ≤ target_high, all positive, entry zone within ±15% of current price
    (a much-wider entry zone would mean the model ignored the anchors)."""
    c = anchors.get("close") or plan.entry_high
    if not (plan.stop_loss > 0 and plan.entry_low > 0 and plan.entry_high > 0
            and plan.target_low > 0 and plan.target_high > 0):
        return False
    if not (plan.stop_loss < plan.entry_low <= plan.entry_high):
        return False
    if not (plan.entry_high < plan.target_low <= plan.target_high):
        return False
    # entry zone must bracket a realistic buy area near the current price
    if plan.entry_high > c * 1.15 or plan.entry_low < c * 0.70:
        return False
    return True


def suggest_prices(cfg: Config, llm: LLMClient, bundle: dict,
                   verdict, df, strategy_text: str = "") -> PricePlan:
    """Stage 5: one price plan per final pick."""
    anchors = compute_anchors(df)
    if not anchors:
        return PricePlan(rationale="K线数据缺失，无法定价", fallback=True)

    code = bundle.get("code", "")
    name = bundle.get("name", "")
    system = (
        f"你是一位 A 股交易定价分析师。基于深度分析结论和技术锚点，为 {code} {name} "
        "给出建议的买入价格区间、止损价和卖出目标区间。\n\n"
        + ASTOCK_RULES
        + (f"\n\n{strategy_text}" if strategy_text else "")
        + "\n\n定价纪律：\n"
        "1. **锚定技术位**：买入区间应落在支撑位附近（MA20/布林下轨/近期低点的聚类区域）；"
        "突破买入策略的买入区间可略高于现价，但不得超过现价的 115%\n"
        "2. **止损**：止损价必须低于买入区间下沿，常用锚点为买入区间下沿再减 2×ATR，"
        "或跌破关键支撑位（如 MA60/近期低点）\n"
        "3. **卖出目标**：目标区间必须高于买入区间上沿；常用锚点为近期高点/布林上轨；"
        "风险收益比（目标价差/止损价差）应 ≥ 2:1\n"
        "4. **仓位**：结合波动率（ATR）和评级强度给出，单只不超过总资金的 20%\n\n"
        "输出**只包含一个 JSON 对象**（所有价格为人民币现价，保留两位小数）：\n"
        '{"entry_low": 建议买入区间下沿, "entry_high": 建议买入区间上沿, '
        '"stop_loss": 止损价, "target_low": 卖出目标下沿, "target_high": 卖出目标上沿, '
        '"holding_period": "建议持有周期，如 2-4周/1-3个月", '
        '"position_pct": "建议仓位，如 ≤10%", "rationale": "定价依据（120字内）"}'
    )

    user_parts = [
        "# 技术锚点（代码计算，含最新收盘价）\n",
        json.dumps(anchors, ensure_ascii=False, indent=1),
        f"\n# 当前行情快照\n最新收盘价: {anchors['close']}\n",
        "\n# 深度分析结论\n",
        f"评分: {verdict.score}  结论: {verdict.verdict}\n买入逻辑: {verdict.buy_logic}\n"
        f"风险: {'; '.join(verdict.risks) or '无'}",
    ]
    # max_tokens 3000：推理模型（glm/deepseek thinkers）需要推理余量——1500
    # 会反复触发空 content 提额重试（见 llm.py）。
    raw = llm.chat_json(system, "\n".join(user_parts), max_tokens=3000, temperature=0.2)

    def _f(key: str) -> float:
        try:
            return round(float(raw.get(key, 0)), 2)
        except (ValueError, TypeError):
            return 0.0

    plan = PricePlan(
        entry_low=_f("entry_low"), entry_high=_f("entry_high"),
        stop_loss=_f("stop_loss"),
        target_low=_f("target_low"), target_high=_f("target_high"),
        holding_period=str(raw.get("holding_period", "")),
        position_pct=str(raw.get("position_pct", "")),
        rationale=str(raw.get("rationale", "")),
        anchors=anchors,
    )
    if not _validate(plan, anchors):
        logger.warning(
            "LLM 定价数值关系不合法（%s: stop=%s entry=[%s,%s] target=[%s,%s] close=%s），回退规则兜底",
            code, plan.stop_loss, plan.entry_low, plan.entry_high,
            plan.target_low, plan.target_high, anchors["close"],
        )
        return _rule_fallback(anchors)
    return plan
