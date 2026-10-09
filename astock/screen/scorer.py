"""Multi-factor scoring for the quant screen (Stage 2, zero LLM calls).

Five factor groups, derived from the tradingagents-astock strategy's
priorities (动量在价先、量价验证、资金流向、板块热度、估值纪律):

- **momentum**: 5/20日涨幅、距20日高点（强势特征，不追已涨停）
- **volume_price**: 量比、换手率适中（散户市里换手率是资金活跃度关键指标）
- **capital**: 当日主力净流入占比（大单是主力动向最硬的证据）
- **heat**: 涨停题材榜匹配、所属板块当日涨幅（政策市 + 板块轮动）
- **valuation**: PE 合理区间（A股 30-50x 常态，不照搬美股标准）

Weights come from the Stage-1 market regime (LLM): a 防守型 regime boosts
valuation/capital, an 进攻型 regime boosts momentum/heat. Bad/missing data
scores 0 — never a bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeWeights:
    """Factor weights, adjusted by the market regime from Stage 1."""
    momentum: float = 0.30
    volume_price: float = 0.15
    capital: float = 0.25
    heat: float = 0.20
    valuation: float = 0.10


# Regime keyword → weight profile. Stage 1's LLM verdict supplies the keywords;
# unknown text keeps the neutral default (never silently re-weighted).
_REGIME_PROFILES = {
    "进攻": RegimeWeights(momentum=0.35, volume_price=0.15, capital=0.20, heat=0.25, valuation=0.05),
    "牛市": RegimeWeights(momentum=0.35, volume_price=0.15, capital=0.20, heat=0.25, valuation=0.05),
    "乐观": RegimeWeights(momentum=0.30, volume_price=0.15, capital=0.25, heat=0.20, valuation=0.10),
    "震荡": RegimeWeights(),  # neutral
    "中性": RegimeWeights(),
    "防守": RegimeWeights(momentum=0.15, volume_price=0.15, capital=0.35, heat=0.10, valuation=0.25),
    "熊市": RegimeWeights(momentum=0.15, volume_price=0.15, capital=0.35, heat=0.10, valuation=0.25),
    "悲观": RegimeWeights(momentum=0.15, volume_price=0.15, capital=0.35, heat=0.10, valuation=0.25),
}


def weights_for_regime(regime_text: str) -> RegimeWeights:
    for keyword, w in _REGIME_PROFILES.items():
        if keyword in (regime_text or ""):
            return w
    return RegimeWeights()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def score_momentum(row: pd.Series) -> float:
    """0-1 momentum score: 20d trend strength without chasing extremes."""
    s = 0.0
    chg20 = row.get("chg_20d")
    if chg20 is not None and pd.notna(chg20):
        # 0%→0分, 10%+→满分, 负趋势按比例扣
        s += _clamp01(chg20 / 10.0) * 0.6
    chg5 = row.get("chg_5d")
    if chg5 is not None and pd.notna(chg5):
        s += _clamp01(chg5 / 5.0) * 0.3
    dist_hi = row.get("dist_to_20d_high")
    if dist_hi is not None and pd.notna(dist_hi):
        # 距高点越近越强（但已涨停的早在过滤阶段排除）
        s += _clamp01(1.0 + dist_hi / 10.0) * 0.1
    return min(1.0, s)


def score_volume_price(row: pd.Series) -> float:
    """0-1 volume/price score: moderate turnover + volume confirmation."""
    s = 0.0
    vr = row.get("volume_ratio")
    if vr is not None and pd.notna(vr):
        # 量比 1~3 为佳（温和放量）；>5 是异常（要么涨停要么出货）
        if 1.0 <= vr <= 3.0:
            s += 0.5
        elif 0.8 <= vr < 1.0 or 3.0 < vr <= 5.0:
            s += 0.3
    tr = row.get("turnover_pct")
    if tr is not None and pd.notna(tr):
        # 换手 3%~15% 活跃；>25% 筹码松动过度
        if 3.0 <= tr <= 15.0:
            s += 0.5
        elif 1.5 <= tr < 3.0 or 15.0 < tr <= 25.0:
            s += 0.3
    return min(1.0, s)


def score_capital(row: pd.Series) -> float:
    """0-1 capital score: today's main net inflow relative to amount."""
    inflow = row.get("main_inflow_yi")
    amount = row.get("amount_yi")
    if inflow is None or pd.isna(inflow) or not amount:
        return 0.0
    ratio = inflow / amount  # 主力净流入占成交额比
    # >10% 强吸筹, 0~10% 按比例, 流出=0
    return _clamp01(ratio / 0.10)


def score_heat(row: pd.Series, hot_codes: set[str], hot_boards: dict[str, float]) -> float:
    """0-1 heat score: hot-stock list membership + board strength."""
    s = 0.0
    if row.get("code") in hot_codes:
        s += 0.5  # 上榜（涨停+题材归因）
    board_pct = hot_boards.get(str(row.get("code")), None)
    if board_pct is not None and board_pct > 0:
        s += _clamp01(board_pct / 5.0) * 0.5  # 所属板块当日涨幅
    return min(1.0, s)


def score_valuation(row: pd.Series) -> float:
    """0-1 valuation score: PE in a reasonable A-share range, PB sanity.

    A股整体 PE 中位数偏高（30-50x 为常态），不照搬美股 15-25x；亏损股（PE<0）
    不打分。20x 以下的低估值蓝筹也加分（但需后续基本面核对）。
    """
    pe = row.get("pe")
    if pe is None or pd.isna(pe):
        return 0.0
    if pe <= 0:
        return 0.0  # 亏损股交给 LLM 深度分析判断
    if 10.0 <= pe <= 45.0:
        return 1.0
    if 45.0 < pe <= 70.0:
        return 0.5
    return 0.2  # >70x：高估值，仅存少量分


def score_universe(df: pd.DataFrame, hot: list[dict], concept_boards: list[dict],
                   weights: RegimeWeights) -> pd.DataFrame:
    """Score every row, add ``score`` column, sort desc. Requires chg_20d/
    dist_to_20d_high — the caller computes momentum_summary per row first
    (fetching K-lines) and merges those columns in."""
    hot_codes = {h["code"] for h in hot if h.get("code")}
    # 板块涨幅映射到个股：用领涨股匹配（近似——精确匹配需额外接口）
    hot_boards = {b.get("leader", ""): b.get("change_pct", 0.0) for b in concept_boards}

    def _total(row) -> float:
        return (
            score_momentum(row) * weights.momentum
            + score_volume_price(row) * weights.volume_price
            + score_capital(row) * weights.capital
            + score_heat(row, hot_codes, hot_boards) * weights.heat
            + score_valuation(row) * weights.valuation
        )

    df = df.copy()
    df["score"] = df.apply(_total, axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
