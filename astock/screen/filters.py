r"""Hard filters for the quant screen (Stage 2).

A-share specific exclusions, mirroring the risk rules from
tradingagents-astock's prompts:

- **ST / \*ST**: regulatory warning, restricted buyer pool, often forced selling
- **次新**: no price limit first 5 days, unstable data (configurable, default
  60 listed days)
- **停牌**: no price → dropped upstream in fetch_universe
- **面值退市风险**: price below min_price (default 2元)
- **流动性**: amount below min_amount (default 1亿)
- **追高风险**: already limit-up today — chasing means buying at the day's
  peak with T+1 lock (configurable)
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_ST_RE = re.compile(r"(ST|\*ST|退)", re.IGNORECASE)


def apply_hard_filters(df: pd.DataFrame, min_price: float, min_amount: float,
                       min_mcap: float, max_mcap: float, min_listed_days: int,
                       exclude_st: bool, exclude_new_high: bool) -> pd.DataFrame:
    """Return only rows passing every hard filter. Excluded counts are logged
    (never silent — a broken filter should be visible in the run report)."""
    total = len(df)
    mask = pd.Series(True, index=df.index)

    if exclude_st:
        mask &= ~df["name"].astype(str).str.upper().str.contains("ST", na=False)
        mask &= ~df["name"].astype(str).str.contains("退", na=False)

    # 次新：listed_days == -1 表示上市日期解析失败——不当作次新（宁可放行，
    # 后续深度分析会再核对），由 LLM 分析兜底。
    if min_listed_days > 0:
        mask &= (df["listed_days"] >= min_listed_days) | (df["listed_days"] == -1)

    mask &= df["price"] >= min_price
    mask &= df["amount_yi"] >= min_amount
    mask &= df["mcap_yi"] >= min_mcap
    if max_mcap > 0:
        mask &= df["mcap_yi"] <= max_mcap

    # 追高风险：当日已涨停（涨跌幅 ≥ 9.5% 主板 / ≥ 19.5% 创科 / ≥ 29.5% 北交）
    if exclude_new_high:
        def _is_limit_up(row) -> bool:
            chg = row["change_pct"]
            if chg is None or pd.isna(chg):
                return False
            code = str(row["code"])
            if code.startswith(("68", "30")):
                return chg >= 19.5
            if code.startswith(("8", "4", "92")):
                return chg >= 29.5
            return chg >= 9.5
        mask &= ~df.apply(_is_limit_up, axis=1)

    out = df[mask].copy()
    logger.info(
        "Hard filters: %d/%d passed (ST=%s, 次新>=%dd, 价>=%.1f, 额>=%.1f亿, "
        "市值>=%.0f亿, 追高排除=%s)",
        len(out), total, exclude_st, min_listed_days, min_price, min_amount,
        min_mcap, exclude_new_high,
    )
    return out
