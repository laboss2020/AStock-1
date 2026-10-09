"""Technical indicators via stockstats on the K-line frame.

Same indicator set as tradingagents-astock's market analyst toolbox, but
computed directly in code (no LLM tool loop) for the screener pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd
from stockstats import wrap

logger = logging.getLogger(__name__)

INDICATORS = (
    "close_5_sma", "close_10_sma", "close_20_sma", "close_60_sma",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr",
    "vwma",
    "mfi",
)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Wrap an OHLCV frame and compute all supported indicators as columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    ss = wrap(out)
    for ind in INDICATORS:
        try:
            out[ind] = ss[ind]
        except Exception:
            out[ind] = pd.NA
    return out


def latest_values(df: pd.DataFrame, indicators: tuple[str, ...] = INDICATORS) -> dict[str, float | None]:
    """Latest non-NaN value per indicator."""
    out: dict[str, float | None] = {}
    if df is None or df.empty:
        return {ind: None for ind in indicators}
    for ind in indicators:
        if ind in df.columns:
            series = df[ind].dropna()
            out[ind] = round(float(series.iloc[-1]), 3) if len(series) else None
        else:
            out[ind] = None
    return out


def momentum_summary(df: pd.DataFrame) -> dict:
    """Compact momentum/quantity snapshot used by the quant screen and LLM prompts.

    Returns pct changes over 1/5/10/20/60 bars, volume ratios (5d vs 20d avg),
    distance to 20d high/low, and the latest indicator values.
    """
    if df is None or df.empty or len(df) < 5:
        return {}
    close = df["Close"]
    vol = df["Volume"]

    def pct(n: int) -> float | None:
        if len(close) <= n:
            return None
        base = close.iloc[-1 - n]
        return round(float(close.iloc[-1] / base - 1) * 100, 2) if base else None

    vol20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol5 = float(vol.tail(5).mean())
    hi20 = float(close.tail(20).max()) if len(close) >= 20 else float(close.max())
    lo20 = float(close.tail(20).min()) if len(close) >= 20 else float(close.min())

    ind = latest_values(compute_indicators(df))
    last_close = float(close.iloc[-1])
    return {
        "close": last_close,
        "chg_1d": pct(1),
        "chg_5d": pct(5),
        "chg_10d": pct(10),
        "chg_20d": pct(20),
        "chg_60d": pct(60),
        "vol_ratio_5v20": round(vol5 / vol20, 2) if vol20 else None,
        "dist_to_20d_high": round((last_close / hi20 - 1) * 100, 2) if hi20 else None,
        "dist_to_20d_low": round((last_close / lo20 - 1) * 100, 2) if lo20 else None,
        **ind,
    }
