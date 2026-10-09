"""Per-stock capital flow: Eastmoney push2 fflow (daily main/super net inflow,
20-day history). Feeds the quant screen's 资金因子 and deep-analysis bundles.
"""

from __future__ import annotations

import logging

from .quotes import em_secid
from .http import em_get

logger = logging.getLogger(__name__)


def stock_fund_flow(code: str, days: int = 20) -> dict:
    """Daily main-capital net flow for ``code``, last ``days`` days (亿元).

    Returns {today: {...}, history: [...], sum_5d, sum_20d} — the 5/20-day
    cumulative direction is the harder capital evidence: 嘴上说什么，不如钱往哪走.
    """
    empty = {"history": [], "sum_5d": None, "sum_20d": None, "today": None}
    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get",
            params={
                "secid": em_secid(code),
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                "klt": "101",
                "lmt": str(days),
            },
            timeout=15,
        )
        klines = (r.json().get("data") or {}).get("klines") or []
        if not klines:
            return empty
        history = []
        for line in klines:
            p = line.split(",")
            if len(p) < 6:
                continue
            history.append({
                "date": p[0],
                "main_net_yi": round(float(p[1]) / 1e8, 3),
                "super_net_yi": round(float(p[5]) / 1e8, 3),
            })
        def _sum(n: int) -> float | None:
            tail = history[-n:]
            return round(sum(h["main_net_yi"] for h in tail), 2) if tail else None
        return {
            "history": history,
            "sum_5d": _sum(5),
            "sum_20d": _sum(20),
            "today": history[-1] if history else None,
        }
    except Exception as e:
        logger.warning("Stock fund flow failed for %s: %s", code, e)
        return empty
