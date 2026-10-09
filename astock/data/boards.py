"""Sector boards & hot stocks: Eastmoney concept/industry boards + 同花顺
limit-up hot stocks with human-curated reason tags (题材归因).

These feed both Stage 1 (板块轮动方向) and the quant screen's 热度因子.
"""

from __future__ import annotations

import logging
from collections import Counter

from .http import em_get, get_json

logger = logging.getLogger(__name__)


def sector_boards(kind: str = "concept", limit: int = 20) -> list[dict]:
    """Top sector boards by change pct. kind: 'concept' (概念) or 'industry' (行业).

    f128 领涨股 f136 领涨股涨跌幅 f62 主力净流入.
    """
    fs = "m:90+t:2" if kind == "concept" else "m:90+t:3"
    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "pn": 1, "pz": limit, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f3", "fs": fs,
                "fields": "f3,f12,f14,f62,f128,f136",
            },
            timeout=15,
        )
        rows = (r.json().get("data") or {}).get("diff") or []
        return [
            {
                "code": row.get("f12", ""),
                "name": row.get("f14", ""),
                "change_pct": row.get("f3", 0.0),
                "main_inflow_yi": round((row.get("f62") or 0) / 1e8, 2),
                "leader": row.get("f128", ""),
                "leader_change": row.get("f136", 0.0),
            }
            for row in rows
        ]
    except Exception as e:
        logger.warning("Sector boards (%s) failed: %s", kind, e)
        return []


def hot_stocks(curr_date: str) -> list[dict]:
    """Limit-up hot stocks with 同花顺 editorial reason tags (题材归因).

    Reason tags like '算力租赁+AI政务' explain WHY a stock surged — the
    screener matches a candidate's theme against these to score 热度.
    """
    try:
        r = get_json(
            f"http://zx.10jqka.com.cn/event/api/getharden/date/{curr_date}"
            "/orderby/date/orderway/desc/charset/GBK/",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0 Safari/537.36",
            },
            timeout=10,
        )
        if r.get("errocode", 0) != 0:
            logger.warning("THS hot stocks API error: %s", r.get("errormsg"))
            return []
        rows = r.get("data") or []
        out = []
        all_tags: list[str] = []
        for row in rows:
            reason = str(row.get("reason", ""))
            out.append({
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "reason": reason,
                "change_pct": row.get("zhangfu", ""),
                "turnover": row.get("huanshou", ""),
                "amount": row.get("chengjiaoe", ""),
            })
            tags = [t.strip() for t in reason.split("+") if t.strip()]
            all_tags.extend(tags)
        return out
    except Exception as e:
        logger.warning("THS hot stocks failed: %s", e)
        return []


def theme_frequency(hot: list[dict], top: int = 20) -> list[tuple[str, int]]:
    """Theme tag frequency across the hot-stock list (top N)."""
    all_tags: list[str] = []
    for h in hot:
        tags = [t.strip() for t in str(h.get("reason", "")).split("+") if t.strip()]
        all_tags.extend(tags)
    return Counter(all_tags).most_common(top)
