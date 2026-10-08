"""News: Eastmoney np-weblist (rolling wire) + 个股新闻 (search-api) +
财联社 cls.cn global wire. All direct HTTP via the shared session.
"""

from __future__ import annotations

import logging
import time

from .http import SESSION, em_get

logger = logging.getLogger(__name__)


def market_news(limit: int = 20) -> list[dict]:
    """Eastmoney rolling market news (np-weblist)."""
    try:
        r = em_get(
            "https://np-weblist.eastmoney.com/comm/web/getFastNewsList",
            params={
                "client": "web",
                "biz": "web_724",
                "fastColumn": "102",
                "sortEnd": "",
                "pageSize": limit,
                "req_trace": str(int(time.time() * 1000)),
            },
            timeout=10,
        )
        items = ((r.json().get("data") or {}).get("fastNewsList")) or []
        return [
            {"time": it.get("showTime", ""), "title": it.get("title", ""),
             "summary": it.get("summary", "")[:200]}
            for it in items
        ]
    except Exception as e:
        logger.warning("EM market news failed: %s", e)
        return []


def cls_news(limit: int = 20) -> list[dict]:
    """财联社 global wire (cls.cn telegraph). Public endpoint, no key."""
    try:
        r = SESSION.post(
            "https://www.cls.cn/nodeapi/updateTelegraphList",
            json={"app": "CailianpressWeb", "os": "web", "sv": "8.4.6",
                  "rn": limit},
            headers={"Content-Type": "application/json",
                     "Referer": "https://www.cls.cn/telegraph"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        items = ((d.get("data") or {}).get("telegraphList")) or []
        out = []
        for it in items:
            out.append({
                "time": it.get("ctime_text", it.get("display_time", "")),
                "title": it.get("title", ""),
                "summary": (it.get("content") or "")[:200],
            })
        return out
    except Exception as e:
        logger.warning("CLS news failed: %s", e)
        return []


def stock_news(code: str, limit: int = 10) -> list[dict]:
    """Per-stock news via Eastmoney search-api."""
    try:
        r = em_get(
            "https://search-api-web.eastmoney.com/search/jsonp",
            params={
                "cb": "",
                "param": (
                    '{"uid":"","keyword":"' + code + '","type":["cmsArticleWebOld"],'
                    '"client":"web","clientType":"web","clientVersion":"curr",'
                    '"param":{"cmsArticleWebOld":{"searchScope":"default",'
                    '"sort":"default","pageIndex":1,"pageSize":' + str(limit) + '}}}'
                ),
            },
            timeout=10,
        )
        text = r.text.strip()
        # jsonp wrapper when cb non-empty; with empty cb it's raw JSON.
        import json

        d = json.loads(text) if text.startswith("{") else {}
        arts = (((d.get("result") or {}).get("cmsArticleWebOld")) or [])
        return [
            {"date": a.get("date", ""), "title": a.get("title", ""),
             "media": a.get("mediaName", "")}
            for a in arts
        ]
    except Exception as e:
        logger.warning("Stock news failed for %s: %s", code, e)
        return []
