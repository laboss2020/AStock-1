"""AStock data layer — pure HTTP, zero TCP, zero third-party data libs.

Sources (all free, direct HTTP):
- 东方财富 push2/push2his/datacenter: 行情、K线、资金流、板块、清单
- 腾讯财经 (qt.gtimg.cn, GBK): 实时估值快照
- 新浪财经: K线 fallback
- 同花顺 10jqka: 涨停热股题材归因、北向资金
- 财联社 cls.cn: 财经快讯

All Eastmoney calls funnel through :func:`.http.em_get` (serial throttle,
anti-ban). All requests share a retrying Session.
"""

from .http import SESSION, em_get, get_json, get_text, market_today
from .quotes import tencent_quotes, em_realtime_quote
from .kline import get_kline, get_index_kline
from .indicators import compute_indicators, latest_values, momentum_summary
from .universe import fetch_universe
from .market import (
    index_snapshot, index_trend_summary, market_fund_flow,
    northbound_realtime, save_northbound_snapshot, northbound_history,
)
from .boards import sector_boards, hot_stocks, theme_frequency
from .capital import stock_fund_flow
from .news import market_news, cls_news, stock_news

__all__ = [
    "SESSION", "em_get", "get_json", "get_text", "market_today",
    "tencent_quotes", "em_realtime_quote",
    "get_kline", "get_index_kline",
    "compute_indicators", "latest_values", "momentum_summary",
    "fetch_universe",
    "index_snapshot", "index_trend_summary", "market_fund_flow", "northbound_realtime",
    "save_northbound_snapshot", "northbound_history",
    "sector_boards", "hot_stocks", "theme_frequency",
    "stock_fund_flow",
    "market_news", "cls_news", "stock_news",
]
