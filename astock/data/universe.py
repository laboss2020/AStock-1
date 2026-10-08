"""Full A-share universe from Eastmoney push2 clist (paged) + Sina fallback.

One paginated API covers all ~5500 A-shares with the fields the quant screen
needs (price, change, turnover, amount, PE, mcap). fs= selects main board
(t:6), SZ SME (t:80), SH STAR (t:2), ChiNext (t:23) — BSE (t:81) excluded by
default: tiny liquidity, 30% daily band, mostly unsuitable for the screener
(set ``include_bse=True`` to add m:0+t:81+s:2048).

Fallback: Sina Market_Center.getHQNodeData (paged, node=hs_a) — same list
without 主力净流入 (scores 0 downstream) and without 上市日期 (次新 filter
passes everything; LLM deep analysis double-checks later).
"""

from __future__ import annotations

import json
import logging
import re
import time as _time

import pandas as pd

from .cache import read_csv_cache, write_csv_cache
from .http import SESSION, em_get

logger = logging.getLogger(__name__)

_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"

_FIELDS = "f2,f3,f5,f6,f8,f9,f10,f12,f14,f15,f16,f20,f23,f26,f62"
# f2 最新价 f3 涨跌幅 f5 成交量(手) f6 成交额 f8 换手率 f9 PE(动) f10 量比
# f12 代码 f14 名称 f15 最高 f16 最低 f20 总市值 f23 PB f26 上市日期 f62 主力净流入

_FS_BASE = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
_FS_BSE = ",m:0+t:81+s:2048"


def fetch_universe(include_bse: bool = False) -> pd.DataFrame:
    """All A-shares as DataFrame[code, name, price, ...]. Empty price rows
    (suspended / new) are dropped here — the quant screen only sees tradable
    names with a real last price.

    Primary: Eastmoney clist. Fallback: Sina Market_Center (when EM is
    banned). Raises RuntimeError when both fail.

    当日缓存（ASTOCK_CACHE=1，默认开）：universe 是盘中快照，按天缓存到
    ``ASTOCK_CACHE_DIR/universe_{date}.csv``——同一天内重复跑直接复用（筛选
    数据本来就是当天的），隔天重拉。读回时 code 列强制 str（CSV 会吃掉
    000001 的前导零）。
    """
    cached = read_csv_cache("universe", f"universe_{pd.Timestamp.now():%Y%m%d}.csv",
                            dtype={"code": str})
    if cached is not None and not cached.empty:
        return cached
    try:
        df = _universe_eastmoney(include_bse)
        if df is not None and not df.empty:
            write_csv_cache(df, "universe", f"universe_{pd.Timestamp.now():%Y%m%d}.csv")
            return df
    except Exception as e:
        # includes RuntimeError("东财 clist 无数据") when EM is banned —
        # both cases fall through to Sina
        logger.warning("EM universe failed, trying Sina: %s", e)
    df = _universe_sina()
    if df is None or df.empty:
        raise RuntimeError(
            "无法获取全市场股票清单（东财 clist 与新浪 Market_Center 均无数据）。"
            "请检查网络，或稍后重试（节假日/非交易时段接口也可能返回空）。"
        )
    write_csv_cache(df, "universe", f"universe_{pd.Timestamp.now():%Y%m%d}.csv")
    return df


def _universe_sina() -> pd.DataFrame | None:
    """Sina Market_Center full A-share list (paged) — EM-ban fallback.

    Returns the same columns as the EM universe. Differences: no 主力净流入
    (main_inflow_yi = NaN → scores 0 downstream, never a bonus), no 上市日期
    (listed_days = -1 → 次新 filter passes, LLM deep analysis double-checks).
    """
    rows: list[dict] = []
    page = 1
    while page <= 80:  # 100/page × 80 — plenty for ~5500 stocks
        try:
            r = SESSION.get(
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/"
                "json_v2.php/Market_Center.getHQNodeData",
                params={"page": page, "num": 100, "sort": "symbol", "asc": 1,
                        "node": "hs_a", "symbol": "", "_s_r_a": "page"},
                timeout=15,
            )
            r.raise_for_status()
            # Sina returns JS with unquoted keys: {symbol:"sh600000",...} —
            # quote the keys, then parse. Values with colons ("15:00:03") are
            # safe: the regex only fires after "{" or ",".
            text = re.sub(r"([{,])(\w+):", r'\1"\2":', r.text)
            batch = json.loads(text)
        except Exception as e:
            logger.warning("Sina universe page %s failed: %s", page, e)
            break
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        _time.sleep(0.2)  # gentle on Sina

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "code": "code", "name": "name", "trade": "price",
        "changepercent": "change_pct", "volume": "volume", "amount": "amount",
        "per": "pe", "pb": "pb", "mktcap": "mcap", "turnoverratio": "turnover_pct",
    })
    for col in ("price", "change_pct", "amount", "turnover_pct", "pe", "pb", "mcap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price", "change_pct"])
    # Sina mktcap unit is 万元 → 亿 = ÷1e4
    df["mcap_yi"] = pd.to_numeric(df["mcap"], errors="coerce") / 1e4
    df["amount_yi"] = df["amount"] / 1e8
    df["main_inflow_yi"] = float("nan")  # Sina list lacks 主力净流入
    df["listed_days"] = -1               # Sina list lacks 上市日期
    df["code"] = df["code"].astype(str)
    logger.info("Sina universe: %d stocks (主力净流入/上市日期缺失，已降级)", len(df))
    return df.reset_index(drop=True)


def _universe_eastmoney(include_bse: bool = False) -> pd.DataFrame:
    fs = _FS_BASE + (_FS_BSE if include_bse else "")
    rows: list[dict] = []
    page = 1
    page_size = 200
    total = None
    while True:
        try:
            r = em_get(
                _CLIST_URL,
                params={
                    "pn": page, "pz": page_size, "po": 1, "np": 1,
                    "fltt": 2, "invt": 2, "fid": "f12", "fs": fs,
                    "fields": _FIELDS,
                },
                timeout=15,
            )
            d = r.json().get("data") or {}
        except Exception as e:
            logger.warning("EM clist page %s failed: %s", page, e)
            break
        if total is None:
            total = d.get("total", 0)
            logger.info("Universe total: %s stocks", total)
        batch = d.get("diff") or []
        if not batch:
            break
        rows.extend(batch)
        if total and len(rows) >= total:
            break
        page += 1

    if not rows:
        raise RuntimeError(
            "东财 clist 无数据（封禁或非交易时段）。请检查网络，或稍后重试。"
        )
    if total and len(rows) < total:
        # 风控断连导致的静默截断不可接受——初筛会漏掉后段市场（按 symbol
        # 排序时整个深市），抛错让 fetch_universe 降级到新浪全量。
        raise RuntimeError(
            f"东财 clist 分页中断（{len(rows)}/{total}）——部分清单不可用作全市场初筛"
        )

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "f2": "price", "f3": "change_pct", "f5": "volume", "f6": "amount",
        "f8": "turnover_pct", "f9": "pe", "f10": "volume_ratio",
        "f12": "code", "f14": "name", "f15": "high", "f16": "low",
        "f20": "mcap", "f23": "pb", "f26": "listed_date", "f62": "main_inflow",
    })
    # Suspended / pre-listing rows come back with "-" or NaN price — drop them.
    for col in ("price", "change_pct", "amount", "turnover_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price", "change_pct"])
    df["mcap_yi"] = pd.to_numeric(df["mcap"], errors="coerce") / 1e8
    df["amount_yi"] = df["amount"] / 1e8
    df["main_inflow_yi"] = pd.to_numeric(df["main_inflow"], errors="coerce") / 1e8
    # 上市日期（YYYYMMDD）→ 已上市交易日数近似（日历日×0.7）；无法解析时给 -1
    # 让过滤器决定怎么处理（解析失败不该静默把老票当次新）。
    ld = pd.to_numeric(df["listed_date"], errors="coerce")
    today_str = int(pd.Timestamp.now().strftime("%Y%m%d"))
    df["listed_days"] = ((today_str - ld) * 0.7 / 10000).round(0)
    df["listed_days"] = df["listed_days"].fillna(-1).astype(int)
    df["code"] = df["code"].astype(str)
    return df.reset_index(drop=True)
