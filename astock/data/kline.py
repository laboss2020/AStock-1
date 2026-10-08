"""K-line history: Eastmoney push2his (primary) + Sina (fallback).

Both are direct HTTP, no TCP (mootdx is unusable on many machines — see the
reference project's fallback chain). Eastmoney push2his serves full daily
bars including today; Sina's CN_MarketData kline API may lag a day, so it is
only a supplement when EM fails entirely.

Caching (two layers, ASTOCK_CACHE=0 to disable):
- in-process memo ``_mem_cache`` — dedups repeated fetches within one run
  (Stage 2 pulls 70d, Stage 4/5 pull the same codes again)
- on-disk ``ASTOCK_CACHE_DIR/kline/{code}.csv`` — subsequent runs only pull
  incremental bars (beg = last cached date). 前复权 (fqt=1) prices shift on
  ex-dividend dates, so the overlapping bar's close is compared on every
  incremental fetch: a >0.5% gap means an ex-dividend event → full refetch.
  Intraday (pre-15:10) bars for today are memo-only, never written to disk —
  the disk cache only ever holds complete trading days.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from .cache import read_csv_cache, write_csv_cache
from .http import SESSION, em_get

logger = logging.getLogger(__name__)

# EM 熔断：push2his 对请求过密的本机会软封（返回空/断连，几分钟到几小时）。
# 连续失败达到阈值后，本进程内后续 K 线直接走新浪 fallback——省掉每只
# 5-7s 的 EM 重试开销（400 只的初筛跑约省 35 分钟）。
_EM_FAIL_LIMIT = 3
_em_fail_count = [0]

# 本跑内去重 + 增量检查标记（按日期失效——同进程多次 run 时重新检查）
_mem_cache: dict[str, pd.DataFrame] = {}
_checked: dict[str, str] = {}  # code → 检查日期 ISO：同跑内不重复增量检查


def _to_secid(code: str) -> str:
    """6-digit code → "<market>.<code>" (1=SH, 0=SZ/BJ)."""
    if code.startswith(("6", "9")) and not code.startswith("92"):
        return f"1.{code}"
    return f"0.{code}"


def _now() -> datetime:
    """Injectable clock (tests monkeypatch this to fake intraday/weekend)."""
    return datetime.now()


def _market_closed(now: datetime) -> bool:
    """True after 15:10 — today's daily bar is complete (10-min buffer past
    the 15:00 close)."""
    return now.hour > 15 or (now.hour == 15 and now.minute >= 10)


def _is_trading_day(d) -> bool:
    """Weekday check (holidays not handled — an extra empty incremental fetch
    on a holiday is harmless)."""
    return d.weekday() < 5


def get_kline(
    code: str,
    days: int = 120,
    end_date: datetime | None = None,
) -> pd.DataFrame | None:
    """Daily OHLCV bars for ``code``, last ``days`` bars up to ``end_date``.

    Returns DataFrame[Date, Open, High, Low, Close, Volume, Amount] with Date
    as datetime, or None when both sources fail. fqt=1 (前复权) keeps moving
    averages comparable across ex-dividend dates.

    Cache-first: memo/disk hit that already covers the requested window
    returns immediately; a stale cache gets an incremental fetch (beg = last
    cached date, overlap-checked for ex-dividend) instead of a full refetch.
    """
    now = _now()
    today = pd.Timestamp(now.date())
    closed = _market_closed(now)

    # ---- 1. memo / disk cache ------------------------------------------
    df = _mem_cache.get(code)
    if df is None:
        df = read_csv_cache("kline", f"{code}.csv", parse_dates=["Date"])
    if df is not None and not df.empty:
        last = pd.Timestamp(df["Date"].max())
        trading_day = _is_trading_day(now.date())
        need_refresh = (
            (last < today and trading_day)          # 昨天为止的数据 + 今天开市 → 增量
            or (last >= today and not closed)        # 盘中拿到的半根 → 重新覆盖今天
        )
        if not need_refresh or _checked.get(code) == today.isoformat():
            return _finalize(df, days, end_date)
        # ---- 2. 增量拉取（熔断后跳过 EM 直接全量 fallback）-------------
        fresh = None
        if _em_fail_count[0] < _EM_FAIL_LIMIT:
            fresh = _kline_eastmoney(code, last.to_pydatetime(), now)
            if fresh is None or fresh.empty:
                _em_fail_count[0] += 1
        if fresh is not None and not fresh.empty:
            _em_fail_count[0] = 0
            merged = _merge_incremental(code, df, fresh)
            if merged is not None:
                _checked[code] = today.isoformat()
                _store(code, merged, today, closed)
                return _finalize(merged, days, end_date)
            # 除权 → 旧缓存作废，掉出本块走下方全量重拉
            df = None
        else:
            # EM 增量失败 → 全量 fallback（EM 熔断后为新浪）；再失败用旧缓存
            full = _fetch_full(code, days, now)
            if full is not None and not full.empty:
                _checked[code] = today.isoformat()
                _store(code, full, today, closed)
                return _finalize(full, days, end_date)
            logger.warning("K线增量失败，使用缓存数据（截至 %s）: %s", last.date(), code)
            _checked[code] = today.isoformat()
            return _finalize(df, days, end_date)

    # ---- 3. 全量拉取（EM → 新浪）----------------------------------------
    df = _fetch_full(code, days, now)
    if df is None or df.empty:
        return None
    _checked[code] = today.isoformat()
    _store(code, df, today, closed)
    return _finalize(df, days, end_date)


def _fetch_full(code: str, days: int, now: datetime) -> pd.DataFrame | None:
    """Full history fetch: EM (throttled, circuit-broken) → Sina fallback."""
    end = now
    beg = end - timedelta(days=days * 2)  # calendar days; trading days ≈ 0.7×

    df = None
    if _em_fail_count[0] < _EM_FAIL_LIMIT:  # 熔断后跳过 EM，直接新浪
        df = _kline_eastmoney(code, beg, end)
        if df is None or df.empty:
            _em_fail_count[0] += 1
            logger.info("EM kline empty for %s (%d/%d), trying sina fallback",
                        code, _em_fail_count[0], _EM_FAIL_LIMIT)
        else:
            _em_fail_count[0] = 0
    if df is None or df.empty:
        df = _kline_sina(code, days)
        if df is None or df.empty:
            return None
        df = df[df["Date"] <= pd.Timestamp(end.date())]
    return df


def _merge_incremental(code: str, old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame | None:
    """Append incremental bars to the cached frame.

    The first fresh bar overlaps a cached date by design (beg = last cached
    date): if its close differs >0.5% from the cached close an ex-dividend
    event moved every 前复权 price — return None so the caller refetches the
    full history. Otherwise drop the cached overlap and append.
    """
    first = pd.Timestamp(fresh["Date"].iloc[0])
    overlap = old[old["Date"] == first]
    if not overlap.empty:
        oc = float(overlap["Close"].iloc[-1])
        nc = float(fresh["Close"].iloc[0])
        if oc > 0 and abs(nc - oc) / oc > 0.005:
            logger.info("除权检测 %s: 重叠日 close %.2f→%.2f（>0.5%%），全量重拉", code, oc, nc)
            return None
    merged = pd.concat([old[old["Date"] < first], fresh], ignore_index=True)
    return merged


def _store(code: str, df: pd.DataFrame, today: pd.Timestamp, closed: bool) -> None:
    """Memo always; disk only when the frame holds no intraday half-bar for
    today (disk cache keeps complete trading days only)."""
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    _mem_cache[code] = df
    if closed or pd.Timestamp(df["Date"].max()) < today:
        write_csv_cache(df, "kline", f"{code}.csv")


def _finalize(df: pd.DataFrame, days: int, end_date: datetime | None) -> pd.DataFrame:
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date.date())]
    return df.tail(days).reset_index(drop=True)


def _kline_eastmoney(code: str, beg: datetime, end: datetime) -> pd.DataFrame | None:
    try:
        r = em_get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": _to_secid(code),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
                "klt": "101",  # daily
                "fqt": "1",  # 前复权
                "beg": beg.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "lmt": "500",
            },
            timeout=15,
        )
        d = r.json().get("data") or {}
        klines = d.get("klines") or []
        if not klines:
            return None
        rows = [line.split(",") for line in klines]
        df = pd.DataFrame(
            rows,
            columns=["Date", "Open", "Close", "High", "Low", "Volume", "Amount"],
        )
        for col in ("Open", "Close", "High", "Low", "Volume", "Amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("EM kline failed for %s: %s", code, e)
        return None


def _kline_sina(code: str, datalen: int) -> pd.DataFrame | None:
    """Sina daily kline (240-min scale = 1 day). May lag by a day."""
    sym = ("sh" if code.startswith(("6", "9")) and not code.startswith("92")
           else "sz") + code
    return _kline_sina_raw(sym, datalen)


def get_index_kline(index_code: str, days: int = 120) -> pd.DataFrame | None:
    """Index daily bars (index_code like '000001' SH Composite).

    Primary: Eastmoney push2his. Fallback: Sina (index symbols carry their
    own sh/sz prefix — 000001→sh000001, 399xxx→sz399xxx).
    """
    end = datetime.now()
    beg = end - timedelta(days=days * 2)
    em_market = "1" if index_code == "000001" or not index_code.startswith("3") else "0"
    try:
        r = em_get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": f"{em_market}.{index_code}",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
                "klt": "101",
                "fqt": "1",
                "beg": beg.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "lmt": "500",
            },
            timeout=15,
        )
        d = r.json().get("data") or {}
        klines = d.get("klines") or []
        if not klines:
            raise ValueError("empty")
        rows = [line.split(",") for line in klines]
        df = pd.DataFrame(rows, columns=["Date", "Open", "Close", "High", "Low", "Volume", "Amount"])
        for col in ("Open", "Close", "High", "Low", "Volume", "Amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        logger.warning("EM index kline failed for %s, trying sina: %s", index_code, e)

    sym = ("sh" if index_code == "000001" or not index_code.startswith("3")
           else "sz") + index_code
    df = _kline_sina_raw(sym, days)
    if df is None or df.empty:
        return None
    return df.tail(days).reset_index(drop=True)


def _kline_sina_raw(symbol: str, datalen: int) -> pd.DataFrame | None:
    """Sina kline for an explicit symbol (with prefix) — shared by index fallback."""
    try:
        r = SESSION.get(
            "https://money.finance.sina.com.cn/quotes_service/api/"
            "json_v2.php/CN_MarketData.getKLineData",
            params={"symbol": symbol, "scale": "240", "ma": "no", "datalen": datalen},
            timeout=15,
        )
        r.raise_for_status()
        import json

        rows = json.loads(r.text)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "day": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ("Open", "Close", "High", "Low", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Amount"] = 0.0
        return df.dropna(subset=["Date", "Close"]).sort_values("Date")[
            ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
        ].reset_index(drop=True)
    except Exception as e:
        logger.warning("Sina index kline failed for %s: %s", symbol, e)
        return None
