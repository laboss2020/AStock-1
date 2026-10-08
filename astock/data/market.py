"""Market-level data: indexes, advance/decline counts, market fund flow,
northbound capital (with local daily cache, upstream stopped history).

These feed Stage 1 (大盘体检) — the market-regime judgment that gates the
whole pick. A股是政策市 + beta 市场：先判断环境，再选股。
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from .http import em_get, get_json
from .kline import get_index_kline, _to_secid

logger = logging.getLogger(__name__)

INDEXES = [
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
    ("000688", "科创50"),
]


def index_snapshot() -> list[dict]:
    """Realtime index quotes + advance/decline counts.

    Primary: Eastmoney ulist (advance/decline counts included). Fallback:
    Tencent index quotes (no advance/decline counts — those fields read 0,
    and the Stage-1 prompt must not read them as "all flat" when the source
    lacks them; the caller labels the field as missing instead).
    """
    secids = ",".join(
        f"1.{c}" if c == "000001" or not c.startswith("3") else f"0.{c}"
        for c, _ in INDEXES
    )
    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/ulist.np/get",
            params={
                "secids": secids,
                "fields": "f2,f3,f4,f12,f14,f104,f105,f106",
                "fltt": 2, "invt": 2,
            },
            timeout=10,
        )
        rows = (r.json().get("data") or {}).get("diff") or []
        if rows:
            return [
                {
                    "code": row.get("f12", ""),
                    "name": row.get("f14", ""),
                    "close": row.get("f2", 0.0),
                    "change_pct": row.get("f3", 0.0),
                    "advance": row.get("f104", 0),   # 上涨家数
                    "decline": row.get("f105", 0),   # 下跌家数
                    "flat": row.get("f106", 0),
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning("EM index snapshot failed, trying Tencent: %s", e)

    # Tencent fallback: index symbols carry their own prefix (上证/科创=sh,
    # 深成/创业板=sz) — parsed directly, not via market_prefix (which would
    # route the SH composite 000001 to "sz").
    from .quotes import fetch_tencent_raw, parse_tencent_text

    index_syms = {"000001": "sh000001", "399001": "sz399001",
                  "399006": "sz399006", "000688": "sh000688"}
    text = fetch_tencent_raw(",".join(index_syms[c] for c, _ in INDEXES))
    tq = parse_tencent_text(text)
    out = []
    for c, name in INDEXES:
        q = tq.get(c)
        if q is None:
            continue
        out.append({
            "code": c, "name": name,
            "close": q["price"], "change_pct": q["change_pct"],
            "advance": 0, "decline": 0, "flat": 0,
            "advance_decline_available": False,
        })
    return out


def index_trend_summary(days: int = 20) -> list[dict]:
    """Per-index trend stats from daily bars: pct over 5/20 days, MA20 position."""
    out = []
    for code, name in INDEXES:
        df = get_index_kline(code, days=days + 10)
        if df is None or df.empty:
            out.append({"code": code, "name": name, "ok": False})
            continue
        close = df["Close"]

        def pct(n: int) -> float | None:
            if len(close) <= n:
                return None
            base = close.iloc[-1 - n]
            return round(float(close.iloc[-1] / base - 1) * 100, 2) if base else None

        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
        last = float(close.iloc[-1])
        out.append({
            "code": code, "name": name, "ok": True,
            "close": round(last, 2),
            "chg_5d": pct(5),
            "chg_20d": pct(20),
            "vs_ma20": round((last / ma20 - 1) * 100, 2) if ma20 else None,
        })
    return out


def market_fund_flow(days: int = 5) -> list[dict]:
    """SH composite main-capital daily net flow, last ``days`` days (亿元)."""
    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get",
            params={
                "secid": _to_secid("000001"),
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                "klt": "101",
                "lmt": str(days),
            },
            timeout=15,
        )
        klines = (r.json().get("data") or {}).get("klines") or []
        out = []
        for line in klines:
            p = line.split(",")
            # f51 date, f52 主力净流入, f53 小单, f54 中单, f55 大单, f56 超大单
            out.append({
                "date": p[0],
                "main_net_yi": round(float(p[1]) / 1e8, 2) if len(p) > 1 else None,
                "super_net_yi": round(float(p[5]) / 1e8, 2) if len(p) > 5 else None,
            })
        return out
    except Exception as e:
        logger.warning("Market fund flow failed: %s", e)
        return []


# --- Northbound (沪深股通) --------------------------------------------------
# 上游 API 自 2024-08 起停发北向历史，只靠本地逐日缓存累积（同 tradingagents-astock）。

def _northbound_cache_path() -> Path:
    cache_dir = os.environ.get("ASTOCK_CACHE_DIR", os.path.expanduser("~/.astock/cache"))
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "northbound_daily.csv"


def northbound_realtime() -> dict | None:
    """Minute-level cumulative net buying for HGT+SGT from 同花顺 hsgtApi."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0 Safari/537.36",
        "Host": "data.hexin.cn",
        "Referer": "https://data.hexin.cn/",
    }
    try:
        d = get_json("https://data.hexin.cn/market/hsgtApi/method/dayChart/",
                     headers=headers, timeout=10)
        times = d.get("time") or []
        hgt = d.get("hgt") or []
        sgt = d.get("sgt") or []
        if not times:
            return None
        return {
            "hgt_close": float(hgt[-1]) if hgt else 0.0,
            "sgt_close": float(sgt[-1]) if sgt else 0.0,
            "total": (float(hgt[-1]) if hgt else 0.0) + (float(sgt[-1]) if sgt else 0.0),
        }
    except Exception as e:
        logger.warning("Northbound realtime failed: %s", e)
        return None


def save_northbound_snapshot(total: float) -> None:
    """Append today's northbound close to the local CSV cache (dedup by date)."""
    from .http import market_today

    path = _northbound_cache_path()
    existing: dict[str, float] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        existing[row[0]] = float(row[1])
                    except ValueError:
                        pass
    existing[market_today().strftime("%Y-%m-%d")] = total
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "total_yi"])
        for d in sorted(existing):
            writer.writerow([d, f"{existing[d]:.2f}"])


def northbound_history(n: int = 10) -> list[tuple[str, float]]:
    """Last N days of northbound close from the local cache."""
    path = _northbound_cache_path()
    if not path.exists():
        return []
    rows: list[tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    rows.append((row[0], float(row[1])))
                except ValueError:
                    pass
    return rows[-n:]
