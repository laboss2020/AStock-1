"""Cache layer tests: disk roundtrip, memo dedup, incremental fetch,
ex-dividend detection, weekend/no-refresh — all with mocked network, zero
real requests (EM bans this dev machine)."""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import astock.data.kline as kline_mod
from astock.data.cache import cache_dir, read_csv_cache, write_csv_cache

TMP = tempfile.mkdtemp(prefix="astock_cache_test_")


def _fake_bars(start: datetime, n: int, base: float = 20.0, jitter: float = 0.4) -> pd.DataFrame:
    """Synthetic daily bars starting at ``start`` (first bar = that date)."""
    rng = np.random.default_rng(int(start.strftime("%Y%m%d")) + n)
    dates = pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)])
    close = base + np.cumsum(rng.normal(0.05, jitter, n))
    return pd.DataFrame({
        "Date": dates, "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close,
        "Volume": 1e5, "Amount": 1e8,
    })


def _reset(fake_now, em_bars=None):
    """Reset module state + install mocks. em_bars: list of DataFrames EM
    returns on successive calls (then None)."""
    kline_mod._mem_cache.clear()
    kline_mod._checked.clear()
    kline_mod._em_fail_count[0] = 0
    os.environ["ASTOCK_CACHE"] = "1"
    os.environ["ASTOCK_CACHE_DIR"] = TMP
    kline_mod._now = fake_now
    calls = {"em": 0, "sina": 0}

    def fake_em(code, beg, end):
        idx = calls["em"]
        calls["em"] += 1
        if em_bars and idx < len(em_bars):
            return em_bars[idx]
        return None

    def fake_sina(code, datalen):
        calls["sina"] += 1
        return None

    kline_mod._kline_eastmoney = fake_em
    kline_mod._kline_sina = fake_sina
    return calls


def case_roundtrip():
    """Disk cache write/read roundtrip keeps code leading zeros."""
    os.environ["ASTOCK_CACHE_DIR"] = TMP  # isolate from the real data_cache/
    df = pd.DataFrame({"code": ["000001", "600000"], "price": [10.0, 20.0]})
    write_csv_cache(df, "universe", "universe_test.csv")
    back = read_csv_cache("universe", "universe_test.csv", dtype={"code": str})
    assert back is not None and list(back["code"]) == ["000001", "600000"], \
        f"code dtype broken: {back['code'].tolist()}"
    print("  roundtrip OK (code 前导零保留)")


def case_disabled():
    """ASTOCK_CACHE=0 → read/write are no-ops."""
    os.environ["ASTOCK_CACHE"] = "0"
    write_csv_cache(pd.DataFrame({"a": [1]}), "x", "y.csv")
    assert read_csv_cache("x", "y.csv") is None, "disabled cache returned data"
    os.environ["ASTOCK_CACHE"] = "1"
    print("  ASTOCK_CACHE=0 禁用 OK")


def case_memo_dedup():
    """Second get_kline (same run) hits memo — zero extra network calls."""
    friday = datetime(2026, 10, 2, 16, 0)  # 周五收盘后
    calls = _reset(lambda: friday, em_bars=[_fake_bars(datetime(2026, 6, 1), 70)])
    df1 = kline_mod.get_kline("600001", days=70)
    df2 = kline_mod.get_kline("600001", days=70)
    assert df1 is not None and df2 is not None
    assert calls["em"] == 1, f"memo dedup failed: em called {calls['em']} times"
    assert len(df2) == 70
    print(f"  memo 去重 OK（2 次调用只 1 次 EM 请求）")


def case_disk_hit():
    """Clear memo → disk cache hit, still zero network."""
    friday = datetime(2026, 10, 2, 16, 0)
    calls = _reset(lambda: friday, em_bars=[_fake_bars(datetime(2026, 6, 1), 70)])
    kline_mod.get_kline("600002", days=70)
    assert (cache_dir() / "kline" / "600002.csv").exists(), "disk cache not written"
    kline_mod._mem_cache.clear()
    calls["em"] = 0
    df = kline_mod.get_kline("600002", days=70)
    assert df is not None and calls["em"] == 0, "disk hit still fetched from network"
    print("  磁盘缓存命中 OK（0 网络请求）")


def _two_bars(d1_close: float, d2_close: float) -> pd.DataFrame:
    """2026-10-02 (overlap) + 2026-10-05 (Monday) bars with explicit closes."""
    return pd.DataFrame({
        "Date": [pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-05")],
        "Open": [d1_close, d2_close * 0.99], "High": [d1_close * 1.01, d2_close * 1.02],
        "Low": [d1_close * 0.98, d2_close * 0.985], "Close": [d1_close, d2_close],
        "Volume": [1e5, 1e5], "Amount": [1e8, 1e8],
    })


def case_incremental():
    """Stale cache (yesterday) → incremental EM fetch (beg=last), merged."""
    today = datetime(2026, 10, 5, 14, 0)       # 周一盘中
    # Cache up to Friday 2026-10-02 (70 calendar days back from 10-02)
    old = _fake_bars(datetime(2026, 7, 25), 70, base=20.0)
    last_close = float(old["Close"].iloc[-1])
    # EM incremental returns bars from beg=last (2026-10-02 overlap + today
    # 10-05); overlap close identical → normal merge
    fresh = _two_bars(last_close, last_close * 1.01)
    calls = _reset(lambda: today, em_bars=[fresh])
    # Seed disk cache as of yesterday (write directly, bypassing _store's
    # closed-day rule which is driven by _now)
    write_csv_cache(old, "kline", "600003.csv")
    df = kline_mod.get_kline("600003", days=120)
    assert df is not None, "incremental failed"
    assert calls["em"] == 1, f"expected 1 incremental call, got {calls['em']}"
    assert df["Date"].max() == pd.Timestamp("2026-10-05"), \
        f"today bar not merged: max={df['Date'].max()}"
    # 盘中半根只进 memo，不写盘（磁盘仍是截至周五的完整交易日）
    assert not (cache_dir() / "kline" / "600003.csv").read_text(encoding="utf-8").count("2026-10-05"), \
        "intraday half-bar leaked to disk cache"
    print("  增量拉取 OK（昨日缓存 → 增量合并，半根不落盘）")


def case_ex_dividend():
    """Overlap close gap >0.5% → ex-dividend → full refetch."""
    today = datetime(2026, 10, 5, 14, 0)
    old = _fake_bars(datetime(2026, 7, 25), 70, base=20.0)
    last_close = float(old["Close"].iloc[-1])
    # EM incremental's first (overlap) bar has a close 10% below the cached
    # one → ex-dividend detected → full refetch (second EM call)
    fresh = _two_bars(last_close * 0.90, last_close * 0.91)
    # full refetch must cover through today (127 calendar days from 06-01)
    calls = _reset(lambda: today, em_bars=[fresh, _fake_bars(datetime(2026, 6, 1), 127, base=18.0)])
    write_csv_cache(old, "kline", "600004.csv")
    df = kline_mod.get_kline("600004", days=120)
    assert df is not None
    assert calls["em"] == 2, f"ex-dividend should trigger full refetch, em calls={calls['em']}"
    assert df["Date"].max() == pd.Timestamp("2026-10-05")
    print("  除权检测 OK（重叠日 close 差 >0.5% → 全量重拉）")


def case_weekend():
    """Weekend + cache up to Friday → no refresh, no network."""
    saturday = datetime(2026, 10, 3, 10, 0)
    calls = _reset(lambda: saturday, em_bars=[])
    old = _fake_bars(datetime(2026, 6, 1), 70)
    write_csv_cache(old, "kline", "600005.csv")
    df = kline_mod.get_kline("600005", days=70)
    assert df is not None and calls["em"] == 0, "weekend still fetched"
    print("  周末不刷新 OK（0 网络请求）")


def main() -> int:
    print("CACHE TESTS:")
    case_roundtrip()
    case_disabled()
    case_memo_dedup()
    case_disk_hit()
    case_incremental()
    case_ex_dividend()
    case_weekend()
    print("ALL CACHE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
