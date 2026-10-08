"""Real-time quotes: Tencent (valuation snapshot) + Eastmoney push2.

Tencent (qt.gtimg.cn, GBK) gives PE/PB/market-cap/turnover in one call.
Eastmoney push2 gives structured realtime quotes. Both are "right now"
snapshots — there is no historical point-in-time version; callers running a
historical replay must label them accordingly (same rule as
tradingagents-astock's ``_snapshot_notice``).
"""

from __future__ import annotations

import logging

from .http import SESSION, em_get, get_text

logger = logging.getLogger(__name__)


def market_prefix(code: str) -> str:
    """6-digit A-share code → market prefix for Tencent / secid for Eastmoney.

    920xxx (Beijing Stock Exchange) must be checked before the leading-9 rule
    — a bare ``startswith("9")`` routes BSE names to Shanghai, whose endpoints
    return empty payloads. 900xxx (Shanghai B shares) is the only legitimate
    ``sh`` nine-prefix.
    """
    if code.startswith("92"):
        return "bj"
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def em_secid(code: str) -> str:
    """6-digit code → Eastmoney secid ("<market>.<code>")."""
    p = market_prefix(code)
    market = {"sh": "1", "sz": "0", "bj": "0"}[p]
    # BSE quotes live under market 0 on push2.
    return f"{market}.{code}"


def tencent_quotes(codes: list[str]) -> dict[str, dict]:
    """Batch realtime quotes from Tencent (GBK). Returns {code: fields}."""
    if not codes:
        return {}
    q = ",".join(f"{market_prefix(c)}{c}" for c in codes)
    return parse_tencent_text(fetch_tencent_raw(q))


def fetch_tencent_raw(query: str) -> str:
    """Raw Tencent GBK payload for a query string like 'sh600519,sz000001'."""
    try:
        return get_text(f"https://qt.gtimg.cn/q={query}", encoding="gbk", timeout=10)
    except Exception as e:
        logger.warning("Tencent quotes failed: %s", e)
        return ""


def parse_tencent_text(text: str) -> dict[str, dict]:
    """Parse Tencent's ~-separated response into {code: fields}.

    The format is positional: the indices below were verified against live
    data (name=1, price=3, change=32, turnover=38, pe_ttm=39, mcap=45, pb=46,
    high=33, low=34, open=5, volume=36, amount=37).
    """
    out: dict[str, dict] = {}
    for line in text.strip().split(";"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        payload = line.split("=", 1)[1].strip('"')
        parts = payload.split("~")
        if len(parts) < 47 or not parts[2]:
            continue
        code = parts[2]

        def _f(idx: str) -> float:
            try:
                return float(parts[int(idx)])
            except (ValueError, IndexError):
                return 0.0

        out[code] = {
            "name": parts[1],
            "price": _f("3"),
            "change_pct": _f("32"),
            "open": _f("5"),
            "volume": _f("36"),  # 手
            "amount": _f("37"),  # 万元
            "turnover_pct": _f("38"),
            "pe_ttm": _f("39"),
            "high": _f("33"),
            "low": _f("34"),
            "mcap_yi": _f("45"),
            "pb": _f("46"),
            "limit_up": _f("47"),
            "limit_down": _f("48"),
        }
    return out


def em_realtime_quote(code: str) -> dict | None:
    """Single realtime quote from Eastmoney push2 (structured fields)."""
    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "secid": em_secid(code),
                "fields": "f43,f44,f45,f45,f46,f47,f48,f50,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170",
            },
            timeout=10,
        )
        d = r.json().get("data") or {}
        if not d.get("f57"):
            return None

        def _f(key: str, scale: float = 1.0) -> float:
            v = d.get(key)
            if v in (None, "-"):
                return 0.0
            return float(v) * scale  # EM price fields come x100 when fltt unset

        return {
            "code": d["f57"],
            "name": d.get("f58", ""),
            "price": _f("f43", 0.01),
            "high": _f("f44", 0.01),
            "low": _f("f45", 0.01),
            "open": _f("f46", 0.01),
            "volume": _f("f47"),
            "turnover_pct": _f("f168", 0.01),
            "pe_ttm": _f("f162", 0.01),
            "pb": _f("f167", 0.01),
            "change_pct": _f("f170", 0.01),
            "mcap": _f("f116"),
            "float_mcap": _f("f117"),
        }
    except Exception as e:
        logger.warning("EM realtime quote failed for %s: %s", code, e)
        return None
