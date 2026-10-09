"""HTTP infrastructure: retrying session + Eastmoney throttling.

Two techniques ported from tradingagents-astock's data layer (Apache-2.0):

- **Every request goes through a retrying Session.** Corporate proxies and
  flaky networks drop connections intermittently — a bare ``requests.get``
  turns one dropped packet into a failed data pull. The Retry adapter
  retries connection errors and 5xx with backoff.
- **All Eastmoney requests funnel through :func:`em_get`.** Eastmoney's
  risk control (measured by tradingagents-astock: >5 req/s, >=10 concurrent,
  or >=200 req/min triggers a temporary IP ban) silently returns empty or
  errors when tripped. A module-level serial throttle (default >=1s between
  calls + jitter) keeps a full screening run (~100+ EM calls) under the ban
  threshold. Non-EM sources (Tencent/Sina/THS) are not throttled.
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# A-share market timezone. "Today" must be computed in market time — a host
# east of UTC+8 (e.g. NZ, UTC+13) has already crossed midnight when Shanghai
# hasn't, which flips "today's run" into "historical replay".
_MARKET_TZ = timezone(timedelta(hours=8))

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)


def market_today() -> datetime:
    """Current date/time in A-share market (Asia/Shanghai), host-TZ independent."""
    return datetime.now(_MARKET_TZ)


def build_session() -> requests.Session:
    """Session with retry adapter + default UA. Reuse it — Keep-Alive matters."""
    session = requests.Session()
    # 默认绕过系统（注册表）代理：Windows 注册表代理会间歇断连甚至整机
    # 拒绝（WinError 10061），而国内数据源与 NIM 直连实测稳定（0.4-0.5s）。
    # 需要走代理的用户设 ASTOCK_TRUST_ENV=1。
    _use_env = os.environ.get("ASTOCK_TRUST_ENV", "0") in ("1", "true", "yes", "True")
    session.trust_env = _use_env
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        # Connection drops (proxy hiccups) are retried by urllib3 core logic.
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": _UA})
    return session


SESSION = build_session()

# --- Eastmoney serial throttle -------------------------------------------
# Interval between EM calls (seconds). Batch/screening runs can raise it via
# EM_MIN_INTERVAL=1.5~2 (same env var as tradingagents-astock).
_EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_em_last_call = [0.0]


def _em_wait() -> None:
    """Serial throttle: ensure >=EM_MIN_INTERVAL since last EM call + jitter."""
    elapsed = time.monotonic() - _em_last_call[0]
    wait = _EM_MIN_INTERVAL - elapsed
    if wait > 0:
        time.sleep(wait)
    _em_last_call[0] = time.monotonic()
    # 0.1~0.5s jitter so request patterns never look machine-like.
    time.sleep(random.uniform(0.1, 0.5))


def em_get(url: str, params: dict | None = None, timeout: int = 15,
           session: requests.Session | None = None, **kwargs) -> requests.Response:
    """GET an Eastmoney endpoint through the throttle. **All EM calls use this.**"""
    _em_wait()
    s = session or SESSION
    return s.get(url, params=params, timeout=timeout, **kwargs)


def get_json(url: str, params: dict | None = None, timeout: int = 15,
             session: requests.Session | None = None, **kwargs) -> dict:
    """GET any (non-EM) JSON endpoint via the shared session."""
    s = session or SESSION
    r = s.get(url, params=params, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r.json()


def get_text(url: str, params: dict | None = None, timeout: int = 15,
             encoding: str | None = None,
             session: requests.Session | None = None, **kwargs) -> str:
    """GET any (non-EM) text endpoint, optionally forcing an encoding (GBK for Tencent)."""
    s = session or SESSION
    r = s.get(url, params=params, timeout=timeout, **kwargs)
    r.raise_for_status()
    if encoding:
        r.encoding = encoding
    return r.text


def is_trading_day(d: datetime | None = None) -> bool:
    """Rough weekday check (Mon-Fri). Holidays are not modelled — callers must
    tolerate empty data on holidays and label it as such."""
    d = d or market_today()
    return d.weekday() < 5
