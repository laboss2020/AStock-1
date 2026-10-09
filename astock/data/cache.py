"""Disk-cache helpers for the data layer (K-lines, universe snapshot).

Two-layer caching lives in the data layer:
- in-process memo (per-run dedup) — see kline.py
- on-disk cache under ``ASTOCK_CACHE_DIR`` (default ``data_cache/``) so
  repeated runs only pull incremental updates

Enabled by default; ``ASTOCK_CACHE=0`` disables both layers (debug). The
cache dir is gitignored and safe to mount as a volume (Docker / GitHub
Actions cache).

Data layer functions never take a Config — they read env directly, same as
http.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_enabled() -> bool:
    """ASTOCK_CACHE=1 (default) / 0 — off disables disk + memo caching."""
    return os.environ.get("ASTOCK_CACHE", "1").strip() in ("1", "true", "yes", "True")


def cache_dir() -> Path:
    """Root dir for on-disk caches (ASTOCK_CACHE_DIR, default data_cache/)."""
    return Path(os.environ.get("ASTOCK_CACHE_DIR", "data_cache"))


def read_csv_cache(*parts: str, dtype: dict | None = None, parse_dates: list | None = None):
    """Read a cached CSV if present, else None. ``dtype`` pins columns like
    code as str (read_csv would otherwise strip leading zeros); ``parse_dates``
    restores datetime columns (Date comes back as str otherwise)."""
    if not cache_enabled():
        return None
    path = cache_dir().joinpath(*parts)
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
        logger.info("磁盘缓存命中: %s", path)
        return df
    except Exception as e:  # corrupted cache → refetch, never block on it
        logger.warning("磁盘缓存读取失败（将重新拉取）: %s: %s", path, e)
        return None


def write_csv_cache(df, *parts: str) -> None:
    """Write a CSV cache entry; failures are logged, never raised (cache is
    best-effort — a read-only volume must not kill the run)."""
    if not cache_enabled():
        return
    path = cache_dir().joinpath(*parts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        logger.warning("磁盘缓存写入失败: %s: %s", path, e)
