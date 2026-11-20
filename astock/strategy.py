"""Strategy versioning: review-driven strategy evolution.

Each run reviews the previous run's report (how did those picks actually
do?), and the LLM distills that into a **strategy version** — a structured
JSON under ``strategies/``:

    strategies/v1.json, v2.json, ...

A version carries: review summary, per-stage prompt calibration blocks
(大盘体检校准/快筛标准/深度分析重点/定价纪律) and factor multipliers
applied on top of the regime weights. The next run loads the latest version
by default; ``--strategy-version N`` / ``ASTOCK_STRATEGY_VERSION`` pins a
specific one (0 = latest, missing dir = default strategy).

Versions are assets (gitignored nothing) — commit them to track evolution.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .screen import RegimeWeights

logger = logging.getLogger(__name__)

_MULT_MIN, _MULT_MAX = 0.5, 2.0  # factor multiplier clamp


@dataclass
class StrategyVersion:
    """One strategy version distilled from reviewing recent runs."""
    version: int
    created: str                      # YYYY-MM-DD
    based_on: str                     # 窗口内最新 run 目录名（复盘窗口的锚点）
    review_summary: str = ""          # 复盘结论（近期选股实际表现、判断准确度）
    regime_calibration: str = ""      # Stage 1 大盘体检的校准提示
    triage_criteria: str = ""         # Stage 3 快筛评分标准的补充要求
    deep_focus: str = ""              # Stage 4 深度分析的补充关注点
    pricing_discipline: str = ""      # Stage 5 定价纪律的补充要求
    factor_multipliers: dict = field(default_factory=dict)  # 因子权重乘数（regime 权重基底上叠加）
    notes: list[str] = field(default_factory=list)          # 人可读的改进点
    review_rows: list[dict] = field(default_factory=list)   # 复盘明细（前后对比+原因，内存传递进报告，不写入版本文件）

    def prompt_blocks(self) -> dict[str, str]:
        """Per-stage prompt supplements (empty fields produce no block)."""
        head = f"### 上次复盘校准（策略 v{self.version}，{self.created} 生成）\n"
        blocks: dict[str, str] = {}
        if self.regime_calibration:
            blocks["regime"] = head + f"- **大盘体检校准**：{self.regime_calibration}"
        if self.triage_criteria:
            blocks["triage"] = head + f"- **快筛评分补充**：{self.triage_criteria}"
        if self.deep_focus:
            blocks["deep"] = head + f"- **深度分析补充**：{self.deep_focus}"
        if self.pricing_discipline:
            blocks["pricing"] = head + f"- **定价纪律补充**：{self.pricing_discipline}"
        return blocks


def strategies_dir() -> Path:
    """Root dir for strategy versions (ASTOCK_STRATEGY_DIR, default strategies/)."""
    return Path(os.environ.get("ASTOCK_STRATEGY_DIR", "strategies"))


def latest_version() -> int:
    """Highest vN.json in strategies/ — 0 when none exist."""
    d = strategies_dir()
    if not d.is_dir():
        return 0
    versions = [int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit()]
    return max(versions) if versions else 0


def load_strategy(version: int = 0) -> StrategyVersion | None:
    """Load a strategy version. ``version=0`` (default) = latest; a missing
    dir/number → None (default strategy, never raises)."""
    v = version if version > 0 else latest_version()
    if v <= 0:
        return None
    if version > 0 and version > latest_version():
        logger.warning("策略版本 v%d 不存在（最新 v%d），使用默认策略", version, latest_version())
        return None
    path = strategies_dir() / f"v{v}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return StrategyVersion(
            version=int(raw.get("version", v)),
            created=str(raw.get("created", "")),
            based_on=str(raw.get("based_on", "")),
            review_summary=str(raw.get("review_summary", "")),
            regime_calibration=str(raw.get("regime_calibration", "")),
            triage_criteria=str(raw.get("triage_criteria", "")),
            deep_focus=str(raw.get("deep_focus", "")),
            pricing_discipline=str(raw.get("pricing_discipline", "")),
            factor_multipliers=dict(raw.get("factor_multipliers", {}) or {}),
            notes=[str(n) for n in raw.get("notes", []) if n],
        )
    except Exception as e:  # corrupted → default strategy, never block
        logger.warning("策略版本读取失败 %s: %s（使用默认策略）", path, e)
        return None


def apply_multipliers(weights: RegimeWeights, mult: dict) -> RegimeWeights:
    """Scale regime weights by the strategy's factor multipliers, then
    renormalize to sum=1. Multipliers clamp to [0.5, 2.0]; unknown keys and
    bad values keep the regime factor untouched."""
    fields = {"momentum", "volume_price", "capital", "heat", "valuation"}
    adjusted = {}
    for f in fields:
        m = mult.get(f, 1.0)
        try:
            m = float(m)
        except (TypeError, ValueError):
            m = 1.0
        m = max(_MULT_MIN, min(_MULT_MAX, m))
        adjusted[f] = getattr(weights, f) * m
    total = sum(adjusted.values())
    if total <= 0:
        return weights
    return RegimeWeights(**{k: v / total for k, v in adjusted.items()})


def save_strategy(sv: StrategyVersion) -> Path:
    """Write strategies/vN.json (best-effort directory creation).

    review_rows（复盘明细）只进运行报告，不写入版本文件——版本文件只留
    可复用的策略参数。"""
    d = strategies_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"v{sv.version}.json"
    payload = asdict(sv)
    payload.pop("review_rows", None)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    logger.info("策略版本已保存: %s", path)
    return path
