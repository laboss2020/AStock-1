"""App configuration: LLM provider presets, screening and pricing parameters.

LLM presets follow the multi-provider design of tradingagents-astock, but all
providers here are OpenAI-compatible (Chat Completions), so a single client
covers them: NVIDIA NIM (free tier), DeepSeek, OpenAI, or any custom gateway.
Each provider reads its own env var; a missing key is reported by name.

Screening parameters are plain dataclass fields — override via CLI flags or
the ``ASTOCK_*`` environment variables (see :func:`load_config`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM provider presets (all OpenAI-compatible Chat Completions)
# ---------------------------------------------------------------------------

# Each preset: (base_url, default_model, env_var_names in priority order).
# NVIDIA NIM is the default: free developer tier, OpenAI-compatible endpoint,
# key from https://build.nvidia.com (nvapi-...).
LLM_PRESETS: dict[str, dict[str, Any]] = {
    "nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "deepseek-ai/deepseek-v4.1-flash",
        "env_vars": ("NVIDIA_API_KEY", "NIM_API_KEY", "OPENAI_API_KEY"),
        "label": "NVIDIA NIM（免费额度）",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_vars": ("DEEPSEEK_API_KEY",),
        "label": "DeepSeek",
    },
    # GitHub Models：GitHub 自家的 OpenAI 兼容推理网关。Actions 里用内置
    # GITHUB_TOKEN（workflow 需声明 permissions: models: read）零配置可调，
    # 作为 NIM 对 CI 网络不通时的兜底通道（probe_llm.py 自动切换）。
    "github_models": {
        "base_url": "https://models.github.ai/inference",
        "default_model": "openai/gpt-4.1-mini",
        "env_vars": ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_MODELS_TOKEN"),
        "label": "GitHub Models（Actions 内置 GITHUB_TOKEN）",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_vars": ("OPENAI_API_KEY",),
        "label": "OpenAI",
    },
    "openai_compatible": {
        "base_url": "",  # must be provided via ASTOCK_BASE_URL / BACKEND_URL
        "default_model": "",  # must be provided via ASTOCK_MODEL
        "env_vars": ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY"),
        "label": "OpenAI 兼容（自定义网关）",
    },
}


@dataclass
class Config:
    """Runtime configuration. Only set the fields you want to override."""

    # --- LLM ---
    provider: str = "nim"
    base_url: str = ""  # empty = preset default (or ASTOCK_BASE_URL)
    model: str = ""  # empty = preset default (or ASTOCK_MODEL)
    api_key: str = ""  # empty = read preset env vars
    llm_timeout: int = 150  # seconds per request (关思考后正常调用 <20s；300 会让挂起
                            # 连接等满 5 分钟才报错——run #42 重试周期 ~5min 致 4h 超时)
    llm_max_retries: int = 2
    llm_retry_delay: float = 5.0  # exponential backoff base: 5s → 10s
    llm_trust_env: bool = False  # False = bypass system proxy (direct); 1 = use it
    llm_thinking: bool = False  # False = 请求关闭思考（仅 NIM 发送 chat_template_kwargs；
                                # deepseek/nemotron 实测生效，GLM 忽略；其他 provider 不发）

    # --- Screening ---
    candidate_count: int = 30  # quant screen → top N candidates for LLM triage
    triage_count: int = 15  # LLM triage → top N for deep analysis
    max_picks: int = 10  # final picks cap (requirement: 10 or fewer)
    min_price: float = 2.0  # exclude penny stocks (面值退市风险)
    min_amount: float = 1.0  # exclude illiquid names, 亿元
    min_mcap: float = 20.0  # exclude micro caps, 亿元
    max_mcap: float = 0  # 0 = no upper bound
    min_listed_days: int = 60  # exclude 次新 (no price limit first 5 days, unstable)
    exclude_st: bool = True
    exclude_new_high: bool = True  # exclude names already limit-up today (追高风险)

    # --- Output ---
    output_dir: str = "reports"  # per-run reports root
    debug: bool = False  # verbose logging

    # --- Review / strategy versioning ---
    review: bool = True  # 复盘上一次 run 并生成新策略版本（每次选股之后）
    strategy_version: int = 0  # 0 = 用最新策略版本（无则默认策略）；N = 指定版本

    def resolved_base_url(self) -> str:
        return (
            self.base_url
            or os.environ.get("ASTOCK_BASE_URL", "")
            or os.environ.get("BACKEND_URL", "")
            or str(LLM_PRESETS[self.provider]["base_url"])
        )

    def resolved_model(self) -> str:
        return (
            self.model
            or os.environ.get("ASTOCK_MODEL", "")
            or str(LLM_PRESETS[self.provider]["default_model"])
        )

    def resolved_fallback_models(self) -> list[str]:
        """Backup models tried in order when the primary answers 404
        (providers rotate their model catalogs — NIM 2026-10-21 removed
        nemotron-3-super overnight). env: ASTOCK_FALLBACK_MODELS (comma
        separated); empty = LLMClient applies a built-in NIM chain."""
        raw = os.environ.get("ASTOCK_FALLBACK_MODELS", "")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        for var in LLM_PRESETS[self.provider]["env_vars"]:
            val = os.environ.get(var, "")
            if val:
                return val
        return ""

    def env_hint(self) -> str:
        preset = LLM_PRESETS[self.provider]
        names = " 或 ".join(preset["env_vars"])
        lines = [f"provider={self.provider}（{preset['label']}）需要设置环境变量：{names}"]
        if self.provider == "openai_compatible":
            lines.append("还必须设置 ASTOCK_BASE_URL=<你的网关地址> 和 ASTOCK_MODEL=<model 名>")
        if self.provider == "nim":
            lines.append("NIM 免费 key 申请地址：https://build.nvidia.com（注册后在任意模型页获取 nvapi- 开头的 key）")
        return "\n".join(lines)


def load_config(**overrides: Any) -> Config:
    """Build a Config from defaults + ASTOCK_* env vars + explicit overrides.

    Env vars (lower precedence than explicit overrides): ASTOCK_PROVIDER,
    ASTOCK_CANDIDATE_COUNT, ASTOCK_MAX_PICKS, ASTOCK_BASE_URL, ASTOCK_MODEL.
    Unknown names are ignored silently for env, but rejected for overrides.
    """
    cfg = Config()

    env_map = {
        "ASTOCK_PROVIDER": ("provider", str),
        "ASTOCK_BASE_URL": ("base_url", str),
        "ASTOCK_MODEL": ("model", str),
        "ASTOCK_API_KEY": ("api_key", str),
        "ASTOCK_CANDIDATE_COUNT": ("candidate_count", int),
        "ASTOCK_TRIAGE_COUNT": ("triage_count", int),
        "ASTOCK_MAX_PICKS": ("max_picks", int),
        "ASTOCK_MIN_PRICE": ("min_price", float),
        "ASTOCK_MIN_AMOUNT": ("min_amount", float),
        "ASTOCK_MIN_MCAP": ("min_mcap", float),
        "ASTOCK_MIN_LISTED_DAYS": ("min_listed_days", int),
        "ASTOCK_OUTPUT_DIR": ("output_dir", str),
        "ASTOCK_LLM_TRUST_ENV": ("llm_trust_env", lambda v: v.strip() in ("1", "true", "yes", "True")),
        "ASTOCK_LLM_TIMEOUT": ("llm_timeout", int),
        "ASTOCK_LLM_MAX_RETRIES": ("llm_max_retries", int),
        "ASTOCK_LLM_THINKING": ("llm_thinking", lambda v: v.strip() in ("1", "true", "yes", "True")),
        "ASTOCK_REVIEW": ("review", lambda v: v.strip() in ("1", "true", "yes", "True")),
        "ASTOCK_STRATEGY_VERSION": ("strategy_version", int),
    }
    for var, (name, cast) in env_map.items():
        raw = os.environ.get(var, "")
        if raw:
            try:
                setattr(cfg, name, cast(raw))
            except ValueError:
                pass  # bad env value → keep default

    known = {f.name for f in fields(Config)}
    for key, val in overrides.items():
        if key not in known:
            raise ValueError(f"Unknown config option: '{key}'")
        if val is not None:
            setattr(cfg, key, val)

    if cfg.provider not in LLM_PRESETS:
        raise ValueError(
            f"Unknown provider '{cfg.provider}'. Choose from: {list(LLM_PRESETS)}"
        )
    return cfg
