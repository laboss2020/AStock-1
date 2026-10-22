"""OpenAI-compatible LLM client (NVIDIA NIM by default).

Single client covers every provider preset in :mod:`astock.config` because
they all speak Chat Completions. Design points borrowed from
tradingagents-astock's LLM layers:

- **Timeouts are mandatory**: a hung gateway must not freeze the pipeline.
  Every call passes an explicit timeout; httpx treats ``None`` as "no timeout".
- **Application-layer retries**: connection errors / 429 / 5xx retry with
  exponential backoff (the SDK's own retries are disabled so the app owns
  the policy).
- **JSON mode with one parse retry**: the model is asked for JSON; a
  malformed reply triggers one retry with an explicit correction notice.
  Code fences are stripped before parsing — models love wrapping JSON in
  ```json blocks.
- **Prompt-anchored JSON**: structured outputs via ``response_format`` are
  requested, but NIM/relays may ignore them, so prompts also state the exact
  JSON shape. Parse failure falls back gracefully rather than crashing a
  30-stock batch.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import Config

logger = logging.getLogger(__name__)

# Errors worth an app-layer retry. Auth failures are deliberately NOT here:
# retrying a 401 never succeeds and only burns time — fail fast with the
# env-var hint instead.
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)

# Built-in fallback chain for NIM (provider rotates its catalog overnight —
# nemotron-3-super-120b-a12b answered 404 from 2026-10-21). All verified
# live 2026-10-22. Override with ASTOCK_FALLBACK_MODELS (comma separated).
_NIM_FALLBACK_MODELS = (
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/nemotron-3-ultra-550b-a55b",
)


class LLMError(RuntimeError):
    """Fatal LLM failure (auth, model missing, retries exhausted)."""


def _is_retryable_status(err: APIStatusError) -> bool:
    status = getattr(err, "status_code", None) or 0
    return status in (408, 409, 429) or status >= 500


class LLMClient:
    """Thin Chat Completions client with timeout, retries, and JSON parsing."""

    def __init__(self, config: Config):
        self.config = config
        api_key = config.resolved_api_key()
        if not api_key:
            raise LLMError(
                "缺少 API Key。\n" + config.env_hint()
            )
        base_url = config.resolved_base_url()
        if not base_url:
            raise LLMError(
                "缺少 base_url（provider=openai_compatible 时必须提供）。\n" + config.env_hint()
            )
        self.model = config.resolved_model()
        if not self.model:
            raise LLMError(
                "缺少 model 名（provider=openai_compatible 时必须提供）。\n" + config.env_hint()
            )
        fallbacks = config.resolved_fallback_models()
        if not fallbacks and config.provider == "nim":
            fallbacks = list(_NIM_FALLBACK_MODELS)
        # 主模型本身不在备选链里（切换到同一个模型没有意义）
        self.fallback_models = [m for m in fallbacks if m and m != self.model]
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.llm_timeout,
            max_retries=0,  # SDK retries off — the app owns the retry policy
            # 绕过系统（注册表）代理直连：Windows 注册表代理对长连接空闲
            # ~240s 即杀、且间歇性完全断连；NIM 直连实测稳定（<1s）。
            # 需要走代理的用户设 ASTOCK_LLM_TRUST_ENV=1。
            http_client=httpx.Client(
                trust_env=config.llm_trust_env,
                timeout=config.llm_timeout,
            ),
        )

    # -- core ------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """One chat completion. Retries transient errors with backoff; on a
        404 "model/function not found" (providers rotate their catalogs)
        switches to the next fallback model and restarts the retry loop —
        the switch sticks for all later calls of this client."""
        tried: set[str] = {self.model}
        while True:
            last_err: Exception | None = None
            for attempt in range(self.config.llm_max_retries + 1):
                try:
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if json_mode:
                        # Ask politely; not every provider honours response_format.
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = self.client.chat.completions.create(**kwargs)
                    msg = resp.choices[0].message
                    content = msg.content or ""
                    if not content.strip():
                        # Reasoning models (glm-5.3 / deepseek thinkers) emit the
                        # answer in content after a separate reasoning_content
                        # span — with a too-small token budget the whole budget
                        # burns on reasoning and content comes back empty. Retry
                        # with a bigger budget; transient empty answers also
                        # deserve a retry.
                        finish = resp.choices[0].finish_reason
                        if finish == "length":
                            max_tokens += 2048
                            logger.warning(
                                "LLM 返回空 content（finish=length，推理吃满 token，"
                                "max_tokens 提升到 %d 重试）", max_tokens,
                            )
                        last_err = ValueError("empty response")
                        delay = self.config.llm_retry_delay * (2**attempt)
                        logger.warning(
                            "LLM 调用第 %d 次失败（empty content），%.0fs 后重试",
                            attempt + 1, delay,
                        )
                        time.sleep(delay)
                        continue
                    return content
                except _RETRYABLE as err:
                    last_err = err
                except APIStatusError as err:
                    if self._switch_model_if_missing(err, tried):
                        last_err = None
                        break  # restart the whole retry loop on the new model
                    if _is_retryable_status(err):
                        last_err = err
                    else:
                        raise LLMError(
                            f"LLM 请求失败（HTTP {getattr(err, 'status_code', '?')}）：{err}\n"
                            f"provider={self.config.provider} model={self.model}"
                        ) from err
                delay = self.config.llm_retry_delay * (2**attempt)
                logger.warning(
                    "LLM 调用第 %d 次失败（%s），%.0fs 后重试",
                    attempt + 1, str(last_err)[:120], delay,
                )
                time.sleep(delay)
            else:
                raise LLMError(
                    f"LLM 调用重试 {self.config.llm_max_retries} 次后仍失败：{last_err}\n"
                    f"provider={self.config.provider} model={self.model}"
                ) from last_err

    def _switch_model_if_missing(self, err: APIStatusError, tried: set[str]) -> bool:
        """404 "Function/Model not found" → the catalog dropped the model
        (NIM removed nemotron-3-super overnight, 2026-10-21). Switch to the
        next untried fallback; True when switched."""
        if getattr(err, "status_code", None) != 404:
            return False
        text = str(err).lower()
        if "not found" not in text or not ("function" in text or "model" in text):
            return False
        nxt = next((m for m in self.fallback_models if m not in tried), "")
        if not nxt:
            return False
        logger.warning("模型 %s 已不可用（HTTP 404 Not Found），切换备选 → %s", self.model, nxt)
        self.model = nxt
        tried.add(nxt)
        return True

    # -- JSON helpers ------------------------------------------------------

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Any:
        """Chat completion parsed as JSON. One parse-retry on malformed reply."""
        content = self.chat(system, user, json_mode=True, temperature=temperature,
                            max_tokens=max_tokens)
        parsed = parse_json(content)
        if parsed is not None:
            return parsed
        # One retry with an explicit correction notice — telling the model
        # what broke fixes most malformed-JSON failures in one round.
        logger.warning("JSON 解析失败，重试一次")
        retry_user = (
            f"{user}\n\n⚠️ 你上一次的回复无法解析为 JSON（可能带了多余文字或代码围栏）。"
            f"请**只输出一个合法的 JSON 对象**，不要任何解释、markdown 或代码围栏。"
        )
        content2 = self.chat(system, retry_user, json_mode=True, temperature=0.0,
                             max_tokens=max_tokens)
        parsed = parse_json(content2)
        if parsed is not None:
            return parsed
        raise LLMError(
            f"LLM 两次回复均无法解析为 JSON，最后一段输出开头：{content2[:200]}"
        )


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Best-effort JSON extraction: direct parse → fenced block → brace scan."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = _CODE_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # Brace scan: find the outermost { ... } or [ ... ] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None
