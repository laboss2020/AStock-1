"""LLM 端点探测（CI 用）：NIM 不通时自动切 GitHub Models。

背景（2026-10 下旬起）：NIM chat 端点对 GitHub Actions runner 的请求
黑洞——TCP 能连上但 150~300s 内零字节响应（Request timed out），同
一 key 从本地秒回。自 10-21 起 CI 无一次真实成功。本脚本在 workflow
跑 main.py 之前各发一次极小 chat（max_tokens=16）验证两个通道：

  1. NIM（secrets.NVIDIA_API_KEY，model 取 ASTOCK_MODEL secret）
  2. GitHub Models（Actions 内置 GITHUB_TOKEN + permissions: models:read）

哪个通就用哪个，结果写 $GITHUB_OUTPUT（provider/model/fallback）供
「运行选股」步骤读取；两个都不通 exit 1 快速失败（省下白拉数据的
时间，日志里也一眼看清是端点层故障）。

本地运行：python probe_llm.py（NVIDIA_API_KEY 走 .env；GITHUB_TOKEN
为空则跳过 GitHub Models 通道）。
"""

from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NIM_DEFAULT_MODEL = "z-ai/glm-5.3-flash"
GH_MODEL = "openai/gpt-4.1-mini"
GH_FALLBACK = "openai/gpt-4o-mini"
PING = [{"role": "user", "content": "Reply with one word: PONG"}]


def _attempt(base_url: str, key: str, model: str, read_timeout: float) -> bool:
    """One tiny chat call; True = endpoint answered HTTP 200."""
    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=10.0, read=read_timeout,
                              write=read_timeout, pool=read_timeout),
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=read_timeout),
    )
    try:
        resp = client.chat.completions.create(
            model=model, messages=PING, max_tokens=16, temperature=0.0,
        )
        ok = bool(resp.choices)
        if ok:
            print(f"  ✓ {base_url} model={model} 可用"
                  f"（finish={resp.choices[0].finish_reason}）")
        return ok
    except Exception as e:  # noqa: BLE001 — 探测只关心通/不通
        print(f"  ✗ {base_url} model={model}：{type(e).__name__}: {str(e)[:120]}")
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def _try_lane(name: str, base_url: str, key: str, model: str,
              attempts: int, read_timeout: float) -> bool:
    if not key:
        print(f"[{name}] 跳过（无 key）")
        return False
    print(f"[{name}] 探测 {model}（{attempts} 次 × {read_timeout:.0f}s）…")
    for i in range(attempts):
        if _attempt(base_url, key, model, read_timeout):
            return True
        if i < attempts - 1:
            time.sleep(2)
    print(f"[{name}] 不可用")
    return False


def _emit(provider: str, model: str, fallback: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"provider={provider}\nmodel={model}\nfallback={fallback}\n")
    print(f"选定通道：provider={provider} model={model}")


def main() -> int:
    nim_key = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NIM_API_KEY", "")
    nim_model = os.environ.get("ASTOCK_MODEL", "") or NIM_DEFAULT_MODEL
    gh_key = os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", "")
    if nim_key or gh_key:
        print("LLM 端点探测：")
    else:
        print("LLM 端点探测：NVIDIA_API_KEY 与 GITHUB_TOKEN 均未设置")
        return 1

    # NIM 平时 <2s 响应；30s 内不回即视为不可用（10-22 实测"慢队列"模式
    # 单请求 4-5 分钟，30 次调用要 2h+，同样不可用于 4h 预算——一并切走）。
    if _try_lane("NIM", "https://integrate.api.nvidia.com/v1", nim_key,
                 nim_model, attempts=2, read_timeout=30.0):
        _emit("nim", nim_model, "")  # fallback 留空 → llm.py 内置 NIM 备选链
        return 0

    # GitHub Models：GitHub 自家机房，Actions 内置 token 即可（需
    # permissions: models: read）；免费层限额紧，gpt-4.1-mini 又快又省。
    if _try_lane("GitHub Models", "https://models.github.ai/inference",
                 gh_key, GH_MODEL, attempts=2, read_timeout=45.0):
        _emit("github_models", GH_MODEL, GH_FALLBACK)
        return 0

    print("✗ 所有 LLM 端点均不可达——今天跑不了，请检查 NVIDIA_API_KEY / "
          "GITHUB_TOKEN（models: read）或稍后重试")
    return 1


if __name__ == "__main__":
    sys.exit(main())
