"""LLM 端点探测（CI 用）：NIM 不通时切到自定义 OpenAI 兼容兜底通道。

背景（2026-10 下旬起）：NIM chat 端点对 GitHub Actions runner 的请求
黑洞——TCP 能连上但 150~300s 内零字节响应（Request timed out），同
一 key 从本地秒回；自 10-21 起 CI 无一次真实成功。原计划的 GitHub
Models 兜底也死了（2026-07-30 整个服务下线，models.github.ai 只剩
通配 "OK" 应答）——所以兜底通道改为**通用 OpenAI 兼容**：注册任意
一家（OpenRouter / 智谱 bigmodel.cn / DeepSeek / …），在 repo secrets
配三个值即可：

  ASTOCK_FALLBACK_BASE_URL  如 https://openrouter.ai/api/v1
  ASTOCK_FALLBACK_KEY       该平台的 API key
  ASTOCK_FALLBACK_MODEL     如 nvidia/nemotron-3-ultra-550b-a55b:free

本脚本在 workflow 跑 main.py 之前各发一次极小 chat（max_tokens=16）：

  1. NIM（secrets.NVIDIA_API_KEY，model 取 ASTOCK_MODEL secret）
  2. 兜底通道（上述三 secret，缺一跳过）

哪个通就用哪个，结果写 $GITHUB_OUTPUT（provider/model/base_url/
fallback）供「运行选股」步骤读取；两个都不通 exit 1 快速失败（省下
白拉数据的时间，日志里也一眼看清是端点层故障）。

本地运行：python probe_llm.py（NVIDIA_API_KEY 走 .env；未配兜底
secret 则跳过兜底通道）。
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
        # 防御：网关异常时 SDK 可能解析出非 ChatCompletion（如裸 str——
        # GitHub Models 下线后 models.github.ai 全域返回 JSON "OK"）。
        if not isinstance(resp, str) and getattr(resp, "choices", None):
            print(f"  ✓ {base_url} model={model} 可用"
                  f"（finish={resp.choices[0].finish_reason}）")
            return True
        print(f"  ✗ {base_url} model={model}：响应异常（非 chat completion）")
        return False
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
    print(f"[{name}] 探测 {model}（{attempts} 次 × {read_timeout:.0f}s）…")
    for i in range(attempts):
        if _attempt(base_url, key, model, read_timeout):
            return True
        if i < attempts - 1:
            time.sleep(2)
    print(f"[{name}] 不可用")
    return False


def _fallback_cfg() -> tuple[str, str, str] | None:
    """通用兜底通道配置（base_url, key, model），缺一返回 None。"""
    base = os.environ.get("ASTOCK_FALLBACK_BASE_URL", "").strip()
    key = os.environ.get("ASTOCK_FALLBACK_KEY", "").strip()
    model = os.environ.get("ASTOCK_FALLBACK_MODEL", "").strip()
    if base and key and model:
        return base, key, model
    return None


def _emit(provider: str, model: str, fallback: str, base_url: str = "") -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"provider={provider}\nmodel={model}\n"
                    f"fallback={fallback}\nbase_url={base_url}\n")
    print(f"选定通道：provider={provider} model={model}"
          + (f" base_url={base_url}" if base_url else ""))


def main() -> int:
    nim_key = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NIM_API_KEY", "")
    nim_model = os.environ.get("ASTOCK_MODEL", "") or NIM_DEFAULT_MODEL
    if not nim_key and not _fallback_cfg():
        print("LLM 端点探测：NVIDIA_API_KEY 与兜底通道 secret 均未设置")
        return 1
    print("LLM 端点探测：")

    # NIM 平时 <2s 响应；30s 内不回即视为不可用（10-22 实测"慢队列"模式
    # 单请求 4-5 分钟，30 次调用要 2h+，同样不可用于 4h 预算——一并切走）。
    if nim_key and _try_lane("NIM", "https://integrate.api.nvidia.com/v1",
                             nim_key, nim_model, attempts=2, read_timeout=30.0):
        _emit("nim", nim_model, "")  # fallback 留空 → llm.py 内置 NIM 备选链
        return 0
    if not nim_key:
        print("[NIM] 跳过（无 NVIDIA_API_KEY）")

    # 通用 OpenAI 兼容兜底（NIM 对 CI 黑洞后的主用通道）。
    fb = _fallback_cfg()
    if fb:
        base, key, model = fb
        if _try_lane("兜底通道", base, key, model, attempts=2, read_timeout=45.0):
            _emit("openai_compatible", model, "", base_url=base)
            return 0
    else:
        print("[兜底通道] 跳过（未配 ASTOCK_FALLBACK_BASE_URL/KEY/MODEL secret）")

    print("✗ 所有 LLM 端点均不可达——今天跑不了。排查：NVIDIA_API_KEY 是否有效 / "
          "是否已配 ASTOCK_FALLBACK_* 三个 secret（注册任意 OpenAI 兼容平台，"
          "如 openrouter.ai 或 bigmodel.cn）/ 稍后重试")
    return 1


if __name__ == "__main__":
    sys.exit(main())
