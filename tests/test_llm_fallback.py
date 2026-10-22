"""LLM 404 model-fallback tests: a 404 "Function/Model not found" switches
to the next fallback model and the call succeeds; non-404 failures don't
switch; exhausted fallbacks raise. All mocked — no network."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from openai import APIStatusError

from astock.config import load_config
from astock.llm import LLMClient

URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _status_err(status: int, detail: str) -> APIStatusError:
    req = httpx.Request("POST", URL)
    body = {"status": status, "title": "Not Found", "detail": detail}
    return APIStatusError(detail, response=httpx.Response(status, request=req, json=body), body=body)


def _err404() -> APIStatusError:
    return _status_err(404, "Function 'ac74040f-9fc9': Not found for account 'Xis'")


def _err401() -> APIStatusError:
    return _status_err(401, "Unauthorized: invalid API key")


def _ok(content: str = "OK"):
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


class _FakeAPI:
    """Plays a scripted list of results (values or exceptions) in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _client(model: str, fallback_env: str) -> LLMClient:
    os.environ["ASTOCK_FALLBACK_MODELS"] = fallback_env
    try:
        cfg = load_config(api_key="mock-key", provider="nim", model=model)
        llm = LLMClient(cfg)
    finally:
        os.environ.pop("ASTOCK_FALLBACK_MODELS", None)
    return llm


def _play(llm: LLMClient, script: list):
    api = _FakeAPI(script)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=api))
    return api


def _cleanup():
    os.environ.pop("ASTOCK_FALLBACK_MODELS", None)


def test_404_switches_and_succeeds():
    llm = _client("m/primary", "m/fb1, m/fb2")
    api = _play(llm, [_err404(), _ok("OK")])  # primary 404 → fb1 succeeds
    out = llm.chat("sys", "user")
    assert out == "OK"
    assert llm.model == "m/fb1", f"should switch to fb1, got {llm.model}"
    assert api.calls == 2
    print("  404 → 切到 fb1 成功 OK")


def test_404_switch_sticks_for_next_calls():
    llm = _client("m/primary", "m/fb1")
    _play(llm, [_err404(), _ok("A"), _ok("B")])
    assert llm.chat("sys", "u1") == "A"
    assert llm.chat("sys", "u2") == "B"  # no more 404 — stays on fb1
    print("  切换对后续调用生效 OK")


def test_404_exhausted_raises():
    llm = _client("m/primary", "m/fb1,m/fb2")
    api = _play(llm, [_err404(), _err404(), _err404()])  # primary + both fallbacks
    try:
        llm.chat("sys", "user")
        raise AssertionError("should raise after all fallbacks 404")
    except Exception as e:
        assert "404" in str(e) or "Not Found" in str(e) or "失败" in str(e), str(e)[:100]
    assert api.calls == 3, f"each model tried once, got {api.calls}"
    print("  备选耗尽 → 报错 OK（每模型只试一次）")


def test_non404_no_switch():
    llm = _client("m/primary", "m/fb1")
    api = _play(llm, [_err401()])
    try:
        llm.chat("sys", "user")
        raise AssertionError("401 should raise immediately")
    except Exception:
        pass
    assert llm.model == "m/primary", "401 must not switch models"
    assert api.calls == 1, "401 must not retry"
    print("  401 不切换不重试 OK")


def test_nim_builtin_defaults():
    llm = _client("nvidia/nemotron-3-super-120b-a12b", "")  # no env → built-in chain
    assert "z-ai/glm-5.3-flash" in llm.fallback_models
    assert "deepseek-ai/deepseek-v4.1-flash" in llm.fallback_models
    assert llm.fallback_models[0] == "z-ai/glm-5.3-flash"
    print(f"  NIM 内置备选链 OK（{len(llm.fallback_models)} 个）")


def test_primary_excluded_from_chain():
    llm = _client("z-ai/glm-5.3-flash", "")  # primary == first builtin fallback
    assert "z-ai/glm-5.3-flash" not in llm.fallback_models, "primary must be excluded"
    assert llm.fallback_models, "remaining fallbacks stay"
    print("  主模型从备选链排除 OK")


def main() -> int:
    print("LLM FALLBACK TESTS:")
    _cleanup()
    test_404_switches_and_succeeds()
    test_404_switch_sticks_for_next_calls()
    test_404_exhausted_raises()
    test_non404_no_switch()
    test_nim_builtin_defaults()
    test_primary_excluded_from_chain()
    _cleanup()
    print("ALL LLM FALLBACK TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
