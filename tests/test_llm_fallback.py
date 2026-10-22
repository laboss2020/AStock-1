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


def test_first_dict_helper():
    from astock.llm import first_dict
    assert first_dict([{"a": 1}, {"b": 2}]) == {"a": 1}
    assert first_dict([1, {"a": 1}, 2]) == {"a": 1}
    assert first_dict([1, 2]) == {}
    assert first_dict({"a": 1}) == {"a": 1}
    assert first_dict("text") == {}
    print("  first_dict helper OK")


def test_deep_pricing_list_wrapped():
    """glm-5.3-flash 把 JSON 对象包进数组——deep_analyze / pricing 不崩，
    取第一个 dict 元素（2026-10-23 CI 实测 003017 因此失败过）。"""
    import tempfile
    from types import SimpleNamespace
    import astock.analysis as analysis_mod
    import astock.pricing as pricing_mod
    from test_e2e_mock import _mock_kline

    class ListLLM:
        def __init__(self, payload):
            self.payload = payload

        def chat_json(self, system, user, **kw):
            return [self.payload]  # 对象包进数组

    cfg = load_config(api_key="mock-key")

    v = analysis_mod.deep_analyze(cfg, ListLLM(
        {"code": "600601", "score": 77, "verdict": "值得买入",
         "buy_logic": "包在数组里", "risks": [], "catalysts": [],
         "dimension_summary": {"技术面": "多头"}}),
        analysis_mod.MarketRegime(), {"code": "600601", "name": "mock"})
    assert v.score == 77 and v.verdict == "值得买入", "wrapped dict not unwrapped"

    df = _mock_kline("600601")
    close = float(df["Close"].iloc[-1])
    e_lo, e_hi = round(close * 0.98, 2), round(close * 1.02, 2)
    plan = pricing_mod.suggest_prices(cfg, ListLLM(
        {"entry_low": e_lo, "entry_high": e_hi,
         "stop_loss": round(e_lo - close * 0.04, 2),
         "target_low": round(e_hi * 1.10, 2),
         "target_high": round(e_hi * 1.25, 2)}),
        {"code": "600601", "name": "mock"},
        SimpleNamespace(score=80, verdict="值得买入", buy_logic="x", risks=[]), df)
    assert not plan.fallback and plan.entry_low > 0, "wrapped pricing not unwrapped"
    print("  deep/pricing 数组包裹防御 OK")


def test_deep_empty_list_degrades():
    """空数组 → 降级为默认"放弃"，不崩（pipeline 兜一层 errors 记录）。"""
    import astock.analysis as analysis_mod

    class EmptyLLM:
        def chat_json(self, system, user, **kw):
            return []

    v = analysis_mod.deep_analyze(load_config(api_key="mock-key"), EmptyLLM(),
                                  analysis_mod.MarketRegime(),
                                  {"code": "600602", "name": "mock"})
    assert v.verdict == "放弃" and v.score == 0
    print("  deep 空数组降级为放弃 OK")


def test_pipeline_skips_failing_deep():
    """深度分析单只 LLMError（如 glm-5.3-flash 超时）→ 只跳过该只，
    其余照常定价出 picks——成功的部分不被报废（2026-10-23 run #37 教训）。"""
    import tempfile
    import astock.pipeline as pipeline_mod
    from astock.llm import LLMError
    from test_e2e_mock import MockLLM as BaseMockLLM, _mock_kline, _mock_universe

    class FlakyDeepLLM(BaseMockLLM):
        deep_calls = 0

        def chat_json(self, system, user, **kw):
            if "做深度分析" in system:
                FlakyDeepLLM.deep_calls += 1
                if FlakyDeepLLM.deep_calls == 1:  # 第一只被深度分析的股票超时
                    raise LLMError("mock: Request timed out.")
            return super().chat_json(system, user, **kw)

    pipeline_mod.fetch_universe = _mock_universe
    pipeline_mod.get_kline = _mock_kline
    pipeline_mod._merge_kline_momentum = lambda pre, limit: pre.head(limit)
    pipeline_mod.momentum_summary = lambda df: {"close": 21.0, "chg_5d": 1.2,
                                                "chg_20d": 3.4, "rsi": 55.0}
    pipeline_mod.stock_fund_flow = lambda code, days=20: {
        "history": [{"date": "2026-09-23", "main_net_yi": 0.5}],
        "sum_5d": 2.5, "sum_20d": 8.0, "today": 0.3}
    pipeline_mod.stock_news = lambda code, limit=8: [
        {"date": "2026-09-23", "title": f"mock 新闻 {code}"}]
    pipeline_mod._collect_market_snapshot = lambda result: {
        "indexes": [], "index_trends": [], "market_flow": [],
        "northbound": None, "boards": [], "industry_boards": [],
        "hot_stocks": [], "themes": [], "news": []}

    out = tempfile.mkdtemp(prefix="astock_skip_test_")
    cfg = load_config(api_key="mock-key", candidate_count=8, triage_count=4,
                      max_picks=3, output_dir=out)
    result = pipeline_mod.run_pipeline(cfg, FlakyDeepLLM())
    assert FlakyDeepLLM.deep_calls >= 2, "deep stage never ran the failing branch"
    assert result.picks, "partial deep failures must not scrap the run"
    assert len(result.picks) == 3, f"expected 3 picks, got {len(result.picks)}"
    assert any(e.startswith("深度分析") and "失败" in e for e in result.errors), \
        f"failure not recorded: {result.errors}"
    print(f"  单只深度分析失败跳过 OK（picks={len(result.picks)}, errors 记录失败）")


def main() -> int:
    print("LLM FALLBACK TESTS:")
    _cleanup()
    test_404_switches_and_succeeds()
    test_404_switch_sticks_for_next_calls()
    test_404_exhausted_raises()
    test_non404_no_switch()
    test_nim_builtin_defaults()
    test_primary_excluded_from_chain()
    test_first_dict_helper()
    test_deep_pricing_list_wrapped()
    test_deep_empty_list_degrades()
    test_pipeline_skips_failing_deep()
    _cleanup()
    print("ALL LLM FALLBACK TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
