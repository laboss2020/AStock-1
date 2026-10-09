"""Hermetic end-to-end pipeline test: mock LLM + mock data, zero network.

Verifies the full five-stage wiring — triage batching, verdict sorting,
price validation (including the rule-fallback path), report rendering and
save_report — without touching the network (EM is often banned on dev
machines; this test always runs).
"""

import json
import re
import sys
import types

import pandas as pd
import numpy as np

import astock.pipeline as pipeline_mod
from astock.config import load_config
from astock.data.http import build_session
from astock.llm import LLMClient


# --- Mock LLM: sniffs the stage from the prompt, returns canned JSON ---------
class MockLLM(LLMClient):
    def __init__(self):
        self.calls = 0
        self.timeout = 30
        self.max_retries = 0

    def chat(self, system, user, **kw):
        self.calls += 1
        return "mock"

    def chat_json(self, system, user, **kw):
        self.calls += 1
        if "建议的买入价格区间" in system:          # Stage 5 定价
            close = float(re.search(r"最新收盘价: ([\d.]+)", user).group(1))
            e_lo, e_hi = round(close * 0.98, 2), round(close * 1.02, 2)
            return {"entry_low": e_lo, "entry_high": e_hi,
                    "stop_loss": round(e_lo - close * 0.04, 2),
                    "target_low": round(e_hi * 1.10, 2),
                    "target_high": round(e_hi * 1.25, 2),
                    "holding_period": "2-4周", "position_pct": "≤10%",
                    "rationale": "mock 定价"}
        if "候选池中快速评审" in system:            # Stage 3 快筛
            codes = re.findall(r'"code": "(\d{6})"', user)
            return [{"code": c, "score": 80 - i * 5, "reason": f"mock {c}"}
                    for i, c in enumerate(codes)]
        if "做深度分析" in system:                   # Stage 4 深度分析
            code = re.search(r"候选股票 (\d{6})", system).group(1)
            return {"code": code, "score": 82, "verdict": "值得买入",
                    "buy_logic": "mock 买入逻辑",
                    "risks": ["mock 风险"], "catalysts": ["mock 催化"],
                    "dimension_summary": {"技术面": "多头", "基本面": "合理",
                                          "资金面": "净流入", "题材政策面": "热",
                                          "风险面": "低"}}
        return {"regime": "进攻", "temperature": "乐观", "style": "中小成长",
                "risk_appetite": "高", "sector_focus": ["AI"], "avoid_sectors": ["地产"],
                "rationale": "mock 大盘体检"}


# --- Mock data: synthetic universe + klines ---------------------------------
def _mock_universe(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "code": f"60060{i}", "name": f"mock股{i}",
            "price": 20.0 + i, "change_pct": 2.0 + i * 0.3,
            "volume": 1e6, "amount": 3e8, "turnover_pct": 5.0,
            "pe": 30.0, "volume_ratio": 1.5, "high": 25.0, "low": 19.0,
            "mcap": (50 + i) * 1e8, "pb": 3.0,
            "listed_date": 20200101, "main_inflow": 5e7,
            "mcap_yi": 50.0 + i, "amount_yi": 3.0, "main_inflow_yi": 0.5,
            "listed_days": 2000,
        })
    return pd.DataFrame(rows)


def _mock_kline(code: str, days: int = 70):
    rng = np.random.default_rng(sum(map(ord, code)))
    base = 20.0
    dates = pd.date_range("2026-06-01", periods=days, freq="B")
    close = base + np.cumsum(rng.normal(0.05, 0.4, days))
    high = close * (1 + rng.uniform(0.005, 0.02, days))
    low = close * (1 - rng.uniform(0.005, 0.02, days))
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, days)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": rng.integers(1e5, 5e5, days).astype(float),
        "Amount": 1e8,
    })


def main() -> int:
    # Patch network-touching functions to their mock twins
    pipeline_mod.fetch_universe = _mock_universe
    pipeline_mod.get_kline = _mock_kline
    # Skip the heavy K-line-merge (mock klines already carry momentum in Stage
    # 4 bundles via _build_bundle); keep the plumbing test focused on wiring
    pipeline_mod._merge_kline_momentum = lambda pre, limit: pre.head(limit)
    pipeline_mod.momentum_summary = lambda df: {"close": 21.0, "chg_5d": 1.2,
                                                "chg_20d": 3.4, "rsi": 55.0}
    # Fully hermetic: mock the remaining per-stock fetches too
    pipeline_mod.stock_fund_flow = lambda code, days=20: {
        "history": [{"date": "2026-09-23", "main_net_yi": 0.5}],
        "sum_5d": 2.5, "sum_20d": 8.0, "today": 0.3,
    }
    pipeline_mod.stock_news = lambda code, limit=8: [
        {"date": "2026-09-23", "title": f"mock 新闻 {code}"}]
    pipeline_mod._collect_market_snapshot = lambda result: {
        "indexes": [], "index_trends": [], "market_flow": [],
        "northbound": None, "boards": [], "industry_boards": [],
        "hot_stocks": [], "themes": [], "news": [],
    }

    cfg = load_config(api_key="mock-key", candidate_count=8, triage_count=4,
                      max_picks=3, output_dir="reports_test")
    llm = MockLLM()

    result = pipeline_mod.run_pipeline(cfg, llm)

    # --- assertions ----------------------------------------------------------
    assert result.universe_count == 12, f"universe_count={result.universe_count}"
    assert result.triaged_count > 0, "triage produced nothing"
    assert result.picks, "no final picks"
    assert len(result.picks) <= 3, f"max_picks violated: {len(result.picks)}"
    assert llm.calls == result.llm_calls, f"llm call count {llm.calls} != {result.llm_calls}"

    for p in result.picks:
        pl = p.price_plan
        assert pl.entry_low > 0 and pl.stop_loss > 0 and pl.target_high > 0, \
            f"{p.code}: prices not populated ({pl})"
        assert pl.stop_loss < pl.entry_low <= pl.entry_high < pl.target_low <= pl.target_high, \
            f"{p.code}: number relationship violated: stop={pl.stop_loss} " \
            f"entry=[{pl.entry_low},{pl.entry_high}] target=[{pl.target_low},{pl.target_high}]"
        assert p.verdict.verdict == "值得买入", f"{p.code}: verdict={p.verdict.verdict}"
        assert p.verdict.dimension_summary, f"{p.code}: no dimension_summary"

    # report + save
    from astock.report import render_report, save_report
    md = render_report(result)
    assert "A股大盘选股报告" in md and "买入区间" in md and "免责声明" in md
    out_dir = save_report(cfg, result)
    picks = json.loads((out_dir / "picks.json").read_text(encoding="utf-8"))
    assert picks["picks"], "picks.json empty"
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["pick_count"] == len(result.picks)

    print(f"E2E MOCK OK: universe=12 triaged={result.triaged_count} "
          f"picks={len(result.picks)} llm_calls={result.llm_calls}")
    print(f"  saved: {out_dir}")
    for p in result.picks:
        pl = p.price_plan
        print(f"  {p.code} {p.name}: entry {pl.entry_low}~{pl.entry_high} "
              f"stop {pl.stop_loss} target {pl.target_low}~{pl.target_high} "
              f"({p.verdict.verdict} {p.verdict.score})")

    # rule-fallback path: LLM prices that violate the relationship
    from astock.pricing import PricePlan, _rule_fallback, _validate
    anchors = {"close": 100.0, "ma20": 98.0, "atr14": 2.0}
    bad = PricePlan(entry_low=110, entry_high=105, stop_loss=120,
                    target_low=100, target_high=90)
    assert not _validate(bad, anchors), "_validate accepted an invalid plan"
    fb = _rule_fallback(anchors)
    assert fb.fallback and _validate(fb, anchors), "_rule_fallback produced an invalid plan"

    print("  fallback path OK (invalid plan rejected, rule fallback valid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
