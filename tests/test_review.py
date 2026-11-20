"""Review / strategy-versioning tests: version roundtrip, factor
multipliers, prompt blocks, outcome computation, window filtering, and a
three-run E2E (run1 → review skipped; run2 reviews run1 → v1; run3 uses
v1 and reviews BOTH in-window runs → v2) — all mocked, zero network."""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import astock.pipeline as pipeline_mod
import astock.review as review_mod
from astock.config import load_config
from astock.screen import RegimeWeights
from astock.strategy import (
    StrategyVersion, apply_multipliers, latest_version,
    load_strategy, save_strategy,
)
from test_e2e_mock import MockLLM as BaseMockLLM
from test_e2e_mock import _mock_kline, _mock_universe

TMP = tempfile.mkdtemp(prefix="astock_review_test_")


class MockLLM(BaseMockLLM):
    """Adds the review stage to the canned E2E mock."""

    def chat_json(self, system, user, **kw):
        self.calls += 1
        if "策略复盘分析师" in system:  # Stage 6 复盘：user 按 run 分组、逐行 JSON
            pairs = []
            for line in user.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("code") and obj.get("run_dir"):
                    pairs.append((str(obj["run_dir"]), str(obj["code"])))
            return {
                "review_summary": "mock 复盘：窗口内多次选股整体符合预期",
                "regime_calibration": "mock 校准：大盘判断偏乐观，需更严格确认资金面",
                "triage_criteria": "mock 快筛补充：资金流向真实性权重高于题材热度",
                "deep_focus": "mock 深度补充：重点核查解禁压力与主力资金背离",
                "pricing_discipline": "mock 定价补充：止损放宽到 2.5×ATR",
                "stock_reviews": [{"run_dir": rd, "code": c,
                                   "change_reason": f"mock 变化原因 {rd} {c}"}
                                  for rd, c in pairs],
                "factor_multipliers": {"momentum": 0.8, "volume_price": 1.5,
                                       "capital": 1.3, "heat": 0.5, "valuation": 2.0},
                "notes": ["mock 改进点1", "mock 改进点2"],
            }
        return super().chat_json(system, user, **kw)


def _mock_quotes(codes):
    return {c: {"name": f"mock股{c}", "price": 25.0, "change_pct": 1.2} for c in codes}


def _patch_pipeline():
    pipeline_mod.fetch_universe = _mock_universe
    pipeline_mod.get_kline = _mock_kline
    pipeline_mod._merge_kline_momentum = lambda pre, limit: pre.head(limit)
    pipeline_mod.momentum_summary = lambda df: {"close": 21.0, "chg_5d": 1.2,
                                                "chg_20d": 3.4, "rsi": 55.0}
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


def _fresh_env():
    os.environ["ASTOCK_STRATEGY_DIR"] = f"{TMP}/strategies"
    os.environ["ASTOCK_CACHE"] = "0"


def test_version_roundtrip():
    assert latest_version() == 0, "fresh dir should have no versions"
    assert load_strategy() is None, "empty dir → default strategy"
    assert load_strategy(1) is None, "missing version → None"
    sv = StrategyVersion(version=1, created="2026-10-09", based_on="run_x",
                         review_summary="s", regime_calibration="r",
                         factor_multipliers={"capital": 1.3}, notes=["n1"])
    save_strategy(sv)
    assert latest_version() == 1
    back = load_strategy()          # 0 = latest
    assert back is not None and back.version == 1 and back.based_on == "run_x"
    back2 = load_strategy(1)        # 指定版本
    assert back2.factor_multipliers == {"capital": 1.3}
    save_strategy(StrategyVersion(version=2, created="2026-10-09", based_on="run_y"))
    assert latest_version() == 2
    assert load_strategy().version == 2
    assert load_strategy(1).based_on == "run_x"  # 指定旧版本仍可加载
    # 损坏的版本文件 → 默认策略（不崩溃）
    p = os.path.join(TMP, "strategies", "v3.json")
    open(p, "w", encoding="utf-8").write("{broken")
    assert load_strategy(3) is None
    os.remove(p)
    print("  版本 roundtrip OK（latest/指定/损坏防御）")


def test_apply_multipliers():
    w = RegimeWeights(momentum=0.30, volume_price=0.15, capital=0.25, heat=0.20, valuation=0.10)
    adj = apply_multipliers(w, {"capital": 1.3, "heat": 0.5})
    total = sum(vars(adj).values())
    assert abs(total - 1.0) < 1e-9, f"weights not normalized: {total}"
    assert adj.capital > adj.heat, "capital×1.3 should beat heat×0.5"
    # clamp：越界乘数收在 [0.5, 2.0]，momentum×2.0 归一化后仍占最大份额
    adj2 = apply_multipliers(w, {"momentum": 99, "valuation": -5})
    assert adj2.momentum == max(vars(adj2).values()), "clamp broken"
    # 非法值 → 按 1.0 处理，权重比例保持 regime 原样
    adj3 = apply_multipliers(w, {"momentum": "abc"})
    assert abs(adj3.momentum - w.momentum) < 1e-9, "bad multiplier should keep regime weight"
    print("  apply_multipliers OK（乘数/clamp/归一化/非法值）")


def test_prompt_blocks():
    sv = StrategyVersion(version=1, created="2026-10-09", based_on="run_x",
                         regime_calibration="校准A", triage_criteria="快筛B",
                         deep_focus="深度C")
    blocks = sv.prompt_blocks()
    assert set(blocks) == {"regime", "triage", "deep"}
    assert "校准A" in blocks["regime"] and "v1" in blocks["regime"]
    empty = StrategyVersion(version=2, created="", based_on="").prompt_blocks()
    assert empty == {}, "empty version should produce no blocks"
    print("  prompt_blocks OK（各阶段嵌入、空版本无块）")


def test_outcome():
    pick = {"price_plan": {"entry_low": 10.0, "entry_high": 11.0,
                           "stop_loss": 9.0, "target_low": 13.0, "target_high": 14.0}}
    q = {"price": 13.5, "change_pct": 2.0}
    assert "卖出目标" in review_mod._outcome(pick, q)
    assert "止损" in review_mod._outcome(pick, {"price": 8.8, "change_pct": -5})
    assert "买入区间内" in review_mod._outcome(pick, {"price": 10.5, "change_pct": 0.5})
    assert "买入区间下方" in review_mod._outcome(pick, {"price": 9.5, "change_pct": -1})
    assert "高于买入区间" in review_mod._outcome(pick, {"price": 12.0, "change_pct": 1})
    assert "[数据缺失]" in review_mod._outcome(pick, None)
    assert "[数据缺失]" in review_mod._outcome(pick, {"price": 0})
    print("  outcome 计算 OK（目标/止损/区间/缺失）")


def test_latest_run():
    assert review_mod._latest_run(f"{TMP}/no_such_dir") is None
    root = f"{TMP}/runs"
    os.makedirs(f"{root}/2026-10-08_100000", exist_ok=True)      # 无 picks.json → 忽略
    os.makedirs(f"{root}/2026-10-08_140000", exist_ok=True)
    open(f"{root}/2026-10-08_140000/picks.json", "w", encoding="utf-8").write('{"picks": []}')
    os.makedirs(f"{root}/2026-10-09_093000", exist_ok=True)
    open(f"{root}/2026-10-09_093000/picks.json", "w", encoding="utf-8").write('{"picks": [1]}')
    got = review_mod._latest_run(root)
    assert got is not None and got.name == "2026-10-09_093000", f"latest run wrong: {got}"
    print("  latest_run OK（无目录/无picks忽略/取最新）")


def test_window_runs():
    root = f"{TMP}/win"
    now = datetime(2026, 11, 19, 10, 0, 0)

    def mk(name, with_picks=True):
        os.makedirs(f"{root}/{name}", exist_ok=True)
        if with_picks:
            open(f"{root}/{name}/picks.json", "w", encoding="utf-8").write('{"picks": [1]}')

    mk("2026-11-19_093000")                    # 当日
    mk("2026-11-06_153000")                    # 13 天前（窗口内）
    mk("2026-11-05_093000")                    # 14 天前（边界日全天在窗口内）
    mk("2026-11-04_093000")                    # 15 天前（窗口外）
    mk("not-a-run-dir")                        # 名字非法 → 忽略
    mk("2026-11-10_100000", with_picks=False)  # 无 picks.json → 忽略
    assert review_mod._window_runs(f"{TMP}/no_such_root", 14, now=now) == []
    runs = review_mod._window_runs(root, 14, now=now)
    assert [r.name for r in runs] == [
        "2026-11-19_093000", "2026-11-06_153000", "2026-11-05_093000"], \
        f"window filter/order wrong: {[r.name for r in runs]}"
    assert [r.name for r in review_mod._window_runs(root, 1, now=now)] == ["2026-11-19_093000"]
    print("  window_runs OK（边界日/非法名/无picks/新→旧/窗口=1）")


def test_build_rows_extra_fields():
    run = Path(f"{TMP}/win/2026-11-12_100000")  # _build_rows 只用 run.name
    picks = [{
        "code": "600000", "name": "mock浦发", "score": 80, "verdict": "值得买入",
        "buy_logic": "量价齐升突破平台",
        "catalysts": ["降准落地", "业绩超预期"],
        "price_plan": {"entry_low": 10.0, "entry_high": 11.0, "stop_loss": 9.0,
                       "target_low": 13.0, "target_high": 14.0,
                       "anchors": {"close": 10.5}},
    }]
    rows = review_mod._build_rows(run, picks,
                                  {"600000": {"price": 13.5, "change_pct": 2.0}},
                                  datetime(2026, 11, 19))
    r = rows[0]
    assert r["run_dir"] == "2026-11-12_100000", "row must carry its source run"
    assert r["days_ago"] == 7, f"days_ago wrong: {r['days_ago']}"
    assert "量价齐升突破平台" in r["当时选股原因"] and "降准落地" in r["当时选股原因"], \
        "buy_logic + catalysts concat broken"
    assert r["累计涨跌%"] == 28.57, f"chg vs anchors.close wrong: {r['累计涨跌%']}"
    assert "已进入卖出目标区间" in r["当前表现"], r["当前表现"]
    print("  build_rows 扩展字段 OK（run_dir/days_ago/原因拼接/累计涨跌/状态判定）")


def test_two_run_e2e():
    """三轮真实时序：run1（首次，窗口内无 run→复盘跳过）→ run2（复盘窗口内
    run1 生成 v1，本身仍用默认策略——版本在 pipeline 开头已定）→ run3（默认
    加载 v1，复盘窗口内 run1+run2 两个 run → v2）→ run4（钉 v1）。"""
    os.environ["ASTOCK_STRATEGY_DIR"] = f"{TMP}/strategies_e2e"  # 独立于 roundtrip 的版本
    _patch_pipeline()
    review_mod.tencent_quotes = _mock_quotes
    out_dir = f"{TMP}/e2e_reports"

    # --- run 1：首次运行，无策略版本 → 复盘跳过 → 落盘 ---
    cfg = load_config(api_key="mock-key", candidate_count=8, triage_count=4,
                      max_picks=3, output_dir=out_dir)
    llm = MockLLM()
    result1 = pipeline_mod.run_pipeline(cfg, llm)
    calls1 = llm.calls
    sv = review_mod.run_review(cfg, llm, out_dir)   # 此时 output_dir 还没有 run
    assert sv is None, "first run should skip review"
    from astock.report import save_report
    run1_dir = save_report(cfg, result1)
    assert latest_version() == 0

    # --- run 2：仍默认策略（v1 还不存在）→ 复盘窗口内 run1 → 生成 v1 ---
    time.sleep(1.1)  # 保证 run 目录名（秒级时间戳）不同
    result2 = pipeline_mod.run_pipeline(cfg, llm)
    assert result2.strategy_version_used == 0, \
        f"v1 not yet generated, run2 should use default, got {result2.strategy_version_used}"
    assert llm.calls > calls1, "run2 made no LLM calls"
    sv1 = review_mod.run_review(cfg, llm, out_dir)  # save 前——窗口内只有 run1
    assert sv1 is not None and sv1.version == 1 and sv1.based_on == run1_dir.name
    assert latest_version() == 1
    assert sv1.review_rows and {r["run_dir"] for r in sv1.review_rows} == {run1_dir.name}, \
        "rows must be tagged with their source run"
    assert all("days_ago" in r for r in sv1.review_rows)
    result2.review = {"based_on": sv1.based_on, "based_on_runs": [sv1.based_on],
                      "window_days": 14, "run_count": 1,
                      "summary": sv1.review_summary,
                      "new_version": sv1.version, "notes": sv1.notes,
                      "rows": sv1.review_rows}
    run2_dir = save_report(cfg, result2)
    md = (run2_dir / "report.md").read_text(encoding="utf-8")
    assert "复盘（最近 14 天全部 run）" in md and "v1" in md, "report should record the review"
    assert f"#### run {run1_dir.name}" in md, "review detail should be grouped by run"
    # 复盘明细：前后对比表 + 逐只原因
    assert "复盘明细" in md and "当时收盘" in md and "累计涨跌" in md, \
        "report should have the price before/after table"
    assert "当时选股原因" in md, "report should show the original buy logic"
    assert "价格变化原因" in md and "mock 变化原因" in md, \
        "report should show per-stock change reasons from the LLM"

    # --- run 3：默认加载 v1 → 复盘窗口内 run1+run2 → 生成 v2 ---
    time.sleep(1.1)
    result3 = pipeline_mod.run_pipeline(cfg, llm)
    assert result3.strategy_version_used == 1, \
        f"run3 should default to v1, got {result3.strategy_version_used}"
    sv2 = review_mod.run_review(cfg, llm, out_dir)
    assert sv2 is not None and sv2.version == 2 and sv2.based_on == run2_dir.name
    n1 = len(json.loads((run1_dir / "picks.json").read_text(encoding="utf-8"))["picks"])
    n2 = len(json.loads((run2_dir / "picks.json").read_text(encoding="utf-8"))["picks"])
    assert {r["run_dir"] for r in sv2.review_rows} == {run1_dir.name, run2_dir.name}, \
        "window review must cover BOTH in-window runs"
    assert len(sv2.review_rows) == n1 + n2, \
        f"expected {n1}+{n2} rows, got {len(sv2.review_rows)}"
    for r in sv2.review_rows:  # 变化原因按 (run_dir, code) 回填——无跨 run 串扰
        assert r["变化原因"] == f"mock 变化原因 {r['run_dir']} {r['code']}"
    result3.review = {"based_on": sv2.based_on,
                      "based_on_runs": [sv2.based_on, run1_dir.name],
                      "window_days": 14, "run_count": 2,
                      "summary": sv2.review_summary,
                      "new_version": sv2.version, "notes": sv2.notes,
                      "rows": sv2.review_rows}
    run3_dir = save_report(cfg, result3)
    md3 = (run3_dir / "report.md").read_text(encoding="utf-8")
    assert "策略版本：v1" in md3, "report should record the version used"
    # 报告按 run 分组：明细表 + 逐只原因两节，每节各含 run1/run2 小节
    assert md3.count(f"#### run {run1_dir.name}") == 2 and \
        md3.count(f"#### run {run2_dir.name}") == 2, \
        "run3 report review should group by both in-window runs"

    # --- run 4：--strategy-version 1 指定旧版本 ---
    cfg4 = load_config(api_key="mock-key", candidate_count=8, triage_count=4,
                       max_picks=3, output_dir=out_dir, strategy_version=1)
    result4 = pipeline_mod.run_pipeline(cfg4, llm)
    assert result4.strategy_version_used == 1, "pinned version should be honored"

    print(f"  三轮 E2E OK：run1(首次,跳过) → run2(复盘→v1) → run3(默认用v1,复盘→v2)"
          f" → run4(指定v1)；picks={len(result3.picks)}")
    return 0


def main() -> int:
    print("REVIEW / STRATEGY TESTS:")
    _fresh_env()
    test_version_roundtrip()
    test_apply_multipliers()
    test_prompt_blocks()
    test_outcome()
    test_latest_run()
    test_window_runs()
    test_build_rows_extra_fields()
    _fresh_env()
    test_two_run_e2e()
    print("ALL REVIEW TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
