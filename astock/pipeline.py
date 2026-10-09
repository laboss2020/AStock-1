"""Pipeline orchestration: the five-stage stock-picking flow.

    Stage 1  大盘体检      — index/flow/board/news data + 1 LLM call
    Stage 2  量化初筛      — hard filters + multi-factor scoring, 0 LLM
    Stage 3  LLM 快筛      — 30 candidates in batches → top 15, ~4 calls
    Stage 4  深度分析      — per-candidate data bundle + 1 call each → top ≤10
    Stage 5  定价          — technical anchors + 1 call per pick → buy/sell prices

Stage 2 fetches K-lines for the filtered universe (the heaviest step, ~1
EM call per stock at ≥1s throttle). Everything is sequential by design —
parallel bursts trip Eastmoney's risk control (see data/http.py).

Every stage records its output into the run report; excluded/dropped items
are counted, never silently dropped.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .analysis import DeepVerdict, MarketRegime, analyze_market, deep_analyze, triage_batch
from .config import Config
from .data import (
    cls_news, fetch_universe, get_kline, hot_stocks, index_snapshot,
    index_trend_summary, market_fund_flow, market_news, momentum_summary,
    northbound_history, northbound_realtime, save_northbound_snapshot,
    sector_boards, stock_fund_flow, stock_news, theme_frequency,
)
from .llm import LLMClient, LLMError
from .pricing import PricePlan, _rule_fallback, compute_anchors, suggest_prices
from .screen import RegimeWeights, apply_hard_filters, score_universe, weights_for_regime
from .strategy import apply_multipliers, load_strategy

logger = logging.getLogger(__name__)


@dataclass
class FinalPick:
    """One final pick with its full analysis chain."""
    code: str
    name: str
    quant_score: float = 0.0
    triage_score: int = 0
    triage_reason: str = ""
    verdict: DeepVerdict = field(default_factory=DeepVerdict)
    price_plan: PricePlan = field(default_factory=PricePlan)
    bundle: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """Complete result of one pipeline run."""
    trade_date: str = ""
    regime: MarketRegime = field(default_factory=MarketRegime)
    universe_count: int = 0
    filtered_count: int = 0
    candidate_count: int = 0
    triaged_count: int = 0
    rejected_count: int = 0
    picks: list[FinalPick] = field(default_factory=list)
    screened_out: list[dict] = field(default_factory=list)  # 候选池 top（未进终选）
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    strategy_version_used: int = 0   # 本次 run 使用的策略版本（0 = 默认策略）
    review: dict = field(default_factory=dict)  # 选股后的复盘结果（见 review.py）


def _llm_call_counting(llm: LLMClient, result: RunResult) -> LLMClient:
    """Wrap chat_json to count LLM calls for the run report."""
    orig = llm.chat_json
    def counted(system, user, **kwargs):
        result.llm_calls += 1
        return orig(system, user, **kwargs)
    llm.chat_json = counted  # type: ignore[method-assign]
    return llm


def run_pipeline(cfg: Config, llm: LLMClient) -> RunResult:
    """Execute the full five-stage flow. Raises LLMError on fatal LLM
    failures (Stage 1 cannot proceed without LLM); downstream per-stock
    failures are caught and recorded in result.errors so one bad stock
    doesn't kill the batch.

    策略版本：默认加载最新版（strategies/ 目录，复盘生成）；cfg.strategy_version
    指定具体版本。版本带各阶段提示词校准块与因子权重乘数。"""
    result = RunResult(trade_date=datetime.now().strftime("%Y-%m-%d"))
    _llm_call_counting(llm, result)

    # ---- 策略版本加载 -----------------------------------------------------
    sv = load_strategy(cfg.strategy_version)
    result.strategy_version_used = sv.version if sv else 0
    if sv:
        logger.info("策略版本: v%d（%s 生成，based_on=%s）", sv.version, sv.created, sv.based_on)
        logger.info("复盘校准: %s", sv.review_summary[:100])
    blocks = sv.prompt_blocks() if sv else {}

    # ---- Stage 1: 大盘体检 ---------------------------------------------
    logger.info("=== Stage 1: 大盘体检 ===")
    snapshot = _collect_market_snapshot(result)
    regime = analyze_market(cfg, llm, snapshot, strategy_text=blocks.get("regime", ""))
    result.regime = regime
    weights: RegimeWeights = weights_for_regime(regime.regime)
    if sv:
        weights = apply_multipliers(weights, sv.factor_multipliers)

    # ---- Stage 2: 量化初筛 ---------------------------------------------
    logger.info("=== Stage 2: 量化初筛 ===")
    universe = fetch_universe()
    result.universe_count = len(universe)
    filtered = apply_hard_filters(
        universe,
        min_price=cfg.min_price, min_amount=cfg.min_amount,
        min_mcap=cfg.min_mcap, max_mcap=cfg.max_mcap,
        min_listed_days=cfg.min_listed_days,
        exclude_st=cfg.exclude_st, exclude_new_high=cfg.exclude_new_high,
    )
    result.filtered_count = len(filtered)
    if filtered.empty:
        result.errors.append("硬性过滤后候选池为空——过滤条件可能过严，或数据异常")
        return result

    # K-line momentum per row (the heavy part: 1 kline call per stock, top
    # 400 by quick pre-score to bound EM calls). Sequential + throttled.
    logger.info("Stage 2: 为 %d 只通过初筛的股票拉取K线计算动量（取快速预分前 400）", len(filtered))
    pre_scored = _pre_score(filtered)
    merged = _merge_kline_momentum(pre_scored, limit=400)
    scored = score_universe(merged, snapshot.get("hot_stocks", []),
                            snapshot.get("boards", []), weights)
    result.candidate_count = min(cfg.candidate_count, len(scored))
    candidates = scored.head(cfg.candidate_count)
    result.screened_out = scored.head(60).to_dict("records")

    # ---- Stage 3: LLM 快筛 ----------------------------------------------
    logger.info("=== Stage 3: LLM 快筛 (%d 只) ===", len(candidates))
    triaged = _triage_all(cfg, llm, regime, candidates, result,
                          strategy_text=blocks.get("triage", ""))

    triage_ranked = sorted(
        triaged, key=lambda t: (t.get("_triage_score", 0), t.get("score", 0.0)),
        reverse=True
    )
    result.triaged_count = len(triage_ranked)
    deep_pool = triage_ranked[: cfg.triage_count]
    result.rejected_count = len(triaged) - len(deep_pool)

    # ---- Stage 4: 深度分析 -----------------------------------------------
    logger.info("=== Stage 4: 深度分析 (%d 只) ===", len(deep_pool))
    hot_codes = {h["code"]: h for h in snapshot.get("hot_stocks", []) if h.get("code")}
    verdicts: list[tuple[dict, DeepVerdict]] = []
    for cand in deep_pool:
        code, name = cand["code"], cand["name"]
        try:
            bundle = _build_bundle(code, name, cand, hot_codes)
            verdict = deep_analyze(cfg, llm, regime, bundle,
                                   strategy_text=blocks.get("deep", ""))
            verdicts.append((cand, verdict))
        except LLMError:
            raise  # fatal: auth/quota — abort the run
        except Exception as e:
            msg = f"深度分析 {code} {name} 失败: {e}"
            logger.warning("%s\n%s", msg, traceback.format_exc())
            result.errors.append(msg)

    verdicts.sort(key=lambda cv: (verdict_rank(cv[1]), cv[1].score), reverse=True)
    top = verdicts[: cfg.max_picks]

    # ---- Stage 5: 定价 ----------------------------------------------------
    logger.info("=== Stage 5: 定价 (%d 只) ===", len(top))
    for cand, verdict in top:
        code, name = cand["code"], cand["name"]
        try:
            df = get_kline(code, days=120)
            plan = suggest_prices(cfg, llm, {"code": code, "name": name}, verdict, df,
                                  strategy_text=blocks.get("pricing", ""))
        except LLMError as e:
            # 一只股票的定价 LLM 失败不应报废整个 run（Stage 1-4 的工作都在
            # 它身后）——规则兜底出价并记录，继续下一只。df 在 try 里已绑定
            # （get_kline 不抛 LLMError）。
            msg = f"定价 {code} {name} LLM 失败（{str(e)[:80]}），已用规则兜底出价"
            logger.warning(msg)
            result.errors.append(msg)
            anchors = compute_anchors(df)
            plan = (_rule_fallback(anchors) if anchors
                    else PricePlan(rationale="K线与LLM定价均失败，无法定价", fallback=True))
        except Exception as e:
            msg = f"定价 {code} {name} 失败: {e}"
            logger.warning("%s\n%s", msg, traceback.format_exc())
            result.errors.append(msg)
            continue
        result.picks.append(FinalPick(
            code=code, name=name,
            quant_score=round(float(cand.get("score", 0.0)), 3),
            triage_score=verdict_triage_score(cand),
            verdict=verdict, price_plan=plan,
            bundle=cand.get("_bundle", {}),
        ))

    logger.info("=== 完成：%d 支最终选股 ===", len(result.picks))
    return result


def verdict_rank(v: DeepVerdict) -> int:
    """verdict → sort rank: 值得买入 > 可关注 > 放弃."""
    return {"值得买入": 2, "可关注": 1}.get(v.verdict, 0)


def verdict_triage_score(cand: dict) -> int:
    try:
        return int(cand.get("_triage_score", 0))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _collect_market_snapshot(result: RunResult) -> dict:
    """Gather all Stage-1 data. Failures degrade per-item (recorded, labeled
    [数据缺失]) — the LLM prompt sees what's missing."""
    snapshot: dict = {}
    snapshot["indexes"] = index_snapshot()
    snapshot["index_trends"] = index_trend_summary()
    snapshot["market_flow"] = market_fund_flow(days=5)

    nb = northbound_realtime()
    if nb:
        save_northbound_snapshot(nb["total"])
        hist = northbound_history(10)
        avg = round(sum(v for _, v in hist) / len(hist), 2) if hist else None
        snapshot["northbound"] = {**nb, "history_avg_10d": avg}
    else:
        snapshot["northbound"] = None

    snapshot["boards"] = sector_boards("concept", limit=15)
    snapshot["industry_boards"] = sector_boards("industry", limit=10)

    hot = hot_stocks(result.trade_date)
    snapshot["hot_stocks"] = hot
    snapshot["themes"] = [[tag, n] for tag, n in theme_frequency(hot, top=10)]

    news = market_news(limit=15) or cls_news(limit=15)
    snapshot["news"] = news
    return snapshot


def _pre_score(filtered) -> "object":
    """Cheap pre-score to bound K-line fetches: rank by the fields already
    present in the universe frame (momentum proxy = change_pct & turnover &
    inflow). Top 400 get real K-lines."""
    df = filtered.copy()
    df["pre"] = (
        df["change_pct"].clip(lower=0) / 10.0 * 0.4
        + (df["turnover_pct"].clip(3, 15) / 15.0) * 0.3
        + (df["main_inflow_yi"].fillna(0).clip(lower=0)
           / (df["amount_yi"] + 1e-9)).clip(0, 1) * 0.3
    )
    return df.sort_values("pre", ascending=False).reset_index(drop=True)


def _merge_kline_momentum(pre_scored, limit: int):
    """Fetch K-lines for the top ``limit`` rows and merge momentum columns.

    Rows whose K-lines fail keep their universe-frame fields and score 0 on
    the momentum factors — visible via the missing-data columns, never
    silently boosted.
    """
    frames = []
    for i, row in pre_scored.head(limit).iterrows():
        code = row["code"]
        df = get_kline(code, days=70)
        if df is not None and not df.empty:
            mom = momentum_summary(df)
        else:
            mom = {}
        row = row.to_dict()
        row.update(mom)
        frames.append(row)
    import pandas as pd

    return pd.DataFrame(frames)


def _triage_all(cfg, llm, regime, candidates, result, strategy_text: str = "") -> list[dict]:
    """Triage in batches of 8; merge scores; keep candidates the model
    scored. Unscored candidates are rejected (logged, never promoted)."""
    out: list[dict] = []
    batch_size = 8
    rows = candidates.to_dict("records")
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        compact = [_compact_candidate(r) for r in batch]
        try:
            scores = triage_batch(cfg, llm, regime, compact, strategy_text=strategy_text)
        except LLMError:
            raise
        except Exception as e:
            msg = f"快筛批次 {i // batch_size + 1} 失败: {e}"
            logger.warning(msg)
            result.errors.append(msg)
            continue
        for r in batch:
            sc = scores.get(r["code"])
            if sc is None:
                continue  # model skipped this candidate → rejected
            r = dict(r)
            r["_triage_score"] = sc["score"]
            r["_triage_reason"] = sc["reason"]
            out.append(r)
    return out


def _compact_candidate(row: dict) -> dict:
    """Compact per-candidate snapshot for the Stage-3 prompt."""
    return {
        "code": row["code"],
        "name": row["name"],
        "price": row.get("price"),
        "change_pct": row.get("change_pct"),
        "turnover_pct": row.get("turnover_pct"),
        "volume_ratio": row.get("volume_ratio"),
        "pe": row.get("pe"),
        "pb": row.get("pb"),
        "mcap_yi": round(row.get("mcap_yi", 0) or 0, 1),
        "amount_yi": round(row.get("amount_yi", 0) or 0, 1),
        "main_inflow_yi": round(row.get("main_inflow_yi", 0) or 0, 2),
        "chg_5d": row.get("chg_5d"),
        "chg_20d": row.get("chg_20d"),
        "dist_to_20d_high": row.get("dist_to_20d_high"),
        "rsi": row.get("rsi"),
        "quant_score": round(float(row.get("score", 0.0)), 3),
    }


def _build_bundle(code: str, name: str, cand: dict, hot_codes: dict) -> dict:
    """Full data bundle for Stage-4 deep analysis. Zero LLM calls — the
    pipeline fetches everything and the LLM reasons over the bundle."""
    kline_df = get_kline(code, days=70)
    mom = momentum_summary(kline_df) if kline_df is not None and not kline_df.empty else {}
    flow = stock_fund_flow(code, days=20)
    news = stock_news(code, limit=8)

    hot = hot_codes.get(code)
    bundle = {
        "code": code,
        "name": name,
        "行情快照": {
            "price": cand.get("price"),
            "change_pct": cand.get("change_pct"),
            "turnover_pct": cand.get("turnover_pct"),
            "pe_ttm": cand.get("pe"),
            "pb": cand.get("pb"),
            "mcap_yi": round(cand.get("mcap_yi", 0) or 0, 1),
            "amount_yi": round(cand.get("amount_yi", 0) or 0, 1),
        },
        "技术面": mom,
        "资金面": {
            "主力净流入_近5日_亿": flow.get("sum_5d"),
            "主力净流入_近20日_亿": flow.get("sum_20d"),
            "今日": flow.get("today"),
            "近20日明细": (flow.get("history") or [])[-10:],
        },
        "题材与板块": {
            "当日是否上涨停热股榜": bool(hot),
            "题材归因": (hot.get("reason", "") if hot else "") or "未上榜",
        },
        "近期新闻": [
            {"date": n.get("date", ""), "title": n.get("title", "")}
            for n in news
        ] or ["[数据缺失]"],
        "量化初筛得分": round(float(cand.get("score", 0.0)), 3),
        "LLM快筛": {"score": cand.get("_triage_score"), "reason": cand.get("_triage_reason")},
    }
    cand["_bundle"] = bundle
    return bundle
