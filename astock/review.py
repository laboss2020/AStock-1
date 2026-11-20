"""Post-run review: track how the recent runs' picks actually performed
and distill the comparison into a new strategy version.

    1. locate the runs to review (all report dirs within the window,
       default 14 calendar days — ASTOCK_REVIEW_WINDOW_DAYS)
    2. read their picks.json + pull current quotes for all codes
       (1 batched call, deduped across runs)
    3. code-computed outcome per pick (已达目标/跌破止损/回落到买入区间…)
    4. one LLM call (grouped per run): review summary → strategy
       improvements (structured JSON)
    5. save strategies/vN.json (version = latest + 1)

Stateless by design: the same pick is re-judged on every run within the
window (a finished trade shows its final state; no "done" bookkeeping).
Failures degrade per-item and are logged — the review never blocks a
finished run. 窗口内无历史 run（首次运行）不生成策略版本。

⚠️ 复盘由 AI 自动生成，仅供研究参考，不构成任何投资建议。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .data import tencent_quotes
from .strategy import StrategyVersion, latest_version, save_strategy

logger = logging.getLogger(__name__)

# 复盘 user prompt 的行数上限：超限按 run 均匀裁剪（防手动同日多跑把
# prompt 撑爆；CI gate 一天一跑，窗口内典型 5~10 个 run，不会触发）。
_MAX_PROMPT_ROWS = 150


def _run_dir_date(name: str) -> datetime | None:
    """'2026-11-19_093045' → datetime; unparseable dir name → None."""
    try:
        return datetime.strptime(name[:15], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def _window_runs(reports_root: str, window_days: int,
                 now: datetime | None = None) -> list[Path]:
    """Report dirs within the last ``window_days`` calendar days (name =
    YYYY-MM-DD_HHMMSS). Newest first. Dirs whose name doesn't parse or
    that lack picks.json are skipped. Cutoff compares dates (the whole
    boundary day stays in the window — no hour-of-day jitter)."""
    now = now or datetime.now()
    cutoff = (now - timedelta(days=window_days)).date()
    root = Path(reports_root)
    if not root.is_dir():
        return []
    runs = []
    for p in root.iterdir():
        if not (p.is_dir() and (p / "picks.json").exists()):
            continue
        d = _run_dir_date(p.name)
        if d is not None and d.date() >= cutoff:
            runs.append(p)
    return sorted(runs, reverse=True)


def _latest_run(reports_root: str) -> Path | None:
    """Latest report dir under ``reports_root`` (degenerate window)."""
    runs = _window_runs(reports_root, 365_000)  # ~1000 年：覆盖全部 run 且不溢出 date
    return runs[0] if runs else None


def _outcome(pick: dict, quote: dict | None) -> str:
    """Code-computed verdict on one pick's performance vs its price plan:
    current price vs 止损/目标/买入区间. Missing data is labeled, never
    guessed."""
    if not quote or not float(quote.get("price", 0)):
        return "[数据缺失]"
    price = float(quote["price"])
    parts = [f"现价 {price}", f"当日涨跌 {quote.get('change_pct', 0)}%"]
    pp = pick.get("price_plan") or {}
    stop = float(pp.get("stop_loss") or 0)
    target_low = float(pp.get("target_low") or 0)
    entry_low = float(pp.get("entry_low") or 0)
    entry_high = float(pp.get("entry_high") or 0)
    if stop and price <= stop:
        parts.append("已跌破止损位")
    elif target_low and price >= target_low:
        parts.append("已进入卖出目标区间")
    elif entry_low and price < entry_low:
        parts.append("回落至买入区间下方")
    elif entry_high and price <= entry_high:
        parts.append("处于建议买入区间内")
    elif entry_high:
        parts.append("高于买入区间上沿（未追入）")
    return "，".join(parts)


def _load_run_picks(run: Path) -> list[dict]:
    """Read one run's picks.json → picks list; read failure / no picks → []
    (per-run tolerance: one broken run must not scrap the review)."""
    try:
        raw = json.loads((run / "picks.json").read_text(encoding="utf-8"))
        return raw.get("picks") or []
    except Exception as e:
        logger.warning("读取报告失败 %s: %s，跳过该 run", run.name, e)
        return []


def _build_rows(run: Path, picks: list[dict], quotes: dict,
                today: datetime) -> list[dict]:
    """One run's picks → review rows (当时 vs 现在), each tagged with its
    source run (run_dir) and age (days_ago)."""
    run_date = _run_dir_date(run.name)
    days_ago = ((today.date() - run_date.date()).days
                if run_date is not None else -1)
    rows = []
    for p in picks:
        code = str(p.get("code", ""))
        quote = quotes.get(code)
        pp = p.get("price_plan") or {}
        then_close = float((pp.get("anchors") or {}).get("close") or 0)
        now_price = float((quote or {}).get("price") or 0)
        chg_since = (round((now_price / then_close - 1) * 100, 2)
                     if then_close and now_price else None)
        reason = str(p.get("buy_logic", "") or "")
        cats = p.get("catalysts") or []
        if cats:
            reason += "；催化剂：" + "；".join(str(c) for c in cats)
        rows.append({
            "run_dir": run.name,
            "days_ago": days_ago,
            "code": code,
            "name": p.get("name", ""),
            "当时评分": p.get("score", ""),
            "当时结论": p.get("verdict", ""),
            "当时收盘价": then_close or "",
            "当前价": now_price or "",
            "累计涨跌%": chg_since,
            "当时买入区间": _fmt_zone(p),
            "当时止损": pp.get("stop_loss", ""),
            "当时卖出目标": _fmt_target(p),
            "当前表现": _outcome(p, quote),
            "当时选股原因": reason[:200],
        })
    return rows


def run_review(cfg, llm, reports_root: str) -> StrategyVersion | None:
    """Review all runs within the window → new strategy version. Returns
    None when there is nothing to review (no in-window run) or fails."""
    window_days = max(1, int(getattr(cfg, "review_window_days", 14) or 14))
    runs = _window_runs(reports_root, window_days)
    if not runs:
        logger.info("窗口 %d 天内无历史 run（首次运行或窗口过小），跳过复盘"
                    "——不生成策略版本", window_days)
        return None

    run_picks: list[tuple[Path, list[dict]]] = []
    for run in runs:
        picks = _load_run_picks(run)
        if picks:
            run_picks.append((run, picks))
        else:
            logger.info("run %s 无选股或读失败，跳过该 run", run.name)
    if not run_picks:
        logger.info("窗口内所有 run 均无选股，跳过复盘")
        return None

    # --- 当前表现：codes 跨 run 合并去重，1 次腾讯批量调用（内部分批）----
    codes: list[str] = []
    seen: set[str] = set()
    for _, picks in run_picks:
        for p in picks:
            c = str(p.get("code", ""))
            if c and c not in seen:
                seen.add(c)
                codes.append(c)
    try:
        quotes = tencent_quotes(codes) if codes else {}
    except Exception as e:
        logger.warning("复盘拉取行情失败: %s（按数据缺失处理）", e)
        quotes = {}

    today = datetime.now()
    rows = []
    for run, picks in run_picks:
        rows.extend(_build_rows(run, picks, quotes, today))
    if len(rows) > _MAX_PROMPT_ROWS:  # 手动同日多跑的防御性裁剪
        keep = max(1, _MAX_PROMPT_ROWS // len(run_picks))
        trimmed = []
        for run, _ in run_picks:  # 按 run 保序，各留前 keep 行
            trimmed.extend([r for r in rows if r["run_dir"] == run.name][:keep])
        logger.warning("复盘行数 %d 超上限 %d，按 run 各裁到 %d 行",
                       len(rows), _MAX_PROMPT_ROWS, keep)
        rows = trimmed

    system = (
        "你是一位 A 股策略复盘分析师。现在复盘最近一段时间（两周内）多次选股 run 的结果："
        "每次 run 的选股与其当前实际表现已逐只列出（按 run 分组，含所属 run 与距今天数），"
        "对比当时的选股判断与实际表现，总结策略哪里准、哪里偏，并把改进落成新的策略参数。\n\n"
        "复盘要点：\n"
        "1. **判断准确度**：各 run 中评为值得买入/可关注的股票现在表现如何（已达目标=判断准，"
        "跌破止损=判断错，未入买入区间=入场时机判断偏早）；同一股票出现在多个 run 时，"
        "对比不同时点的判断与后续走势，重点看反复入选却持续走弱的标的\n"
        "2. **共性问题**：表现差的选股有哪些共同特征（如追高、题材热度权重过高、资金流向"
        "判断不准、大盘环境判断偏乐观/悲观）\n"
        "3. **改进落点**：把教训落成可执行的策略参数——因子权重调整（乘数 0.5~2.0）与"
        "各阶段提示词校准\n\n"
        "输出**只包含一个 JSON 对象**：\n"
        '{"review_summary": "复盘结论（150字内：近期选股实际表现、判断准确度、共性问题）", '
        '"regime_calibration": "大盘体检的校准提示（下次体检时参考，100字内；没有偏差就写 保持现有判断框架）", '
        '"triage_criteria": "快筛评分标准的补充要求（100字内；如 更重视资金流向真实性，题材热度降权）", '
        '"deep_focus": "深度分析的补充关注点（100字内；如 重点核查解禁压力与主力资金背离）", '
        '"pricing_discipline": "定价纪律的补充要求（80字内；没有就写空字符串）", '
        '"stock_reviews": [{"run_dir": "所属run目录名", "code": "股票代码", "change_reason": "该股从当时收盘价到当前价的变化原因（60字内；结合当时选股逻辑与实际走势，如题材发酵/利好兑现/大盘拖累/资金撤离）"}], '
        '"factor_multipliers": {"momentum": 0.5-2.0, "volume_price": 0.5-2.0, "capital": 0.5-2.0, "heat": 0.5-2.0, "valuation": 0.5-2.0}, '
        '"notes": ["策略改进点1", "改进点2（每条40字内）"]}'
    )
    user = (f"# 复盘窗口：最近 {window_days} 天的选股 run"
            f"（共 {len(run_picks)} 个 run，{len(rows)} 只）\n")
    for run_name in [r.name for r, _ in run_picks]:
        grp = [r for r in rows if r["run_dir"] == run_name]
        user += f"\n## run {run_name}（{grp[0]['days_ago']} 天前）\n"
        for r in grp:
            user += json.dumps(r, ensure_ascii=False) + "\n"
    user += f"\n# 当前日期\n{today:%Y-%m-%d}\n"

    raw = llm.chat_json(system, user, max_tokens=4500, temperature=0.2)
    if isinstance(raw, list):  # 某些模型把对象包进数组——取第一个 dict
        raw = next((x for x in raw if isinstance(x, dict)), {})
    if not raw:
        logger.warning("复盘 LLM 输出为空，跳过策略版本生成")
        return None

    # 逐只价格变化原因（LLM 结合当时选股逻辑与实际走势解释）。键 = (run_dir,
    # code)；模型不守 schema 只回裸 code 时二级 fallback 按 code 兜底。
    change_reasons = {
        (str(r.get("run_dir", "")), str(r.get("code", ""))): str(r.get("change_reason", ""))
        for r in raw.get("stock_reviews", []) or [] if isinstance(r, dict)
    }
    by_code = {
        str(r.get("code", "")): str(r.get("change_reason", ""))
        for r in raw.get("stock_reviews", []) or [] if isinstance(r, dict)
    }
    for r in rows:
        r["变化原因"] = change_reasons.get((r["run_dir"], r["code"]),
                                          by_code.get(r["code"], "（未给出）"))

    sv = StrategyVersion(
        version=latest_version() + 1,
        created=f"{today:%Y-%m-%d}",
        based_on=runs[0].name,  # 窗口内最新 run（窗口化复盘的锚点）
        review_summary=str(raw.get("review_summary", "")),
        regime_calibration=str(raw.get("regime_calibration", "")),
        triage_criteria=str(raw.get("triage_criteria", "")),
        deep_focus=str(raw.get("deep_focus", "")),
        pricing_discipline=str(raw.get("pricing_discipline", "")),
        factor_multipliers=dict(raw.get("factor_multipliers", {}) or {}),
        notes=[str(n) for n in raw.get("notes", []) if n],
        review_rows=rows,
    )
    save_strategy(sv)
    logger.info("复盘完成: v%d（窗口 %d 天，%d 个 run，%d 行，based_on=%s）",
                sv.version, window_days, len(run_picks), len(rows), sv.based_on)
    return sv


def _fmt_zone(pick: dict) -> str:
    pp = pick.get("price_plan") or {}
    lo, hi = pp.get("entry_low"), pp.get("entry_high")
    if lo and hi:
        return f"{lo}~{hi}"
    return "[数据缺失]"


def _fmt_target(pick: dict) -> str:
    pp = pick.get("price_plan") or {}
    lo, hi = pp.get("target_low"), pp.get("target_high")
    if lo and hi:
        return f"{lo}~{hi}"
    return "[数据缺失]"
