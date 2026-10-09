"""LLM analysis stages: market regime (Stage 1), rapid triage (Stage 3),
deep analysis (Stage 4).

The prompts embed the A-stock strategy distilled from
tradingagents-astock's 7-Analyst framework — compressed from seven tool-loop
analysts into self-contained prompts with the same priorities:

- 市场/技术面：量在价先、放量突破缩量回调、换手率是资金活跃度关键
- 基本面：A股 PE 30-50x 常态、30x 消化框架、扣非净利、现金流匹配
- 政策面：A股是政策市，政策力度分级（指导意见<部委通知<国务院文件<法律法规）
- 资金面：龙虎榜/大单/主力流向是短线定价核心；嘴上说什么不如钱往哪走
- 情绪面：先看资金再看新闻，消息面与资金面背离是最有价值的信号
- 解禁/减持：破发/破净/分红不达标时控股股东减持路径被封死

All verdicts are JSON; malformed replies get one parse-retry (see llm.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import Config
from .llm import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 共享 A 股规则块（参照 tradingagents-astock 各分析师提示词压缩）
# ---------------------------------------------------------------------------

ASTOCK_RULES = """⚠️ A 股市场特殊规则（分析时必须纳入考量）：
- **涨跌停制度**：主板 ±10%，科创板/创业板 ±20%，北交所 ±30%。风险警示股(ST/*ST)不另设更窄的限制（主板 ST/*ST 已于 2026-07-06 起与主板普通股一致为 ±10%）。触及涨跌停后流动性骤降，技术指标可能失真。
- **T+1 交易制度**：当日买入次日才能卖出，短线策略的可执行性受限，追高风险是核心结构风险。
- **政策市**：A 股是全球最典型的政策市。政策力度分级：指导意见（弱）< 部委通知（中）< 国务院文件（强）< 法律法规（最强）。政策受益/受损逻辑链：政策 → 行业影响 → 公司业务映射。
- **估值参照系**：A 股整体 PE 中位数偏高（30-50x 为常态），不能照搬美股 15-25x 标准，应对标同行业 A 股公司。30x 前向 PE 消化框架是成长股估值锚：消化时间超过 5 年即为高估。
- **量价关系**：「量在价先」规律显著，放量突破和缩量回调是核心信号。换手率是判断资金活跃度和筹码松动的关键指标。
- **资金面**：主力资金（超大单+大单）净流向是短线定价的核心力量；嘴上说什么不如钱往哪走。消息面正面但主力资金持续净流出（或相反）是重要背离信号。
- **解禁/减持**：解禁市值占流通市值 >20% 为重大供给压力。破发/破净/分红不达标时控股股东不得通过集中竞价或大宗交易减持（《减持管理暂行办法》第十、十一条）。大股东减持需提前 15 个交易日预披露。
- **散户结构**：散户占比超过 60%，情绪对短期股价影响大；情绪一致性过高（极度乐观/悲观）往往是反转信号。"""

# ---------------------------------------------------------------------------
# Stage 1: 大盘体检
# ---------------------------------------------------------------------------


@dataclass
class MarketRegime:
    """Stage-1 verdict: the market environment that gates the whole pick."""
    regime: str = "中性"           # 进攻/乐观/震荡/防守/悲观
    temperature: str = "中性"      # 极度乐观/乐观/中性/悲观/极度悲观
    style: str = ""                # 风格偏好，如 大盘蓝筹/中小成长/题材轮动
    risk_appetite: str = "中性"    # 高/中/低
    sector_focus: list[str] = field(default_factory=list)   # 建议关注的板块方向
    avoid_sectors: list[str] = field(default_factory=list)  # 建议回避的板块
    rationale: str = ""
    raw: str = ""


def analyze_market(cfg: Config, llm: LLMClient, snapshot: dict,
                   strategy_text: str = "") -> MarketRegime:
    """Stage 1: judge the market environment from index/flow/board/news data.

    ``strategy_text`` (from the loaded strategy version) carries the review
    calibration for this stage — empty when running the default strategy."""
    system = (
        "你是一位资深的大盘策略分析师，负责在个股分析之前判断 A 股整体市场环境。"
        "你的判断决定后续选股的进攻/防守取向。\n\n"
        + ASTOCK_RULES
        + (f"\n\n{strategy_text}" if strategy_text else "")
        + "\n\n分析框架：\n"
        "1. **指数趋势**：主要指数（上证/深成/创业板/科创50）的近期走势、均线位置，判断趋势方向\n"
        "2. **市场宽度**：上涨/下跌家数比，判断赚钱效应\n"
        "3. **资金面**：大盘主力资金净流入（近5日），判断增量资金态度\n"
        "4. **板块轮动**：当日领涨/领跌板块，判断热点主线和轮动节奏\n"
        "5. **消息面**：政策与宏观快讯，识别可能改变环境的事件\n\n"
        "输出**只包含一个 JSON 对象**，结构如下：\n"
        '{"regime": "进攻|乐观|震荡|防守|悲观 之一", '
        '"temperature": "极度乐观|乐观|中性|悲观|极度悲观 之一", '
        '"style": "风格偏好一句话，如：大盘蓝筹/中小成长/题材轮动", '
        '"risk_appetite": "高|中|低", '
        '"sector_focus": ["建议关注的板块方向1", "方向2"], '
        '"avoid_sectors": ["建议回避的板块1", "板块2"], '
        '"rationale": "300字以内的判断依据"}'
    )

    user_parts = ["# 当日大盘数据\n"]
    user_parts.append("## 指数行情\n" + json.dumps(
        snapshot.get("indexes", []), ensure_ascii=False, indent=1))
    user_parts.append("## 指数趋势（近5/20日）\n" + json.dumps(
        snapshot.get("index_trends", []), ensure_ascii=False, indent=1))
    user_parts.append("## 大盘主力资金（近5日，亿元）\n" + json.dumps(
        snapshot.get("market_flow", []), ensure_ascii=False, indent=1))
    northbound = snapshot.get("northbound")
    if northbound:
        user_parts.append("## 北向资金（亿元）\n" + json.dumps(northbound, ensure_ascii=False))
    else:
        user_parts.append("## 北向资金\n[数据缺失]")
    user_parts.append("## 概念板块涨幅榜 Top15\n" + json.dumps(
        snapshot.get("boards", []), ensure_ascii=False, indent=1))
    user_parts.append("## 行业板块涨幅榜 Top10\n" + json.dumps(
        snapshot.get("industry_boards", []), ensure_ascii=False, indent=1))
    user_parts.append("## 涨停题材热点 Top10\n" + json.dumps(
        snapshot.get("themes", []), ensure_ascii=False, indent=1))
    news_items = [n for n in (snapshot.get("news") or [])[:15]]
    if news_items:
        user_parts.append("## 财经快讯\n" + "\n".join(
            f"- [{n.get('time','')}] {n.get('title','')}"
            for n in news_items if isinstance(n, dict)))
    else:
        user_parts.append("## 财经快讯\n[数据缺失]")

    raw = llm.chat_json(system, "\n\n".join(user_parts), max_tokens=2000)
    if isinstance(raw, list):
        # 某些模型（glm-5.3-flash）会把 JSON 对象包进数组——取第一个 dict；
        # 完全没有 dict 时降级为空（下方字段取默认值，不崩溃）。
        raw = next((x for x in raw if isinstance(x, dict)), {})
        if not raw:
            logger.warning("Stage1 LLM 输出为空数组，已降级为默认中性环境")
    regime = MarketRegime(
        regime=str(raw.get("regime", "中性")),
        temperature=str(raw.get("temperature", "中性")),
        style=str(raw.get("style", "")),
        risk_appetite=str(raw.get("risk_appetite", "中性")),
        sector_focus=[str(s) for s in raw.get("sector_focus", []) if s],
        avoid_sectors=[str(s) for s in raw.get("avoid_sectors", []) if s],
        rationale=str(raw.get("rationale", "")),
        raw=json.dumps(raw, ensure_ascii=False),
    )
    logger.info("Stage1 大盘体检: regime=%s temperature=%s", regime.regime, regime.temperature)
    return regime


# ---------------------------------------------------------------------------
# Stage 3: LLM 快筛
# ---------------------------------------------------------------------------


def triage_batch(cfg: Config, llm: LLMClient, regime: MarketRegime,
                 candidates: list[dict], strategy_text: str = "") -> dict[str, dict]:
    """Score a batch of candidates (0-100) with one-line reasons.

    Returns {code: {score, reason}} — missing codes mean the model skipped
    them; the pipeline treats absence as rejected (never silently promoted).
    """
    system = (
        "你是一位 A 股选股助手，负责从量化初筛出的候选池中快速评审每只股票的买入价值。"
        "依据当日大盘环境和候选股票的数据快照打分（0-100）。\n\n"
        + ASTOCK_RULES
        + (f"\n\n{strategy_text}" if strategy_text else "")
        + "\n\n评分标准：\n"
        "- 70-100：多因素共振（趋势/资金/题材/估值），值得深度分析\n"
        "- 40-69：部分因素成立，可关注\n"
        "- 0-39：缺乏亮点或有明显风险，放弃\n\n"
        "注意：打分要独立思考，不要因为板块热就给所有成分股高分；资金流向和趋势质量权重高于题材噱头。\n\n"
        "输出**只包含一个 JSON 对象**，key 为 stocks，value 为数组，数组长度必须等于"
        "候选股票数量（每只恰好一个元素，按输入顺序，不要省略）：\n"
        '{"stocks": [{"code": "6位代码", "score": 0-100的整数, "reason": "一句话理由（40字内）"}, ...]}'
    )

    user_parts = [
        f"# 大盘环境\nregime={regime.regime} 风格={regime.style} "
        f"风险偏好={regime.risk_appetite}\n建议关注: {', '.join(regime.sector_focus) or '无'}\n"
        f"建议回避: {', '.join(regime.avoid_sectors) or '无'}\n",
        "# 候选股票数据快照（按量化得分排序）\n",
    ]
    for c in candidates:
        user_parts.append(json.dumps(c, ensure_ascii=False))

    raw = llm.chat_json(system, "\n".join(user_parts), max_tokens=4000)
    if not isinstance(raw, list):
        raw = raw.get("stocks", []) if isinstance(raw, dict) else []

    out: dict[str, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        if len(code) == 6 and code.isdigit():
            try:
                score = int(item.get("score", 0))
            except (ValueError, TypeError):
                score = 0
            out[code] = {"score": max(0, min(100, score)),
                         "reason": str(item.get("reason", ""))[:80]}
    return out


# ---------------------------------------------------------------------------
# Stage 4: 深度分析
# ---------------------------------------------------------------------------


@dataclass
class DeepVerdict:
    """Stage-4 verdict: comprehensive per-stock analysis."""
    code: str = ""
    name: str = ""
    score: int = 0
    verdict: str = "放弃"        # 值得买入/可关注/放弃
    buy_logic: str = ""          # 买入逻辑（多维证据）
    risks: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    dimension_summary: dict = field(default_factory=dict)  # 各维度评价
    raw: str = ""


def deep_analyze(cfg: Config, llm: LLMClient, regime: MarketRegime,
                 bundle: dict, strategy_text: str = "") -> DeepVerdict:
    """Stage 4: comprehensive analysis of one candidate from its data bundle.

    The bundle carries compact but complete per-dimension evidence (技术面
    指标快照、基本面、资金流、题材、新闻) — built by the pipeline with zero
    LLM calls.
    """
    code = bundle.get("code", "")
    name = bundle.get("name", "")
    system = (
        f"你是一位 A 股资深投研分析师。现在对候选股票 {code} {name} 做深度分析，"
        "判断其是否值得买入。你需要像投研团队一样从多个维度独立评估。\n\n"
        + ASTOCK_RULES
        + (f"\n\n{strategy_text}" if strategy_text else "")
        + "\n\n分析维度（每个维度都必须给出明确评价，数据缺失时如实标注）：\n"
        "1. **技术面**：趋势方向（均线多空排列）、量价配合（放量/缩量）、关键支撑阻力、RSI/MACD 信号\n"
        "2. **基本面**：估值水平（对标同行业）、盈利质量、成长性；PE>70x 且无高增长支撑即为高估\n"
        "3. **资金面**：主力资金近5日/20日净流向、量比异动；消息面与资金面背离必须指出\n"
        "4. **题材/政策面**：所属题材热度、政策受益逻辑、板块轮动位置\n"
        "5. **风险面**：解禁/减持、股权质押、商誉、退市风险、追高风险（T+1 下当日买入无法止损）\n\n"
        "评分标准（0-100）：\n"
        "- 75-100：多维共振、证据扎实，值得买入\n"
        "- 55-74：多数维度成立，可关注\n"
        "- 0-54：证据不足或风险显著，放弃\n\n"
        "输出**只包含一个 JSON 对象**：\n"
        '{"code": "6位代码", "score": 0-100的整数, '
        '"verdict": "值得买入|可关注|放弃 之一", '
        '"buy_logic": "买入逻辑（150字内，多维证据）", '
        '"risks": ["风险1", "风险2"], '
        '"catalysts": ["催化因素1", "催化因素2"], '
        '"dimension_summary": {"技术面": "一句话", "基本面": "一句话", "资金面": "一句话", "题材政策面": "一句话", "风险面": "一句话"}}'
    )

    user_parts = [
        f"# 大盘环境\nregime={regime.regime} 风格={regime.style} "
        f"风险偏好={regime.risk_appetite}\n",
        "# 候选股票完整数据包\n",
        json.dumps(bundle, ensure_ascii=False, indent=1, default=str),
    ]
    raw = llm.chat_json(system, "\n\n".join(user_parts), max_tokens=3000)
    verdict = DeepVerdict(
        code=str(raw.get("code", code)),
        name=str(raw.get("name", name)),
        score=_safe_int(raw.get("score", 0)),
        verdict=str(raw.get("verdict", "放弃")),
        buy_logic=str(raw.get("buy_logic", "")),
        risks=[str(r) for r in raw.get("risks", []) if r],
        catalysts=[str(c) for c in raw.get("catalysts", []) if c],
        dimension_summary=dict(raw.get("dimension_summary", {}) or {}),
        raw=json.dumps(raw, ensure_ascii=False),
    )
    logger.info("Stage4 深度分析 %s %s: score=%s verdict=%s",
                verdict.code, verdict.name, verdict.score, verdict.verdict)
    return verdict


def _safe_int(v, default: int = 0) -> int:
    try:
        return max(0, min(100, int(v)))
    except (ValueError, TypeError):
        return default
