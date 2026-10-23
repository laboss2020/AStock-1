# AStock — A股大盘选股 App

先判大盘环境（政策市/beta），再五阶段筛选，从全 A 股选出 **≤10 支值得买入的股票**，给出分析结果、建议**买入价格**与**卖出价格**。

策略源自 [tradingagents-astock](https://github.com/simonlin1212/tradingagents-astock)（v0.5.20，Apache-2.0）的分析框架；AI 接入 **NVIDIA NIM 免费 API** 或其他 OpenAI 兼容 provider。

## 功能

- **大盘体检**（Stage 1）：LLM 综合指数涨跌、涨跌家数、主力资金、北向资金、市场新闻，判定 regime（进攻/防守/震荡）、温度、风格、回避板块
- **量化初筛**（Stage 2，零 LLM）：硬过滤（ST/次新/低价/低流动性/追高排除）+ 多因子打分（动量/量价/资金/热度/估值），权重随 regime 自动切换
- **LLM 快筛**（Stage 3）：批量打分 0-100 + 一句话理由
- **深度分析**（Stage 4）：5 维度评价（基本面/技术面/资金面/消息面/风险）→ 值得买入/可关注/放弃
- **买卖定价**（Stage 5）：技术锚点（MA/BOLL/ATR/高低点）+ LLM 定价 + 硬校验 + 规则兜底，输出买入区间、止损、两档卖出价、仓位建议
- **复盘与策略版本**（每次选股后）：对比上一次 run 的判断与这些股票的当前实际表现（已达目标/跌破止损/回落买入区间…），LLM 复盘总结得失并输出结构化策略改进 → `strategies/vN.json`；下次运行**默认使用最新版本**，也可 `--strategy-version N` 指定
- **报告落盘**：`reports/YYYY-MM-DD_HHMMSS/` 下 report.md + picks.json + run_meta.json

A股规则（T+1、涨跌停、PE 框架、政策市、量在价先、减持新规、散户情绪）内嵌于所有 LLM 提示词与定价校验。

## 快速开始

```bash
# 1. 安装（Python 3.10+，实测 3.14 可用）
pip install -e .

# 2. 配置 AI（NIM 免费 ~1000 credits）
cp .env.example .env
# 编辑 .env，填入 NVIDIA_API_KEY=nvapi-xxx
# key 在 https://build.nvidia.com 免费申请

# 3. 跑一遍选股
python main.py
```

输出示例（console）+ `reports/2026-09-24_093000/report.md`。

## Provider 配置

| 方案 | provider | base_url | 模型 | key 来源 |
|---|---|---|---|---|
| A（默认） | nim | `https://integrate.api.nvidia.com/v1` | `deepseek-ai/deepseek-v4.1-flash` | [build.nvidia.com](https://build.nvidia.com) 免费申请 |
| B | deepseek | `https://api.deepseek.com/v1` | `deepseek-chat` | platform.deepseek.com |
| C | openai | `https://api.openai.com/v1` | `gpt-4o-mini` | platform.openai.com |
| D | openai_compatible | 自填 | 自填 | 任意 OpenAI 兼容服务 |

```bash
python main.py -p deepseek -m deepseek-chat
```

也可用环境变量：`ASTOCK_PROVIDER`、`ASTOCK_MODEL`、`ASTOCK_BASE_URL`、`ASTOCK_API_KEY`。

**思考控制与提速**：NIM 上默认发送 `chat_template_kwargs` 关闭思考过程（deepseek/nemotron 生效、耗时降 3-10×；GLM 系模型无参数可关、照常思考）——结论直接进 content，不再等 reasoning。`ASTOCK_LLM_THINKING=1` 恢复思考；`ASTOCK_LLM_TIMEOUT`（默认 150s，connect 固定 10s）与 `ASTOCK_LLM_MAX_RETRIES`（默认 2）控制失败代价。

## CLI 选项

```
python main.py [--provider/-p NAME] [--model/-m NAME] [--base-url URL]
               [--max-picks N] [--candidates N] [--output/-o DIR]
               [--strategy-version N] [--no-review] [--debug]
```

- `--max-picks 8`：最终选股数量上限（默认 10）
- `--candidates 40`：LLM 快筛候选数（默认 30）
- `--strategy-version 1`：指定策略版本（默认最新复盘版本；无版本时用内置默认策略）
- `--no-review`：跳过选股后的复盘（不生成新策略版本）
- `--debug`：输出数据层 WARNING 日志（东财封禁/fallback 触发可见）

## 复盘与策略版本

每次选股完成后自动复盘**上一次 run**：读取其 `picks.json`，拉取这些股票的当前行情，逐只判定实际表现（已进入卖出目标区间 / 已跌破止损位 / 处于建议买入区间内 / 回落至买入区间下方 / 高于买入区间上沿 / `[数据缺失]`），然后 1 次 LLM 调用把对比结论落成**新策略版本**：

```
strategies/
├── v1.json   # 第 2 次 run 复盘第 1 次 run 生成
├── v2.json   # 第 3 次 run 复盘第 2 次 run 生成
└── ...
```

版本内容（由 LLM 输出、代码校验后落盘）：

- **复盘结论** `review_summary`：上次选股实际表现、判断准确度、共性问题
- **各阶段提示词校准**：大盘体检校准 / 快筛评分补充 / 深度分析补充 / 定价纪律——嵌入后续 run 的对应阶段提示词
- **因子权重乘数** `factor_multipliers`：momentum/volume_price/capital/heat/valuation 各 0.5~2.0，作用在 regime 基底权重上（clamp 后归一化 sum=1）
- **改进点** `notes`：人可读的改进记录

版本控制：

- 下次运行**默认加载最新版本**（`--strategy-version` 缺省即最新）
- `--strategy-version 2` / `ASTOCK_STRATEGY_VERSION=2`：钉住特定版本（可回滚对比）
- 首次运行（无上一次 run）跳过复盘，不生成版本；版本文件损坏时自动回退默认策略
- `ASTOCK_REVIEW=0` / `--no-review`：关闭复盘
- 建议把 `strategies/` 提交进 git，跟踪策略演进

## Web UI（可选）

```bash
pip install -e ".[web]"
streamlit run web/app.py
```

## 数据缓存

两层缓存（`ASTOCK_CACHE=0` 关闭，默认开）：

- **本跑内去重**：同一进程内同股票的 K 线只拉一次（Stage 2 初筛与 Stage 4/5 深度分析、定价之间共享）
- **磁盘缓存** `data_cache/`：K 线按股票落盘，**后续运行只拉增量**（从缓存最后交易日起）；universe 清单按天缓存（同一天内复用）。盘中（15:10 前）拿到的当日半根 K 线只进内存不落盘——磁盘缓存只存完整交易日；除权日（重叠日收盘价差 >0.5%）自动全量重拉，前复权序列不失真

```bash
ASTOCK_CACHE_DIR=/path/to/cache python main.py   # 自定义缓存目录
```

## GitHub Actions 定时运行

`.github/workflows/run-picker.yml`：**每个工作日 7 个定时触发点**（北京 06:37 / 08:37 / 10:37 / 12:37 / 13:07 / 13:37 / 14:07）。GitHub 定时调度延迟严重且不可预测（实测 2~8 小时、甚至整轮丢弃），单点 cron 定不准，所以工作流内置「gate + 时间窗」检查：每个触发先看当天是否已成功跑过（已跑秒退）；距北京 13:10 超 3 小时的太早触发直接跳过；不足 3 小时的**睡到 13:10 开跑**（吸收调度延迟）；已过 13:10 的立即开跑。效果：调度延迟 0~7 小时的触发都能让选股在 **13:10~13:41 开跑、约 14:00 前完成**（job 全程约 20 分钟）；延迟 >7h 的极端天傍晚出结果，不空缺。第一个成功的 run 写当日标记，后续触发秒退；失败的 run 自动由后续触发重试。结果在 workflow 运行日志中查看（完整报告也会打印到日志），并上传 `astock-report-*` artifact（保留 30 天）。

前置一步：仓库 **Settings → Secrets and variables → Actions** 添加 `NVIDIA_API_KEY`（免费 key 申请：[build.nvidia.com](https://build.nvidia.com)）。可选添加 secret `ASTOCK_MODEL` 换模型（优先于 workflow 默认值）；主模型被平台下架（404）时 `llm.py` 会自动切换到备选模型链，选股不中断。

- 手动触发：Actions 页 → A股选股定时运行 → **Run workflow**
- K 线缓存跨 run 增量累积（actions/cache，昨天为止的缓存 → 今天只拉增量），复盘生成的策略版本随缓存一并累积
- 注：GitHub 定时任务不保证精确到分（高负载时延迟或偶发跳过），cron 刻意错开整点（14:07）以提高触发可靠性

## Docker

```bash
# 选股（一次性跑批，结果落盘宿主 ./reports/，缓存与策略版本复用宿主目录）
docker build -t astock .
docker run --rm --env-file .env \
  -v ./reports:/app/reports -v ./data_cache:/app/data_cache \
  -v ./strategies:/app/strategies astock

# Web UI（http://localhost:8501）
docker run --rm --env-file .env -p 8501:8501 \
  -v ./reports:/app/reports -v ./data_cache:/app/data_cache astock \
  python -m streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0
```

或用 compose（`env_file` 指向 `.env`，缺省可跳过）：

```bash
docker compose run --rm picker   # 选股
docker compose up web            # Web UI
```

国内网络构建慢时换源：`docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t astock .`

## 数据源

纯 HTTP，零 TCP（mootdx 在很多机器上不可用），全部免费：

- **腾讯财经** qt.gtimg.cn：实时估值快照（PE/PB/市值/换手，一次调用）
- **东方财富** push2/push2his/clist/fflow：K线（前复权）、全A清单、资金流、指数
- **新浪财经**：K线 fallback（可能滞后一天）
- **同花顺**：涨停热股题材归因、北向资金分钟级
- **财联社** cls.cn：财经快讯

东财调用经串行节流（1s + 抖动）防封；任何源失败自动降级并在报告中标注 `[数据缺失]`。

## 架构

```
astock/
├── config.py        # provider 预设 + 环境变量加载
├── llm.py           # OpenAI SDK 封装（应用层重试、JSON 解析）
├── data/            # 数据层（http/cache/quotes/kline/indicators/universe/market/boards/capital/news）
├── screen/          # 硬过滤 + 多因子打分（权重随 regime）
├── analysis.py      # Stage 1 大盘体检 + Stage 3 快筛 + Stage 4 深度分析提示词
├── pricing.py       # Stage 5 定价（锚点 + 校验 + 规则兜底）
├── pipeline.py      # 五阶段编排
├── strategy.py      # 策略版本管理（复盘生成 vN.json + 加载/乘数）
├── review.py        # 复盘：上次 run 判断 vs 实际表现 → 新策略版本
├── report.py        # Markdown/JSON 报告
└── cli.py           # typer CLI
strategies/          # 策略版本（复盘自动生成，v1.json v2.json …）
.claude/skills/astock-picker/SKILL.md   # 大盘选股策略 skill（Claude Code 用）
web/app.py          # Streamlit UI
.github/workflows/run-picker.yml   # 工作日 14:00 北京时间定时运行
Dockerfile / docker-compose.yml    # Docker 运行（选股 + Web UI）
```

## 策略 Skill

本仓库的策略同时打包为 Claude Code skill：`.claude/skills/astock-picker/SKILL.md`。在 Claude Code 中对项目说"选股"即可触发，skill 文档包含完整策略规则、定价方法与降级链说明。

## 免责声明

本工具输出仅为 AI 生成的分析参考，不构成投资建议。A 股有风险，入市需谨慎。

## 致谢 / 许可

- 策略框架：[simonlin1212/tradingagents-astock](https://github.com/simonlin1212/tradingagents-astock)（Apache-2.0），其上游 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
- 见 [NOTICE](NOTICE)
