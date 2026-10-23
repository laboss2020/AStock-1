# CLAUDE.md

A股大盘选股 app。策略源自 simonlin1212/tradingagents-astock（本地 clone 在 `D:\src\ai\TradingAgents-astock`，旧版在 `TradingAgents-astock-old`）。

## 命令

```bash
# 跑选股（需 .env 里 NVIDIA_API_KEY）
.venv/Scripts/python main.py

# 导入冒烟测试
PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "import astock; import astock.cli; import astock.pipeline; print('OK')"

# 数据层单测（EM 被封时走 fallback）
PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "from astock import data; print(data.index_snapshot())"

# 封闭 E2E 测试（mock LLM + mock 数据，零网络，永远可跑）
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tests/test_e2e_mock.py

# 缓存层单测（mock 网络：命中/增量/除权重拉/周末）
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tests/test_cache.py

# 复盘/策略版本单测（mock：版本 roundtrip/乘数/提示词块/三轮 E2E 时序）
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tests/test_review.py
```

## 缓存（两层，`ASTOCK_CACHE=0` 关闭，默认开）

- **本跑内去重**：`kline.py::_mem_cache`——同进程内同股票 K 线只拉一次
- **磁盘缓存** `data_cache/`（`ASTOCK_CACHE_DIR` 可配）：K 线按股票 CSV 落盘、后续只拉增量（beg=缓存最后交易日）；universe 按天缓存 `universe_{date}.csv`（读回必须 `dtype={"code": str}`）
- **除权检测**：增量首根（重叠日）close 差 >0.5% → 全量重拉（前复权失真防御）
- **盘中半根不落盘**：15:10 前拉的当日 K 线只进内存——磁盘只存完整交易日；`_now()` 可 monkeypatch（test_cache.py 用）
- GitHub Actions 用 actions/cache 跨 run 保留 data_cache/（增量累积）；Docker 挂 volume

## 复盘与策略版本

- 每次选股后自动复盘**上一次 run**（`review.py::run_review`）：读上次 picks.json → 腾讯批量拉当前表现（1 call）→ `_outcome` 逐只判定（已达目标/跌破止损/回落区间/缺失）→ 1 次 LLM 调用（schema 含 `stock_reviews` 逐只价格变化原因）→ `strategies/vN.json`（版本 = latest+1）
- **复盘明细进报告**（report.py 复盘节）：前后对比表（当时收盘/现价/累计涨跌%（现价 vs anchors.close）/当时挂的价位/当前状态）+ 逐只"当时选股原因"（picks.json 的 buy_logic+催化剂）与"价格变化原因"（LLM stock_reviews）；明细挂 `StrategyVersion.review_rows`（内存传递，不写入版本文件）
- **复盘调用在 save_report 之前**（cli.py）——本次 run 还没写盘，_latest_run 拿到的是上一次 run；复盘失败只记录不阻塞选股结果
- 版本结构（`strategy.py::StrategyVersion`）：review_summary + 各阶段提示词校准（regime/triage/deep/pricing）+ 因子权重乘数（clamp 0.5-2.0，归一化 sum=1）+ notes
- 下次运行默认加载最新版本（`cfg.strategy_version=0` 语义：0=最新/无则默认策略）；`--strategy-version N` / `ASTOCK_STRATEGY_VERSION=N` 钉住特定版本
- pipeline 开头加载版本 → `sv.prompt_blocks()` 嵌入各阶段提示词（ASTOCK_RULES 之后）→ Stage 2 权重 `apply_multipliers` 叠加乘数
- Docker/CI 挂/缓存 `strategies/`（版本跨 run 演进）；建议提交 strategies/ 进 git

## 定时运行 / Docker

- `.github/workflows/run-picker.yml`：**七 cron**（北京 06:37/08:37/10:37/12:37/13:07/13:37/14:07，首个为 `37 22 * * 0-4` 其余 `1-5` 同日档）+ **precheck「gate + 时间窗」**：当天已跑秒退；距北京 13:10 超 3h 跳过；不足 3h **睡到 13:10 开跑**（吸收调度延迟）；已过 13:10 立即跑——延迟 0~7h 都能在 ~14:00 前完成（job ~20 分钟）。背景：GitHub schedule 延迟不可预测（实测 +2.3h~+8h、整天丢弃；猜 cron 没用，2026-10-16 由"赌延迟"改为"睡到点"）。gate 按北京时间去重（`.run_gate` 走 cache，`TZ=Asia/Shanghai date +%F`——**读写都必须带 TZ**，06:37 触发在 UTC 前一天）；第一个成功 run 写 gate，当天后续触发秒退；手动 workflow_dispatch 忽略 gate 与时间窗强制跑；失败不写 gate 后续触发重试。concurrency 组 `stock-picker` 排队防并发双跑。**CI 缓存必须含 reports/**（复盘读上一次 run 的 picks.json——reports/ 在 .gitignore，不进缓存则 CI 每天都是"首次运行"、复盘永远跳过）。需 repo secret `NVIDIA_API_KEY`
- `Dockerfile`（python:3.12-slim，`pip install .[web]`）+ `docker-compose.yml`（picker 跑批 / web UI 8501）+ `.dockerignore`
- 无 `.env` 时纯环境变量可用（env_map）；workflow 的 ASTOCK_MODEL 用 `${{ secrets.ASTOCK_MODEL || 'z-ai/glm-5.3-flash' }}`（secret 优先）

## 环境注意（本机）

- **控制台 cp932**：跑任何输出中文的命令必须加 `PYTHONIOENCODING=utf-8`
- **系统代理已死**：Windows 注册表代理（192.168.88.7:808）彻底断连（WinError 10061）——数据层与 LLM 层均已默认 `trust_env=False` 直连（国内数据源与 NIM 直连全通）；需要走代理设 `ASTOCK_TRUST_ENV=1` / `ASTOCK_LLM_TRUST_ENV=1`
- **NIM 模型现状（2026-10-22 实测）**：`/v1/models` 列表与实际部署不一致（83 个里大半 404），且**模型会隔夜下架**（nemotron-3-super-120b-a12b 10-20 还在、10-21 全天 404，导致当日 6 次 CI run 全失败）；**实测可用**：`z-ai/glm-5.3-flash`（1.2s，现主力）、`nvidia/nemotron-3-ultra-550b-a55b`（0.8s）、`deepseek-ai/deepseek-v4.1-flash`（3.2s）、`stepfun-ai/step-5-preview`（2.6s）、`openai/gpt-oss-20b`（18.8s）、`nvidia/nemotron-3.5-super-vl-120b-a12b`（0.7s，视觉模型）；**已死**：nemotron-3-super、glm-5.3（超时）、diffusiongemma（超时）。llm.py 有 404 防御：模型缺失自动切备选链（`ASTOCK_FALLBACK_MODELS` 可配，nim 默认 glm-5.3-flash → deepseek-v4.1-flash → nemotron-3-ultra）；workflow 的 ASTOCK_MODEL 用 `${{ secrets.ASTOCK_MODEL || '默认' }}`——secret 优先，勿在 workflow 硬编码盖掉
- **NIM 思考控制（2026-10-23 实测）**：llm.py 默认对 NIM 发 `chat_template_kwargs: {"thinking": false, "enable_thinking": false}` 关思考——deepseek/nemotron **生效**（reasoning 归零、耗时降 3-10×），GLM 系忽略照常思考（NIM 上 GLM 无参数可关）；**顶层参数会被 NIM 400 拒绝**（Unsupported parameter），勿改成顶层；`ASTOCK_LLM_THINKING=1` 恢复思考。单请求 timeout 150s（connect 独立 10s）、失败重试 2 次（`ASTOCK_LLM_TIMEOUT` / `ASTOCK_LLM_MAX_RETRIES` 可调）
- **东财防封**：东财 push2/push2his 对请求过密的本机会临时封禁（RemoteDisconnected），几分钟到几小时后解除。所有东财调用必须经 `em_get()` 节流（勿绕过直连 requests）；kline.py 有熔断（连续 3 次失败后本进程直接走新浪）；测试时优先用腾讯/新浪/同花顺路径
- **mootdx 不可用**：本机 TCP 连得上但服务器返回空数据——数据层是纯 HTTP，勿尝试改回 mootdx

## 架构约定

- 五阶段流水线在 `astock/pipeline.py`：大盘体检（1 call）→ 量化初筛（0）→ LLM 快筛（~4）→ 深度分析（~15）→ 定价（~10）
- A 股规则块 `ASTOCK_RULES` 在 `astock/analysis.py`，所有 LLM 提示词共享
- 定价校验在 `astock/pricing.py::validate`：`stop < entry_low ≤ entry_high < target_low ≤ target_high`，LLM 输出不满足时走 `_rule_fallback` 兜底
- 数据缺失一律标注 `[数据缺失]`，不静默跳过；fallback 链见 `.claude/skills/astock-picker/SKILL.md` 的数据源表
- 策略 skill：`.claude/skills/astock-picker/SKILL.md`（改策略时同步更新）

## 已知坑（勿重复踩）

- 新浪 K 线返回**小写**列名（day/open/high/low/close/volume），需全部 rename
- 腾讯指数 fallback 必须直接传完整符号（sh000001/sz399001）给 `parse_tencent_text`，经 `market_prefix` 会把上证 000001 路由成 sz
- BSE 920xxx 判断须先于 leading-9 规则（见 `quotes.py::market_prefix`）
- `filters.py` 的 docstring 含 `\*ST`，须保持 raw 字符串
- typer CLI 用 `@app.callback(invoke_without_command=True)` 保住裸跑（参考项目血泪教训）
- `json_mode=True` → response_format json_object **强制顶层是 JSON 对象**，数组永远出不来——Stage 3 快筛用 `{"stocks": [...]}` 包裹；个别模型（glm-5.3-flash）会把对象包进数组：Stage 1 有就地防御，Stage 4/5 统一走 `llm.py::first_dict`（取第一个 dict，空则降级默认值/规则兜底）
- 推理模型（glm/deepseek/nemotron thinkers）max_tokens 太小时推理吃满 token → content 空（finish=length）；llm.py 空 content 自动提额 max_tokens+=2048 重试；定价 max_tokens 起步 3000
- 定价 LLM 返回全 0 数值（如对"放弃"评级的股票给空壳 JSON）→ pricing.validate 拦住走规则兜底（2026-10-08 真实跑验证）
- **错误隔离**（2026-10-23 教训，run #37 单只超时报废整跑）：快筛批次失败/深度分析单只失败只记录 `result.errors` 跳过，**不报废整个 run**——成功的部分照常定价出报告；全部失败时 errors 留明确记录（"今日无选股产出"）。仅 Stage 1 大盘体检失败是 fatal（LLM 不通后面全白跑）
- **run #42 4h 超时教训**（2026-10-23）：NIM 网关高峰单请求挂 4-5 分钟才报 Connection error，timeout=300 + 重试 3 次 → 一次失败烧 ~15 分钟，几只失败股票拖满 4h 被 GitHub 取消——现已 timeout 150/connect 10s/重试 2/关思考（失败代价 ~5 分钟）。另：deepseek-v4.1-flash **开思考时 content 为空**（reasoning 吃满但 finish=stop，不触发提额重试），关思考根治
- 磁盘缓存读 CSV 时 Date 列是字符串——`read_csv_cache` 必须传 `parse_dates=["Date"]`；code 列必须 `dtype={"code": str}`（否则 000001 前导零丢失）
- universe 是盘中快照：缓存按天（同日复用、隔天重拉），勿做跨天缓存
- 复盘时序：run_review 在 save_report 前调用；测试里"用 vN"的断言出现在第 N+1 轮 run（版本在 pipeline 开头已定，当轮复盘生成的新版本下一轮才生效）
