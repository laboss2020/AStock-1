"""CLI entry: one-shot stock picking (``astock`` command / ``python main.py``).

Uses typer+rich like tradingagents-astock's CLI, with the same principle that
the default command must work bare — every doc points to a single command.
"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .llm import LLMClient, LLMError
from .pipeline import run_pipeline
from .report import render_report, save_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM 提供商：nim / deepseek / openai / openai_compatible（默认 nim）",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名（默认按 provider 取预设）"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="自定义 API 端点（openai_compatible 时必填）"),
    max_picks: Optional[int] = typer.Option(None, "--max-picks", help="最终选股数量上限（默认 10）"),
    candidates: Optional[int] = typer.Option(None, "--candidates", help="量化初筛候选数（默认 30）"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="报告输出目录（默认 reports/）"),
    strategy_version: Optional[int] = typer.Option(
        None, "--strategy-version", help="策略版本（默认最新复盘版本；0/未生成时用默认策略）"),
    no_review: bool = typer.Option(False, "--no-review", help="跳过选股后的复盘（不生成新策略版本）"),
    debug: bool = typer.Option(False, "--debug", help="调试日志"),
):
    """A股大盘选股：五阶段流程（大盘体检→量化初筛→LLM快筛→深度分析→定价），
    选出 ≤10 支值得买入的股票并给出建议买入/卖出价格。
    选股完成后自动复盘上一次 run，生成新策略版本（下次运行默认使用）。

    ⚠️ 仅供学习研究，不构成任何投资建议。"""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(
            provider=provider, model=model, base_url=base_url,
            max_picks=max_picks, candidate_count=candidates,
            output_dir=output_dir, strategy_version=strategy_version,
            review=not no_review, debug=debug,
        )
        llm = LLMClient(cfg)
    except (ValueError, LLMError) as e:
        console.print(f"[red]✗ 配置错误：[/red]{e}")
        raise typer.Exit(1)

    console.print(f"[bold cyan]A股大盘选股[/bold cyan] provider={cfg.provider} "
                  f"model={cfg.model or cfg.resolved_model()}")
    console.print("[yellow]⚠️ 仅供学习研究，不构成任何投资建议[/yellow]\n")

    try:
        result = run_pipeline(cfg, llm)
    except LLMError as e:
        console.print(f"[red]✗ LLM 调用失败：[/red]{e}")
        raise typer.Exit(2)

    # --- 复盘（选股之后；在 save_report 前跑——复盘对象是上一次 run，---
    # --- 本次 run 还没写盘。复盘失败只记录，不阻塞已完成的选股）---------
    if cfg.review:
        from .review import run_review

        try:
            sv_new = run_review(cfg, llm, cfg.output_dir)
            if sv_new:
                rows = sv_new.review_rows  # 逐只明细：前后对比+选股/变化原因
                run_dirs = list(dict.fromkeys(r.get("run_dir", "") for r in rows))
                result.review = {
                    "based_on": sv_new.based_on,       # 窗口内最新 run（兼容旧字段）
                    "based_on_runs": run_dirs,          # 窗口内全部 run（新→旧）
                    "window_days": getattr(cfg, "review_window_days", 14),
                    "run_count": len(run_dirs),
                    "summary": sv_new.review_summary,
                    "new_version": sv_new.version,
                    "notes": sv_new.notes,
                    "rows": rows,
                }
        except Exception as e:
            msg = f"复盘失败（不影响选股结果）: {e}"
            console.print(f"[yellow]⚠ {msg}[/yellow]")
            result.errors.append(msg)

    run_dir = save_report(cfg, result)

    # --- 大盘体检摘要 ---
    r = result.regime
    console.print("[bold]一、大盘体检[/bold]")
    console.print(f"  市场环境: {r.regime}（{r.temperature}）  风格: {r.style}")
    console.print(f"  关注: {', '.join(r.sector_focus) or '无'}  回避: {', '.join(r.avoid_sectors) or '无'}")
    console.print()

    # --- 筛选漏斗 ---
    console.print("[bold]二、筛选漏斗[/bold]")
    console.print(f"  全市场 {result.universe_count} → 硬过滤 {result.filtered_count} "
                  f"→ 初筛候选 {result.candidate_count} → 快筛通过 {result.triaged_count} "
                  f"→ 终选 [bold green]{len(result.picks)}[/bold green]（LLM 调用 {result.llm_calls} 次）")
    console.print()

    # --- 最终选股表 ---
    console.print("[bold]三、最终选股[/bold]")
    if result.picks:
        table = Table(show_header=True, header_style="bold")
        for col in ("#", "代码", "名称", "买入区间", "止损价", "卖出目标", "评分", "结论"):
            table.add_column(col)
        for i, p in enumerate(result.picks, 1):
            pl = p.price_plan
            color = "green" if p.verdict.verdict == "值得买入" else "cyan"
            table.add_row(
                str(i), p.code, p.name,
                f"{pl.entry_low}~{pl.entry_high}",
                str(pl.stop_loss),
                f"{pl.target_low}~{pl.target_high}",
                str(p.verdict.score),
                f"[{color}]{p.verdict.verdict}[/{color}]",
            )
        console.print(table)
    else:
        console.print("  今日无符合策略的选股。")

    if result.errors:
        console.print(f"\n[yellow]⚠ {len(result.errors)} 个非致命错误[/yellow]（详见报告）")

    # --- 复盘摘要 ---
    if result.review:
        rv = result.review
        console.print("\n[bold]四、复盘[/bold]")
        runs = rv.get("based_on_runs") or []
        console.print(
            f"  复盘对象: 最近{rv.get('window_days', 14)}天 {len(runs)} 个 run"
            f"（最新 {runs[0] if runs else rv.get('based_on', '—')}）"
            f"共 {len(rv.get('rows') or [])} 只"
            f"  新策略版本: [bold green]v{rv.get('new_version', '?')}[/bold green]（下次运行默认使用）")
        if rv.get("summary"):
            console.print(f"  结论: {rv['summary']}")
        for n in rv.get("notes", []):
            console.print(f"  - {n}")
    elif result.strategy_version_used == 0:
        console.print("\n[bold]四、复盘[/bold]\n  窗口内无历史 run（首次运行），未生成策略版本。")

    console.print(f"\n[bold]完整报告[/bold]: {run_dir / 'report.md'}")
    console.print("[yellow]⚠️ 以上内容由 AI 生成，仅供研究参考，不构成投资建议[/yellow]")


if __name__ == "__main__":
    app()
