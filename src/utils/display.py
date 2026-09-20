"""Rich console display utilities for real-time research monitoring."""

import sys
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text

console = Console(legacy_windows=False, force_terminal=True)


def print_banner():
    """Print the stylized welcome banner."""
    banner_text = Text()
    banner_text.append("LangGraph Autonomous AI Research Agent\n", style="bold cyan")
    banner_text.append("Autonomous tool decision-making | Source citation tracking | Hard step budget safety", style="dim white")
    console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))


def print_step_header(step: int, max_steps: int):
    """Print step counter header."""
    console.print(f"\n[bold magenta]=== STEP {step}/{max_steps} ===[/bold magenta]")


def start_stream(title: str = "Streaming Final Report"):
    """Display an indicator before token streaming begins."""
    console.print(f"\n[bold green]>>> {title} (Live Tokens):[/bold green]\n")


def stream_chunk(token: str):
    """Stream token chunk directly to standard output with immediate flush."""
    sys.stdout.write(token)
    sys.stdout.flush()


def end_stream():
    """Conclude streaming block."""
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def print_decision(thought: str, action: str, params: Dict[str, Any]):
    """Display the agent's thought process and chosen tool."""
    action_colors = {
        "SEARCH": "blue",
        "FETCH": "green",
        "SUMMARISE": "yellow",
        "FINISH": "bright_magenta",
    }
    color = action_colors.get(action.upper(), "white")

    content = f"[bold italic white]Thought:[/bold italic white] {thought}\n\n"
    content += f"[bold]Selected Tool:[/bold] [{color}]{action.upper()}[/{color}]\n"

    param_lines = []
    for k, v in params.items():
        if v:
            param_lines.append(f"- [bold]{k}:[/bold] {v}")
    if param_lines:
        content += "\n" + "\n".join(param_lines)

    console.print(Panel(content, title="[bold]Agent Reasoning & Decision[/bold]", border_style=color))


def print_tool_status(action: str, status: str, summary: str, details: Optional[str] = None):
    """Display the execution outcome of a tool."""
    status_badges = {
        "success": "[bold green][SUCCESS][/bold green]",
        "failed": "[bold red][FAILED][/bold red]",
        "empty": "[bold yellow][EMPTY][/bold yellow]",
        "warning": "[bold yellow][WARNING][/bold yellow]",
        "in_progress": "[bold cyan][RUNNING][/bold cyan]",
    }
    badge = status_badges.get(status, f"[{status.upper()}]")

    text = f"{badge} [bold]{action.upper()}[/bold]: {summary}"
    if details:
        text += f"\n[dim]{details}[/dim]"

    border_color = "green" if status == "success" else ("yellow" if status in ("empty", "warning") else "red")
    console.print(Panel(text, border_style=border_color, padding=(0, 1)))


def print_final_answer(answer: str):
    """Display the grounded final answer formatted in Markdown."""
    console.print("\n" + "=" * 70)
    console.print(Panel(
        Markdown(answer),
        title="[bold green]FINAL RESEARCH REPORT (GROUNDED & CITED)[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


def print_references_table(sources: Dict[str, Any], citations: List[str]):
    """Display a clean table of cited sources and all fetched sources."""
    if not sources:
        console.print("[dim italic]No web sources were successfully fetched.[/dim italic]")
        return

    table = Table(title="[bold cyan]Verified Source References[/bold cyan]", show_lines=True, expand=True)
    table.add_column("ID", style="bold cyan", width=6, justify="center")
    table.add_column("Cited?", style="bold green", width=8, justify="center")
    table.add_column("Title", style="bold white", ratio=2)
    table.add_column("URL", style="dim underline", ratio=3)

    for sid in sorted(sources.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
        src = sources[sid]
        is_cited = "YES [bold]*[/bold]" if sid in citations else "No"
        cited_style = "green" if sid in citations else "dim"
        table.add_row(
            sid,
            f"[{cited_style}]{is_cited}[/{cited_style}]",
            src.get("title", "Untitled"),
            src.get("url", ""),
        )

    console.print(table)


def print_research_summary(current_step: int, max_steps: int, num_sources: int, num_summaries: int, status: str):
    """Display summary statistics of the research run."""
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()

    stats.add_row("Steps Consumed:", f"{current_step} / {max_steps}")
    stats.add_row("Sources Fetched:", f"{num_sources} web sources (S1..)")
    stats.add_row("Summaries Created:", f"{num_summaries} source extractions")
    stats.add_row("Final Status:", f"[bold cyan]{status.upper()}[/bold cyan]")

    console.print(Panel(stats, title="[bold]Research Run Statistics[/bold]", border_style="blue"))
