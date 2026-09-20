"""CLI Interface for the LangGraph Autonomous Research Agent."""

import argparse
import sys
from typing import Optional

from src.config import Settings, get_settings
from src.state import ResearchState
from src.agent import build_research_agent, create_initial_state, MockLLM
from src.utils.display import (
    console,
    print_banner,
    print_final_answer,
    print_references_table,
    print_research_summary,
)


def run_cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangGraph Autonomous AI Research Agent with Source ID Grounding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        type=str,
        help="The research question to investigate (optional; interactive prompt if omitted)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Hard maximum step budget (default from config or 8)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/offline mode without calling external LLM or requiring API keys",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. gpt-4o-mini, openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live token streaming",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the Green & Black Terminal Web Interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the Green & Black Terminal Web Interface (default 8000)",
    )

    args = parser.parse_args()

    if args.web:
        import web_app
        web_app.start(port=args.port)
        return

    print_banner()

    # Determine question
    question = args.question
    if not question:
        try:
            console.print("[bold cyan]Enter your research question:[/bold cyan]")
            question = input("❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted by user.[/dim]")
            sys.exit(0)

    if not question:
        console.print("[bold red]Error:[/bold red] Research question cannot be empty.")
        sys.exit(1)

    settings = get_settings()
    if args.max_steps:
        settings.max_steps = args.max_steps
    if args.model:
        settings.model_name = args.model
    if args.no_stream:
        settings.streaming = False
    if args.mock:
        settings.llm_provider = "mock"
        settings.search_provider = "mock"

    has_api_key = bool(settings.openai_api_key or settings.groq_api_key)
    llm = MockLLM() if (args.mock or settings.llm_provider == "mock" or not has_api_key) else None

    if llm is not None and not args.mock and not has_api_key:
        console.print(
            "[yellow]Note: No API key (OPENAI_API_KEY or GROQ_API_KEY) found. Automatically running in Mock Mode for demonstration.[/yellow]\n"
        )

    active_provider = "groq" if (settings.groq_api_key and not settings.openai_api_key) or settings.llm_provider == "groq" else settings.llm_provider
    if llm is not None:
        active_provider = "mock"

    console.print(f"[bold]Researching:[/bold] [italic white]\"{question}\"[/italic white]")
    console.print(f"[dim]Step Budget: {settings.max_steps} steps | Provider: {active_provider}[/dim]\n")

    initial_state = create_initial_state(question=question, max_steps=settings.max_steps)
    app = build_research_agent(llm=llm, settings=settings, show_display=True)

    try:
        final_state: ResearchState = app.invoke(initial_state)
    except Exception as e:
        console.print(f"\n[bold red]Fatal Agent Error:[/bold red] {e}")
        sys.exit(1)

    # Display results
    if final_state.get("final_answer"):
        print_final_answer(final_state["final_answer"])

    print_references_table(
        sources=final_state.get("fetched_sources", {}),
        citations=final_state.get("citations", []),
    )

    print_research_summary(
        current_step=final_state.get("current_step", 0),
        max_steps=final_state.get("max_steps", settings.max_steps),
        num_sources=len(final_state.get("fetched_sources", {})),
        num_summaries=len(final_state.get("summaries", [])),
        status=final_state.get("status", "unknown"),
    )


if __name__ == "__main__":
    run_cli()
