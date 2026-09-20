"""Top-level entry point for the AI Research Agent."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.cli import run_cli

if __name__ == "__main__":
    run_cli()
