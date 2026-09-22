"""Vercel Serverless Function entrypoint for the LangGraph Research Agent FastAPI app."""

import sys
from pathlib import Path

# Ensure workspace root directory is in sys.path for module resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from web_app import app

# Vercel's Python runtime requires an ASGI application instance named 'app'
__all__ = ["app"]
