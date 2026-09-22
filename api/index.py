"""Vercel Serverless Function entrypoint for the LangGraph Research Agent FastAPI app."""

import sys
import traceback
from pathlib import Path

# Ensure workspace root, api directory, and serverless /var/task are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent

for p in [str(ROOT_DIR), str(API_DIR), "/var/task"]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from web_app import app
except Exception:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    err_trace = traceback.format_exc()
    print("FATAL ERROR DURING APP INITIALIZATION:\n", err_trace)

    app = FastAPI(title="Initialization Error")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(full_path: str = ""):
        debug_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Vercel App Initialization Error</title>
  <style>
    body {{ background: #080303; color: #ff3355; font-family: monospace; padding: 24px; line-height: 1.6; }}
    h1 {{ color: #ff5566; }}
    pre {{ background: #1a0508; padding: 16px; border: 1px solid #ff2244; overflow-x: auto; color: #ffccdd; }}
    .box {{ background: #110505; border: 1px solid #441111; padding: 12px; margin-top: 14px; color: #aaa; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>[!] App Initialization Failed on Vercel</h1>
  <p>The FastAPI application encountered an unhandled exception during startup. Traceback:</p>
  <pre>{err_trace}</pre>
  <div class="box">
    <strong>Diagnostic Info:</strong><br>
    Python: {sys.version}<br>
    Root Dir: {ROOT_DIR} (exists: {ROOT_DIR.exists()})<br>
    sys.path: {sys.path}<br>
  </div>
</body>
</html>"""
        return HTMLResponse(content=debug_html, status_code=500)

__all__ = ["app"]
