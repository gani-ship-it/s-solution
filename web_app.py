"""FastAPI Web Server providing a real-time Terminal Web Interface for the Research Agent."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import Settings, get_settings
from src.agent import build_research_agent, create_initial_state, MockLLM

logger = logging.getLogger("web_app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LangGraph Research Agent Terminal")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ResearchRequest(BaseModel):
    question: str
    mock: bool = False
    max_steps: Optional[int] = 8


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the terminal web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/api/research")
async def run_research_stream(req: ResearchRequest):
    """Execute research agent and stream real-time events to the web terminal."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Research question cannot be empty.")

    settings = get_settings()
    if req.max_steps:
        settings.max_steps = req.max_steps

    is_mock = req.mock or settings.llm_provider == "mock" or (not settings.openai_api_key and not settings.groq_api_key)
    if is_mock:
        settings.llm_provider = "mock"
        settings.search_provider = "mock"

    llm = MockLLM() if is_mock else None

    # Async queue for SSE streaming
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def event_callback(event: dict):
        """Thread-safe callback invoked by the LangGraph agent nodes."""
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    # Compile agent with event callback and no terminal CLI output (web handles it)
    agent = build_research_agent(
        llm=llm,
        settings=settings,
        show_display=False,
        on_event=event_callback,
    )

    initial_state = create_initial_state(question=question, max_steps=settings.max_steps)

    async def execute_agent():
        """Worker executing LangGraph synchronously in background thread."""
        try:
            await asyncio.to_thread(agent.invoke, initial_state)
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            event_callback({"type": "error", "message": str(e)})
        finally:
            event_callback({"type": "end_of_stream"})

    # Launch agent task
    asyncio.create_task(execute_agent())

    async def sse_generator():
        """Generator streaming Server-Sent Events to the client."""
        while True:
            event = await event_queue.get()
            if event.get("type") == "end_of_stream":
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


import socket


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8000, max_attempts: int = 10, host: str = "127.0.0.1") -> int:
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    return start_port


def start(port: Optional[int] = None):
    """Start uvicorn server with automatic port conflict handling."""
    target_port = port or int(os.getenv("PORT", "8000"))
    chosen_port = target_port

    if is_port_in_use(chosen_port):
        alt_port = find_available_port(chosen_port + 1)
        print(f"\n[!] Note: Port {chosen_port} is already in use (an instance may already be running).")
        print(f"--> Automatically launching on available port {alt_port}!\n")
        chosen_port = alt_port

    print(f"\n=======================================================")
    print(f"  LANGGRAPH RESEARCH AGENT WEB TERMINAL RUNNING")
    print(f"  URL: http://127.0.0.1:{chosen_port}")
    print(f"=======================================================\n")
    uvicorn.run("web_app:app", host="127.0.0.1", port=chosen_port, reload=False)


if __name__ == "__main__":
    start()
