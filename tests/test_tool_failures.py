"""Tests for graceful error handling during tool failures."""

import json
from unittest.mock import patch, MagicMock
import requests

from src.tools.fetcher import fetch_webpage, clean_html_content
from src.tools.summarizer import summarise_source
from src.agent import build_research_agent, MockLLM, MockResponse, create_initial_state
from src.config import Settings


def test_fetch_webpage_http_404_error():
    """Verify HTTP 404 is caught and returns an error without crashing."""
    fetched_sources = {}

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        source, error = fetch_webpage("https://example.com/not-found", fetched_sources=fetched_sources)

        assert source is None
        assert "HTTP 404" in error
        assert "not-found" in error


def test_fetch_webpage_timeout_handling():
    """Verify connection timeouts are handled cleanly."""
    fetched_sources = {}

    with patch("requests.get", side_effect=requests.exceptions.Timeout("Read timed out")):
        source, error = fetch_webpage("https://slow-server.org/article", fetched_sources=fetched_sources, timeout=2)

        assert source is None
        assert "timed out" in error


def test_fetch_webpage_unsupported_content_type():
    """Verify unsupported binary content (e.g. zip or binary stream) is rejected cleanly."""
    fetched_sources = {}

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/zip"}
        mock_get.return_value = mock_resp

        source, error = fetch_webpage("https://example.com/archive.zip", fetched_sources=fetched_sources)

        assert source is None
        assert "Unsupported Content-Type" in error


def test_summarise_source_nonexistent_id():
    """Verify summarising a non-existent source ID returns an informative error."""
    fetched_sources = {}
    summary, error = summarise_source("S99", "What are the findings?", fetched_sources=fetched_sources)

    assert summary is None
    assert "S99" in error
    assert "does not exist" in error


def test_agent_recovers_from_fetch_failure_and_tries_next_source():
    """Verify agent encounters a fetch failure on first URL, logs it, and succeeds with second URL."""
    # Scripted LLM decisions:
    # 1. FETCH url1 (will fail)
    # 2. FETCH url2 (will succeed)
    # 3. SUMMARISE S1
    # 4. FINISH
    decisions = [
        {
            "thought": "Fetching first URL from search results.",
            "action": "FETCH",
            "fetch_url": "https://broken-link.com/404",
            "search_query": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "First URL failed. Fetching alternative source.",
            "action": "FETCH",
            "fetch_url": "https://working-link.com/valid",
            "search_query": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "Source S1 fetched. Now summarising.",
            "action": "SUMMARISE",
            "summarise_source_id": "S1",
            "search_query": None,
            "fetch_url": None,
            "final_answer": None,
        },
        {
            "thought": "Formulating final answer from S1.",
            "action": "FINISH",
            "final_answer": "Verified findings from the valid source show stability [S1].",
            "search_query": None,
            "fetch_url": None,
            "summarise_source_id": None,
        },
    ]

    llm = MockLLM(scripted_decisions=decisions)
    settings = Settings(max_steps=8, llm_provider="mock")
    agent = build_research_agent(llm=llm, settings=settings, show_display=False)

    initial_state = create_initial_state(question="Test failure recovery", max_steps=8)

    # Mock requests to fail on broken-link and succeed on working-link
    def mock_get(url, **kwargs):
        resp = MagicMock()
        if "broken-link" in url:
            resp.status_code = 404
            return resp
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><h1>Valid Page</h1><p>Verified laboratory tests show remarkable room-temperature stability.</p></body></html>"
        return resp

    with patch("requests.get", side_effect=mock_get):
        final_state = agent.invoke(initial_state)

    # Verify agent state
    assert len(final_state["failed_urls"]) == 1
    assert final_state["failed_urls"][0]["url"] == "https://broken-link.com/404"
    assert "S1" in final_state["fetched_sources"]
    assert final_state["fetched_sources"]["S1"]["url"] == "https://working-link.com/valid"
    assert final_state["status"] == "completed"
    assert "[S1]" in final_state["final_answer"]
