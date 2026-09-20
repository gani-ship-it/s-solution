"""Shared fixtures and mocks for testing the Research Agent."""

import pytest
from typing import Dict, Any, List
from src.state import ResearchState, FetchedSource, SearchResult
from src.agent import create_initial_state


@pytest.fixture
def initial_state() -> ResearchState:
    """Provide a standard initial state."""
    return create_initial_state(question="What are the latest breakthroughs in room-temperature superconductors?", max_steps=8)


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Sample search results for testing."""
    return [
        {
            "title": "Room-Temperature Superconductivity Updates 2026",
            "url": "https://example.com/superconductor-2026",
            "snippet": "Researchers announce verified magnetic levitation and zero electrical resistance at 295K.",
        },
        {
            "title": "Independent Replication Report on Ambient Superconductors",
            "url": "https://example.com/replication-report",
            "snippet": "Independent laboratories test the material under 1 atm pressure with reproducible data.",
        },
    ]


@pytest.fixture
def sample_html_page() -> str:
    """Realistic HTML page for parser testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breakthrough in Ambient Superconductivity</title>
        <script>console.log('tracker');</script>
        <style>body { font-family: sans-serif; }</style>
    </head>
    <body>
        <header><nav><a href="/">Home</a></nav></header>
        <main>
            <h1>Breakthrough in Ambient Superconductivity</h1>
            <p>Scientists have synthesized a modified copper-substituted apatite structure that exhibits zero resistance at 294 Kelvin under ambient atmospheric pressure.</p>
            <p>The critical current density exceeded 10^5 A/cm2, enabling practical lossless power transmission cables.</p>
        </main>
        <footer><p>Copyright 2026 Science Daily</p></footer>
    </body>
    </html>
    """
