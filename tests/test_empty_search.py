"""Tests for handling empty search results and query refinement."""

from unittest.mock import patch, MagicMock
from src.tools.web_search import search_web
from src.agent import build_research_agent, MockLLM, create_initial_state
from src.config import Settings


def test_search_web_returns_empty_list_on_no_matches():
    """Verify search_web returns empty list cleanly when no results found."""
    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.text.return_value = []
        mock_ddgs.return_value = instance

        results = search_web("unfindable obscure query string 12345xyz")
        assert results == []
        assert isinstance(results, list)


def test_search_web_handles_exception_gracefully():
    """Verify search_web does not raise exceptions when DDGS encounters network error."""
    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.text.side_effect = ConnectionResetError("Connection dropped by peer")
        mock_ddgs.return_value = instance

        results = search_web("any query")
        assert results == []


def test_agent_handles_empty_search_and_refines_query():
    """Verify agent encounters 0 search results, records error, and can execute a refined search."""
    # Scripted sequence:
    # 1. SEARCH "obscure term" -> returns []
    # 2. SEARCH "refined term" -> returns results
    # 3. FINISH
    decisions = [
        {
            "thought": "Searching with obscure query.",
            "action": "SEARCH",
            "search_query": "obscure query with 0 results",
            "fetch_url": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "Previous search returned 0 results. Refining query to broader keywords.",
            "action": "SEARCH",
            "search_query": "broader refined query",
            "fetch_url": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "Search succeeded. Proceeding to finalize report.",
            "action": "FINISH",
            "final_answer": "Report finalized after finding results.",
            "search_query": None,
            "fetch_url": None,
            "summarise_source_id": None,
        },
    ]

    llm = MockLLM(scripted_decisions=decisions)
    settings = Settings(max_steps=6, llm_provider="mock")
    agent = build_research_agent(llm=llm, settings=settings, show_display=False)

    initial_state = create_initial_state(question="Test empty search handling", max_steps=6)

    def mock_ddg_text(query, **kwargs):
        if "obscure" in query:
            return []
        return [{"title": "Refined Result", "href": "https://example.com/item", "body": "Found content"}]

    with patch("src.tools.web_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.text.side_effect = mock_ddg_text
        mock_ddgs.return_value = instance

        final_state = agent.invoke(initial_state)

    # First search should be recorded as empty
    assert final_state["action_history"][0]["action"] == "SEARCH"
    assert final_state["action_history"][0]["status"] == "empty"

    # Second search should be recorded as success
    assert final_state["action_history"][1]["action"] == "SEARCH"
    assert final_state["action_history"][1]["status"] == "success"
    assert len(final_state["search_results"]) == 1
