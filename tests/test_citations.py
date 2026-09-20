"""Tests for source ID tracking (S1, S2, S3) and citation validation."""

from unittest.mock import patch, MagicMock
from src.tools.fetcher import fetch_webpage
from src.agent import extract_citations, build_research_agent, MockLLM, create_initial_state
from src.config import Settings


def test_sequential_source_id_allocation():
    """Verify that fetched sources are assigned S1, S2, S3 sequentially."""
    fetched_sources = {}

    def make_mock_resp(title, text):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = f"<html><head><title>{title}</title></head><body><p>{text}</p></body></html>"
        return resp

    with patch("requests.get") as mock_get:
        # First source
        mock_get.return_value = make_mock_resp("Doc 1", "First document content for scientific analysis with detailed factual findings.")
        s1, err1 = fetch_webpage("https://example.com/one", fetched_sources=fetched_sources)
        assert err1 is None
        assert s1["id"] == "S1"
        fetched_sources["S1"] = s1

        # Second source
        mock_get.return_value = make_mock_resp("Doc 2", "Second document content for scientific analysis with detailed factual findings.")
        s2, err2 = fetch_webpage("https://example.com/two", fetched_sources=fetched_sources)
        assert err2 is None
        assert s2["id"] == "S2"
        fetched_sources["S2"] = s2

        # Third source
        mock_get.return_value = make_mock_resp("Doc 3", "Third document content for scientific analysis with detailed factual findings.")
        s3, err3 = fetch_webpage("https://example.com/three", fetched_sources=fetched_sources)
        assert err3 is None
        assert s3["id"] == "S3"
        fetched_sources["S3"] = s3

    assert list(fetched_sources.keys()) == ["S1", "S2", "S3"]


def test_fetch_existing_url_returns_same_source_id_without_duplicate():
    """Verify fetching an already-cached URL returns the existing source without incrementing."""
    fetched_sources = {
        "S1": {
            "id": "S1",
            "url": "https://example.com/existing",
            "title": "Existing Doc",
            "content": "Existing content",
            "char_count": 16,
        }
    }

    source, err = fetch_webpage("https://example.com/existing", fetched_sources=fetched_sources)
    assert err is None
    assert source["id"] == "S1"
    assert len(fetched_sources) == 1


def test_extract_citations_utility():
    """Verify regex extraction of citation tags."""
    text = (
        "Quantum coherence was maintained for 1.4 milliseconds [S1]. "
        "A 40% reduction in error rate was observed under surface code lattice configurations [S2]. "
        "Further improvements were noted at low temperatures [S1]."
    )
    citations = extract_citations(text)
    assert citations == ["S1", "S2"]


def test_final_answer_citations_match_fetched_sources():
    """Verify agent output citations correspond strictly to fetched source IDs."""
    decisions = [
        {
            "thought": "Fetching source one.",
            "action": "FETCH",
            "fetch_url": "https://example.com/paper1",
            "search_query": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "Fetching source two.",
            "action": "FETCH",
            "fetch_url": "https://example.com/paper2",
            "search_query": None,
            "summarise_source_id": None,
            "final_answer": None,
        },
        {
            "thought": "Summarising S1.",
            "action": "SUMMARISE",
            "summarise_source_id": "S1",
            "search_query": None,
            "fetch_url": None,
            "final_answer": None,
        },
        {
            "thought": "Summarising S2.",
            "action": "SUMMARISE",
            "summarise_source_id": "S2",
            "search_query": None,
            "fetch_url": None,
            "final_answer": None,
        },
        {
            "thought": "Finishing with grounded citations.",
            "action": "FINISH",
            "final_answer": "Finding A is confirmed by [S1], while benchmark B is verified by [S2].",
            "search_query": None,
            "fetch_url": None,
            "summarise_source_id": None,
        },
    ]

    llm = MockLLM(scripted_decisions=decisions)
    settings = Settings(max_steps=8, llm_provider="mock")
    agent = build_research_agent(llm=llm, settings=settings, show_display=False)

    initial_state = create_initial_state(question="Citation test query", max_steps=8)

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = f"<html><body><p>Substantive text from {url} to pass length check for source storage.</p></body></html>"
        return resp

    with patch("requests.get", side_effect=mock_get):
        final_state = agent.invoke(initial_state)

    # Check that S1 and S2 exist
    assert "S1" in final_state["fetched_sources"]
    assert "S2" in final_state["fetched_sources"]
    assert final_state["citations"] == ["S1", "S2"]
    for c in final_state["citations"]:
        assert c in final_state["fetched_sources"]
