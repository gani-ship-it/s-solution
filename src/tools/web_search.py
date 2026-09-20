import logging
import warnings
from typing import List, Optional

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from src.state import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MOCK_RESULTS: List[SearchResult] = [
    {
        "title": "Recent Breakthroughs and Empirical Evaluations in Applied Research",
        "url": "https://example.org/research/solid-state-breakthroughs",
        "snippet": "Experimental investigations demonstrate 12 mS/cm ionic conductivity at room temperature, representing a 35% efficiency boost over prior baselines.",
    },
    {
        "title": "Industrial Scaling and Operational Stability Benchmarks in Multi-Cell Trials",
        "url": "https://example.org/industry/scaling-and-cycling-benchmarks",
        "snippet": "Multi-center operational evaluations verified 88% capacity retention across 1,200 fast cycles under ambient atmospheric pressure.",
    },
    {
        "title": "Comparative Architectural Analysis and Manufacturing Feasibility Review",
        "url": "https://example.org/reviews/architectural-analysis",
        "snippet": "Comprehensive overview comparing interface engineering, dendrite suppression mechanisms, and automated fabrication pipelines.",
    },
]


def search_web(
    query: str,
    max_results: int = 5,
    provider: str = "duckduckgo",
    tavily_key: Optional[str] = None,
    mock_results: Optional[List[SearchResult]] = None,
) -> List[SearchResult]:
    """Execute a web search and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        provider: 'duckduckgo' or 'tavily'.
        tavily_key: Optional API key for Tavily search.
        mock_results: Optional pre-defined results for testing or offline demos.

    Returns:
        A list of SearchResult dictionaries with title, url, and snippet.
    """
    if mock_results is not None:
        return mock_results[:max_results]

    if provider == "mock":
        return DEFAULT_MOCK_RESULTS[:max_results]

    results: List[SearchResult] = []
    clean_query = query.strip()
    if not clean_query:
        return results

    # 1. Try Tavily if configured and key provided
    if provider == "tavily" and tavily_key:
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": clean_query, "max_results": max_results},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    })
                return results
        except Exception as e:
            logger.warning(f"Tavily search error: {e}. Falling back to DuckDuckGo.")

    # 2. Primary free engine: DuckDuckGo
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ddgs = DDGS()
        raw_results = list(ddgs.text(clean_query, max_results=max_results))
        for item in raw_results:
            title = item.get("title", "No Title")
            url = item.get("href") or item.get("url", "")
            snippet = item.get("body") or item.get("snippet", "")
            if url:
                results.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": snippet.strip(),
                })
    except Exception as e:
        logger.error(f"DuckDuckGo search error for '{clean_query}': {e}")
        # Return empty list gracefully so the agent can react
        return []

    return results
