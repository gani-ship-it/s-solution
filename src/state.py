"""State definitions for the LangGraph Research Agent."""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchResult(TypedDict):
    """Single search result item."""
    title: str
    url: str
    snippet: str


class FetchedSource(TypedDict):
    """Successfully fetched and cleaned webpage content with source ID."""
    id: str  # e.g., 'S1', 'S2'
    url: str
    title: str
    content: str
    char_count: int


class SourceSummary(TypedDict):
    """Summary of relevant facts from a fetched source."""
    source_id: str  # References FetchedSource.id
    url: str
    summary: str
    key_points: List[str]


class ActionRecord(TypedDict):
    """Audit log entry for an autonomous agent action."""
    step: int
    action: str  # SEARCH, FETCH, SUMMARISE, FINISH
    params: Dict[str, Any]
    status: str  # 'success', 'failed', 'empty', 'in_progress'
    details: str


class AgentDecision(BaseModel):
    """Structured decision produced by the LLM reasoning node."""
    thought: str = Field(
        description="Reasoning about the current state, what is known, what is missing, and why this action was chosen."
    )
    action: str = Field(
        description="The action to perform next: 'SEARCH', 'FETCH', 'SUMMARISE', or 'FINISH'."
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Search query string if action is 'SEARCH'."
    )
    fetch_url: Optional[str] = Field(
        default=None,
        description="URL to fetch if action is 'FETCH'. Must be chosen from search results."
    )
    summarise_source_id: Optional[str] = Field(
        default=None,
        description="Source ID (e.g., 'S1', 'S2') to summarise if action is 'SUMMARISE'."
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Comprehensive answer with inline citations (e.g. [S1], [S2]) if action is 'FINISH'."
    )


class ResearchState(TypedDict):
    """Complete LangGraph state for the autonomous research process."""
    question: str
    search_results: List[SearchResult]
    fetched_sources: Dict[str, FetchedSource]  # Keyed by 'S1', 'S2', etc.
    failed_urls: List[Dict[str, str]]
    summaries: List[SourceSummary]
    current_step: int
    max_steps: int
    action_history: List[ActionRecord]
    next_action: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    citations: List[str]
    status: str  # 'in_progress', 'completed', 'budget_exceeded', 'error'
    last_error: Optional[str]
