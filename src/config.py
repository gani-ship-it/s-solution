"""Configuration settings for the AI Research Agent."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def _parse_int(val: Optional[str], default: int) -> int:
    """Safely parse integer from environment variable, handling empty strings and invalid formats."""
    if val is None or not str(val).strip():
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def _parse_float(val: Optional[str], default: float) -> float:
    """Safely parse float from environment variable, handling empty strings and invalid formats."""
    if val is None or not str(val).strip():
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


@dataclass
class Settings:
    """Application configuration settings."""
    llm_provider: str = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY") or None
    model_name: str = (os.getenv("MODEL_NAME") or "gpt-4o-mini").strip()
    temperature: float = _parse_float(os.getenv("TEMPERATURE"), 0.2)
    max_steps: int = _parse_int(os.getenv("MAX_STEPS"), 8)
    search_provider: str = (os.getenv("SEARCH_PROVIDER") or "duckduckgo").strip().lower()
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY") or None
    fetch_timeout: int = _parse_int(os.getenv("FETCH_TIMEOUT"), 10)
    max_content_chars: int = _parse_int(os.getenv("MAX_CONTENT_CHARS"), 6000)
    max_search_results: int = _parse_int(os.getenv("MAX_SEARCH_RESULTS"), 5)
    streaming: bool = (os.getenv("STREAMING") or "true").strip().lower() == "true"


def get_settings() -> Settings:
    """Retrieve settings instance with current environment values."""
    return Settings()

