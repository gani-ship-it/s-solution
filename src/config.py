"""Configuration settings for the AI Research Agent."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


@dataclass
class Settings:
    """Application configuration settings."""
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_steps: int = int(os.getenv("MAX_STEPS", "8"))
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    fetch_timeout: int = int(os.getenv("FETCH_TIMEOUT", "10"))
    max_content_chars: int = int(os.getenv("MAX_CONTENT_CHARS", "6000"))
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    streaming: bool = os.getenv("STREAMING", "true").lower() == "true"


def get_settings() -> Settings:
    """Retrieve settings instance with current environment values."""
    return Settings()
