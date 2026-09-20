"""Summarisation tool to extract question-relevant information from fetched sources."""

import json
import logging
import re
from typing import Optional, Tuple, Dict, Any

from src.state import FetchedSource, SourceSummary
from src.prompts import SUMMARISE_SYSTEM_PROMPT, SUMMARISE_USER_TEMPLATE

logger = logging.getLogger(__name__)


def summarise_source(
    source_id: str,
    question: str,
    fetched_sources: Dict[str, FetchedSource],
    llm: Any = None,
) -> Tuple[Optional[SourceSummary], Optional[str]]:
    """Extract and summarize only the information relevant to the research question from a source.

    Args:
        source_id: Unique identifier of the source (e.g., 'S1', 'S2').
        question: The user's research query.
        fetched_sources: Dictionary of currently fetched sources.
        llm: LangChain compatible LLM or MockLLM instance.

    Returns:
        Tuple of (SourceSummary if successful, error_message if failed).
    """
    if source_id not in fetched_sources:
        available = list(fetched_sources.keys())
        return None, f"Source ID '{source_id}' does not exist. Available sources: {available}"

    source = fetched_sources[source_id]
    content = source.get("content", "").strip()
    if not content:
        return None, f"Source '{source_id}' ({source.get('url')}) has empty content."

    # If an LLM is provided, use it for contextual question-focused extraction
    if llm is not None:
        import time
        user_prompt = SUMMARISE_USER_TEMPLATE.format(
            question=question,
            source_id=source_id,
            url=source.get("url", ""),
            title=source.get("title", ""),
            content=content[:2500],  # Moderate character budget to conserve tokens
        )

        messages = [
            {"role": "system", "content": SUMMARISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(3):
            try:
                response = llm.invoke(messages)
                raw_text = response.content if hasattr(response, "content") else str(response)

                # Extract JSON block
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    summary = data.get("summary", "")
                    key_points = data.get("key_points", [])
                    if isinstance(key_points, str):
                        key_points = [key_points]

                    if summary:
                        return {
                            "source_id": source_id,
                            "url": source.get("url", ""),
                            "summary": summary.strip(),
                            "key_points": [str(kp).strip() for kp in key_points if str(kp).strip()],
                        }, None
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < 2:
                    time.sleep(2.0)
                    continue
                logger.warning(f"LLM summarisation for {source_id} failed: {e}. Falling back to heuristic summariser.")
                break

    # Fallback heuristic summariser if LLM is unavailable or failed
    sentences = re.split(r"(?<=[.?!])\s+", content)
    query_keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]

    # Rank sentences by keyword overlap with the question
    relevant_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 20:
            continue
        score = sum(1 for kw in query_keywords if kw in s_clean.lower())
        if score > 0:
            relevant_sentences.append((score, s_clean))

    relevant_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in relevant_sentences[:5]]

    if not top_sentences:
        top_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]

    summary_text = " ".join(top_sentences)
    if not summary_text:
        summary_text = content[:500]

    return {
        "source_id": source_id,
        "url": source.get("url", ""),
        "summary": summary_text,
        "key_points": top_sentences[:4],
    }, None
