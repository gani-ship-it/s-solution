"""LangGraph-based Autonomous AI Research Agent."""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from src.state import (
    ResearchState,
    AgentDecision,
    SearchResult,
    FetchedSource,
    SourceSummary,
    ActionRecord,
)
from src.prompts import (
    REASONING_SYSTEM_PROMPT,
    REASONING_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from src.tools.web_search import search_web
from src.tools.fetcher import fetch_webpage
from src.tools.summarizer import summarise_source
from src.config import Settings, get_settings
from src.utils.display import (
    print_step_header,
    print_decision,
    print_tool_status,
)

logger = logging.getLogger(__name__)


class MockLLM:
    """Deterministic Mock LLM for offline testing and demonstrations."""

    def __init__(self, scripted_decisions: Optional[List[Dict[str, Any]]] = None):
        self.scripted_decisions = scripted_decisions or []
        self.call_count = 0

    def stream(self, messages: List[Dict[str, str]]):
        """Simulate token chunk streaming."""
        resp = self.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        words = content.split(" ")
        for i, w in enumerate(words):
            yield MockResponse(w + (" " if i < len(words) - 1 else ""))

    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """Simulate LLM response based on prompt context or scripted queue."""
        self.call_count += 1
        prompt_text = " ".join([m.get("content", "") for m in messages])

        # If summarisation prompt
        if "expert research analyst" in prompt_text.lower():
            return MockResponse(
                '{"summary": "Research findings demonstrate key advances and measured metrics relevant to the query.", "key_points": ["Key metric improved by 35%", "Novel mechanism confirmed in peer-reviewed trials"]}'
            )

        # If synthesis prompt
        if "senior scientific research synthesizer" in prompt_text.lower():
            return MockResponse(
                "## Synthesis\n\nRecent scientific investigations demonstrate significant breakthroughs. "
                "The primary architecture showed a 35% efficiency boost under standard conditions [S1]. "
                "Furthermore, complementary multi-center trials verified long-term operational stability [S2].\n\n"
                "### Key Outcomes\n"
                "- High operational stability confirmed across temperature ranges [S1].\n"
                "- Scalability demonstrated in automated fabrication pipelines [S2]."
            )

        # Check scripted decisions first
        if self.scripted_decisions and len(self.scripted_decisions) > 0:
            decision = self.scripted_decisions.pop(0)
            return MockResponse(json.dumps(decision))

        # Autonomous heuristic simulation for mock runs
        if "=== CURRENT SEARCH RESULTS (0 found) ===" in prompt_text or "No searches performed yet" in prompt_text:
            return MockResponse(json.dumps({
                "thought": "No search results exist yet. I will start by searching for primary literature and recent findings on this topic.",
                "action": "SEARCH",
                "search_query": "advances breakthroughs analysis",
                "fetch_url": None,
                "summarise_source_id": None,
                "final_answer": None
            }))

        # Parse search results, fetched sources, and failed URLs from prompt
        search_urls = []
        if "=== CURRENT SEARCH RESULTS" in prompt_text and "=== FETCHED SOURCES" in prompt_text:
            search_section = prompt_text.split("=== CURRENT SEARCH RESULTS")[1].split("=== FETCHED SOURCES")[0]
            search_urls = re.findall(r"URL: (https?://[^\s\n]+)", search_section)

        failed_urls = set()
        if "=== FAILED URLS" in prompt_text and "=== SOURCE SUMMARIES" in prompt_text:
            failed_section = prompt_text.split("=== FAILED URLS")[1].split("=== SOURCE SUMMARIES")[0]
            failed_urls = set(re.findall(r"URL: (https?://[^\s\n]+)", failed_section))

        fetched_urls = set()
        if "=== FETCHED SOURCES" in prompt_text and "=== FAILED URLS" in prompt_text:
            fetched_section = prompt_text.split("=== FETCHED SOURCES")[1].split("=== FAILED URLS")[0]
            fetched_urls = set(re.findall(r"URL: (https?://[^\s\n]+)", fetched_section))

        # Available URLs to try
        available_urls = [u for u in search_urls if u not in failed_urls and u not in fetched_urls]

        # If search results exist but 0 sources fetched, fetch first available URL
        if "=== FETCHED SOURCES (0 sources) ===" in prompt_text and search_urls:
            if available_urls:
                target_url = available_urls[0]
                return MockResponse(json.dumps({
                    "thought": "I have search results. I will fetch an available source to extract verified content.",
                    "action": "FETCH",
                    "search_query": None,
                    "fetch_url": target_url,
                    "summarise_source_id": None,
                    "final_answer": None,
                }))
            else:
                return MockResponse(json.dumps({
                    "thought": "All search results failed to fetch. I will conduct a refined search.",
                    "action": "SEARCH",
                    "search_query": "solid state battery electrolytes advances",
                    "fetch_url": None,
                    "summarise_source_id": None,
                    "final_answer": None,
                }))

        # If S1 is fetched but not summarized
        if "ID: S1" in prompt_text and "=== SOURCE SUMMARIES & NOTES (0 summaries) ===" in prompt_text:
            return MockResponse(json.dumps({
                "thought": "Source S1 has been successfully fetched. I must now summarise it to extract facts relevant to the research question.",
                "action": "SUMMARISE",
                "search_query": None,
                "fetch_url": None,
                "summarise_source_id": "S1",
                "final_answer": None,
            }))

        # If S1 is summarized, check if we can fetch S2 from available URLs
        if "ID: S1" in prompt_text and ("=== SOURCE SUMMARIES & NOTES (1 summaries) ===" in prompt_text or "=== SOURCE SUMMARIES & NOTES (1 " in prompt_text):
            if available_urls and "ID: S2" not in prompt_text:
                return MockResponse(json.dumps({
                    "thought": "I have one summarized source. Fetching a second complementary source to verify findings.",
                    "action": "FETCH",
                    "search_query": None,
                    "fetch_url": available_urls[0],
                    "summarise_source_id": None,
                    "final_answer": None,
                }))
            else:
                return MockResponse(json.dumps({
                    "thought": "I have sufficient verified information from fetched sources to formulate a well-grounded answer with citations.",
                    "action": "FINISH",
                    "search_query": None,
                    "fetch_url": None,
                    "summarise_source_id": None,
                    "final_answer": (
                        "## Research Findings\n\n"
                        "Recent published benchmarks demonstrate substantial progress in this domain [S1]. "
                        "The observed efficiency gained over previous baselines exceeds 35% with high reproducibility [S1].\n\n"
                        "### Conclusion\n"
                        "The technology is transitioning rapidly from benchtop prototypes to scalable implementations [S1]."
                    )
                }))

        # If S2 fetched but not summarized
        if "ID: S2" in prompt_text and "=== SOURCE SUMMARIES & NOTES (1 summaries) ===" in prompt_text:
            return MockResponse(json.dumps({
                "thought": "Source S2 is fetched. Now summarising key findings from S2.",
                "action": "SUMMARISE",
                "search_query": None,
                "fetch_url": None,
                "summarise_source_id": "S2",
                "final_answer": None,
            }))

        # Ready to finish
        return MockResponse(json.dumps({
            "thought": "I have collected and summarized multiple independent sources. Ready to generate the final cited synthesis.",
            "action": "FINISH",
            "search_query": None,
            "fetch_url": None,
            "summarise_source_id": None,
            "final_answer": (
                "## Executive Summary\n\n"
                "Investigation into this subject reveals two major developments. "
                "First, core performance metrics showed a 35% enhancement over conventional systems [S1]. "
                "Second, field trials and independent stress evaluations confirmed operational stability [S2].\n\n"
                "### Detailed Breakdown\n"
                "- Primary architecture yields increased efficiency with lower power dissipation [S1].\n"
                "- Independent replication across multiple test environments verifies consistency [S2]."
            )
        }))


class MockResponse:
    """Wrapper matching LangChain message response object."""

    def __init__(self, content: str):
        self.content = content


def invoke_or_stream(
    llm: Any,
    messages: List[Dict[str, str]],
    stream: bool = False,
    stream_title: Optional[str] = None,
    on_token: Optional[Any] = None,
) -> str:
    """Invoke LLM with optional token-by-token streaming to standard output.

    Args:
        llm: LangChain or Mock LLM instance.
        messages: Prompt messages.
        stream: Whether to stream tokens live to stdout.
        stream_title: Optional title banner to display above streamed tokens.
        on_token: Optional callback invoked for each generated token string.

    Returns:
        The complete generated text.
    """
    if stream and hasattr(llm, "stream"):
        from src.utils.display import start_stream, stream_chunk, end_stream
        if stream_title:
            start_stream(stream_title)

        chunks = []
        for chunk in llm.stream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            chunks.append(token)
            if stream_title:
                stream_chunk(token)
            if on_token:
                on_token(token)

        if stream_title:
            end_stream()

        return "".join(chunks)
    else:
        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        if on_token:
            on_token(text)
        return text


def extract_citations(text: str) -> List[str]:
    """Extract unique source citation tags like [S1], [S2] from text."""
    matches = re.findall(r"\[(S\d+)\]", text)
    # preserve order while making unique
    seen = set()
    citations = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            citations.append(m)
    return citations


def build_research_agent(
    llm: Any = None,
    settings: Optional[Settings] = None,
    show_display: bool = True,
    on_event: Optional[Any] = None,
):
    """Build and compile the LangGraph StateGraph for the Research Agent.

    Args:
        llm: LangChain LLM or MockLLM instance.
        settings: Application settings.
        show_display: Whether to print rich real-time CLI status panels.

    Returns:
        Compiled LangGraph runnable.
    """
    cfg = settings or get_settings()

    # Determine LLM instance
    if llm is None:
        if cfg.llm_provider == "groq" or (cfg.groq_api_key and not cfg.openai_api_key):
            from langchain_openai import ChatOpenAI
            # Free-tier Groq model fallback chain (tried in order until one works)
            _OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "text-davinci")
            _GROQ_FALLBACK_CHAIN = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "allam-2-7b",
            ]
            # Only remap pure OpenAI cloud model names (not Groq's openai/gpt-oss-* models)
            _OPENAI_CLOUD_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
                                    "gpt-4-turbo", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"}
            model = (
                cfg.model_name
                if cfg.model_name not in _OPENAI_CLOUD_MODELS
                else _GROQ_FALLBACK_CHAIN[0]
            )
            logger.info(f"Using Groq LLM provider with model: {model}")
            if on_event:
                on_event({"type": "llm_info", "provider": "groq", "model": model})

            def _make_groq_llm(m: str):
                return ChatOpenAI(
                    model=m,
                    temperature=cfg.temperature,
                    api_key=cfg.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    max_retries=2,
                )

            # Auto-fallback: if the configured model returns 404, try the chain
            _groq_model_chain = [model] + [m for m in _GROQ_FALLBACK_CHAIN if m != model]

            class _GroqWithFallback:
                """Wraps ChatOpenAI with automatic model fallback on 404 errors."""
                def __init__(self):
                    self._idx = 0
                    self._client = _make_groq_llm(_groq_model_chain[0])

                def invoke(self, messages):
                    for i, m in enumerate(_groq_model_chain):
                        try:
                            client = _make_groq_llm(m)
                            result = client.invoke(messages)
                            if i > 0:
                                logger.info(f"Groq fallback succeeded with model: {m}")
                                if on_event:
                                    on_event({"type": "llm_info", "provider": "groq", "model": m})
                            return result
                        except Exception as e:
                            if any(code in str(e) for code in ["404", "400", "model_not_found", "model_decommissioned"]):
                                logger.warning(f"Groq model {m!r} unavailable ({type(e).__name__}), trying next fallback...")
                                continue
                            raise
                    raise RuntimeError(f"All Groq models failed: {_groq_model_chain}")

            llm = _GroqWithFallback()
        elif cfg.llm_provider == "openai" and cfg.openai_api_key:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=cfg.model_name,
                temperature=cfg.temperature,
                api_key=cfg.openai_api_key,
            )
        else:
            logger.info("Using built-in MockLLM (no API key configured).")
            llm = MockLLM()

    # Define Node Functions

    def reasoning_node(state: ResearchState) -> Dict[str, Any]:
        """LLM analyzes state and chooses the next tool or FINISH."""
        step = state.get("current_step", 0)
        max_steps = state.get("max_steps", cfg.max_steps)

        # Hard step limit guardrail: if budget reached, force finish
        if step >= max_steps:
            if show_display:
                print_step_header(step, max_steps)
                print_tool_status("BUDGET", "warning", f"Hard step limit ({max_steps}) reached. Forcing final synthesis.")
            if on_event:
                on_event({"type": "step", "step": step, "max_steps": max_steps})
                on_event({"type": "tool_result", "action": "BUDGET", "status": "warning", "summary": f"Step limit ({max_steps}) reached."})
            return {
                "next_action": {"action": "FINISH", "reason": "step_budget_exceeded"},
                "status": "budget_exceeded",
            }

        # Build prompt inputs with token efficiency
        search_results = state.get("search_results", [])
        if search_results:
            search_lines = []
            for i, r in enumerate(search_results, 1):
                snippet = r['snippet'][:180] + ("..." if len(r['snippet']) > 180 else "")
                search_lines.append(f"{i}. Title: {r['title']}\n   URL: {r['url']}\n   Snippet: {snippet}")
            search_text = "\n".join(search_lines)
        else:
            search_text = "No searches performed yet or previous search yielded 0 results."

        fetched = state.get("fetched_sources", {})
        if fetched:
            fetched_lines = []
            for sid, src in fetched.items():
                fetched_lines.append(f"- ID: {sid} | Title: {src['title']} | URL: {src['url']} ({src['char_count']} chars)")
            fetched_text = "\n".join(fetched_lines)
        else:
            fetched_text = "No sources successfully fetched yet."

        failed = state.get("failed_urls", [])
        if failed:
            failed_lines = [f"- URL: {f.get('url')} | Reason: {f.get('error')}" for f in failed]
            failed_text = "\n".join(failed_lines)
        else:
            failed_text = "None"

        summaries = state.get("summaries", [])
        if summaries:
            sum_lines = []
            for s in summaries:
                sum_text = s['summary'][:250] + ("..." if len(s['summary']) > 250 else "")
                sum_lines.append(f"- Source {s['source_id']}:\n  Summary: {sum_text}\n  Key Points: {', '.join(s.get('key_points', [])[:3])}")
            summaries_text = "\n".join(sum_lines)
        else:
            summaries_text = "No summaries extracted yet."

        last_error = state.get("last_error")
        last_action_status = f"Previous error: {last_error}" if last_error else "Normal"

        user_content = REASONING_USER_TEMPLATE.format(
            question=state.get("question", ""),
            current_step=step,
            max_steps=max_steps,
            last_action_status=last_action_status,
            num_search_results=len(search_results),
            search_results_text=search_text,
            num_fetched=len(fetched),
            fetched_sources_text=fetched_text,
            num_failed=len(failed),
            failed_urls_text=failed_text,
            num_summaries=len(summaries),
            summaries_text=summaries_text,
        )

        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if show_display:
            print_step_header(step + 1, max_steps)

        if on_event:
            on_event({"type": "step", "step": step + 1, "max_steps": max_steps})

        # Query LLM with automatic 429 rate-limit backoff
        import time
        data = None
        for attempt in range(3):
            try:
                resp = llm.invoke(messages)
                raw = resp.content if hasattr(resp, "content") else str(resp)
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not json_match:
                    raise ValueError(f"No JSON found in LLM output: {raw[:200]}")
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    data = parsed
                    break
                raise ValueError(f"LLM returned non-dict JSON: {type(parsed)}")
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "rate_limit" in err_msg.lower()) and attempt < 2:
                    time.sleep(2.0)
                    continue
                logger.error(f"Reasoning error (attempt {attempt + 1}): {e}")
                if on_event:
                    on_event({
                        "type": "llm_error",
                        "attempt": attempt + 1,
                        "error": err_msg[:300],
                    })
                break  # don't retry on non-rate-limit errors

        # Fallback if data is still None after all attempts (LLM completely failed / returned null)
        if not data or not isinstance(data, dict):
            logger.warning("LLM returned no usable data after all retries — applying safe fallback.")
            if not search_results:
                data = {
                    "thought": "LLM unavailable, initiating fallback search.",
                    "action": "SEARCH",
                    "search_query": state.get("question", ""),
                    "fetch_url": None,
                    "summarise_source_id": None,
                    "final_answer": None,
                }
            elif fetched and len(summaries) < len(fetched):
                unsum = [sid for sid in fetched.keys() if not any(s["source_id"] == sid for s in summaries)]
                target_id = unsum[0] if unsum else list(fetched.keys())[0]
                data = {
                    "thought": "LLM unavailable, summarising fetched source.",
                    "action": "SUMMARISE",
                    "search_query": None,
                    "fetch_url": None,
                    "summarise_source_id": target_id,
                    "final_answer": None,
                }
            else:
                data = {
                    "thought": "LLM unavailable, finishing with collected evidence.",
                    "action": "FINISH",
                    "search_query": None,
                    "fetch_url": None,
                    "summarise_source_id": None,
                    "final_answer": None,
                }

        thought = data.get("thought", "Autonomous decision made.")
        action = data.get("action", "SEARCH").upper()
        params = {
            "search_query": data.get("search_query"),
            "fetch_url": data.get("fetch_url"),
            "summarise_source_id": data.get("summarise_source_id"),
        }

        if show_display:
            print_decision(thought, action, params)

        if on_event:
            on_event({
                "type": "decision",
                "thought": thought,
                "action": action,
                "params": params,
            })

        return {
            "next_action": data,
            "last_error": None,  # Reset last error after reasoning has evaluated it
        }

    def search_node(state: ResearchState) -> Dict[str, Any]:
        """Execute web search and record results."""
        action_data = state.get("next_action") or {}
        query = action_data.get("search_query") or state.get("question", "")
        step = state.get("current_step", 0) + 1

        try:
            results = search_web(
                query=query,
                max_results=cfg.max_search_results,
                provider=cfg.search_provider,
                tavily_key=cfg.tavily_api_key,
            )
        except Exception as e:
            results = []
            logger.error(f"Search tool raised exception: {e}")

        status = "success" if results else "empty"
        details = f"Retrieved {len(results)} search results for '{query}'" if results else f"No results found for '{query}'"

        if show_display:
            print_tool_status("SEARCH", status, details)

        if on_event:
            on_event({
                "type": "tool_result",
                "action": "SEARCH",
                "status": status,
                "summary": details,
            })

        record: ActionRecord = {
            "step": step,
            "action": "SEARCH",
            "params": {"query": query},
            "status": status,
            "details": details,
        }

        return {
            "search_results": results,
            "current_step": step,
            "action_history": state.get("action_history", []) + [record],
            "last_error": None if results else f"Search for '{query}' returned 0 results.",
        }

    def fetch_node(state: ResearchState) -> Dict[str, Any]:
        """Fetch and clean webpage content, assigning unique source ID."""
        action_data = state.get("next_action") or {}
        url = action_data.get("fetch_url")
        step = state.get("current_step", 0) + 1
        fetched = dict(state.get("fetched_sources", {}))
        failed = list(state.get("failed_urls", []))

        if not url:
            # Pick first unfetched URL from search results
            existing_urls = {s["url"] for s in fetched.values()} | {f["url"] for f in failed}
            candidate = None
            for r in state.get("search_results", []):
                if r["url"] not in existing_urls:
                    candidate = r["url"]
                    break
            url = candidate or (state.get("search_results", [{}])[0].get("url") if state.get("search_results") else "")

        source, error = fetch_webpage(
            url=url,
            fetched_sources=fetched,
            timeout=cfg.fetch_timeout,
            max_chars=cfg.max_content_chars,
        )

        if source:
            source_id = source["id"]
            fetched[source_id] = source
            status = "success"
            details = f"Stored as [{source_id}] '{source['title']}' ({source['char_count']} chars)"
            last_error = None
        else:
            failed.append({"url": url, "error": error or "Unknown error"})
            status = "failed"
            details = f"Failed to fetch {url}: {error}"
            last_error = f"Fetch failed for {url}: {error}"

        if show_display:
            print_tool_status("FETCH", status, details)

        if on_event:
            on_event({
                "type": "tool_result",
                "action": "FETCH",
                "status": status,
                "summary": details,
            })

        record: ActionRecord = {
            "step": step,
            "action": "FETCH",
            "params": {"url": url},
            "status": status,
            "details": details,
        }

        return {
            "fetched_sources": fetched,
            "failed_urls": failed,
            "current_step": step,
            "action_history": state.get("action_history", []) + [record],
            "last_error": last_error,
        }

    def summarise_node(state: ResearchState) -> Dict[str, Any]:
        """Extract question-relevant facts from a fetched source."""
        action_data = state.get("next_action") or {}
        source_id = action_data.get("summarise_source_id")
        step = state.get("current_step", 0) + 1
        fetched = state.get("fetched_sources", {})
        summaries = list(state.get("summaries", []))

        if not source_id or source_id not in fetched:
            # Pick an un-summarized source ID if available
            summarized_ids = {s["source_id"] for s in summaries}
            unsummarized = [sid for sid in fetched.keys() if sid not in summarized_ids]
            source_id = unsummarized[0] if unsummarized else (list(fetched.keys())[0] if fetched else None)

        if not source_id:
            status = "failed"
            details = "No fetched sources available to summarise."
            last_error = "Cannot summarise: No sources have been fetched yet."
        else:
            summary_obj, error = summarise_source(
                source_id=source_id,
                question=state.get("question", ""),
                fetched_sources=fetched,
                llm=llm,
            )
            if summary_obj:
                # Replace existing summary if updated or append
                summaries = [s for s in summaries if s["source_id"] != source_id]
                summaries.append(summary_obj)
                status = "success"
                details = f"Extracted {len(summary_obj.get('key_points', []))} key points from [{source_id}]"
                last_error = None
            else:
                status = "failed"
                details = f"Summarisation failed for [{source_id}]: {error}"
                last_error = details

        if show_display:
            print_tool_status("SUMMARISE", status, details)

        if on_event:
            on_event({
                "type": "tool_result",
                "action": "SUMMARISE",
                "status": status,
                "summary": details,
            })

        record: ActionRecord = {
            "step": step,
            "action": "SUMMARISE",
            "params": {"source_id": source_id},
            "status": status,
            "details": details,
        }

        return {
            "summaries": summaries,
            "current_step": step,
            "action_history": state.get("action_history", []) + [record],
            "last_error": last_error,
        }

    def finish_node(state: ResearchState) -> Dict[str, Any]:
        """Generate or validate grounded final answer with strict citations."""
        action_data = state.get("next_action") or {}
        candidate_answer = action_data.get("final_answer")
        fetched = state.get("fetched_sources", {})
        summaries = state.get("summaries", [])
        question = state.get("question", "")

        # If answer is missing or lacks citations, synthesize using SYNTHESIS prompt
        citations = extract_citations(candidate_answer) if candidate_answer else []
        valid_citations = [c for c in citations if c in fetched]

        if not candidate_answer or len(valid_citations) == 0:
            if fetched:
                # Build context for synthesis
                source_docs = []
                for sid, src in fetched.items():
                    src_sum = next((s for s in summaries if s["source_id"] == sid), None)
                    summary_text = src_sum["summary"] if src_sum else src["content"][:2000]
                    points = "\n  - " + "\n  - ".join(src_sum.get("key_points", [])) if src_sum else ""
                    source_docs.append(f"[{sid}] Title: {src['title']}\nURL: {src['url']}\nSummary:\n{summary_text}{points}")

                verified_text = "\n\n".join(source_docs)
                prompt = [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": SYNTHESIS_USER_TEMPLATE.format(
                            question=question,
                            verified_sources_text=verified_text,
                        ),
                    },
                ]
                if on_event:
                    on_event({"type": "start_stream", "title": "Synthesizing Final Grounded Report"})

                final_answer = invoke_or_stream(
                    llm=llm,
                    messages=prompt,
                    stream=cfg.streaming and show_display,
                    stream_title="Streaming Final Grounded Synthesis",
                    on_token=lambda tok: on_event({"type": "token", "token": tok}) if on_event else None,
                )
            else:
                final_answer = (
                    "## Research Incomplete\n\n"
                    "No web sources could be successfully fetched or verified to answer the research question. "
                    "Please verify network access, target URLs, or search terms."
                )
        else:
            final_answer = candidate_answer

        final_citations = extract_citations(final_answer)

        if show_display:
            print_tool_status(
                "FINISH",
                "success",
                f"Generated grounded final answer with {len(final_citations)} citations: {final_citations}",
            )

        if on_event:
            on_event({
                "type": "complete",
                "final_answer": final_answer,
                "citations": final_citations,
                "sources": fetched,
                "steps": state.get("current_step", 0),
                "max_steps": cfg.max_steps,
                "status": "completed",
            })

        return {
            "final_answer": final_answer,
            "citations": final_citations,
            "status": "completed",
        }

    def force_synthesize_node(state: ResearchState) -> Dict[str, Any]:
        """Force synthesis when step budget is exhausted."""
        fetched = state.get("fetched_sources", {})
        summaries = state.get("summaries", [])
        question = state.get("question", "")

        if fetched:
            source_docs = []
            for sid, src in fetched.items():
                src_sum = next((s for s in summaries if s["source_id"] == sid), None)
                summary_text = src_sum["summary"] if src_sum else src["content"][:1500]
                points = "\n  - " + "\n  - ".join(src_sum.get("key_points", [])) if src_sum else ""
                source_docs.append(f"[{sid}] Title: {src['title']}\nURL: {src['url']}\nSummary:\n{summary_text}{points}")

            verified_text = "\n\n".join(source_docs)
            prompt = [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SYNTHESIS_USER_TEMPLATE.format(
                        question=question,
                        verified_sources_text=verified_text,
                    ),
                },
            ]
            if on_event:
                on_event({"type": "start_stream", "title": "Streaming Forced Synthesis Report"})

            answer_text = invoke_or_stream(
                llm=llm,
                messages=prompt,
                stream=cfg.streaming and show_display,
                stream_title="Streaming Forced Synthesis Report",
                on_token=lambda tok: on_event({"type": "token", "token": tok}) if on_event else None,
            )
            citations = extract_citations(answer_text)
            final_answer = (
                f"> [!NOTE]\n"
                f"> **Research terminated**: Maximum step limit ({cfg.max_steps} steps) reached. "
                f"Below is the synthesis generated from {len(fetched)} successfully retrieved source(s).\n\n"
                + answer_text
            )
        else:
            citations = []
            final_answer = (
                f"> [!WARNING]\n"
                f"> **Maximum step limit reached ({cfg.max_steps} steps)** before any web sources could be successfully fetched.\n\n"
                "The research agent was unable to confirm facts against verified web content."
            )

        if on_event:
            on_event({
                "type": "complete",
                "final_answer": final_answer,
                "citations": citations,
                "sources": fetched,
                "steps": state.get("current_step", 0),
                "max_steps": cfg.max_steps,
                "status": "budget_exceeded",
            })

        return {
            "final_answer": final_answer,
            "citations": citations,
            "status": "budget_exceeded",
        }

    # Conditional Routing Logic

    def route_decision(state: ResearchState) -> str:
        """Route conditionally to the appropriate tool or finish node."""
        step = state.get("current_step", 0)
        max_steps = state.get("max_steps", cfg.max_steps)

        if step >= max_steps:
            return "force_synthesize"

        next_act = state.get("next_action") or {}
        action = str(next_act.get("action", "")).upper()

        if action == "SEARCH":
            return "search"
        elif action == "FETCH":
            return "fetch"
        elif action == "SUMMARISE":
            return "summarise"
        elif action == "FINISH":
            return "finish"
        else:
            return "force_synthesize"

    # Assemble StateGraph
    workflow = StateGraph(ResearchState)

    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("summarise", summarise_node)
    workflow.add_node("finish", finish_node)
    workflow.add_node("force_synthesize", force_synthesize_node)

    # Entry point
    workflow.set_entry_point("reasoning")

    # Conditional branching from reasoning
    workflow.add_conditional_edges(
        "reasoning",
        route_decision,
        {
            "search": "search",
            "fetch": "fetch",
            "summarise": "summarise",
            "finish": "finish",
            "force_synthesize": "force_synthesize",
        },
    )

    # Tools loop back to reasoning node for autonomous decision making
    workflow.add_edge("search", "reasoning")
    workflow.add_edge("fetch", "reasoning")
    workflow.add_edge("summarise", "reasoning")

    # Terminations
    workflow.add_edge("finish", END)
    workflow.add_edge("force_synthesize", END)

    return workflow.compile()


def create_initial_state(question: str, max_steps: int = 8) -> ResearchState:
    """Instantiate a clean initial state for research."""
    return {
        "question": question.strip(),
        "search_results": [],
        "fetched_sources": {},
        "failed_urls": [],
        "summaries": [],
        "current_step": 0,
        "max_steps": max_steps,
        "action_history": [],
        "next_action": None,
        "final_answer": None,
        "citations": [],
        "status": "in_progress",
        "last_error": None,
    }
