"""Tests for hard step budget enforcement and loop prevention."""

import json
from src.agent import build_research_agent, MockLLM, MockResponse, create_initial_state
from src.config import Settings


class InfiniteLoopLLM:
    """Mock LLM that endlessly requests SEARCH actions to test budget guardrails."""

    def __init__(self):
        self.invocations = 0

    def invoke(self, messages):
        self.invocations += 1
        prompt = " ".join([m.get("content", "") for m in messages])
        if "senior scientific research synthesizer" in prompt.lower():
            return MockResponse(
                "## Forced Synthesis\nBudget exhausted. Partial findings recorded with available data."
            )
        # Always return SEARCH action
        return MockResponse(
            json.dumps({
                "thought": "I need more information, so I will search again endlessly.",
                "action": "SEARCH",
                "search_query": f"endless query iteration {self.invocations}",
                "fetch_url": None,
                "summarise_source_id": None,
                "final_answer": None,
            })
        )


def test_step_budget_prevents_infinite_loop():
    """Verify that agent terminates when reaching max_steps even if LLM wants to continue."""
    max_steps = 4
    settings = Settings(max_steps=max_steps, llm_provider="mock")
    looping_llm = InfiniteLoopLLM()

    agent = build_research_agent(llm=looping_llm, settings=settings, show_display=False)
    initial_state = create_initial_state(question="Test step limit query", max_steps=max_steps)

    final_state = agent.invoke(initial_state)

    # Assertions
    assert final_state["current_step"] == max_steps
    assert final_state["status"] == "budget_exceeded"
    assert final_state["final_answer"] is not None
    assert "step limit" in final_state["final_answer"].lower() or "budget" in final_state["final_answer"].lower()
    assert len(final_state["action_history"]) == max_steps


def test_default_8_step_budget_respected():
    """Verify default budget of 8 steps terminates safely."""
    max_steps = 8
    settings = Settings(max_steps=max_steps, llm_provider="mock")
    looping_llm = InfiniteLoopLLM()

    agent = build_research_agent(llm=looping_llm, settings=settings, show_display=False)
    initial_state = create_initial_state(question="Test 8 steps limit", max_steps=max_steps)

    final_state = agent.invoke(initial_state)

    assert final_state["current_step"] == 8
    assert final_state["status"] == "budget_exceeded"
