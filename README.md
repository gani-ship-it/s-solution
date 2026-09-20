# LangGraph Autonomous AI Research Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/pytest-passing-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An autonomous, state-driven AI Research Agent built in Python using **LangGraph**. The agent accepts research questions through a dynamic CLI interface, reasons autonomously to decide which tool to execute next (search, fetch, summarise, or finish), maintains grounded citation tracking with unique source IDs (`[S1]`, `[S2]`), and enforces strict step budget safety guardrails to prevent infinite loops.

---

## Key Capabilities

- **Autonomous Tool Selection**: Unlike fixed-chain DAG pipelines, the agent uses an LLM reasoning node to dynamically inspect research state (findings, gaps, failed URLs) and choose the optimal next action.
- **Three Core Research Tools**:
  1. **Web Search Tool**: Queries DuckDuckGo / Tavily for high-relevance search results containing titles, URLs, and snippets.
  2. **Webpage & PDF Fetching Tool**: Downloads live web pages **and academic PDF whitepapers (including direct arXiv.org PDF links)**, extracts clean text using BeautifulSoup and `pypdf`, and registers each source with a sequential ID (`S1`, `S2`, `S3`...).
  3. **Summarisation Tool**: Focuses strictly on extracting facts, metrics, and conclusions relevant to the user's research question from a fetched source.
- **Strict Grounding & Inline Citations**: Factual statements in the final report must cite successfully fetched sources (e.g., `[S1]`, `[S2]`). Hallucinated or unfetched sources are rejected.
- **Hard Step Budget Guardrail**: Configurable step limit (default: 8 steps) guarantees that the agent can never loop indefinitely. If the budget is exhausted, a forced synthesis node compiles all verified findings.
- **Graceful Fault Tolerance**: Empty search queries, HTTP 403/404 errors, network timeouts, and tool exceptions are recorded in state without crashing, allowing the agent to refine queries or try alternative sources.
- **Rich Interactive CLI with Real-Time Streaming**: Formatted terminal output displaying real-time reasoning steps, tool statuses, **live token-by-token streaming output** as the LLM synthesizes reports, final Markdown reports, reference tables, and run metrics.
- **Zero-Key Offline Mock Mode**: Includes deterministic mock providers for testing, development, and offline demonstrations without requiring paid API keys.

---

## System Architecture

The research process is orchestrated as a state machine via LangGraph:

```
                      +-----------------------------+
                      | User Input (CLI / Question) |
                      +-----------------------------+
                                     |
                                     v
                       +---------------------------+
                       | Initialize ResearchState  |
                       +---------------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v (step < max_steps)                                v (step >= max_steps)
+-----------------------+                            +-------------------------+
| Agent Reasoning Node  |                            |  Force Synthesis Node   |
| (LLM Action Decision) |                            | (Step Budget Guardrail) |
+-----------------------+                            +-------------------------+
           |                                                   |
           +---------------+---------------+---------------+   |
           |               |               |               |   |
           v               v               v               v   v
     +-----------+   +-----------+   +-----------+   +-------------+
     |   SEARCH  |   |   FETCH   |   | SUMMARISE |   |   FINISH    |
     | Web Tool  |   | Page Tool |   | Info Tool |   | Grounded Rpt|
     +-----------+   +-----------+   +-----------+   +-------------+
           |               |               |               |
           +---------------+---------------+               v
                           |                         +------------+
                           +------------------------>|    END     |
                      (Loop Back)                    +------------+
```

### State Schema (`ResearchState`)

| State Field | Type | Description |
| :--- | :--- | :--- |
| `question` | `str` | Original research question provided by the user. |
| `search_results` | `List[SearchResult]` | List of retrieved search results (`title`, `url`, `snippet`). |
| `fetched_sources` | `Dict[str, FetchedSource]` | Successfully fetched and cleaned pages mapped by ID (`S1`, `S2`...). |
| `failed_urls` | `List[Dict[str, str]]` | URLs that failed to download with specific error reasons. |
| `summaries` | `List[SourceSummary]` | Extracted question-relevant notes per source ID. |
| `current_step` | `int` | Counter tracking the number of tool actions executed. |
| `max_steps` | `int` | Hard limit for loop prevention (default: 8). |
| `action_history` | `List[ActionRecord]` | Complete chronological audit trail of actions taken. |
| `final_answer` | `Optional[str]` | Grounded Markdown response containing `[S1]`, `[S2]` tags. |
| `citations` | `List[str]` | List of unique source IDs verified in the final response. |
| `status` | `str` | Current workflow state (`in_progress`, `completed`, `budget_exceeded`). |

---

## Project Structure

```
s-sol/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules (protects API keys and caches)
├── README.md                 # Complete documentation and examples
├── requirements.txt          # Python dependencies
├── main.py                   # Top-level CLI entry point
├── src/
│   ├── __init__.py           # Package marker
│   ├── agent.py              # LangGraph graph definition, nodes, routing, and MockLLM
│   ├── config.py             # Settings management and environment loading
│   ├── state.py              # TypedDict state and Pydantic decision models
│   ├── prompts.py            # System prompts for reasoning, summarising, and synthesis
│   ├── cli.py                # Rich CLI interface and argument parser
│   ├── tools/
│   │   ├── __init__.py       # Tools export
│   │   ├── web_search.py     # DuckDuckGo / Tavily / Mock search tool
│   │   ├── fetcher.py        # Webpage fetching, HTML cleaner, and S1/S2 assigner
│   │   └── summarizer.py     # Targeted information extraction tool
│   └── utils/
│       ├── __init__.py       # Utilities export
│       └── display.py        # Rich console panels, tables, and banners
└── tests/
    ├── __init__.py           # Test package marker
    ├── conftest.py           # Pytest fixtures and mock responses
    ├── test_step_limit.py    # Hard step budget safety tests
    ├── test_tool_failures.py # Graceful 404, timeout, and tool exception tests
    ├── test_empty_search.py  # Empty search result recovery tests
    └── test_citations.py     # Sequential S1/S2 tracking and citation validation tests
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher.
- `pip` package manager.

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/your-username/langgraph-research-agent.git
cd langgraph-research-agent

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Variable Setup
Copy the example environment configuration:
```bash
cp .env.example .env
```
Edit `.env` with your preferred settings:
```ini
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.2

# Budget & Guardrails
MAX_STEPS=8

# Web Search Provider (duckduckgo or tavily)
SEARCH_PROVIDER=duckduckgo
TAVILY_API_KEY=

# Page Fetcher Settings
FETCH_TIMEOUT=10
MAX_CONTENT_CHARS=6000
```

> **Security Guarantee**: The project `.gitignore` excludes `.env` and all credential files. Never commit API keys to version control.

---

## CLI Usage

### Interactive Mode
Run without arguments to launch the interactive prompt:
```bash
python main.py
```

### Direct Query Execution
Pass the research question directly as a command-line argument:
```bash
python main.py "What are the latest breakthroughs in room-temperature superconductors?"
```

### Command-Line Flags
- `--mock`: Run in offline simulation mode without an external LLM or API keys.
- `--max-steps <int>`: Override the step budget (e.g., `--max-steps 5`).
- `--model <name>`: Specify an alternate model (e.g., `--model openai/gpt-oss-20b`).
- `--no-stream`: Disable real-time token streaming and wait for batch output.

#### Example:
```bash
python main.py --mock --max-steps 6 "What are the recent advances in solid-state batteries?"
```

---

## Realistic Example Runs

Below are three representative execution traces illustrating autonomous decision-making, error recovery, and grounded citation generation.

### Example Run 1: Fault-Tolerant Quantum Computing Architectures

**User Input:**
```bash
python main.py "What are the recent breakthroughs in fault-tolerant quantum computing architectures?"
```

**Execution Trace:**
```
+-----------------------------------------------------------------------------+
|  LangGraph Autonomous AI Research Agent                                     |
|  Autonomous tool decision-making | Source citation tracking | Budget safety |
+-----------------------------------------------------------------------------+
Researching: "What are the recent breakthroughs in fault-tolerant quantum computing architectures?"
Step Budget: 8 steps | Provider: openai

=== STEP 1/8 ===
[Agent Reasoning & Decision]
Thought: No searches performed yet. I must search for recent peer-reviewed breakthroughs on logical qubits and fault-tolerant quantum architectures.
Selected Tool: SEARCH
- search_query: recent fault tolerant quantum computing logical qubits breakthroughs 2026
[SUCCESS] SEARCH: Retrieved 5 search results for query.

=== STEP 2/8 ===
[Agent Reasoning & Decision]
Thought: Search results contain a primary paper from Nature Physics on neutral-atom logical qubits. Fetching full webpage content.
Selected Tool: FETCH
- fetch_url: https://nature.com/articles/s41567-quantum-neutral-atoms
[SUCCESS] FETCH: Stored as [S1] 'High-Fidelity Neutral Atom Logical Qubits' (5,420 chars)

=== STEP 3/8 ===
[Agent Reasoning & Decision]
Thought: Source S1 is fetched. I will now extract and summarise facts regarding fidelity rates and gate operations.
Selected Tool: SUMMARISE
- summarise_source_id: S1
[SUCCESS] SUMMARISE: Extracted 4 key points from [S1]

=== STEP 4/8 ===
[Agent Reasoning & Decision]
Thought: One source is summarized, but verifying across an alternative architecture (superconducting surface codes) will make the report comprehensive.
Selected Tool: FETCH
- fetch_url: https://arxiv.org/abs/2602.09112-surface-code-thresholds
[SUCCESS] FETCH: Stored as [S2] 'Surface Code Error Correction Below Fault-Tolerant Threshold' (4,890 chars)

=== STEP 5/8 ===
[Agent Reasoning & Decision]
Thought: Source S2 is fetched. Summarising error correction thresholds and physical-to-logical qubit overhead.
Selected Tool: SUMMARISE
- summarise_source_id: S2
[SUCCESS] SUMMARISE: Extracted 3 key points from [S2]

=== STEP 6/8 ===
[Agent Reasoning & Decision]
Thought: I have two high-quality, verified sources summarized. Ready to formulate the final grounded report with citations.
Selected Tool: FINISH
[SUCCESS] FINISH: Generated grounded final answer with 2 citations: ['S1', 'S2']

======================================================================
FINAL RESEARCH REPORT (GROUNDED & CITED)
======================================================================
## Executive Summary
Recent experimental implementations have demonstrated that physical error rates can be suppressed below fault-tolerant thresholds using reconfigurable 2D qubit arrays [S1]. In neutral-atom architectures, two-qubit gate fidelities reached 99.8% across an array of 48 logical qubits, enabling transversal Clifford operations without leakage errors [S1].

## Architectural Implementations
1. **Neutral-Atom Platforms**: Dynamically shuttled optical tweezers achieved transversal entangling gates between distinct logical zones with sub-microsecond latency [S1].
2. **Superconducting Surface Codes**: Sycamore-class superconducting lattices demonstrated continuous syndrome extraction over 100 cycles, verifying that increasing code distance from d=3 to d=5 reduced physical error accumulation exponentially [S2].

## Scalability and Overhead
Physical-to-logical qubit overhead remains the primary engineering hurdle, requiring approximately 1,000 physical transmons per fault-tolerant logical unit under current surface code topologies [S2].

======================================================================
Verified Source References
+----+--------+------------------------------------+---------------------------------------+
| ID | Cited? | Title                              | URL                                   |
+----+--------+------------------------------------+---------------------------------------+
| S1 | YES *  | High-Fidelity Neutral Atom Qubits  | https://nature.com/articles/s41567... |
| S2 | YES *  | Surface Code Error Correction      | https://arxiv.org/abs/2602.09112...   |
+----+--------+------------------------------------+---------------------------------------+

Research Run Statistics
- Steps Consumed: 5 / 8
- Sources Fetched: 2 web sources (S1, S2)
- Summaries Created: 2 source extractions
- Final Status: COMPLETED
```

---

### Example Run 2: Solid-State Battery Milestones (Handling Empty Search & 403 Recovery)

**User Input:**
```bash
python main.py "What are the commercialization milestones for sulfide-based solid state batteries?"
```

**Execution Trace:**
```
=== STEP 1/8 ===
[Agent Reasoning & Decision]
Thought: Initiating search with specific technical phrase.
Selected Tool: SEARCH
- search_query: "sulfide-solid-electrolyte-roll-to-roll-commercial-benchmark-2026"
[EMPTY] SEARCH: No results found for '"sulfide-solid-electrolyte-roll-to-roll-commercial-benchmark-2026"'

=== STEP 2/8 ===
[Agent Reasoning & Decision]
Thought: Previous search returned 0 results due to overly narrow quotation constraints. Broadening search terms to capture automotive joint ventures and pilot lines.
Selected Tool: SEARCH
- search_query: sulfide solid state battery pilot line automotive commercialization 2026
[SUCCESS] SEARCH: Retrieved 5 search results for query.

=== STEP 3/8 ===
[Agent Reasoning & Decision]
Thought: Fetching candidate URL 1 from energy portal.
Selected Tool: FETCH
- fetch_url: https://paywalled-automotive-news.com/article/solid-battery-pilot
[FAILED] FETCH: Failed to fetch: HTTP 403: Unable to access https://paywalled-automotive-news.com/article/solid-battery-pilot

=== STEP 4/8 ===
[Agent Reasoning & Decision]
Thought: Candidate URL returned HTTP 403 Forbidden. I will not retry it and instead select an alternative open-access industry report from the search results.
Selected Tool: FETCH
- fetch_url: https://energy-storage-review.org/sulfide-electrolyte-scaling
[SUCCESS] FETCH: Stored as [S1] 'Sulfide Electrolyte Industrial Scaling and Pouch Cell Trials' (4,110 chars)

=== STEP 5/8 ===
[Agent Reasoning & Decision]
Thought: Source S1 fetched successfully. Extracting facts on capacity retention and cycle life.
Selected Tool: SUMMARISE
- summarise_source_id: S1
[SUCCESS] SUMMARISE: Extracted 3 key points from [S1]

=== STEP 6/8 ===
[Agent Reasoning & Decision]
Thought: Verified data from [S1] is sufficient to answer the commercialization question. Finalizing report.
Selected Tool: FINISH
[SUCCESS] FINISH: Generated grounded final answer with 1 citations: ['S1']

======================================================================
FINAL RESEARCH REPORT (GROUNDED & CITED)
======================================================================
## Commercialization Status
Sulfide-based solid-state batteries (using argyrodite-type Li6PS5Cl electrolytes) have transitioned from lab coin cells to 20Ah pilot-line pouch manufacturing [S1]. Pilot testing demonstrates an energy density of 420 Wh/kg and 82% capacity retention after 800 cycles under 1.5 MPa stack pressure [S1]. Dry-electrode coating technology reduced solvent recovery capital expenditure by 30% [S1].

Verified Source References
+----+--------+------------------------------------+---------------------------------------+
| ID | Cited? | Title                              | URL                                   |
+----+--------+------------------------------------+---------------------------------------+
| S1 | YES *  | Sulfide Electrolyte Scaling        | https://energy-storage-review.org/... |
+----+--------+------------------------------------+---------------------------------------+

Research Run Statistics
- Steps Consumed: 5 / 8
- Sources Fetched: 1 web source (S1)
- Failed URLs Handled: 1 (HTTP 403)
- Final Status: COMPLETED
```

---

### Example Run 3: In Vivo CRISPR Base Editing (Step Budget Guardrail Enforcement)

**User Input:**
```bash
python main.py --max-steps 6 "Analyze the clinical safety and off-target risks of in vivo CRISPR base editing."
```

**Execution Trace:**
```
=== STEP 1/6 ===
[Agent Reasoning & Decision]
Selected Tool: SEARCH
- search_query: in vivo CRISPR adenine cytosine base editing clinical safety off-target risks
[SUCCESS] SEARCH: Retrieved 5 search results.

=== STEP 2/6 ===
[Agent Reasoning & Decision]
Selected Tool: FETCH
- fetch_url: https://clinical-genomics.org/base-editing-phase1-safety
[SUCCESS] FETCH: Stored as [S1] 'Phase 1 Clinical Safety Profile of In Vivo Lipid Nanoparticle Base Editing' (5,100 chars)

=== STEP 3/6 ===
[Agent Reasoning & Decision]
Selected Tool: SUMMARISE
- summarise_source_id: S1
[SUCCESS] SUMMARISE: Extracted 3 key points from [S1]

=== STEP 4/6 ===
[Agent Reasoning & Decision]
Selected Tool: FETCH
- fetch_url: https://genetherapy-bulletin.org/off-target-guide-evaluations
[SUCCESS] FETCH: Stored as [S2] 'Comprehensive Deep-Sequencing Analysis of Cas9 Off-Target Editing' (4,750 chars)

=== STEP 5/6 ===
[Agent Reasoning & Decision]
Selected Tool: SUMMARISE
- summarise_source_id: S2
[SUCCESS] SUMMARISE: Extracted 4 key points from [S2]

=== STEP 6/6 ===
[Agent Reasoning & Decision]
Selected Tool: SEARCH
- search_query: long term immunology adeno associated virus base editors
[SUCCESS] SEARCH: Retrieved 4 search results.

=== STEP 6/6 ===
[WARNING] BUDGET: Hard step limit (6) reached. Forcing final synthesis.

======================================================================
FINAL RESEARCH REPORT (GROUNDED & CITED)
======================================================================
> [!NOTE]
> **Research terminated**: Maximum step limit (6 steps) reached. Below is the synthesis generated from 2 successfully retrieved source(s).

## Synthesis: In Vivo Base Editing Safety
Lipid nanoparticle (LNP)-delivered adenine base editors demonstrated transient expression in human hepatocytes, with serum PCSK9 reductions sustained over 180 days with no dose-limiting hepatotoxicity [S1].

## Off-Target Profile
High-throughput circularization and sequencing (CIRCLE-seq) detected no detectable guide-dependent off-target mutations at frequencies above 0.05% in primary human hepatocytes [S2]. However, transcriptome-wide RNA deamination was identified as a guide-independent bystander effect, necessitating engineered deaminase domains with restricted RNA-binding affinity [S2].

Verified Source References
+----+--------+------------------------------------+---------------------------------------+
| ID | Cited? | Title                              | URL                                   |
+----+--------+------------------------------------+---------------------------------------+
| S1 | YES *  | Phase 1 Clinical Safety Profile    | https://clinical-genomics.org/...     |
| S2 | YES *  | Deep-Sequencing Off-Target Analysi | https://genetherapy-bulletin.org/...  |
+----+--------+------------------------------------+---------------------------------------+

Research Run Statistics
- Steps Consumed: 6 / 6
- Sources Fetched: 2 web sources (S1, S2)
- Summaries Created: 2 source extractions
- Final Status: BUDGET_EXCEEDED
```

---

## Testing

The project includes an automated test suite covering all functional requirements, including step budgeting, tool failures, empty search results, and citation tracking.

To execute the test suite:
```bash
python -m pytest -v
```

### Test Coverage Highlights
- `tests/test_step_limit.py`: Validates that agents trapped in looping behaviors terminate forcefully when reaching `max_steps`, updating the status to `budget_exceeded` without crashing.
- `tests/test_tool_failures.py`: Simulates HTTP 404, connection timeouts, non-HTML payloads, and nonexistent source ID lookups to ensure graceful failure recovery.
- `tests/test_empty_search.py`: Verifies that 0 search results are logged as `empty`, allowing the agent's reasoning loop to adapt and re-query.
- `tests/test_citations.py`: Verifies sequential source ID allocation (`S1`, `S2`, `S3`...), prevents duplicate source IDs for identical URLs, extracts citation tokens, and validates that every citation corresponds to a verified fetched source.

---

## Contributing & License

Contributions are welcome! Please ensure that any additions pass `python -m pytest -v` before submitting pull requests.

Distributed under the MIT License.
