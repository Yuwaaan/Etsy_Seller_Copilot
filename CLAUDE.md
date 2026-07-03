# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependency management is via [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run the app: `uv run streamlit run src/etsy_seller_copilot/ui/app.py`
- Run all tests: `uv run pytest`
- Run a single test file: `uv run pytest tests/test_analytics_metrics.py`
- Run a single test: `uv run pytest tests/test_analytics_metrics.py::TestTotalRevenue::test_sums_price_times_quantity`

There is no configured lint/format command yet.

The app and full test suite work with **no API key configured** -- see
"Business Analyst Agent" below for why.

## Architecture

`src/etsy_seller_copilot/`:

- `analytics/metrics.py` — pure Pandas functions over an orders DataFrame
  (total revenue, top products, sales trends, period-over-period comparisons,
  etc). Every function assumes one row per order line item and raises
  `ValueError` (via `_require_columns`) if a required column is missing,
  rather than failing silently.
- `analytics/health.py` — deterministic 0-100 shop health scoring, built on
  top of `metrics.py`'s period comparisons. Never hardcoded and never
  LLM-generated: every point on the score is explainable via
  `HealthScore.components`.
- `data/loaders.py` — loads Etsy CSV exports (Orders/Listings/Traffic) from a
  file path into a DataFrame, with strict required-column validation
  (`MissingColumnsError`). Also has `load_sample_orders()`, a hardcoded
  in-code sample dataset (not read from `data/raw/`, which is gitignored).
- `data/utils.py` — flexible column mapping and cleaning for uploaded CSVs:
  maps header variants (`Qty` -> `Quantity`, `Title` -> `Item Name`, etc.)
  onto canonical names, coerces numeric/date columns, drops duplicates/bad
  rows, and raises `FriendlyColumnError` with a message safe to show
  directly in the UI.
- `config/llm.py` — `get_chat_model()`: provider-agnostic LLM loader (OpenAI
  or Anthropic, auto-detected from whichever `*_API_KEY` env var is set, or
  forced via `MODEL_PROVIDER`). Returns `None` — never raises — when nothing
  is configured, which is the signal every caller below uses to fall back to
  a deterministic path.
- `ui/app.py`, `ui/components.py`, `ui/charts.py` — the Streamlit dashboard.
  `components.py` has no analytics/cleaning logic of its own; it only calls
  into `analytics/`, `data/`, and `agents/health_report.py`.

### Business Analyst Agent (`agents/`)

The "Ask a question" chat box is a LangGraph graph (`agents/graph.py`) with
two paths:

1. **Primary — `business_analyst_node`**: delegates to
   `agents.business_analyst.run_business_analyst`, which runs a manual
   tool-calling loop (bind tools -> invoke -> execute any tool calls -> feed
   results back -> repeat, up to `MAX_TOOL_ITERATIONS`). The LLM decides
   *which* of the tools in `agents/analyst_tools.py` to call, with what
   arguments, and in what order — nothing here is keyword-routed. Each tool
   is a thin, JSON-safe wrapper around a real `analytics/metrics.py`
   function, so every number the LLM cites is traceable back to a
   deterministic computation, not invented.
2. **Fallback — `detect_intent` -> `plan` -> `dispatch` -> `respond`**: the
   original rule-based pipeline (`services/intent.py` -> `services/planner.py`
   -> `services/dispatcher.py`), unchanged. `agents/graph.py` routes here via
   a conditional edge whenever `business_analyst_node` doesn't produce a
   `response` — i.e. no LLM is configured, or the LLM/tool-calling loop fails
   for any reason at runtime (network error, bad key, ...). This is why the
   whole app — including live LLM behavior at a code level — is testable and
   runnable with zero API keys: `get_chat_model()` returning `None` is a
   first-class, always-tested code path, not just an error case.

`agents/health_report.py` follows the same pattern for the dashboard's "AI
Insights" panel: `analytics/health.py` always computes the score and signals
deterministically; if an LLM is available, it's asked (via
`with_structured_output`) to turn those grounded signals into an executive
summary and recommendations — it never invents the score itself. Without an
LLM, simple templates generate the same fields from the same signals.

`agents/tools.py` + `services/*` predate the Business Analyst Agent and still
power the deterministic fallback path only; they're intentionally left
as-is rather than merged into `agents/analyst_tools.py`, which is Business
Analyst-specific and calls `analytics/metrics.py` directly (no `Intent` enum
involved).

- `chains/`, `tools/` — placeholders, currently empty.

Tests in `tests/` mirror this structure 1:1 (e.g. `test_analytics_metrics.py`
tests `analytics/metrics.py`). LLM-dependent code (`business_analyst.py`,
`health_report.py`) is tested by injecting a hand-rolled fake chat model
(see `FakeChatModel` in `test_agents_business_analyst.py`) rather than
calling a real provider — no test requires an API key.
