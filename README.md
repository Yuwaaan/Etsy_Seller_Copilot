# EtsySellerCopilot

An AI Business Analytics Copilot for Etsy sellers: upload a **Sold Orders**
CSV export and get a full dashboard (revenue, top products, sales trends, an
AI-scored shop health panel) plus a chat assistant that reasons about your
business the way an analyst would — not a fixed list of commands.

## What it does

Upload your Etsy Sold Orders CSV (or click "Load sample Etsy order data" to
try it without one) and the app shows:

- **Key metrics**: total revenue, total orders, average order value, and
  repeat customer rate, each with a plain-language explanation.
- **AI Insights**: a 0-100 business health score (always computed from real
  metrics, never hardcoded) with a full breakdown of *why*, plus what's
  improving, what's declining, risks, and recommended actions.
- **Top Selling Products**: a table of products ranked by revenue, with units
  sold alongside.
- **Sales Trend**: a bar chart of revenue by month or by day.
- **Data Preview**: the first rows of your cleaned data, so you can sanity
  check what was uploaded.
- **Ask a question**: a chat box that understands natural business questions
  — "why are sales down?", "give me a health check", "which products deserve
  more promotion?" — not just fixed keywords.

## How the AI actually works

The chat box and AI Insights panel are powered by a **Business Analyst
Agent**: the LLM is given a set of analytics tools (total revenue, sales
trend, period-over-period comparison, per-product performance, etc.) and
decides for itself which ones to call, with what arguments, and in what
order, based on your question. It then writes the final answer — for
open-ended questions, structured as Executive Summary / Key Findings /
Possible Causes / Recommendations. No analytics function is ever
keyword-routed for this path; the LLM plans and reasons, the tools only
compute.

**This works with no API key configured.** Every number the LLM cites comes
from a real, independently-tested Pandas function (`analytics/metrics.py`,
`analytics/health.py`) — the LLM never invents figures. And if no LLM
provider is configured (or a live call fails for any reason — network, rate
limit, etc.), the app automatically falls back to a deterministic, rule-based
engine for both the chat and the AI Insights score, so the dashboard is never
broken by a missing or failing API key. Natural-language reasoning about
open-ended questions ("why are sales down?") does require a real key, though
— the deterministic fallback only understands a fixed set of phrasings.

To enable it, set **one** of these in `.env` (see `.env.example`):

```
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

The provider is auto-detected from whichever key is set (set `MODEL_PROVIDER`
to force one explicitly). Default models are small and cheap
(`gpt-4o-mini` / `claude-haiku-4-5`) — enough to reason over a handful of
pre-computed metrics without needing a large model.

## Tech Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- [LangGraph](https://www.langchain.com/langgraph) for orchestrating the agent workflow
- [LangChain](https://www.langchain.com/) (`langchain-openai` / `langchain-anthropic`) for provider-agnostic LLM tool-calling
- [Pandas](https://pandas.pydata.org/) for shop/listing data analysis
- [Streamlit](https://streamlit.io/) for the user interface
- [Plotly](https://plotly.com/python/) for interactive charts in the UI
- [python-dotenv](https://saurabh-kumar.com/python-dotenv/) for loading local environment variables
- [Pydantic](https://docs.pydantic.dev/) for typed config and structured LLM output

## Project Structure

```
EtsySellerCopilot/
├── src/etsy_seller_copilot/
│   ├── agents/       # Business Analyst Agent (LLM tool-calling loop) +
│   │                 # deterministic fallback pipeline, wired via LangGraph
│   ├── services/      # Deterministic intent detection / planning / dispatch
│   │                  # (used only by the fallback path)
│   ├── analytics/     # Pure Pandas analytics + deterministic health scoring
│   ├── data/          # CSV loading (loaders.py) and cleaning/column
│   │                  # mapping (utils.py)
│   ├── config/        # Provider-agnostic LLM loading (llm.py)
│   ├── ui/            # Streamlit application (app.py, components.py, charts.py)
│   ├── chains/         # (placeholder)
│   └── tools/          # (placeholder)
├── tests/           # Test suite -- LLM-dependent code is tested with a
│                    # hand-rolled fake chat model, never a real API call
└── data/            # Local data files (raw/processed), gitignored
```

## Getting Started

Install dependencies:

```
uv sync
```

Copy the environment template (optional — the dashboard, CSV insights, and
chat all work fully offline via the deterministic fallback; only the
LLM-powered reasoning needs a key):

```
cp .env.example .env
```

Run the Streamlit app:

```
uv run streamlit run src/etsy_seller_copilot/ui/app.py
```

Run tests:

```
uv run pytest
```

## Supported CSV format

The app is built for Etsy's **Sold Orders** CSV export (Shop Manager >
Settings > Options > Download Data), one row per order line item. At
minimum it needs a column for each of:

- **Item name** — e.g. `Item Name`, `Title`, `Item Title`, `Product Name`
- **Quantity** — e.g. `Quantity`, `Qty`, `Quantity Purchased`
- **Price** — e.g. `Price`, `Order Value`, `Sale Amount`, `Subtotal`

Column names are matched case- and spacing-insensitively, so `"QTY"`,
`" Qty "`, and `"Quantity"` are all treated the same. If none of the known
spellings for a required field are found, the app shows an error listing
the columns it did find and what's still needed, instead of crashing.

Optional columns improve the results but aren't required:

- **Order ID** (e.g. `Order Number`) — without it, each row is treated as
  its own order.
- **Sale Date** (e.g. `Order Date`, `Purchase Date`) — without it, the sales
  trend chart, AI Insights trends, and period-over-period tools are skipped.
- **Currency**, **Buyer** (e.g. `Buyer Name`, `Customer`) — used for display
  and for the repeat customer rate metric.

The app also handles messy real-world exports without failing:

- Duplicate rows are removed.
- Non-numeric quantity/price values (and rows with them) are dropped.
- Currency symbols and commas in price (e.g. `"$18.50"`) are stripped before
  parsing.
- Unparseable dates are kept in totals but excluded from trend charts.

Any rows dropped or values that couldn't be parsed are reported back as
warnings in the UI rather than failing silently. The same applies to
questions the data structurally can't answer — e.g. conversion rate needs
shop visit/traffic data that a Sold Orders export doesn't contain, so the
assistant explains that gap and suggests exporting Etsy's Shop Stats instead,
rather than failing or guessing.

## Future improvements

- Support Etsy's Listings and Traffic/Stats exports in the same dashboard —
  once visit data is available, conversion rate becomes a real, computable
  metric instead of an explained gap.
- Add streaming responses for the chat box, so multi-tool-call answers show
  progress instead of appearing all at once.
- Persist uploaded data (and chat history) across sessions so sellers don't
  need to re-upload each time.
- Add response caching / a cheaper model tier for simple factual questions,
  reserving the full tool-calling loop for genuinely open-ended ones.
