"""LangChain tool definitions for the Business Analyst Agent.

Tools are organized in two tiers, deliberately not one flat list of metric
wrappers:

- **Direct lookups** (``get_total_revenue`` and friends) answer a single,
  literal factual question with one number.
- **Business investigation tools** (``investigate_sales_decline``,
  ``business_health_check``, ...) represent the business tasks a real
  analyst would perform -- each one internally calls several
  ``analytics/metrics.py`` (and ``analytics/health.py``) functions and
  returns one structured evidence bundle. This is the level the LLM should
  usually be choosing at: it decides *which investigation* answers the
  question, not which individual metric to fetch, so it can no longer
  produce an incomplete answer by forgetting to also check e.g. repeat
  customer rate when diagnosing a sales dip -- the investigation tool always
  gathers everything relevant to it.

No analytics logic is reimplemented here -- every number still traces back
to a pure function in ``etsy_seller_copilot.analytics`` -- and no routing
decision is hardcoded either; the LLM still decides which tool(s) to call
and in what order.
"""

import pandas as pd
from langchain_core.tools import BaseTool, tool

from etsy_seller_copilot.analytics.health import compute_health_score
from etsy_seller_copilot.analytics.metrics import (
    average_order_value,
    compare_periods,
    daily_sales_trend,
    number_of_orders,
    product_performance_comparison,
    repeat_customer_rate,
    revenue_by_month,
    top_selling_listings,
    total_revenue,
)

CONVERSION_RATE_UNAVAILABLE_MESSAGE = (
    "Conversion rate can't be computed from this data: an Etsy Sold Orders CSV "
    "only lists completed orders, not shop visits or listing views, so there's "
    "no visitor count to divide orders by. Ask the seller to also export Etsy's "
    "Shop Stats / Traffic report (Shop Manager > Stats), which includes visits, "
    "to calculate this."
)

# A product revenue drop at least this steep (vs a non-trivial prior base) is
# flagged as an anomaly worth naming individually. Matches the threshold
# `analytics.health` already uses for the same judgment call.
_ANOMALY_DROP_THRESHOLD_PCT = -50.0
_ANOMALY_MIN_PREVIOUS_REVENUE = 1.0


def _series_to_dict(series: pd.Series) -> dict[str, float]:
    return {str(period): float(value) for period, value in series.items()}


def _top_listing_records(orders: pd.DataFrame, top_n: int) -> list[dict]:
    listings = top_selling_listings(orders, top_n=top_n)
    return [
        {"item_name": str(row["Item Name"]), "quantity_sold": int(row["quantity_sold"]), "revenue": float(row["revenue"])}
        for _, row in listings.iterrows()
    ]


def _product_performance_records(orders: pd.DataFrame, period_days: int, limit: int = 10) -> list[dict]:
    df = product_performance_comparison(orders, period_days=period_days).head(limit)
    return [
        {
            "item_name": str(row["Item Name"]),
            "recent_revenue": float(row["recent_revenue"]),
            "previous_revenue": float(row["previous_revenue"]),
            "pct_change": None if pd.isna(row["pct_change"]) else float(row["pct_change"]),
        }
        for _, row in df.iterrows()
    ]


def build_analyst_tools(orders: pd.DataFrame) -> list[BaseTool]:
    """Build the full analyst tool set for the Business Analyst Agent, bound to ``orders``."""

    # -- Direct lookups: one literal number, for simple factual questions. --

    @tool
    def get_available_columns() -> list[str]:
        """List the column names actually present in the uploaded order data.

        Call this before claiming a metric is unavailable, so you can name
        the real gap (e.g. "no Sale Date column") instead of guessing.
        """
        return list(orders.columns)

    @tool
    def get_total_revenue() -> float:
        """Total revenue across all orders (price x quantity, summed)."""
        return total_revenue(orders)

    @tool
    def get_number_of_orders() -> int:
        """The total number of distinct orders."""
        return number_of_orders(orders)

    @tool
    def get_average_order_value() -> float:
        """The average revenue per order."""
        return average_order_value(orders)

    @tool
    def explain_conversion_rate_unavailable() -> str:
        """Call this whenever asked about conversion rate, shop visits, listing
        views, or traffic -- an Etsy Sold Orders CSV cannot answer these."""
        return CONVERSION_RATE_UNAVAILABLE_MESSAGE

    # -- Business investigations: the evidence bundle for a whole task. --

    @tool
    def investigate_sales_decline(period_days: int = 30) -> dict:
        """Investigate why sales might be down or up: recent-vs-previous-period
        revenue/order/AOV trend, the day-by-day sales trend, which products
        are the biggest movers, and the repeat customer rate -- all in one
        call.

        Use this for "why are sales down", "why did revenue drop", or any
        other sales-decline-style question. It already gathers everything an
        analyst would check, so you normally won't need any other tool
        alongside it. Pick `period_days` to match the timeframe implied by
        the question (e.g. 7 for "this week", 30 for "this month"). Requires
        a Sale Date column.
        """
        evidence: dict = {
            "period_days": period_days,
            "period_over_period": compare_periods(orders, period_days=period_days),
            "daily_sales_trend": _series_to_dict(daily_sales_trend(orders)),
            "product_movers": _product_performance_records(orders, period_days),
        }
        try:
            evidence["repeat_customer_rate"] = repeat_customer_rate(orders)
        except ValueError as exc:
            evidence["repeat_customer_rate"] = f"unavailable: {exc}"
        return evidence

    @tool
    def business_health_check(period_days: int = 30) -> dict:
        """Run a full shop health check: an overall 0-100 score, a
        plain-English breakdown of what's driving it, and lists of what's
        improving, declining, and at risk.

        Use this for open-ended "how is my shop doing", "give me a health
        check", or "is anything wrong" questions -- it's the best single
        starting point for a general diagnostic, and often the only tool
        you'll need.
        """
        health = compute_health_score(orders, period_days=period_days)
        return {
            "score": health.score,
            "score_breakdown": [f"{c.label}: {c.points:+.1f} pts -- {c.explanation}" for c in health.components],
            "improving": list(health.improving),
            "declining": list(health.declining),
            "risks": list(health.risks),
        }

    @tool
    def analyze_customer_behavior() -> dict:
        """Analyze buyer behavior: the fraction of customers who've placed
        more than one order.

        Use this for "who are my customers", "are people coming back", or
        "repeat buyer rate" questions. Requires a Buyer column.
        """
        return {"repeat_customer_rate": repeat_customer_rate(orders)}

    @tool
    def analyze_product_performance(period_days: int = 30, top_n: int = 10) -> dict:
        """Analyze which products are driving (or dragging on) revenue:
        current top sellers by revenue and units sold, plus how each
        product's revenue moved in the most recent `period_days` days vs the
        period before.

        Use this for "what's selling well", "what should I promote or
        restock", or "which products are underperforming" questions.
        """
        return {
            "period_days": period_days,
            "top_listings": _top_listing_records(orders, top_n),
            "product_performance_changes": _product_performance_records(orders, period_days),
        }

    @tool
    def analyze_revenue_trends(period_days: int = 30) -> dict:
        """Analyze revenue over time: month-by-month totals, day-by-day
        totals, and a recent-vs-previous-period comparison.

        Use this for "how has revenue changed", "show me my sales trend", or
        "am I growing" questions. Requires a Sale Date column.
        """
        return {
            "revenue_by_month": _series_to_dict(revenue_by_month(orders)),
            "daily_sales_trend": _series_to_dict(daily_sales_trend(orders)),
            "period_over_period": compare_periods(orders, period_days=period_days),
        }

    @tool
    def detect_business_anomalies(period_days: int = 30) -> dict:
        """Scan for notable anomalies: the overall period-over-period swing
        in revenue/orders/AOV, and individual products whose revenue dropped
        sharply (50%+ from a non-trivial base) vs the previous period.

        Use this for "is anything unusual", "any red flags", or "detect
        anomalies" questions.
        """
        performance = product_performance_comparison(orders, period_days=period_days)
        drops = performance[
            performance["pct_change"].notna()
            & (performance["pct_change"] <= _ANOMALY_DROP_THRESHOLD_PCT)
            & (performance["previous_revenue"] >= _ANOMALY_MIN_PREVIOUS_REVENUE)
        ]
        return {
            "period_days": period_days,
            "period_over_period": compare_periods(orders, period_days=period_days),
            "sharp_product_drops": [
                {
                    "item_name": str(row["Item Name"]),
                    "previous_revenue": float(row["previous_revenue"]),
                    "recent_revenue": float(row["recent_revenue"]),
                    "pct_change": float(row["pct_change"]),
                }
                for _, row in drops.iterrows()
            ],
        }

    return [
        get_available_columns,
        get_total_revenue,
        get_number_of_orders,
        get_average_order_value,
        explain_conversion_rate_unavailable,
        investigate_sales_decline,
        business_health_check,
        analyze_customer_behavior,
        analyze_product_performance,
        analyze_revenue_trends,
        detect_business_anomalies,
    ]
