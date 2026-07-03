"""LangChain tool definitions over the analytics capabilities.

Not wired into ``graph.py`` yet — this milestone's routing is fully
deterministic via the services layer (see ``nodes.py``). These tools exist as
the seam a future LLM-driven node (e.g. a fallback for questions the
rule-based intent classifier can't resolve) will bind to, without requiring
any change to ``services/``. Each tool is a one-line delegation to
``services.dispatcher.dispatch`` — no analytics logic is reimplemented here.
"""

import pandas as pd
from langchain_core.tools import BaseTool, tool

from etsy_seller_copilot.services.dispatcher import ListingRecord, dispatch
from etsy_seller_copilot.services.intent import Intent
from etsy_seller_copilot.services.planner import AnalyticsPlan


def build_tools(orders: pd.DataFrame) -> list[BaseTool]:
    """Build one LangChain tool per analytics capability, bound to ``orders``."""

    @tool
    def get_total_revenue() -> float:
        """Total revenue across all orders."""
        return dispatch(AnalyticsPlan(intent=Intent.TOTAL_REVENUE), orders).value

    @tool
    def get_number_of_orders() -> int:
        """The total number of distinct orders."""
        return dispatch(AnalyticsPlan(intent=Intent.NUMBER_OF_ORDERS), orders).value

    @tool
    def get_average_order_value() -> float:
        """The average revenue per order."""
        return dispatch(AnalyticsPlan(intent=Intent.AVERAGE_ORDER_VALUE), orders).value

    @tool
    def get_revenue_by_month() -> dict[str, float]:
        """Revenue broken down by calendar month."""
        return dispatch(AnalyticsPlan(intent=Intent.REVENUE_BY_MONTH), orders).value

    @tool
    def get_top_selling_listings(top_n: int = 10) -> list[ListingRecord]:
        """The top-selling listings ranked by revenue."""
        plan = AnalyticsPlan(intent=Intent.TOP_SELLING_LISTINGS, kwargs={"top_n": top_n})
        return dispatch(plan, orders).value

    @tool
    def get_repeat_customer_rate() -> float:
        """The fraction of customers who have placed more than one order."""
        return dispatch(AnalyticsPlan(intent=Intent.REPEAT_CUSTOMER_RATE), orders).value

    return [
        get_total_revenue,
        get_number_of_orders,
        get_average_order_value,
        get_revenue_by_month,
        get_top_selling_listings,
        get_repeat_customer_rate,
    ]
