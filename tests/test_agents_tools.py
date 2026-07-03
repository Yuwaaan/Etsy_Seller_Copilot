import pandas as pd

from etsy_seller_copilot.agents.tools import build_tools
from etsy_seller_copilot.services.dispatcher import dispatch
from etsy_seller_copilot.services.intent import Intent
from etsy_seller_copilot.services.planner import AnalyticsPlan

SAMPLE_ORDERS = pd.DataFrame(
    [
        {
            "Order ID": 1,
            "Sale Date": "2026-01-05",
            "Item Name": "Mug",
            "Quantity": 2,
            "Price": 10.0,
            "Buyer": "alice",
        },
        {
            "Order ID": 2,
            "Sale Date": "2026-02-10",
            "Item Name": "Coaster",
            "Quantity": 3,
            "Price": 5.0,
            "Buyer": "bob",
        },
    ]
)


def _tool_by_name(tools, name: str):
    return next(t for t in tools if t.name == name)


class TestBuildTools:
    def test_builds_one_tool_per_capability(self) -> None:
        tools = build_tools(SAMPLE_ORDERS)

        names = {t.name for t in tools}
        assert names == {
            "get_total_revenue",
            "get_number_of_orders",
            "get_average_order_value",
            "get_revenue_by_month",
            "get_top_selling_listings",
            "get_repeat_customer_rate",
        }

    def test_get_total_revenue_matches_dispatcher(self) -> None:
        tools = build_tools(SAMPLE_ORDERS)
        tool = _tool_by_name(tools, "get_total_revenue")

        expected = dispatch(AnalyticsPlan(intent=Intent.TOTAL_REVENUE), SAMPLE_ORDERS).value

        assert tool.invoke({}) == expected

    def test_get_repeat_customer_rate_matches_dispatcher(self) -> None:
        tools = build_tools(SAMPLE_ORDERS)
        tool = _tool_by_name(tools, "get_repeat_customer_rate")

        expected = dispatch(AnalyticsPlan(intent=Intent.REPEAT_CUSTOMER_RATE), SAMPLE_ORDERS).value

        assert tool.invoke({}) == expected

    def test_get_top_selling_listings_passes_top_n(self) -> None:
        tools = build_tools(SAMPLE_ORDERS)
        tool = _tool_by_name(tools, "get_top_selling_listings")

        expected = dispatch(
            AnalyticsPlan(intent=Intent.TOP_SELLING_LISTINGS, kwargs={"top_n": 1}),
            SAMPLE_ORDERS,
        ).value

        assert tool.invoke({"top_n": 1}) == expected

    def test_get_revenue_by_month_matches_dispatcher(self) -> None:
        tools = build_tools(SAMPLE_ORDERS)
        tool = _tool_by_name(tools, "get_revenue_by_month")

        expected = dispatch(AnalyticsPlan(intent=Intent.REVENUE_BY_MONTH), SAMPLE_ORDERS).value

        assert tool.invoke({}) == expected

    def test_tools_are_isolated_per_dataset(self) -> None:
        other_orders = SAMPLE_ORDERS.copy()
        other_orders["Price"] = other_orders["Price"] * 2

        tools_a = build_tools(SAMPLE_ORDERS)
        tools_b = build_tools(other_orders)

        revenue_a = _tool_by_name(tools_a, "get_total_revenue").invoke({})
        revenue_b = _tool_by_name(tools_b, "get_total_revenue").invoke({})

        assert revenue_b == revenue_a * 2
