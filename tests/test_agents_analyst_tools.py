import pandas as pd

from etsy_seller_copilot.agents.analyst_tools import CONVERSION_RATE_UNAVAILABLE_MESSAGE, build_analyst_tools

SAMPLE_ORDERS = pd.DataFrame(
    [
        {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "alice"},
        {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 2, "Price": 10.0, "Buyer": "bob"},
        {"Order ID": 3, "Sale Date": "2026-01-16", "Item Name": "Scarf", "Quantity": 1, "Price": 30.0, "Buyer": "alice"},
    ]
)


def _tool_by_name(tools, name: str):
    return next(t for t in tools if t.name == name)


class TestBuildAnalystTools:
    def test_builds_two_tiers_of_tools(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        names = {t.name for t in tools}
        assert names == {
            # Direct lookups.
            "get_available_columns",
            "get_total_revenue",
            "get_number_of_orders",
            "get_average_order_value",
            "explain_conversion_rate_unavailable",
            # Business investigations.
            "investigate_sales_decline",
            "business_health_check",
            "analyze_customer_behavior",
            "analyze_product_performance",
            "analyze_revenue_trends",
            "detect_business_anomalies",
        }

    def test_every_tool_has_a_docstring_description(self) -> None:
        # The LLM only knows what a tool does from its description -- an
        # undocumented tool is effectively unusable to the planner.
        tools = build_analyst_tools(SAMPLE_ORDERS)

        for t in tools:
            assert t.description, f"{t.name} has no description"

    def test_get_available_columns_lists_real_columns(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "get_available_columns").invoke({})

        assert set(result) == set(SAMPLE_ORDERS.columns)

    def test_get_total_revenue_matches_analytics_function(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "get_total_revenue").invoke({})

        assert result == 60.0  # (1*10) + (2*10) + (1*30)

    def test_explain_conversion_rate_unavailable_names_the_missing_data(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "explain_conversion_rate_unavailable").invoke({})

        assert result == CONVERSION_RATE_UNAVAILABLE_MESSAGE
        assert "visits" in result or "views" in result

    def test_tools_are_isolated_per_dataset(self) -> None:
        other_orders = SAMPLE_ORDERS.copy()
        other_orders["Price"] = other_orders["Price"] * 2

        tools_a = build_analyst_tools(SAMPLE_ORDERS)
        tools_b = build_analyst_tools(other_orders)

        revenue_a = _tool_by_name(tools_a, "get_total_revenue").invoke({})
        revenue_b = _tool_by_name(tools_b, "get_total_revenue").invoke({})

        assert revenue_b == revenue_a * 2


class TestInvestigateSalesDecline:
    def test_returns_a_full_evidence_bundle(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "investigate_sales_decline").invoke({"period_days": 10})

        assert result["period_days"] == 10
        assert isinstance(result["period_over_period"], dict)
        assert isinstance(result["daily_sales_trend"], dict)
        assert isinstance(result["product_movers"], list)
        assert result["repeat_customer_rate"] == 0.5  # alice: 2 orders, bob: 1 order

    def test_marks_repeat_customer_rate_unavailable_instead_of_failing(self) -> None:
        orders_without_buyer = SAMPLE_ORDERS.drop(columns=["Buyer"])
        tools = build_analyst_tools(orders_without_buyer)

        result = _tool_by_name(tools, "investigate_sales_decline").invoke({"period_days": 10})

        assert isinstance(result["period_over_period"], dict)
        assert "unavailable" in result["repeat_customer_rate"]


class TestBusinessHealthCheck:
    def test_wraps_the_deterministic_health_score(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "business_health_check").invoke({"period_days": 10})

        assert isinstance(result["score"], int)
        assert 0 <= result["score"] <= 100
        assert len(result["score_breakdown"]) > 0
        assert all("pts" in line for line in result["score_breakdown"])
        assert isinstance(result["improving"], list)
        assert isinstance(result["declining"], list)
        assert isinstance(result["risks"], list)


class TestAnalyzeCustomerBehavior:
    def test_returns_repeat_customer_rate(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "analyze_customer_behavior").invoke({})

        assert result == {"repeat_customer_rate": 0.5}

    def test_raises_when_no_buyer_column(self) -> None:
        orders_without_buyer = SAMPLE_ORDERS.drop(columns=["Buyer"])
        tools = build_analyst_tools(orders_without_buyer)

        try:
            _tool_by_name(tools, "analyze_customer_behavior").invoke({})
            raised = False
        except Exception:
            raised = True

        # This is expected to raise -- the business_analyst loop is what
        # catches it and feeds the error back to the LLM as a tool result.
        assert raised


class TestAnalyzeProductPerformance:
    def test_returns_json_safe_records(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "analyze_product_performance").invoke({"period_days": 10, "top_n": 1})

        assert result["top_listings"] == [{"item_name": "Mug", "quantity_sold": 3, "revenue": 30.0}]
        assert isinstance(result["top_listings"][0]["quantity_sold"], int)
        assert isinstance(result["top_listings"][0]["revenue"], float)
        assert isinstance(result["product_performance_changes"], list)
        assert all("pct_change" in row for row in result["product_performance_changes"])


class TestAnalyzeRevenueTrends:
    def test_returns_month_day_and_period_breakdowns(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "analyze_revenue_trends").invoke({"period_days": 10})

        assert isinstance(result["revenue_by_month"], dict)
        assert isinstance(result["daily_sales_trend"], dict)
        assert result["period_over_period"]["period_days"] == 10


class TestDetectBusinessAnomalies:
    def test_flags_a_sharp_product_drop(self) -> None:
        orders = pd.DataFrame(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 10, "Price": 10.0, "Buyer": "alice"},
                {"Order ID": 2, "Sale Date": "2026-01-20", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "bob"},
            ]
        )
        tools = build_analyst_tools(orders)

        result = _tool_by_name(tools, "detect_business_anomalies").invoke({"period_days": 10})

        assert result["period_days"] == 10
        assert isinstance(result["period_over_period"], dict)
        assert len(result["sharp_product_drops"]) == 1
        assert result["sharp_product_drops"][0]["item_name"] == "Mug"
        assert result["sharp_product_drops"][0]["pct_change"] <= -50.0

    def test_no_drops_when_nothing_fell_sharply(self) -> None:
        tools = build_analyst_tools(SAMPLE_ORDERS)

        result = _tool_by_name(tools, "detect_business_anomalies").invoke({"period_days": 10})

        assert result["sharp_product_drops"] == []
