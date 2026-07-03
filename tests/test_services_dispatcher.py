import pandas as pd
import pytest

from etsy_seller_copilot.services.dispatcher import DispatchError, dispatch
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
            "Order ID": 1,
            "Sale Date": "2026-01-05",
            "Item Name": "Coaster",
            "Quantity": 1,
            "Price": 5.0,
            "Buyer": "alice",
        },
        {
            "Order ID": 2,
            "Sale Date": "2026-02-10",
            "Item Name": "Mug",
            "Quantity": 1,
            "Price": 10.0,
            "Buyer": "bob",
        },
        {
            "Order ID": 3,
            "Sale Date": "2026-02-15",
            "Item Name": "Coaster",
            "Quantity": 3,
            "Price": 5.0,
            "Buyer": "alice",
        },
    ]
)


class TestDispatch:
    def test_total_revenue(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.TOTAL_REVENUE), SAMPLE_ORDERS)

        assert result.intent is Intent.TOTAL_REVENUE
        assert result.value == 50.0

    def test_number_of_orders(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.NUMBER_OF_ORDERS), SAMPLE_ORDERS)

        assert result.value == 3

    def test_average_order_value(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.AVERAGE_ORDER_VALUE), SAMPLE_ORDERS)

        assert result.value == pytest.approx(50.0 / 3)

    def test_repeat_customer_rate(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.REPEAT_CUSTOMER_RATE), SAMPLE_ORDERS)

        assert result.value == pytest.approx(0.5)

    def test_revenue_by_month_returns_json_safe_dict(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.REVENUE_BY_MONTH), SAMPLE_ORDERS)

        assert isinstance(result.value, dict)
        assert result.value == {"2026-01": 25.0, "2026-02": 25.0}

    def test_top_selling_listings_returns_json_safe_records(self) -> None:
        plan = AnalyticsPlan(intent=Intent.TOP_SELLING_LISTINGS, kwargs={"top_n": 1})

        result = dispatch(plan, SAMPLE_ORDERS)

        assert result.value == [{"item_name": "Mug", "quantity_sold": 3, "revenue": 30.0}]

    def test_unrecognized_intent_returns_friendly_help_message(self) -> None:
        # Intent.UNKNOWN has a real handler (unlike the old behavior), so it
        # succeeds like any other intent — just with a help message as the value.
        result = dispatch(AnalyticsPlan(intent=Intent.UNKNOWN), SAMPLE_ORDERS)

        assert result.intent is Intent.UNKNOWN
        assert isinstance(result.value, str)
        assert "total revenue" in result.value

    def test_conversion_rate_explains_data_is_unavailable(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.CONVERSION_RATE), SAMPLE_ORDERS)

        assert isinstance(result.value, str)
        assert "visits" in result.value or "views" in result.value

    def test_daily_sales_trend_returns_json_safe_dict(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.DAILY_SALES_TREND), SAMPLE_ORDERS)

        assert result.value == {"2026-01-05": 25.0, "2026-02-10": 10.0, "2026-02-15": 15.0}

    def test_shop_summary_combines_headline_metrics(self) -> None:
        result = dispatch(AnalyticsPlan(intent=Intent.SHOP_SUMMARY), SAMPLE_ORDERS)

        assert result.value == {
            "total_revenue": 50.0,
            "number_of_orders": 3,
            "average_order_value": pytest.approx(50.0 / 3),
            "top_product": "Mug",
        }

    def test_unregistered_intent_raises_dispatch_error(self) -> None:
        # A plan with an intent that has no dispatch-table entry at all
        # (unreachable via the real planner, which only ever produces real
        # Intent values) still raises rather than silently doing nothing.
        plan = AnalyticsPlan(intent="not_a_real_intent")  # type: ignore[arg-type]

        with pytest.raises(DispatchError):
            dispatch(plan, SAMPLE_ORDERS)
