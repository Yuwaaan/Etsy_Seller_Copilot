from etsy_seller_copilot.agents.prompts import format_error_response, format_response
from etsy_seller_copilot.services.intent import Intent


class TestFormatResponse:
    def test_total_revenue(self) -> None:
        assert format_response(Intent.TOTAL_REVENUE, 1234.5) == "Your total revenue is $1,234.50."

    def test_number_of_orders(self) -> None:
        assert format_response(Intent.NUMBER_OF_ORDERS, 7) == "You have 7 orders."

    def test_average_order_value(self) -> None:
        assert format_response(Intent.AVERAGE_ORDER_VALUE, 42.0) == "Your average order value is $42.00."

    def test_repeat_customer_rate(self) -> None:
        assert format_response(Intent.REPEAT_CUSTOMER_RATE, 0.5) == "50.0% of your customers are repeat buyers."

    def test_revenue_by_month(self) -> None:
        result = format_response(Intent.REVENUE_BY_MONTH, {"2026-01": 25.0, "2026-02": 25.0})

        assert result == "Here is your revenue by month:\n- 2026-01: $25.00\n- 2026-02: $25.00"

    def test_revenue_by_month_empty(self) -> None:
        assert format_response(Intent.REVENUE_BY_MONTH, {}) == "There's no revenue data to break down by month yet."

    def test_top_selling_listings(self) -> None:
        value = [{"item_name": "Mug", "quantity_sold": 3, "revenue": 30.0}]

        result = format_response(Intent.TOP_SELLING_LISTINGS, value)

        assert result == "Here are your top-selling listings:\n- Mug: 3 sold ($30.00)"

    def test_top_selling_listings_empty(self) -> None:
        assert format_response(Intent.TOP_SELLING_LISTINGS, []) == "There are no listings with sales yet."

    def test_daily_sales_trend(self) -> None:
        result = format_response(Intent.DAILY_SALES_TREND, {"2026-01-05": 25.0, "2026-01-06": 10.0})

        assert result == "Here is your revenue by day:\n- 2026-01-05: $25.00\n- 2026-01-06: $10.00"

    def test_daily_sales_trend_empty(self) -> None:
        assert format_response(Intent.DAILY_SALES_TREND, {}) == "There's no revenue data to break down by day yet."

    def test_shop_summary(self) -> None:
        value = {
            "total_revenue": 100.0,
            "number_of_orders": 4,
            "average_order_value": 25.0,
            "top_product": "Mug",
        }

        result = format_response(Intent.SHOP_SUMMARY, value)

        assert result == (
            "Here's a quick summary of your shop:\n"
            "- Total revenue: $100.00\n"
            "- Orders: 4\n"
            "- Average order value: $25.00\n"
            "- Top product: Mug"
        )

    def test_shop_summary_without_top_product(self) -> None:
        value = {
            "total_revenue": 0.0,
            "number_of_orders": 0,
            "average_order_value": 0.0,
            "top_product": None,
        }

        result = format_response(Intent.SHOP_SUMMARY, value)

        assert "Top product" not in result

    def test_conversion_rate_passes_message_through(self) -> None:
        assert format_response(Intent.CONVERSION_RATE, "no visits data") == "no visits data"

    def test_unknown_passes_message_through(self) -> None:
        assert format_response(Intent.UNKNOWN, "here's what I can answer") == "here's what I can answer"


class TestFormatErrorResponse:
    def test_wraps_error_message(self) -> None:
        assert format_error_response("boom") == "I couldn't answer that question: boom"
