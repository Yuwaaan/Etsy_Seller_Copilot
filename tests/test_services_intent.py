import pytest

from etsy_seller_copilot.services.intent import Intent, detect_intent


class TestDetectIntent:
    @pytest.mark.parametrize(
        "text, expected_intent",
        [
            ("What is my total revenue?", Intent.TOTAL_REVENUE),
            ("How much money have I made?", Intent.TOTAL_REVENUE),
            ("What were my total sales", Intent.TOTAL_REVENUE),
            ("How many orders do I have?", Intent.NUMBER_OF_ORDERS),
            ("What's my order count", Intent.NUMBER_OF_ORDERS),
            ("What is my average order value?", Intent.AVERAGE_ORDER_VALUE),
            ("What's my AOV", Intent.AVERAGE_ORDER_VALUE),
            ("Show me revenue by month", Intent.REVENUE_BY_MONTH),
            ("What's my monthly revenue trend", Intent.REVENUE_BY_MONTH),
            ("What are my top selling listings?", Intent.TOP_SELLING_LISTINGS),
            ("Show me my best sellers", Intent.TOP_SELLING_LISTINGS),
            ("Show me my top 5 listings", Intent.TOP_SELLING_LISTINGS),
            ("What is my repeat customer rate?", Intent.REPEAT_CUSTOMER_RATE),
            ("How many returning customers do I have", Intent.REPEAT_CUSTOMER_RATE),
        ],
    )
    def test_matches_expected_intent(self, text: str, expected_intent: Intent) -> None:
        assert detect_intent(text).intent is expected_intent

    @pytest.mark.parametrize(
        "text, expected_intent",
        [
            # Short/bare order queries.
            ("orders", Intent.NUMBER_OF_ORDERS),
            ("order", Intent.NUMBER_OF_ORDERS),
            ("how many orders", Intent.NUMBER_OF_ORDERS),
            # Short/bare revenue queries.
            ("revenue", Intent.TOTAL_REVENUE),
            ("sales", Intent.TOTAL_REVENUE),
            ("money", Intent.TOTAL_REVENUE),
            # Conversion rate has no handler that computes a number, but it's
            # still a recognized intent, not UNKNOWN.
            ("conversion rate", Intent.CONVERSION_RATE),
            ("conversion", Intent.CONVERSION_RATE),
            # Product queries.
            ("products", Intent.TOP_SELLING_LISTINGS),
            ("top products", Intent.TOP_SELLING_LISTINGS),
            ("best seller", Intent.TOP_SELLING_LISTINGS),
            ("best sellers", Intent.TOP_SELLING_LISTINGS),
            # Trend queries.
            ("trend", Intent.REVENUE_BY_MONTH),
            ("monthly", Intent.REVENUE_BY_MONTH),
            ("daily", Intent.DAILY_SALES_TREND),
            # Overview queries.
            ("summary", Intent.SHOP_SUMMARY),
            ("overview", Intent.SHOP_SUMMARY),
            ("performance", Intent.SHOP_SUMMARY),
        ],
    )
    def test_matches_short_and_imperfect_queries(self, text: str, expected_intent: Intent) -> None:
        assert detect_intent(text).intent is expected_intent

    def test_is_case_insensitive(self) -> None:
        assert detect_intent("TOTAL REVENUE PLEASE").intent is Intent.TOTAL_REVENUE

    def test_unrecognized_text_is_unknown(self) -> None:
        assert detect_intent("What's the weather today?").intent is Intent.UNKNOWN

    def test_empty_text_is_unknown(self) -> None:
        assert detect_intent("").intent is Intent.UNKNOWN

    def test_more_specific_intent_wins_over_generic_revenue_wording(self) -> None:
        # "revenue" alone would match TOTAL_REVENUE's generic pattern, but this
        # phrasing should resolve to the more specific REVENUE_BY_MONTH intent.
        assert detect_intent("Give me a revenue breakdown by month").intent is Intent.REVENUE_BY_MONTH

    def test_extracts_top_n_from_text(self) -> None:
        result = detect_intent("Show me my top 3 best sellers")

        assert result.intent is Intent.TOP_SELLING_LISTINGS
        assert result.top_n == 3

    def test_top_n_is_none_when_not_specified(self) -> None:
        result = detect_intent("Show me my top selling listings")

        assert result.top_n is None

    def test_raw_text_is_preserved_unmodified(self) -> None:
        original = "  What is my TOTAL revenue?  "
        result = detect_intent(original)

        assert result.raw_text == original
