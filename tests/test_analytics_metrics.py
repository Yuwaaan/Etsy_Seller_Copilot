import pandas as pd
import pytest

from etsy_seller_copilot.analytics.metrics import (
    average_order_value,
    compare_periods,
    daily_sales_trend,
    monthly_sales_trend,
    number_of_orders,
    product_performance_comparison,
    quantity_sold_by_product,
    repeat_customer_rate,
    revenue_by_month,
    revenue_by_product,
    top_selling_listings,
    total_revenue,
)


def _orders(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


SAMPLE_ORDERS = _orders(
    [
        # Order 1: two line items, same order, same buyer.
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
        # Order 2: single item, different buyer, different month.
        {
            "Order ID": 2,
            "Sale Date": "2026-02-10",
            "Item Name": "Mug",
            "Quantity": 1,
            "Price": 10.0,
            "Buyer": "bob",
        },
        # Order 3: alice's second order (repeat customer), same month as order 2.
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


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(columns=["Order ID", "Sale Date", "Item Name", "Quantity", "Price", "Buyer"])


class TestTotalRevenue:
    def test_sums_price_times_quantity(self) -> None:
        # (2*10) + (1*5) + (1*10) + (3*5) = 20 + 5 + 10 + 15 = 50
        assert total_revenue(SAMPLE_ORDERS) == 50.0

    def test_empty_orders_is_zero(self) -> None:
        assert total_revenue(_empty_orders()) == 0.0

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="Price"):
            total_revenue(pd.DataFrame({"Quantity": [1]}))


class TestNumberOfOrders:
    def test_counts_distinct_order_ids_not_rows(self) -> None:
        assert number_of_orders(SAMPLE_ORDERS) == 3

    def test_empty_orders_is_zero(self) -> None:
        assert number_of_orders(_empty_orders()) == 0

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Order ID"):
            number_of_orders(pd.DataFrame({"Item Name": ["Mug"]}))


class TestAverageOrderValue:
    def test_divides_revenue_by_order_count(self) -> None:
        assert average_order_value(SAMPLE_ORDERS) == pytest.approx(50.0 / 3)

    def test_empty_orders_is_zero_not_error(self) -> None:
        assert average_order_value(_empty_orders()) == 0.0


class TestRevenueByMonth:
    def test_groups_by_calendar_month(self) -> None:
        result = revenue_by_month(SAMPLE_ORDERS)

        assert list(result.index.astype(str)) == ["2026-01", "2026-02"]
        assert result.loc["2026-01"] == 25.0
        assert result.loc["2026-02"] == 25.0

    def test_empty_orders_returns_empty_series(self) -> None:
        result = revenue_by_month(_empty_orders())

        assert isinstance(result, pd.Series)
        assert result.empty


class TestTopSellingListings:
    def test_ranks_by_revenue_descending(self) -> None:
        result = top_selling_listings(SAMPLE_ORDERS)

        assert list(result["Item Name"]) == ["Mug", "Coaster"]
        # Coaster: (1*5) + (3*5) = 20; Mug: (2*10) + (1*10) = 30
        assert result.loc[result["Item Name"] == "Mug", "revenue"].iloc[0] == 30.0
        assert result.loc[result["Item Name"] == "Coaster", "revenue"].iloc[0] == 20.0

    def test_respects_top_n(self) -> None:
        result = top_selling_listings(SAMPLE_ORDERS, top_n=1)

        assert len(result) == 1
        assert result.iloc[0]["Item Name"] == "Mug"

    def test_empty_orders_returns_empty_dataframe_with_columns(self) -> None:
        result = top_selling_listings(_empty_orders())

        assert list(result.columns) == ["Item Name", "quantity_sold", "revenue"]
        assert result.empty


class TestQuantitySoldByProduct:
    def test_sums_quantity_per_item_sorted_descending(self) -> None:
        result = quantity_sold_by_product(SAMPLE_ORDERS)

        # Mug: 2 + 1 = 3; Coaster: 1 + 3 = 4
        assert list(result.index) == ["Coaster", "Mug"]
        assert result.loc["Coaster"] == 4
        assert result.loc["Mug"] == 3

    def test_empty_orders_returns_empty_series(self) -> None:
        result = quantity_sold_by_product(_empty_orders())

        assert isinstance(result, pd.Series)
        assert result.empty


class TestRevenueByProduct:
    def test_sums_revenue_per_item_sorted_descending(self) -> None:
        result = revenue_by_product(SAMPLE_ORDERS)

        assert list(result.index) == ["Mug", "Coaster"]
        assert result.loc["Mug"] == 30.0
        assert result.loc["Coaster"] == 20.0

    def test_empty_orders_returns_empty_series(self) -> None:
        result = revenue_by_product(_empty_orders())

        assert isinstance(result, pd.Series)
        assert result.empty


class TestDailySalesTrend:
    def test_groups_by_calendar_day(self) -> None:
        result = daily_sales_trend(SAMPLE_ORDERS)

        assert list(result.index.astype(str)) == ["2026-01-05", "2026-02-10", "2026-02-15"]
        assert result.loc["2026-01-05"] == 25.0
        assert result.loc["2026-02-10"] == 10.0
        assert result.loc["2026-02-15"] == 15.0

    def test_empty_orders_returns_empty_series(self) -> None:
        result = daily_sales_trend(_empty_orders())

        assert isinstance(result, pd.Series)
        assert result.empty


class TestMonthlySalesTrend:
    def test_matches_revenue_by_month(self) -> None:
        result = monthly_sales_trend(SAMPLE_ORDERS)
        expected = revenue_by_month(SAMPLE_ORDERS)

        pd.testing.assert_series_equal(result, expected)


class TestComparePeriods:
    def test_compares_recent_vs_previous_window(self) -> None:
        orders = _orders(
            [
                {"Order ID": 10, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
                {"Order ID": 11, "Sale Date": "2026-01-05", "Item Name": "Scarf", "Quantity": 1, "Price": 20.0},
                {"Order ID": 12, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 2, "Price": 10.0},
                {"Order ID": 13, "Sale Date": "2026-01-20", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
            ]
        )

        result = compare_periods(orders, period_days=10)

        assert result["period_days"] == 10
        assert result["recent_start"] == "2026-01-11"
        assert result["recent_end"] == "2026-01-20"
        assert result["previous_start"] == "2026-01-01"
        assert result["previous_end"] == "2026-01-10"
        assert result["revenue"] == {"recent": 30.0, "previous": 30.0, "pct_change": 0.0}
        assert result["orders"] == {"recent": 2, "previous": 2, "pct_change": 0.0}
        assert result["average_order_value"] == {"recent": 15.0, "previous": 15.0, "pct_change": 0.0}

    def test_pct_change_is_none_when_previous_period_is_empty(self) -> None:
        orders = _orders([{"Order ID": 1, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 1, "Price": 10.0}])

        result = compare_periods(orders, period_days=10)

        assert result["revenue"] == {"recent": 10.0, "previous": 0.0, "pct_change": None}

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="Sale Date"):
            compare_periods(pd.DataFrame({"Price": [1]}))

    def test_no_valid_dates_returns_empty_comparison(self) -> None:
        orders = _orders([{"Order ID": 1, "Sale Date": "not-a-date", "Item Name": "Mug", "Quantity": 1, "Price": 10.0}])

        result = compare_periods(orders, period_days=10)

        assert result["recent_start"] is None
        assert result["revenue"] == {"recent": 0.0, "previous": 0.0, "pct_change": None}

    def test_empty_orders_returns_empty_comparison(self) -> None:
        result = compare_periods(_empty_orders(), period_days=10)

        assert result["recent_start"] is None
        assert result["orders"] == {"recent": 0, "previous": 0, "pct_change": None}


class TestProductPerformanceComparison:
    def test_compares_per_product_revenue_across_periods(self) -> None:
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
                {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 3, "Price": 10.0},
                {"Order ID": 3, "Sale Date": "2026-01-16", "Item Name": "Coaster", "Quantity": 1, "Price": 5.0},
            ]
        )

        result = product_performance_comparison(orders, period_days=10)

        mug_row = result[result["Item Name"] == "Mug"].iloc[0]
        assert mug_row["recent_revenue"] == 30.0
        assert mug_row["previous_revenue"] == 10.0
        assert mug_row["pct_change"] == pytest.approx(200.0)

        coaster_row = result[result["Item Name"] == "Coaster"].iloc[0]
        assert coaster_row["recent_revenue"] == 5.0
        assert coaster_row["previous_revenue"] == 0.0
        assert pd.isna(coaster_row["pct_change"])

    def test_sorted_by_magnitude_of_dollar_change_descending(self) -> None:
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "SmallMover", "Quantity": 1, "Price": 1.0},
                {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "SmallMover", "Quantity": 1, "Price": 1.0},
                {"Order ID": 3, "Sale Date": "2026-01-01", "Item Name": "BigMover", "Quantity": 1, "Price": 100.0},
                {"Order ID": 4, "Sale Date": "2026-01-15", "Item Name": "BigMover", "Quantity": 10, "Price": 100.0},
            ]
        )

        result = product_performance_comparison(orders, period_days=10)

        assert list(result["Item Name"]) == ["BigMover", "SmallMover"]

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="Item Name"):
            product_performance_comparison(pd.DataFrame({"Price": [1]}))

    def test_empty_orders_returns_empty_dataframe(self) -> None:
        result = product_performance_comparison(_empty_orders(), period_days=10)

        assert list(result.columns) == ["Item Name", "recent_revenue", "previous_revenue", "pct_change"]
        assert result.empty


class TestRepeatCustomerRate:
    def test_computes_fraction_of_repeat_buyers(self) -> None:
        # alice: orders 1 and 3 (repeat); bob: order 2 only. 1/2 buyers repeat.
        assert repeat_customer_rate(SAMPLE_ORDERS) == pytest.approx(0.5)

    def test_empty_orders_is_zero(self) -> None:
        assert repeat_customer_rate(_empty_orders()) == 0.0

    def test_missing_buyer_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Buyer"):
            repeat_customer_rate(pd.DataFrame({"Order ID": [1, 2]}))

    def test_no_repeat_buyers_is_zero(self) -> None:
        single_order_per_buyer = _orders(
            [
                {
                    "Order ID": 1,
                    "Sale Date": "2026-01-01",
                    "Item Name": "Mug",
                    "Quantity": 1,
                    "Price": 10.0,
                    "Buyer": "alice",
                },
                {
                    "Order ID": 2,
                    "Sale Date": "2026-01-02",
                    "Item Name": "Mug",
                    "Quantity": 1,
                    "Price": 10.0,
                    "Buyer": "bob",
                },
            ]
        )

        assert repeat_customer_rate(single_order_per_buyer) == 0.0
