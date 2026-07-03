import pandas as pd
import pytest

from etsy_seller_copilot.data.utils import (
    FriendlyColumnError,
    check_required_columns,
    clean_orders,
    map_columns,
)


class TestMapColumns:
    def test_leaves_exact_canonical_columns_untouched(self) -> None:
        df = pd.DataFrame(columns=["Order ID", "Item Name", "Quantity", "Price"])

        result = map_columns(df)

        assert list(result.columns) == ["Order ID", "Item Name", "Quantity", "Price"]

    def test_maps_known_aliases_case_and_spacing_insensitively(self) -> None:
        df = pd.DataFrame(
            [{"Order Number": 1, "  Title ": "Mug", "QTY": 2, "Sale Amount": 10.0}]
        )

        result = map_columns(df)

        assert set(["Order ID", "Item Name", "Quantity", "Price"]).issubset(result.columns)

    def test_does_not_clobber_existing_canonical_column(self) -> None:
        # "Title" would normally map to "Item Name", but it's already present.
        df = pd.DataFrame([{"Item Name": "Mug", "Title": "Other Title"}])

        result = map_columns(df)

        assert list(result.columns) == ["Item Name", "Title"]


class TestCheckRequiredColumns:
    def test_passes_when_all_required_present(self) -> None:
        df = pd.DataFrame(columns=["Item Name", "Quantity", "Price"])
        check_required_columns(df)  # should not raise

    def test_raises_friendly_error_listing_missing_and_available(self) -> None:
        df = pd.DataFrame(columns=["Order ID", "Currency"])

        with pytest.raises(FriendlyColumnError) as exc_info:
            check_required_columns(df)

        message = str(exc_info.value)
        assert "Item Name" in message
        assert "Quantity" in message
        assert "Price" in message
        assert "Order ID" in message  # lists available columns too


class TestCleanOrders:
    def test_maps_aliases_coerces_types_and_returns_no_warnings_for_clean_data(self) -> None:
        df = pd.DataFrame(
            [
                {"Title": "Mug", "Qty": "2", "Order Value": "10.00", "Order Number": 1, "Sale Date": "2026-01-05"},
            ]
        )

        cleaned, warnings = clean_orders(df)

        assert cleaned.loc[0, "Item Name"] == "Mug"
        assert cleaned.loc[0, "Quantity"] == 2.0
        assert cleaned.loc[0, "Price"] == 10.0
        assert warnings == []

    def test_raises_friendly_error_on_missing_required_columns(self) -> None:
        df = pd.DataFrame([{"Order ID": 1}])

        with pytest.raises(FriendlyColumnError):
            clean_orders(df)

    def test_raises_friendly_error_on_completely_empty_csv(self) -> None:
        df = pd.DataFrame()

        with pytest.raises(FriendlyColumnError):
            clean_orders(df)

    def test_drops_duplicate_rows_with_warning(self) -> None:
        df = pd.DataFrame(
            [
                {"Item Name": "Mug", "Quantity": 1, "Price": 10.0},
                {"Item Name": "Mug", "Quantity": 1, "Price": 10.0},
            ]
        )

        cleaned, warnings = clean_orders(df)

        assert len(cleaned) == 1
        assert any("duplicate" in warning.lower() for warning in warnings)

    def test_drops_non_numeric_price_or_quantity_with_warning(self) -> None:
        df = pd.DataFrame(
            [
                {"Item Name": "Mug", "Quantity": "two", "Price": 10.0},
                {"Item Name": "Coaster", "Quantity": 1, "Price": "not-a-price"},
                {"Item Name": "Scarf", "Quantity": 1, "Price": 32.0},
            ]
        )

        cleaned, warnings = clean_orders(df)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["Item Name"] == "Scarf"
        assert any("non-numeric" in warning.lower() for warning in warnings)

    def test_strips_currency_symbols_from_price(self) -> None:
        df = pd.DataFrame([{"Order ID": 1, "Item Name": "Mug", "Quantity": 1, "Price": "$18.50"}])

        cleaned, warnings = clean_orders(df)

        assert cleaned.iloc[0]["Price"] == 18.50
        assert warnings == []

    def test_unparseable_dates_produce_warning_but_keep_row(self) -> None:
        df = pd.DataFrame(
            [{"Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Sale Date": "not-a-date"}]
        )

        cleaned, warnings = clean_orders(df)

        assert len(cleaned) == 1
        assert pd.isna(cleaned.iloc[0]["Sale Date"])
        assert any("date" in warning.lower() for warning in warnings)

    def test_missing_order_id_is_synthesized_with_warning(self) -> None:
        df = pd.DataFrame([{"Item Name": "Mug", "Quantity": 1, "Price": 10.0}])

        cleaned, warnings = clean_orders(df)

        assert "Order ID" in cleaned.columns
        assert cleaned.iloc[0]["Order ID"] == 1
        assert any("order id" in warning.lower() for warning in warnings)
