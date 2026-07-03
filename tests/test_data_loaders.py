from pathlib import Path

import pandas as pd
import pytest

from etsy_seller_copilot.data.loaders import (
    EtsyFileNotFoundError,
    InvalidEtsyFileError,
    MissingColumnsError,
    load_listings,
    load_orders,
    load_sample_orders,
    load_traffic,
)

ORDERS_HEADER = "Order ID,Sale Date,Item Name,Quantity,Price,Currency"
ORDERS_ROW = "12345,2026-01-15,Handmade Mug,1,18.50,USD"

LISTINGS_HEADER = "TITLE,DESCRIPTION,PRICE,CURRENCY_CODE,QUANTITY,TAGS"
LISTINGS_ROW = "Handmade Mug,A lovely mug,18.50,USD,10,mug;ceramic;gift"

TRAFFIC_HEADER = "Date,Listing ID,Views,Visits,Orders,Revenue"
TRAFFIC_ROW = "2026-01-15,987654,120,45,3,55.50"

LOADER_CASES = [
    pytest.param(load_orders, ORDERS_HEADER, ORDERS_ROW, "Order ID", id="orders"),
    pytest.param(load_listings, LISTINGS_HEADER, LISTINGS_ROW, "TITLE", id="listings"),
    pytest.param(load_traffic, TRAFFIC_HEADER, TRAFFIC_ROW, "Date", id="traffic"),
]


@pytest.mark.parametrize("loader, header, row, first_column", LOADER_CASES)
def test_load_success(tmp_path: Path, loader, header: str, row: str, first_column: str) -> None:
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(f"{header}\n{row}\n")

    df = loader(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert first_column in df.columns


@pytest.mark.parametrize("loader, header, row, first_column", LOADER_CASES)
def test_load_missing_required_column(
    tmp_path: Path, loader, header: str, row: str, first_column: str
) -> None:
    # Drop the last required column from both the header and the data row.
    truncated_header = ",".join(header.split(",")[:-1])
    truncated_row = ",".join(row.split(",")[:-1])
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(f"{truncated_header}\n{truncated_row}\n")

    with pytest.raises(MissingColumnsError):
        loader(csv_path)


@pytest.mark.parametrize("loader, header, row, first_column", LOADER_CASES)
def test_load_file_not_found(
    tmp_path: Path, loader, header: str, row: str, first_column: str
) -> None:
    missing_path = tmp_path / "does-not-exist.csv"

    with pytest.raises(EtsyFileNotFoundError):
        loader(missing_path)


@pytest.mark.parametrize("loader, header, row, first_column", LOADER_CASES)
def test_load_empty_file(
    tmp_path: Path, loader, header: str, row: str, first_column: str
) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    with pytest.raises(InvalidEtsyFileError):
        loader(csv_path)


def test_load_orders_strips_header_whitespace(tmp_path: Path) -> None:
    header = "Order ID , Sale Date,Item Name,Quantity,Price,Currency"
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(f"{header}\n{ORDERS_ROW}\n")

    df = load_orders(csv_path)

    assert "Order ID" in df.columns


def test_load_orders_zero_rows_is_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(f"{ORDERS_HEADER}\n")

    df = load_orders(csv_path)

    assert len(df) == 0
    assert "Order ID" in df.columns


def test_missing_columns_error_lists_missing_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("Order ID,Sale Date\n1,2026-01-15\n")

    with pytest.raises(MissingColumnsError, match="Item Name"):
        load_orders(csv_path)


def test_load_orders_maps_known_column_aliases(tmp_path: Path) -> None:
    header = "Order Number,Order Date,Title,Qty,Order Value,Currency Code"
    row = "12345,2026-01-15,Handmade Mug,1,18.50,USD"
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(f"{header}\n{row}\n")

    df = load_orders(csv_path)

    assert list(df.loc[0, ["Order ID", "Sale Date", "Item Name", "Quantity", "Price", "Currency"]]) == [
        12345,
        "2026-01-15",
        "Handmade Mug",
        1,
        18.50,
        "USD",
    ]


def test_load_sample_orders_returns_usable_dataframe() -> None:
    df = load_sample_orders()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for column in ("Order ID", "Sale Date", "Item Name", "Quantity", "Price"):
        assert column in df.columns
