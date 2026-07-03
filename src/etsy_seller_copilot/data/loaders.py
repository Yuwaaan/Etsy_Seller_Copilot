"""Load Etsy CSV exports (Orders, Listings, Traffic) into Pandas DataFrames.

This module only loads and validates raw exports. It does not transform,
aggregate, or otherwise interpret the data.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from etsy_seller_copilot.data.utils import ORDER_COLUMN_ALIASES, map_columns


class EtsyDataError(Exception):
    """Base class for errors raised while loading Etsy CSV exports."""


class EtsyFileNotFoundError(EtsyDataError):
    """Raised when the given CSV path does not exist."""


class InvalidEtsyFileError(EtsyDataError):
    """Raised when the CSV file exists but is empty or cannot be parsed."""


class MissingColumnsError(EtsyDataError):
    """Raised when the CSV is missing one or more required columns."""


# Etsy "Orders" CSV export (Shop Manager > Settings > Options > Download Data).
ORDERS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Order ID",
    "Sale Date",
    "Item Name",
    "Quantity",
    "Price",
    "Currency",
)

# Etsy "Listings" CSV export/template, which uses UPPER_SNAKE_CASE headers.
LISTINGS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "TITLE",
    "DESCRIPTION",
    "PRICE",
    "CURRENCY_CODE",
    "QUANTITY",
    "TAGS",
)

# Etsy does not provide a standard downloadable CSV for shop traffic/stats
# (unlike Orders and Listings). This schema is a best-effort placeholder based
# on the metrics shown in Shop Manager > Stats, and may need to be adjusted
# once a real traffic export is available to validate against.
TRAFFIC_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "Listing ID",
    "Views",
    "Visits",
    "Orders",
    "Revenue",
)


def _load_csv(
    path: str | Path,
    dataset: str,
    required_columns: Sequence[str],
    *,
    apply_column_aliases: bool = False,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise EtsyFileNotFoundError(f"{dataset} file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise InvalidEtsyFileError(f"{dataset} file at {csv_path} is empty") from exc
    except pd.errors.ParserError as exc:
        raise InvalidEtsyFileError(
            f"{dataset} file at {csv_path} could not be parsed as CSV"
        ) from exc

    df.columns = df.columns.str.strip()
    if apply_column_aliases:
        # Best-effort: map header variants (e.g. "Qty", "Title") onto the
        # canonical names this app expects, without changing behavior when
        # the exact expected headers are already present.
        df = map_columns(df, ORDER_COLUMN_ALIASES)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        available = ", ".join(df.columns) if len(df.columns) else "(no columns found)"
        raise MissingColumnsError(
            f"{dataset} file at {csv_path} is missing required columns: "
            f"{', '.join(missing_columns)}. Columns found: {available}."
        )

    return df


def load_orders(path: str | Path) -> pd.DataFrame:
    """Load an Etsy Orders CSV export into a DataFrame.

    Column headers that look like known variants (e.g. "Qty" for "Quantity",
    "Title" for "Item Name") are mapped onto the canonical names below before
    validation, so mildly different export formats still work.
    """
    return _load_csv(path, "Orders", ORDERS_REQUIRED_COLUMNS, apply_column_aliases=True)


def load_listings(path: str | Path) -> pd.DataFrame:
    """Load an Etsy Listings CSV export into a DataFrame."""
    return _load_csv(path, "Listings", LISTINGS_REQUIRED_COLUMNS)


def load_traffic(path: str | Path) -> pd.DataFrame:
    """Load an Etsy Traffic/Stats CSV export into a DataFrame.

    Etsy has no standard "Download Data" export for traffic, so this schema
    is a placeholder pending validation against a real export.
    """
    return _load_csv(path, "Traffic", TRAFFIC_REQUIRED_COLUMNS)


def load_sample_orders() -> pd.DataFrame:
    """Return a small built-in sample Orders dataset for trying out the app.

    Defined in code (rather than read from a file) so it's always available,
    including on a fresh clone where local CSVs under ``data/raw/`` are
    gitignored.
    """
    return pd.DataFrame(
        [
            {"Order ID": 1001, "Sale Date": "2026-01-05", "Item Name": "Handmade Ceramic Mug", "Quantity": 2, "Price": 18.50, "Currency": "USD", "Buyer": "alice123"},
            {"Order ID": 1001, "Sale Date": "2026-01-05", "Item Name": "Ceramic Coaster Set", "Quantity": 1, "Price": 12.00, "Currency": "USD", "Buyer": "alice123"},
            {"Order ID": 1002, "Sale Date": "2026-01-18", "Item Name": "Handmade Ceramic Mug", "Quantity": 1, "Price": 18.50, "Currency": "USD", "Buyer": "bob456"},
            {"Order ID": 1003, "Sale Date": "2026-02-02", "Item Name": "Knit Wool Scarf", "Quantity": 1, "Price": 32.00, "Currency": "USD", "Buyer": "carol789"},
            {"Order ID": 1004, "Sale Date": "2026-02-14", "Item Name": "Handmade Ceramic Mug", "Quantity": 3, "Price": 18.50, "Currency": "USD", "Buyer": "alice123"},
            {"Order ID": 1005, "Sale Date": "2026-02-20", "Item Name": "Ceramic Coaster Set", "Quantity": 2, "Price": 12.00, "Currency": "USD", "Buyer": "dave321"},
            {"Order ID": 1006, "Sale Date": "2026-03-01", "Item Name": "Knit Wool Scarf", "Quantity": 1, "Price": 32.00, "Currency": "USD", "Buyer": "alice123"},
            {"Order ID": 1007, "Sale Date": "2026-03-15", "Item Name": "Handmade Ceramic Mug", "Quantity": 1, "Price": 18.50, "Currency": "USD", "Buyer": "erin654"},
        ]
    )
