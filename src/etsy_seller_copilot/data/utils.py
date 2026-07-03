"""CSV cleaning and flexible column mapping for uploaded Etsy order exports.

Real-world Etsy "Sold Orders" CSV exports vary in header naming/casing across
shops, export tools, and Etsy CSV format changes over time. This module
normalizes headers and maps common synonyms (e.g. "Qty" -> "Quantity") onto
the canonical column names the rest of the app expects, then coerces types
and drops bad rows rather than letting the app crash.
"""

import re
from collections.abc import Sequence

import pandas as pd

# Only these are truly required to compute any insight at all. "Order ID",
# "Sale Date", "Currency", and "Buyer" are nice-to-have: the app degrades
# individual features (e.g. no sales-trend chart) rather than refusing the
# whole file when they're missing.
REQUIRED_ORDER_COLUMNS: tuple[str, ...] = ("Item Name", "Quantity", "Price")

# Canonical column name -> alternate header spellings seen in the wild,
# normalized (lowercase, non-alphanumerics collapsed to single spaces).
ORDER_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Order ID": ("order id", "order number", "order no", "order #"),
    "Sale Date": ("sale date", "date", "order date", "purchase date", "date paid"),
    "Item Name": ("item name", "title", "item title", "product name", "product", "listing title"),
    "Quantity": ("quantity", "qty", "quantity purchased", "quantity sold", "item quantity"),
    "Price": (
        "price",
        "order value",
        "sale amount",
        "subtotal",
        "item total",
        "price per unit",
        "unit price",
        "item price",
    ),
    "Currency": ("currency", "currency code"),
    "Buyer": ("buyer", "buyer username", "buyer name", "customer", "customer name"),
}


class FriendlyColumnError(ValueError):
    """Raised when a CSV can't be used, with a message meant to be shown as-is in the UI."""


def _normalize(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


def map_columns(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]] = ORDER_COLUMN_ALIASES) -> pd.DataFrame:
    """Rename columns that look like known aliases onto their canonical name.

    Only renames onto a canonical name that isn't already present, and only
    the first matching column is used per canonical name, so this never maps
    two different columns onto the same target.
    """
    df = df.rename(columns=lambda c: str(c).strip())

    normalized_lookup: dict[str, str] = {}
    for column in reversed(df.columns):
        normalized_lookup[_normalize(column)] = column

    rename_map: dict[str, str] = {}
    for canonical, alias_list in aliases.items():
        if canonical in df.columns:
            continue
        for alias in alias_list:
            actual = normalized_lookup.get(alias)
            if actual is not None:
                rename_map[actual] = canonical
                break

    return df.rename(columns=rename_map)


def check_required_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_ORDER_COLUMNS) -> None:
    """Raise a friendly, actionable error if any required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if not missing:
        return

    available = ", ".join(str(c) for c in df.columns) if len(df.columns) else "(no columns found)"
    raise FriendlyColumnError(
        f"We couldn't find the column(s) {', '.join(missing)} in your CSV.\n\n"
        f"Columns found in your file: {available}.\n\n"
        "An Etsy Sold Orders CSV needs a column for each of: item name "
        "(e.g. 'Item Name', 'Title'), quantity (e.g. 'Quantity', 'Qty'), and "
        "price (e.g. 'Price', 'Order Value', 'Subtotal'). Rename the "
        "relevant column(s) and re-upload, or check that you exported the "
        "'Sold Orders' report from Etsy (Shop Manager > Settings > Options > "
        "Download Data)."
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # Strip currency symbols/commas/spaces (e.g. "$18.50", "1,200") before
    # parsing, so common price formatting doesn't get treated as invalid.
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", None)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map columns, validate, and coerce types for an uploaded orders CSV.

    Returns the cleaned DataFrame plus a list of human-readable warnings
    describing any rows that were dropped or values that couldn't be parsed,
    so the caller can surface them instead of the app failing outright.
    """
    if df.empty and len(df.columns) == 0:
        raise FriendlyColumnError(
            "This CSV file appears to be empty. Please upload an Etsy Sold "
            "Orders export with at least a header row."
        )

    working = map_columns(df)
    check_required_columns(working)

    warnings: list[str] = []

    duplicate_count = int(working.duplicated().sum())
    if duplicate_count:
        working = working.drop_duplicates().reset_index(drop=True)
        warnings.append(f"Removed {duplicate_count} duplicate row(s).")

    working["Quantity"] = _coerce_numeric(working["Quantity"])
    working["Price"] = _coerce_numeric(working["Price"])

    bad_numeric = working["Quantity"].isna() | working["Price"].isna()
    bad_numeric_count = int(bad_numeric.sum())
    if bad_numeric_count:
        working = working.loc[~bad_numeric].reset_index(drop=True)
        warnings.append(f"Dropped {bad_numeric_count} row(s) with a non-numeric quantity or price.")

    if "Sale Date" in working.columns:
        parsed_dates = pd.to_datetime(working["Sale Date"], errors="coerce")
        bad_date_count = int(parsed_dates.isna().sum())
        if bad_date_count:
            warnings.append(
                f"{bad_date_count} row(s) have a date we couldn't parse; they're "
                "included in totals but excluded from sales-trend charts."
            )
        working["Sale Date"] = parsed_dates

    if "Order ID" not in working.columns:
        working.insert(0, "Order ID", range(1, len(working) + 1))
        warnings.append("No order id column found; treating each row as its own order.")

    return working, warnings
