"""Reusable analytics functions over Etsy order data.

These functions take plain Pandas DataFrames (not file paths), so they work
regardless of where the data came from and stay decoupled from the loading
layer in ``etsy_seller_copilot.data``.

Every function assumes one row per order *line item* (an order with multiple
items has multiple rows sharing the same "Order ID"), and that "Price" is a
per-unit price rather than a line total, so revenue is always computed as
``Price * Quantity``.
"""

from collections.abc import Sequence
from typing import TypedDict

import pandas as pd


class MetricComparison(TypedDict):
    recent: float
    previous: float
    pct_change: float | None  # None when `previous` is 0 (percent change is undefined)


class PeriodComparison(TypedDict):
    period_days: int
    recent_start: str | None
    recent_end: str | None
    previous_start: str | None
    previous_end: str | None
    revenue: MetricComparison
    orders: MetricComparison
    average_order_value: MetricComparison


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} requires column(s) {', '.join(missing)}, "
            f"which are not present in the given DataFrame"
        )


def _line_item_revenue(orders: pd.DataFrame) -> pd.Series:
    _require_columns(orders, ["Price", "Quantity"], "revenue calculations")
    return orders["Price"] * orders["Quantity"]


def total_revenue(orders: pd.DataFrame) -> float:
    """Total revenue across all order line items (Price * Quantity, summed)."""
    return float(_line_item_revenue(orders).sum())


def number_of_orders(orders: pd.DataFrame) -> int:
    """Count of distinct orders (unique "Order ID" values, not rows)."""
    _require_columns(orders, ["Order ID"], "number_of_orders")
    return int(orders["Order ID"].nunique())


def average_order_value(orders: pd.DataFrame) -> float:
    """Total revenue divided by number of orders; 0.0 when there are no orders."""
    orders_count = number_of_orders(orders)
    if orders_count == 0:
        return 0.0
    return total_revenue(orders) / orders_count


def revenue_by_month(orders: pd.DataFrame) -> pd.Series:
    """Revenue grouped by calendar month, sorted chronologically.

    Returns a Series with a monthly ``PeriodIndex`` and float revenue values.
    """
    _require_columns(orders, ["Sale Date"], "revenue_by_month")
    if orders.empty:
        return pd.Series(dtype=float, name="revenue")

    revenue = _line_item_revenue(orders)
    months = pd.to_datetime(orders["Sale Date"]).dt.to_period("M")
    return revenue.groupby(months).sum().sort_index().rename("revenue")


def top_selling_listings(orders: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Listings ranked by revenue, with quantity sold alongside.

    Returns a DataFrame with columns ["Item Name", "quantity_sold", "revenue"],
    sorted by revenue descending and limited to ``top_n`` rows.
    """
    _require_columns(orders, ["Item Name", "Price", "Quantity"], "top_selling_listings")
    if orders.empty:
        return pd.DataFrame(columns=["Item Name", "quantity_sold", "revenue"])

    working = orders.assign(revenue=_line_item_revenue(orders))
    grouped = (
        working.groupby("Item Name", as_index=False)
        .agg(quantity_sold=("Quantity", "sum"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    return grouped.head(top_n)


def quantity_sold_by_product(orders: pd.DataFrame) -> pd.Series:
    """Total units sold per item, sorted by quantity descending."""
    _require_columns(orders, ["Item Name", "Quantity"], "quantity_sold_by_product")
    if orders.empty:
        return pd.Series(dtype=float, name="quantity_sold")

    return (
        orders.groupby("Item Name")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .rename("quantity_sold")
    )


def revenue_by_product(orders: pd.DataFrame) -> pd.Series:
    """Total revenue per item (Price * Quantity, summed), sorted descending."""
    _require_columns(orders, ["Item Name", "Price", "Quantity"], "revenue_by_product")
    if orders.empty:
        return pd.Series(dtype=float, name="revenue")

    working = orders.assign(revenue=_line_item_revenue(orders))
    return working.groupby("Item Name")["revenue"].sum().sort_values(ascending=False).rename("revenue")


def daily_sales_trend(orders: pd.DataFrame) -> pd.Series:
    """Revenue grouped by calendar day, sorted chronologically.

    Returns a Series with a daily ``PeriodIndex`` and float revenue values.
    """
    _require_columns(orders, ["Sale Date"], "daily_sales_trend")
    if orders.empty:
        return pd.Series(dtype=float, name="revenue")

    revenue = _line_item_revenue(orders)
    days = pd.to_datetime(orders["Sale Date"]).dt.to_period("D")
    return revenue.groupby(days).sum().sort_index().rename("revenue")


def monthly_sales_trend(orders: pd.DataFrame) -> pd.Series:
    """Revenue grouped by calendar month, sorted chronologically.

    Alias for :func:`revenue_by_month`, named to match how sellers ask for it.
    """
    return revenue_by_month(orders)


def repeat_customer_rate(orders: pd.DataFrame) -> float:
    """Fraction of unique buyers with more than one distinct order.

    Returns 0.0 when there are no buyers, to avoid dividing by zero.
    """
    _require_columns(orders, ["Buyer", "Order ID"], "repeat_customer_rate")
    if orders.empty:
        return 0.0

    orders_per_buyer = orders.groupby("Buyer")["Order ID"].nunique()
    total_buyers = len(orders_per_buyer)
    if total_buyers == 0:
        return 0.0

    repeat_buyers = int((orders_per_buyer > 1).sum())
    return repeat_buyers / total_buyers


def _pct_change(current: float, prior: float) -> float | None:
    """Percent change from `prior` to `current`; None when `prior` is 0 (undefined, not infinite)."""
    if prior == 0:
        return None
    return (current - prior) / prior * 100


def _split_into_periods(orders: pd.DataFrame, period_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Split `orders` into a recent window and the equal-length window before it.

    The recent window ends at the latest "Sale Date" *in the data* (not
    today's real-world date), so comparisons are correct on historical or
    sample exports, not just live ones. Rows with an unparseable date are
    excluded from both windows (they can't be placed in time).
    """
    dates = pd.to_datetime(orders["Sale Date"], errors="coerce")
    valid = orders.loc[dates.notna()].assign(_sale_date=dates.loc[dates.notna()])

    latest = valid["_sale_date"].max()
    recent_start = latest - pd.Timedelta(days=period_days - 1)
    previous_start = recent_start - pd.Timedelta(days=period_days)
    previous_end = recent_start - pd.Timedelta(days=1)

    recent = valid[valid["_sale_date"] >= recent_start]
    previous = valid[(valid["_sale_date"] >= previous_start) & (valid["_sale_date"] <= previous_end)]
    return recent, previous, recent_start, latest, previous_start


def compare_periods(orders: pd.DataFrame, period_days: int = 30) -> PeriodComparison:
    """Compare revenue, order count, and average order value between the most
    recent `period_days` days of data and the equal-length period before it.

    This is the building block for "how am I trending" / "has anything
    changed" style questions: it's plain period-over-period comparison, not
    statistical anomaly detection, so every number in the result is directly
    traceable back to the underlying orders.
    """
    _require_columns(orders, ["Sale Date", "Price", "Quantity", "Order ID"], "compare_periods")

    if orders.empty or pd.to_datetime(orders["Sale Date"], errors="coerce").notna().sum() == 0:
        empty_metric: MetricComparison = {"recent": 0.0, "previous": 0.0, "pct_change": None}
        return {
            "period_days": period_days,
            "recent_start": None,
            "recent_end": None,
            "previous_start": None,
            "previous_end": None,
            "revenue": dict(empty_metric),
            "orders": {"recent": 0, "previous": 0, "pct_change": None},
            "average_order_value": dict(empty_metric),
        }

    recent, previous, recent_start, latest, previous_start = _split_into_periods(orders, period_days)

    recent_revenue, previous_revenue = total_revenue(recent), total_revenue(previous)
    recent_orders, previous_orders = number_of_orders(recent), number_of_orders(previous)
    recent_aov, previous_aov = average_order_value(recent), average_order_value(previous)

    return {
        "period_days": period_days,
        "recent_start": str(recent_start.date()),
        "recent_end": str(latest.date()),
        "previous_start": str(previous_start.date()),
        "previous_end": str((recent_start - pd.Timedelta(days=1)).date()),
        "revenue": {
            "recent": recent_revenue,
            "previous": previous_revenue,
            "pct_change": _pct_change(recent_revenue, previous_revenue),
        },
        "orders": {
            "recent": recent_orders,
            "previous": previous_orders,
            "pct_change": _pct_change(recent_orders, previous_orders),
        },
        "average_order_value": {
            "recent": recent_aov,
            "previous": previous_aov,
            "pct_change": _pct_change(recent_aov, previous_aov),
        },
    }


def product_performance_comparison(orders: pd.DataFrame, period_days: int = 30) -> pd.DataFrame:
    """Per-product revenue in the most recent `period_days` days vs the equal-
    length period before it, with percent change.

    Returns a DataFrame with columns ["Item Name", "recent_revenue",
    "previous_revenue", "pct_change"], sorted by the size of the dollar
    change (largest swings, up or down, first) so the most notable movers
    surface at the top.
    """
    _require_columns(orders, ["Sale Date", "Item Name", "Price", "Quantity"], "product_performance_comparison")

    columns = ["Item Name", "recent_revenue", "previous_revenue", "pct_change"]
    if orders.empty or pd.to_datetime(orders["Sale Date"], errors="coerce").notna().sum() == 0:
        return pd.DataFrame(columns=columns)

    recent, previous, *_ = _split_into_periods(orders, period_days)
    recent_rev = revenue_by_product(recent) if not recent.empty else pd.Series(dtype=float)
    previous_rev = revenue_by_product(previous) if not previous.empty else pd.Series(dtype=float)

    all_products = sorted(set(recent_rev.index) | set(previous_rev.index))
    rows = [
        {
            "Item Name": product,
            "recent_revenue": float(recent_rev.get(product, 0.0)),
            "previous_revenue": float(previous_rev.get(product, 0.0)),
            "pct_change": _pct_change(float(recent_rev.get(product, 0.0)), float(previous_rev.get(product, 0.0))),
        }
        for product in all_products
    ]

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result

    abs_change = (result["recent_revenue"] - result["previous_revenue"]).abs()
    return result.loc[abs_change.sort_values(ascending=False).index].reset_index(drop=True)
