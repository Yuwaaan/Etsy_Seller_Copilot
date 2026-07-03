"""Execute an ``AnalyticsPlan`` by calling the real analytics functions.

This module owns two things:
1. The mapping from ``Intent`` to the actual function in
   ``etsy_seller_copilot.analytics.metrics`` that fulfills it — every
   ``Intent`` has an entry here, including ``Intent.CONVERSION_RATE`` and
   ``Intent.UNKNOWN``, whose "value" is a canned explanatory message rather
   than a computed number, so the pipeline never has to fail outright.
2. Converting each function's pandas-shaped return value into a plain,
   JSON-safe result — the analytics layer itself stays pandas-native and is
   never modified to accommodate callers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from etsy_seller_copilot.analytics.metrics import (
    average_order_value,
    daily_sales_trend,
    number_of_orders,
    repeat_customer_rate,
    revenue_by_month,
    top_selling_listings,
    total_revenue,
)
from etsy_seller_copilot.services.intent import Intent
from etsy_seller_copilot.services.planner import AnalyticsPlan


class ListingRecord(TypedDict):
    item_name: str
    quantity_sold: int
    revenue: float


class ShopSummary(TypedDict):
    total_revenue: float
    number_of_orders: int
    average_order_value: float
    top_product: str | None


DispatchValue = float | int | str | dict[str, float] | list[ListingRecord] | ShopSummary


@dataclass(frozen=True)
class DispatchResult:
    intent: Intent
    value: DispatchValue


class DispatchError(Exception):
    """Raised when a plan's intent has no registered analytics function."""


CONVERSION_RATE_UNAVAILABLE_MESSAGE = (
    "I can't calculate conversion rate from this file: an Etsy Sold Orders "
    "CSV only lists completed orders, not shop visits or listing views, so "
    "there's no visitor count to divide orders by. Check Shop Manager > "
    "Stats in Etsy directly for conversion rate."
)

SUPPORTED_QUESTIONS_MESSAGE = (
    "I'm not sure how to answer that. Try asking about: total revenue, "
    "number of orders, average order value, top products, repeat customer "
    "rate, sales trend (daily or monthly), or ask for a shop summary."
)


def _dispatch_total_revenue(orders: pd.DataFrame) -> float:
    return total_revenue(orders)


def _dispatch_number_of_orders(orders: pd.DataFrame) -> int:
    return number_of_orders(orders)


def _dispatch_average_order_value(orders: pd.DataFrame) -> float:
    return average_order_value(orders)


def _dispatch_repeat_customer_rate(orders: pd.DataFrame) -> float:
    return repeat_customer_rate(orders)


def _dispatch_revenue_by_month(orders: pd.DataFrame) -> dict[str, float]:
    series = revenue_by_month(orders)
    return {str(period): float(value) for period, value in series.items()}


def _dispatch_daily_sales_trend(orders: pd.DataFrame) -> dict[str, float]:
    series = daily_sales_trend(orders)
    return {str(period): float(value) for period, value in series.items()}


def _dispatch_top_selling_listings(orders: pd.DataFrame, *, top_n: int = 10) -> list[ListingRecord]:
    listings = top_selling_listings(orders, top_n=top_n)
    return [
        ListingRecord(
            item_name=str(row["Item Name"]),
            quantity_sold=int(row["quantity_sold"]),
            revenue=float(row["revenue"]),
        )
        for _, row in listings.iterrows()
    ]


def _dispatch_shop_summary(orders: pd.DataFrame) -> ShopSummary:
    top = top_selling_listings(orders, top_n=1)
    top_product = str(top.iloc[0]["Item Name"]) if not top.empty else None
    return ShopSummary(
        total_revenue=total_revenue(orders),
        number_of_orders=number_of_orders(orders),
        average_order_value=average_order_value(orders),
        top_product=top_product,
    )


def _dispatch_conversion_rate(orders: pd.DataFrame) -> str:
    return CONVERSION_RATE_UNAVAILABLE_MESSAGE


def _dispatch_unknown(orders: pd.DataFrame) -> str:
    return SUPPORTED_QUESTIONS_MESSAGE


_DISPATCH_TABLE: dict[Intent, Callable[..., DispatchValue]] = {
    Intent.TOTAL_REVENUE: _dispatch_total_revenue,
    Intent.NUMBER_OF_ORDERS: _dispatch_number_of_orders,
    Intent.AVERAGE_ORDER_VALUE: _dispatch_average_order_value,
    Intent.REPEAT_CUSTOMER_RATE: _dispatch_repeat_customer_rate,
    Intent.REVENUE_BY_MONTH: _dispatch_revenue_by_month,
    Intent.DAILY_SALES_TREND: _dispatch_daily_sales_trend,
    Intent.TOP_SELLING_LISTINGS: _dispatch_top_selling_listings,
    Intent.SHOP_SUMMARY: _dispatch_shop_summary,
    Intent.CONVERSION_RATE: _dispatch_conversion_rate,
    Intent.UNKNOWN: _dispatch_unknown,
}


def dispatch(plan: AnalyticsPlan, orders: pd.DataFrame) -> DispatchResult:
    """Execute ``plan`` against ``orders`` and return a JSON-safe result."""
    handler = _DISPATCH_TABLE.get(plan.intent)
    if handler is None:
        raise DispatchError(f"No analytics function registered for intent: {plan.intent}")

    value = handler(orders, **plan.kwargs)
    return DispatchResult(intent=plan.intent, value=value)
