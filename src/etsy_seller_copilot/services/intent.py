"""Detect which business question a user is asking, from raw text.

Deterministic, keyword/regex based — no LLM involved. Detection runs in two
passes:

1. Specific phrase patterns (e.g. "average order value", "top selling"),
   checked first and in most-specific-first order, so a precise phrase isn't
   swallowed by a more generic bucket.
2. A broader single-keyword fallback (e.g. a bare "orders", "revenue",
   "products") for short, imprecise queries that don't match any phrase.

If neither pass matches, ``Intent.UNKNOWN`` is returned. This is *not*
treated as an error anywhere downstream — ``services.dispatcher`` has a
handler for it that returns a friendly message listing what the assistant
can answer, so no user input ever produces a hard failure.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    TOTAL_REVENUE = "total_revenue"
    NUMBER_OF_ORDERS = "number_of_orders"
    AVERAGE_ORDER_VALUE = "average_order_value"
    REVENUE_BY_MONTH = "revenue_by_month"
    DAILY_SALES_TREND = "daily_sales_trend"
    TOP_SELLING_LISTINGS = "top_selling_listings"
    REPEAT_CUSTOMER_RATE = "repeat_customer_rate"
    CONVERSION_RATE = "conversion_rate"
    SHOP_SUMMARY = "shop_summary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedIntent:
    intent: Intent
    raw_text: str
    top_n: int | None = None


# Pass 1: specific phrases, most-specific-first so e.g. "average order value"
# is matched before a more generic pattern could claim it.
_PHRASE_PATTERNS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.CONVERSION_RATE,
        (
            r"\bconversion( rate)?\b",
            r"\bcvr\b",
        ),
    ),
    (
        Intent.AVERAGE_ORDER_VALUE,
        (
            r"\baverage order value\b",
            r"\baov\b",
            r"\baverage order\b",
            r"\baverage (sale|purchase|basket)\b",
        ),
    ),
    (
        Intent.REPEAT_CUSTOMER_RATE,
        (
            r"\brepeat customers?\b",
            r"\breturning customers?\b",
            r"\bcustomer retention\b",
            r"\brepeat (rate|buyers?)\b",
        ),
    ),
    (
        Intent.SHOP_SUMMARY,
        (
            r"\b(shop |overall )?summary\b",
            r"\boverview\b",
            r"\bperformance\b",
            r"\bhow('?s| is) my shop\b",
            r"\bhow am i doing\b",
        ),
    ),
    (
        Intent.TOP_SELLING_LISTINGS,
        (
            r"\btop[- ]?selling\b",
            r"\bbest[- ]?sell(ing|ers?)\b",
            r"\btop \d+\b",
            r"\btop (listings?|products?)\b",
            r"\bbestsellers?\b",
        ),
    ),
    (
        Intent.DAILY_SALES_TREND,
        (
            r"\bdaily\b",
            r"\brevenue (by|per|each) day\b",
            r"\bday[- ]over[- ]day\b",
        ),
    ),
    (
        Intent.REVENUE_BY_MONTH,
        (
            r"\brevenue (by|per|each) month\b",
            r"\bmonthly\b",
            r"\bmonth[- ]over[- ]month\b",
            r"\brevenue (trend|breakdown)\b",
            r"\bsales trend\b",
            r"\btrend\b",
        ),
    ),
    (
        Intent.NUMBER_OF_ORDERS,
        (
            r"\bhow many orders\b",
            r"\bnumber of orders\b",
            r"\border count\b",
            r"\btotal orders\b",
        ),
    ),
    (
        Intent.TOTAL_REVENUE,
        (
            r"\btotal revenue\b",
            r"\btotal sales\b",
            r"\bhow much (revenue|money|have i made)\b",
        ),
    ),
)

# Pass 2: fallback single-keyword matching, only consulted when nothing in
# pass 1 matched. This is what makes short/imprecise queries like a lone
# "orders" or "revenue" work, without letting those generic words hijack the
# more specific phrases handled above.
_KEYWORD_FALLBACKS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (Intent.NUMBER_OF_ORDERS, ("orders", "order")),
    (Intent.TOTAL_REVENUE, ("revenue", "sales", "money", "income", "earnings")),
    (Intent.TOP_SELLING_LISTINGS, ("products", "product", "listings", "listing", "bestsellers", "bestseller")),
)

_TOP_N_PATTERN = re.compile(r"\btop\s+(\d+)\b")


def detect_intent(text: str) -> DetectedIntent:
    """Classify a natural-language business question into an ``Intent``.

    Falls back to ``Intent.UNKNOWN`` when nothing matches at all — callers
    treat that as "explain what I can answer", not an error.
    """
    normalized = text.strip().lower()
    top_n = _extract_top_n(normalized)

    for intent, patterns in _PHRASE_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return DetectedIntent(intent=intent, raw_text=text, top_n=top_n)

    for intent, keywords in _KEYWORD_FALLBACKS:
        if any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords):
            return DetectedIntent(intent=intent, raw_text=text, top_n=top_n)

    return DetectedIntent(intent=Intent.UNKNOWN, raw_text=text, top_n=top_n)


def _extract_top_n(normalized_text: str) -> int | None:
    match = _TOP_N_PATTERN.search(normalized_text)
    if match is None:
        return None
    return int(match.group(1))
