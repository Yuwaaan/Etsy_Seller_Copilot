"""Turn a dispatch result into a natural-language response.

No LLM is involved here — these are plain string templates. The name
"prompts" reflects their role (producing the user-facing text), not an
implementation detail; a future LLM-driven node can live alongside this
without changing it.
"""

from etsy_seller_copilot.services.dispatcher import DispatchValue
from etsy_seller_copilot.services.intent import Intent

_SCALAR_TEMPLATES: dict[Intent, str] = {
    Intent.TOTAL_REVENUE: "Your total revenue is ${value:,.2f}.",
    Intent.NUMBER_OF_ORDERS: "You have {value} orders.",
    Intent.AVERAGE_ORDER_VALUE: "Your average order value is ${value:,.2f}.",
    Intent.REPEAT_CUSTOMER_RATE: "{value:.1%} of your customers are repeat buyers.",
}

# These intents' dispatched value is already a complete, user-facing message
# (there's no computed number to format) — pass it through unchanged. This is
# how an unrecognized question or a "conversion rate" request get answered
# without ever going through the error path.
_PASSTHROUGH_MESSAGE_INTENTS = {Intent.CONVERSION_RATE, Intent.UNKNOWN}

_PERIOD_BREAKDOWN_LABELS: dict[Intent, str] = {
    Intent.REVENUE_BY_MONTH: "month",
    Intent.DAILY_SALES_TREND: "day",
}


def format_response(intent: Intent, value: DispatchValue) -> str:
    """Format a successful ``DispatchResult.value`` into a user-facing message."""
    if intent in _PASSTHROUGH_MESSAGE_INTENTS:
        assert isinstance(value, str)
        return value

    template = _SCALAR_TEMPLATES.get(intent)
    if template is not None:
        return template.format(value=value)

    period_label = _PERIOD_BREAKDOWN_LABELS.get(intent)
    if period_label is not None:
        assert isinstance(value, dict)
        if not value:
            return f"There's no revenue data to break down by {period_label} yet."
        lines = "\n".join(f"- {period}: ${amount:,.2f}" for period, amount in value.items())
        return f"Here is your revenue by {period_label}:\n{lines}"

    if intent is Intent.TOP_SELLING_LISTINGS:
        assert isinstance(value, list)
        if not value:
            return "There are no listings with sales yet."
        lines = "\n".join(
            f"- {item['item_name']}: {item['quantity_sold']} sold (${item['revenue']:,.2f})"
            for item in value
        )
        return f"Here are your top-selling listings:\n{lines}"

    if intent is Intent.SHOP_SUMMARY:
        assert isinstance(value, dict)
        lines = [
            f"- Total revenue: ${value['total_revenue']:,.2f}",
            f"- Orders: {value['number_of_orders']}",
            f"- Average order value: ${value['average_order_value']:,.2f}",
        ]
        if value.get("top_product"):
            lines.append(f"- Top product: {value['top_product']}")
        return "Here's a quick summary of your shop:\n" + "\n".join(lines)

    raise ValueError(f"No response template registered for intent: {intent}")


def format_error_response(error: str) -> str:
    """Format a pipeline error into a user-facing message."""
    return f"I couldn't answer that question: {error}"
