import pandas as pd

from etsy_seller_copilot.services import answer
from etsy_seller_copilot.services.intent import Intent

SAMPLE_ORDERS = pd.DataFrame(
    [
        {
            "Order ID": 1,
            "Sale Date": "2026-01-05",
            "Item Name": "Mug",
            "Quantity": 2,
            "Price": 10.0,
            "Buyer": "alice",
        },
        {
            "Order ID": 2,
            "Sale Date": "2026-01-10",
            "Item Name": "Mug",
            "Quantity": 1,
            "Price": 10.0,
            "Buyer": "bob",
        },
    ]
)


def test_answer_runs_full_pipeline_for_recognized_question() -> None:
    result = answer("What is my total revenue?", SAMPLE_ORDERS)

    assert result.intent is Intent.TOTAL_REVENUE
    assert result.value == 30.0


def test_answer_returns_friendly_help_for_unrecognized_question() -> None:
    result = answer("What's the weather like?", SAMPLE_ORDERS)

    assert result.intent is Intent.UNKNOWN
    assert isinstance(result.value, str)
    assert "total revenue" in result.value
