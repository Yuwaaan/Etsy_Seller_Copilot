import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from etsy_seller_copilot.agents.graph import build_graph

ORDERS_WITH_BUYER = pd.DataFrame(
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

ORDERS_WITHOUT_BUYER = ORDERS_WITH_BUYER.drop(columns=["Buyer"])


class TestGraphEndToEnd:
    def test_happy_path_produces_final_response(self) -> None:
        graph = build_graph()

        final_state = graph.invoke({"question": "What is my total revenue?", "orders": ORDERS_WITH_BUYER})

        assert final_state.get("error") is None
        assert final_state["response"] == "Your total revenue is $30.00."

    def test_top_selling_listings_with_extracted_top_n(self) -> None:
        graph = build_graph()

        final_state = graph.invoke(
            {"question": "Show me my top 1 best sellers", "orders": ORDERS_WITH_BUYER}
        )

        assert final_state["response"] == "Here are your top-selling listings:\n- Mug: 3 sold ($30.00)"

    def test_unrecognized_question_yields_friendly_help_not_an_error(self) -> None:
        graph = build_graph()

        final_state = graph.invoke({"question": "What's the weather like?", "orders": ORDERS_WITH_BUYER})

        assert final_state.get("error") is None
        assert "total revenue" in final_state["response"]

    def test_short_query_orders_is_understood(self) -> None:
        graph = build_graph()

        final_state = graph.invoke({"question": "orders", "orders": ORDERS_WITH_BUYER})

        assert final_state.get("error") is None
        assert final_state["response"] == "You have 2 orders."

    def test_conversion_rate_query_explains_data_is_unavailable(self) -> None:
        graph = build_graph()

        final_state = graph.invoke({"question": "conversion rate", "orders": ORDERS_WITH_BUYER})

        assert final_state.get("error") is None
        assert "conversion rate" in final_state["response"].lower()

    def test_summary_query_returns_shop_summary(self) -> None:
        graph = build_graph()

        final_state = graph.invoke({"question": "summary", "orders": ORDERS_WITH_BUYER})

        assert final_state.get("error") is None
        assert "summary" in final_state["response"].lower()

    def test_dispatch_failure_is_handled_gracefully(self) -> None:
        graph = build_graph()

        final_state = graph.invoke(
            {"question": "What is my repeat customer rate?", "orders": ORDERS_WITHOUT_BUYER}
        )

        assert "couldn't answer" in final_state["response"]
        assert "Buyer" in final_state["response"]


class _FakeChatModel:
    """Minimal test double for the tool-calling chat model interface."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


class _FailingChatModel:
    """Simulates a configured-but-unreachable LLM (network error, bad key, ...)."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("simulated LLM outage")


class TestGraphWithLLMConfigured:
    def test_uses_business_analyst_agent_when_llm_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_llm = _FakeChatModel(
            [
                AIMessage(content="This asks about overall performance; I have enough context to answer."),
                AIMessage(content="Your shop is trending up this month."),
            ]
        )
        monkeypatch.setattr("etsy_seller_copilot.agents.business_analyst.get_chat_model", lambda: fake_llm)

        graph = build_graph()
        final_state = graph.invoke({"question": "how is my business doing?", "orders": ORDERS_WITH_BUYER})

        assert final_state["response"] == "Your shop is trending up this month."

    def test_falls_back_to_deterministic_pipeline_when_llm_call_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "etsy_seller_copilot.agents.business_analyst.get_chat_model", lambda: _FailingChatModel()
        )

        graph = build_graph()
        final_state = graph.invoke({"question": "What is my total revenue?", "orders": ORDERS_WITH_BUYER})

        # Even though an LLM is "configured", its failure at call time still
        # produces a real answer via the deterministic fallback -- not a crash.
        assert final_state["response"] == "Your total revenue is $30.00."
