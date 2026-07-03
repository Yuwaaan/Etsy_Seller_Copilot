import pandas as pd
import pytest

from etsy_seller_copilot.agents.business_analyst import BusinessAnalystUnavailableError
from etsy_seller_copilot.agents.nodes import (
    business_analyst_node,
    detect_intent_node,
    dispatch_node,
    plan_node,
    respond_node,
)
from etsy_seller_copilot.services.dispatcher import DispatchResult
from etsy_seller_copilot.services.intent import DetectedIntent, Intent
from etsy_seller_copilot.services.planner import AnalyticsPlan

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


class TestBusinessAnalystNode:
    def test_sets_response_when_the_agent_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "etsy_seller_copilot.agents.nodes.run_business_analyst",
            lambda question, orders: "Your shop is doing well.",
        )
        state = {"question": "how am I doing?", "orders": ORDERS_WITH_BUYER}

        update = business_analyst_node(state)

        assert update == {"response": "Your shop is doing well."}

    def test_returns_empty_update_when_llm_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(question, orders):
            raise BusinessAnalystUnavailableError("no provider configured")

        monkeypatch.setattr("etsy_seller_copilot.agents.nodes.run_business_analyst", _raise)
        state = {"question": "how am I doing?", "orders": ORDERS_WITH_BUYER}

        assert business_analyst_node(state) == {}

    def test_returns_empty_update_on_any_runtime_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(question, orders):
            raise RuntimeError("simulated network outage")

        monkeypatch.setattr("etsy_seller_copilot.agents.nodes.run_business_analyst", _raise)
        state = {"question": "how am I doing?", "orders": ORDERS_WITH_BUYER}

        assert business_analyst_node(state) == {}

    def test_short_circuits_when_error_already_set(self) -> None:
        state = {"question": "irrelevant", "orders": ORDERS_WITH_BUYER, "error": "already broken"}

        assert business_analyst_node(state) == {}


class TestDetectIntentNode:
    def test_detects_intent_from_question(self) -> None:
        state = {"question": "What is my total revenue?", "orders": ORDERS_WITH_BUYER}

        update = detect_intent_node(state)

        assert update["detected_intent"].intent is Intent.TOTAL_REVENUE

    def test_short_circuits_when_error_already_set(self) -> None:
        state = {"question": "irrelevant", "orders": ORDERS_WITH_BUYER, "error": "already broken"}

        assert detect_intent_node(state) == {}


class TestPlanNode:
    def test_builds_plan_for_known_intent(self) -> None:
        state = {
            "detected_intent": DetectedIntent(intent=Intent.TOTAL_REVENUE, raw_text="x"),
        }

        update = plan_node(state)

        assert update["plan"] == AnalyticsPlan(intent=Intent.TOTAL_REVENUE)

    def test_builds_plan_for_unknown_intent_without_erroring(self) -> None:
        state = {
            "detected_intent": DetectedIntent(intent=Intent.UNKNOWN, raw_text="asdf"),
        }

        update = plan_node(state)

        assert "error" not in update
        assert update["plan"] == AnalyticsPlan(intent=Intent.UNKNOWN)

    def test_short_circuits_when_error_already_set(self) -> None:
        state = {"detected_intent": None, "error": "already broken"}

        assert plan_node(state) == {}


class TestDispatchNode:
    def test_executes_plan_against_orders(self) -> None:
        state = {
            "plan": AnalyticsPlan(intent=Intent.TOTAL_REVENUE),
            "orders": ORDERS_WITH_BUYER,
        }

        update = dispatch_node(state)

        assert update["result"] == DispatchResult(intent=Intent.TOTAL_REVENUE, value=30.0)

    def test_sets_error_when_analytics_function_raises(self) -> None:
        state = {
            "plan": AnalyticsPlan(intent=Intent.REPEAT_CUSTOMER_RATE),
            "orders": ORDERS_WITHOUT_BUYER,
        }

        update = dispatch_node(state)

        assert "error" in update
        assert "Buyer" in update["error"]

    def test_short_circuits_when_error_already_set(self) -> None:
        state = {"plan": None, "orders": ORDERS_WITH_BUYER, "error": "already broken"}

        assert dispatch_node(state) == {}


class TestRespondNode:
    def test_formats_successful_result(self) -> None:
        state = {
            "result": DispatchResult(intent=Intent.TOTAL_REVENUE, value=30.0),
        }

        update = respond_node(state)

        assert update["response"] == "Your total revenue is $30.00."

    def test_formats_error(self) -> None:
        state = {"error": "boom"}

        update = respond_node(state)

        assert update["response"] == "I couldn't answer that question: boom"
