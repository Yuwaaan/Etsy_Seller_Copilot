import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from etsy_seller_copilot.agents.business_analyst import (
    MAX_TOOL_ITERATIONS,
    BusinessAnalystUnavailableError,
    run_business_analyst,
)

SAMPLE_ORDERS = pd.DataFrame(
    [
        {"Order ID": 1, "Sale Date": "2026-01-05", "Item Name": "Mug", "Quantity": 2, "Price": 10.0, "Buyer": "alice"},
        {"Order ID": 2, "Sale Date": "2026-02-10", "Item Name": "Coaster", "Quantity": 3, "Price": 5.0, "Buyer": "bob"},
    ]
)


class FakeChatModel:
    """Minimal test double for a LangChain chat model's tool-calling interface.

    Only implements what ``run_business_analyst`` actually calls
    (``bind_tools`` and ``invoke``), returning a scripted sequence of
    ``AIMessage`` objects -- no network, no real provider needed.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.bound_tool_names: list[str] | None = None
        self.invoke_call_count = 0

    def bind_tools(self, tools):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.invoke_call_count += 1
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of scripted responses")
        return self._responses.pop(0)


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id}


class TestRunBusinessAnalyst:
    def test_raises_when_no_llm_available(self) -> None:
        with pytest.raises(BusinessAnalystUnavailableError):
            run_business_analyst("what is my revenue", SAMPLE_ORDERS, llm=None)

    def test_answers_directly_when_no_tool_call_is_needed(self) -> None:
        fake = FakeChatModel([AIMessage(content="Hi! Ask me about your shop.")])

        result = run_business_analyst("hello", SAMPLE_ORDERS, llm=fake)

        assert result == "Hi! Ask me about your shop."
        assert fake.invoke_call_count == 1

    def test_calls_a_tool_and_uses_its_result(self) -> None:
        fake = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[_tool_call("get_total_revenue", {})]),
                AIMessage(content="Your total revenue is $35.00."),
            ]
        )

        result = run_business_analyst("what's my revenue", SAMPLE_ORDERS, llm=fake)

        assert result == "Your total revenue is $35.00."
        assert fake.invoke_call_count == 2

    def test_handles_multiple_tool_calls_in_one_round(self) -> None:
        fake = FakeChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("get_total_revenue", {}, call_id="call_1"),
                        _tool_call("get_number_of_orders", {}, call_id="call_2"),
                    ],
                ),
                AIMessage(content="You have 2 orders totaling $35.00."),
            ]
        )

        result = run_business_analyst("give me an overview", SAMPLE_ORDERS, llm=fake)

        assert result == "You have 2 orders totaling $35.00."

    def test_feeds_tool_error_back_to_the_llm_instead_of_crashing(self) -> None:
        orders_without_buyer = SAMPLE_ORDERS.drop(columns=["Buyer"])
        fake = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[_tool_call("analyze_customer_behavior", {})]),
                AIMessage(content="I can't compute repeat customer rate -- there's no Buyer column."),
            ]
        )

        result = run_business_analyst("repeat rate?", orders_without_buyer, llm=fake)

        assert "Buyer column" in result

    def test_unknown_tool_call_does_not_crash(self) -> None:
        fake = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[_tool_call("not_a_real_tool", {})]),
                AIMessage(content="Sorry, I couldn't find that data."),
            ]
        )

        result = run_business_analyst("do something weird", SAMPLE_ORDERS, llm=fake)

        assert result == "Sorry, I couldn't find that data."

    def test_stops_after_max_iterations_and_asks_for_a_final_summary(self) -> None:
        # One tool-call round per iteration, exhausting the whole budget.
        looping_responses = [
            AIMessage(content="", tool_calls=[_tool_call("get_total_revenue", {})]) for _ in range(MAX_TOOL_ITERATIONS)
        ]
        fake = FakeChatModel([*looping_responses, AIMessage(content="Best guess: revenue is around $35.")])

        result = run_business_analyst("keep digging", SAMPLE_ORDERS, llm=fake)

        assert result == "Best guess: revenue is around $35."
        assert fake.invoke_call_count == MAX_TOOL_ITERATIONS + 1

    def test_binds_the_full_analyst_tool_set(self) -> None:
        fake = FakeChatModel([AIMessage(content="ok")])

        run_business_analyst("hello", SAMPLE_ORDERS, llm=fake)

        assert fake.bound_tool_names is not None
        assert "get_total_revenue" in fake.bound_tool_names
        assert "investigate_sales_decline" in fake.bound_tool_names
