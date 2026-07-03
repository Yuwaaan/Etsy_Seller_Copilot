import pandas as pd

from etsy_seller_copilot.agents.health_report import _NarrativeOutput, generate_shop_health_report

GROWING_SHOP_ORDERS = pd.DataFrame(
    [
        {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "alice"},
        {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 3, "Price": 10.0, "Buyer": "bob"},
        {"Order ID": 3, "Sale Date": "2026-01-20", "Item Name": "Mug", "Quantity": 3, "Price": 10.0, "Buyer": "carol"},
    ]
)


class FakeStructuredLLM:
    def __init__(self, output: _NarrativeOutput) -> None:
        self._output = output

    def invoke(self, messages):
        return self._output


class FakeChatModelWithStructuredOutput:
    def __init__(self, output: _NarrativeOutput) -> None:
        self._output = output

    def with_structured_output(self, schema):
        assert schema is _NarrativeOutput
        return FakeStructuredLLM(self._output)


class FailingChatModel:
    def with_structured_output(self, schema):
        raise RuntimeError("simulated LLM outage")


class TestGenerateShopHealthReport:
    def test_falls_back_to_templated_narrative_when_no_llm(self) -> None:
        report = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=None)

        assert report.ai_generated is False
        assert isinstance(report.score, int)
        assert 0 <= report.score <= 100
        assert report.executive_summary  # non-empty
        assert len(report.recommendations) >= 1

    def test_uses_llm_narrative_when_available(self) -> None:
        fake_output = _NarrativeOutput(
            executive_summary="Business is trending up nicely.",
            recommendations=["Restock Mug", "Run a promo for new customers"],
        )
        fake_llm = FakeChatModelWithStructuredOutput(fake_output)

        report = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=fake_llm)

        assert report.ai_generated is True
        assert report.executive_summary == "Business is trending up nicely."
        assert report.recommendations == ("Restock Mug", "Run a promo for new customers")

    def test_score_is_identical_with_and_without_llm(self) -> None:
        # The score must come from the deterministic layer either way -- the
        # LLM should never be able to change the actual number.
        fake_output = _NarrativeOutput(executive_summary="whatever", recommendations=["whatever"])
        fake_llm = FakeChatModelWithStructuredOutput(fake_output)

        without_llm = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=None)
        with_llm = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=fake_llm)

        assert without_llm.score == with_llm.score
        assert without_llm.score_breakdown == with_llm.score_breakdown

    def test_falls_back_gracefully_when_llm_call_fails(self) -> None:
        report = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=FailingChatModel())

        assert report.ai_generated is False
        assert report.executive_summary  # still produced something usable

    def test_score_breakdown_is_always_present_and_explains_each_component(self) -> None:
        report = generate_shop_health_report(GROWING_SHOP_ORDERS, period_days=10, llm=None)

        assert len(report.score_breakdown) > 0
        for line in report.score_breakdown:
            assert "pts" in line
