from etsy_seller_copilot.services.intent import DetectedIntent, Intent
from etsy_seller_copilot.services.planner import (
    DEFAULT_TOP_N,
    MAX_TOP_N,
    MIN_TOP_N,
    build_plan,
)


class TestBuildPlan:
    def test_unknown_intent_produces_a_usable_plan_not_an_error(self) -> None:
        detected = DetectedIntent(intent=Intent.UNKNOWN, raw_text="asdf")

        plan = build_plan(detected)

        assert plan.intent is Intent.UNKNOWN
        assert plan.kwargs == {}

    def test_simple_intent_produces_empty_kwargs(self) -> None:
        detected = DetectedIntent(intent=Intent.TOTAL_REVENUE, raw_text="total revenue")

        plan = build_plan(detected)

        assert plan.intent is Intent.TOTAL_REVENUE
        assert plan.kwargs == {}

    def test_top_selling_listings_defaults_top_n_when_unspecified(self) -> None:
        detected = DetectedIntent(intent=Intent.TOP_SELLING_LISTINGS, raw_text="best sellers")

        plan = build_plan(detected)

        assert plan.kwargs == {"top_n": DEFAULT_TOP_N}

    def test_top_selling_listings_uses_extracted_top_n(self) -> None:
        detected = DetectedIntent(
            intent=Intent.TOP_SELLING_LISTINGS, raw_text="top 5 listings", top_n=5
        )

        plan = build_plan(detected)

        assert plan.kwargs == {"top_n": 5}

    def test_top_n_is_clamped_to_minimum(self) -> None:
        detected = DetectedIntent(
            intent=Intent.TOP_SELLING_LISTINGS, raw_text="top 0 listings", top_n=0
        )

        plan = build_plan(detected)

        assert plan.kwargs == {"top_n": MIN_TOP_N}

    def test_top_n_is_clamped_to_maximum(self) -> None:
        detected = DetectedIntent(
            intent=Intent.TOP_SELLING_LISTINGS, raw_text="top 9999 listings", top_n=9999
        )

        plan = build_plan(detected)

        assert plan.kwargs == {"top_n": MAX_TOP_N}
