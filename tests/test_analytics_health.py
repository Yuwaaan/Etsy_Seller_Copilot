import pandas as pd

from etsy_seller_copilot.analytics.health import BASE_SCORE, compute_health_score


def _orders(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestComputeHealthScore:
    def test_growing_shop_scores_above_baseline_and_lists_improving_signals(self) -> None:
        orders = _orders(
            [
                # Previous 10 days: modest activity.
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "alice"},
                {"Order ID": 2, "Sale Date": "2026-01-05", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "bob"},
                # Recent 10 days: revenue and orders both roughly doubled.
                {"Order ID": 3, "Sale Date": "2026-01-12", "Item Name": "Mug", "Quantity": 2, "Price": 10.0, "Buyer": "carol"},
                {"Order ID": 4, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 2, "Price": 10.0, "Buyer": "dave"},
                {"Order ID": 5, "Sale Date": "2026-01-20", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "alice"},
                {"Order ID": 6, "Sale Date": "2026-01-20", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "erin"},
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert health.score > BASE_SCORE
        assert any("Revenue" in signal for signal in health.improving)
        assert health.declining == ()

    def test_shrinking_shop_scores_below_baseline_and_lists_declining_signals(self) -> None:
        orders = _orders(
            [
                # Previous 10 days: strong activity.
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 5, "Price": 10.0},
                {"Order ID": 2, "Sale Date": "2026-01-05", "Item Name": "Mug", "Quantity": 5, "Price": 10.0},
                # Recent 10 days: revenue collapsed.
                {"Order ID": 3, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert health.score < BASE_SCORE
        assert any("Revenue" in signal for signal in health.declining)

    def test_score_is_always_between_0_and_100(self) -> None:
        # Extreme swing (previous period had one tiny sale, recent has a lot)
        # shouldn't push the score out of range.
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 1.0},
                *[
                    {"Order ID": i, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 10, "Price": 100.0}
                    for i in range(2, 20)
                ],
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert 0 <= health.score <= 100

    def test_insufficient_date_history_returns_baseline_with_risk_note(self) -> None:
        orders = _orders([{"Order ID": 1, "Sale Date": "not-a-date", "Item Name": "Mug", "Quantity": 1, "Price": 10.0}])

        health = compute_health_score(orders, period_days=30)

        assert health.score == int(BASE_SCORE)
        assert any("enough" in risk.lower() for risk in health.risks)

    def test_missing_required_columns_returns_baseline_with_risk_note(self) -> None:
        health = compute_health_score(pd.DataFrame({"Price": [1.0]}), period_days=30)

        assert health.score == int(BASE_SCORE)
        assert len(health.risks) == 1

    def test_low_repeat_rate_is_flagged_as_a_risk(self) -> None:
        orders = _orders(
            [
                {"Order ID": i, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": f"buyer{i}"}
                for i in range(10)
            ]
        )

        health = compute_health_score(orders, period_days=30)

        assert any("repeat" in risk.lower() for risk in health.risks)

    def test_missing_buyer_column_does_not_error_and_skips_repeat_rate_component(self) -> None:
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
                {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert all(c.label != "Repeat customer rate" for c in health.components)

    def test_big_product_revenue_drop_is_flagged_as_a_risk_with_product_name(self) -> None:
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Scarf", "Quantity": 10, "Price": 10.0},
                {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Scarf", "Quantity": 1, "Price": 10.0},
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert any("Scarf" in risk for risk in health.risks)

    def test_components_always_explain_their_contribution(self) -> None:
        orders = _orders(
            [
                {"Order ID": 1, "Sale Date": "2026-01-01", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "alice"},
                {"Order ID": 2, "Sale Date": "2026-01-15", "Item Name": "Mug", "Quantity": 1, "Price": 10.0, "Buyer": "bob"},
            ]
        )

        health = compute_health_score(orders, period_days=10)

        assert len(health.components) > 0
        for component in health.components:
            assert component.label
            assert component.explanation
