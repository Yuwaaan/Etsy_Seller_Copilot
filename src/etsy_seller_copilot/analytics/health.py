"""Deterministic shop health scoring.

Computes a 0-100 health score from real period-over-period metrics -- never
a hardcoded or hallucinated number -- plus a plain-English breakdown of
exactly how each component contributed. This is the grounding layer for the
"AI Insights" dashboard: the score and its explanation always come from this
module, whether or not an LLM is available to add narrative on top.
"""

from dataclasses import dataclass, field

import pandas as pd

from etsy_seller_copilot.analytics.metrics import (
    compare_periods,
    product_performance_comparison,
    repeat_customer_rate,
)

BASE_SCORE = 50.0

# How much each trend can move the score, in either direction.
_REVENUE_MAX_POINTS = 15.0
_ORDERS_MAX_POINTS = 9.0
_AOV_MAX_POINTS = 6.0
_REPEAT_RATE_MAX_POINTS = 10.0

# A trend has to move at least this much to be called out as "improving" or
# "declining" -- small day-to-day noise shouldn't read as a signal.
_NOTABLE_PCT_CHANGE = 5.0

# A product revenue drop of at least this magnitude (and coming from a
# non-trivial base) is flagged as a risk worth naming individually.
_PRODUCT_DROP_RISK_THRESHOLD_PCT = -50.0
_PRODUCT_DROP_MIN_PREVIOUS_REVENUE = 1.0
_MAX_PRODUCT_RISKS = 3


@dataclass(frozen=True)
class ScoreComponent:
    """One term in the health score, always paired with why it was applied."""

    label: str
    points: float
    explanation: str


@dataclass(frozen=True)
class HealthScore:
    score: int
    components: tuple[ScoreComponent, ...] = field(default_factory=tuple)
    improving: tuple[str, ...] = field(default_factory=tuple)
    declining: tuple[str, ...] = field(default_factory=tuple)
    risks: tuple[str, ...] = field(default_factory=tuple)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_trend(
    components: list[ScoreComponent],
    improving: list[str],
    declining: list[str],
    *,
    label: str,
    pct_change: float | None,
    max_points: float,
    period_days: int,
) -> None:
    """Score one period-over-period trend and record it as improving/declining if notable."""
    if pct_change is None:
        return

    points = _clamp(pct_change, -100.0, 100.0) / 100.0 * max_points
    direction = "up" if pct_change >= 0 else "down"
    components.append(
        ScoreComponent(
            label=label,
            points=points,
            explanation=f"{label} is {direction} {abs(pct_change):.1f}% vs the previous {period_days} days.",
        )
    )
    if pct_change >= _NOTABLE_PCT_CHANGE:
        improving.append(f"{label} is up {pct_change:.1f}% vs the previous {period_days} days.")
    elif pct_change <= -_NOTABLE_PCT_CHANGE:
        declining.append(f"{label} is down {abs(pct_change):.1f}% vs the previous {period_days} days.")


def compute_health_score(orders: pd.DataFrame, period_days: int = 30) -> HealthScore:
    """Score overall shop health from 0-100 using period-over-period trends.

    Falls back to a neutral baseline score with a clear "not enough data"
    risk note when there isn't enough dated order history to compare two
    periods -- it never fabricates a number.
    """
    components: list[ScoreComponent] = [
        ScoreComponent(label="Baseline", points=BASE_SCORE, explanation="Starting point before trend signals are applied.")
    ]
    improving: list[str] = []
    declining: list[str] = []
    risks: list[str] = []

    try:
        comparison = compare_periods(orders, period_days=period_days)
    except ValueError as exc:
        risks.append(f"Can't measure recent trends: {exc}")
        return HealthScore(score=int(BASE_SCORE), components=tuple(components), risks=tuple(risks))

    if comparison["recent_start"] is None:
        risks.append("Not enough dated order history to measure recent trends.")
        return HealthScore(score=int(BASE_SCORE), components=tuple(components), risks=tuple(risks))

    _score_trend(
        components, improving, declining,
        label="Revenue", pct_change=comparison["revenue"]["pct_change"],
        max_points=_REVENUE_MAX_POINTS, period_days=period_days,
    )
    _score_trend(
        components, improving, declining,
        label="Order count", pct_change=comparison["orders"]["pct_change"],
        max_points=_ORDERS_MAX_POINTS, period_days=period_days,
    )
    _score_trend(
        components, improving, declining,
        label="Average order value", pct_change=comparison["average_order_value"]["pct_change"],
        max_points=_AOV_MAX_POINTS, period_days=period_days,
    )

    try:
        rate = repeat_customer_rate(orders)
        # Reward being *high*, not just "changing" -- a stable 40% repeat
        # rate is good even though it isn't a period-over-period trend.
        rate_points = _clamp((rate * 100.0) - 20.0, -_REPEAT_RATE_MAX_POINTS, _REPEAT_RATE_MAX_POINTS)
        components.append(
            ScoreComponent(
                label="Repeat customer rate",
                points=rate_points,
                explanation=f"{rate:.1%} of customers are repeat buyers.",
            )
        )
        if rate >= 0.3:
            improving.append(f"Repeat customer rate is healthy at {rate:.1%}.")
        elif rate < 0.1:
            risks.append(f"Repeat customer rate is low ({rate:.1%}); most buyers aren't coming back.")
    except ValueError:
        pass  # No Buyer column -- simply skip this component, don't penalize for it.

    try:
        performance = product_performance_comparison(orders, period_days=period_days)
        big_drops = performance[
            performance["pct_change"].notna()
            & (performance["pct_change"] <= _PRODUCT_DROP_RISK_THRESHOLD_PCT)
            & (performance["previous_revenue"] >= _PRODUCT_DROP_MIN_PREVIOUS_REVENUE)
        ]
        for _, row in big_drops.head(_MAX_PRODUCT_RISKS).iterrows():
            risks.append(f"\"{row['Item Name']}\" revenue dropped {abs(row['pct_change']):.0f}% vs the previous period.")
    except ValueError:
        pass  # No Item Name / Sale Date -- skip product-level risk detection.

    score = int(round(_clamp(sum(c.points for c in components), 0.0, 100.0)))
    return HealthScore(
        score=score,
        components=tuple(components),
        improving=tuple(improving),
        declining=tuple(declining),
        risks=tuple(risks),
    )
