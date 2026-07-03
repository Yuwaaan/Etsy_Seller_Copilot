"""Convert a detected intent into a concrete analytics execution plan.

This is where business rules about *how* to fulfill an intent live: default
and validate parameters. It never touches pandas or the analytics layer
directly — that's the dispatcher's job.
"""

from dataclasses import dataclass, field

from etsy_seller_copilot.services.intent import DetectedIntent, Intent

DEFAULT_TOP_N = 10
MIN_TOP_N = 1
MAX_TOP_N = 50


@dataclass(frozen=True)
class AnalyticsPlan:
    intent: Intent
    kwargs: dict[str, int] = field(default_factory=dict)


def build_plan(detected: DetectedIntent) -> AnalyticsPlan:
    """Build an ``AnalyticsPlan`` from a ``DetectedIntent``.

    Every ``Intent`` — including ``Intent.UNKNOWN`` — has a registered
    handler in ``services.dispatcher``, so this always produces a usable
    plan. An unrecognized question still gets answered, just with a message
    explaining what the assistant can help with instead of a computed value.
    """
    if detected.intent is Intent.TOP_SELLING_LISTINGS:
        top_n = detected.top_n if detected.top_n is not None else DEFAULT_TOP_N
        top_n = max(MIN_TOP_N, min(top_n, MAX_TOP_N))
        return AnalyticsPlan(intent=detected.intent, kwargs={"top_n": top_n})

    return AnalyticsPlan(intent=detected.intent)
