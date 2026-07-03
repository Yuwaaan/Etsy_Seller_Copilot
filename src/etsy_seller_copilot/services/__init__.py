import pandas as pd

from .dispatcher import (
    DispatchError,
    DispatchResult,
    ListingRecord,
    ShopSummary,
    dispatch,
)
from .intent import DetectedIntent, Intent, detect_intent
from .planner import AnalyticsPlan, build_plan

__all__ = [
    "AnalyticsPlan",
    "DetectedIntent",
    "DispatchError",
    "DispatchResult",
    "Intent",
    "ListingRecord",
    "ShopSummary",
    "answer",
    "build_plan",
    "detect_intent",
    "dispatch",
]


def answer(text: str, orders: pd.DataFrame) -> DispatchResult:
    """Detect intent, build a plan, and execute it — the full reasoning layer.

    This is the entry point future callers (a LangGraph node, a CLI, tests)
    are expected to use, rather than composing the three steps themselves.
    Always returns a ``DispatchResult``, even for an unrecognized question
    (``Intent.UNKNOWN``) — its value is then a friendly message explaining
    what the assistant can help with, rather than an exception.
    """
    detected = detect_intent(text)
    plan = build_plan(detected)
    return dispatch(plan, orders)
