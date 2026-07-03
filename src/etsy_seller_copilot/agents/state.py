"""Shared runtime state for the LangGraph orchestration layer.

This only holds data nodes need to pass to each other between steps. The
actual intent/plan/result types are owned by the services layer; this module
just assembles them into one state schema instead of redefining them.
"""

from typing import NotRequired, TypedDict

import pandas as pd

from etsy_seller_copilot.services.dispatcher import DispatchResult
from etsy_seller_copilot.services.intent import DetectedIntent
from etsy_seller_copilot.services.planner import AnalyticsPlan


class AgentState(TypedDict):
    # Required at invocation time.
    question: str
    orders: pd.DataFrame

    # Filled in progressively as nodes run.
    detected_intent: NotRequired[DetectedIntent | None]
    plan: NotRequired[AnalyticsPlan | None]
    result: NotRequired[DispatchResult | None]
    response: NotRequired[str | None]
    error: NotRequired[str | None]
