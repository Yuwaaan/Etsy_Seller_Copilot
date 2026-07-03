"""LangGraph node functions.

Every node is a thin wrapper around a lower layer: call it, translate its
outcome (or its specific exception) into a state update.

``business_analyst_node`` is the primary path: it lets the LLM plan, call
analytics tools, and write the final answer -- no node decides *which*
analytics function to call, that's entirely the LLM's reasoning inside
``agents.business_analyst``. If no LLM is configured, or the tool-calling
loop fails at runtime for any reason, this node leaves ``response`` unset
rather than raising; ``agents.graph`` then routes to the original
deterministic ``detect_intent`` -> ``plan`` -> ``dispatch`` -> ``respond``
pipeline as a fallback, unchanged from before the Business Analyst Agent
existed.

Once ``error`` is set on state, every deterministic-path node except
``respond_node`` becomes a no-op, so that pipeline doesn't need conditional
edges of its own to short-circuit a failed run.
"""

from etsy_seller_copilot.agents.business_analyst import run_business_analyst
from etsy_seller_copilot.agents.prompts import format_error_response, format_response
from etsy_seller_copilot.agents.state import AgentState
from etsy_seller_copilot.services.dispatcher import DispatchError, dispatch
from etsy_seller_copilot.services.intent import detect_intent
from etsy_seller_copilot.services.planner import build_plan


def business_analyst_node(state: AgentState) -> dict:
    """Try the LLM-driven Business Analyst Agent.

    On success, sets ``response`` directly. On any failure (no LLM
    configured, network error, malformed tool call, ...) it returns an empty
    update, leaving ``response`` unset so ``agents.graph`` routes to the
    deterministic fallback pipeline instead of failing the request.
    """
    if state.get("error"):
        return {}
    try:
        return {"response": run_business_analyst(state["question"], state["orders"])}
    except Exception:
        return {}


def detect_intent_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    return {"detected_intent": detect_intent(state["question"])}


def plan_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    return {"plan": build_plan(state["detected_intent"])}


def dispatch_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    try:
        result = dispatch(state["plan"], state["orders"])
    except (DispatchError, ValueError) as exc:
        return {"error": str(exc)}
    return {"result": result}


def respond_node(state: AgentState) -> dict:
    error = state.get("error")
    if error:
        return {"response": format_error_response(error)}

    result = state["result"]
    return {"response": format_response(result.intent, result.value)}
