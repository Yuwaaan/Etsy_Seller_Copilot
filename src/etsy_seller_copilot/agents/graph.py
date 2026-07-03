"""Graph orchestration for the Etsy business-question agent.

This module only wires nodes together -- all decision-making and
computation live in the layers ``agents.nodes`` calls into.

Every question is first attempted by the LLM-driven ``business_analyst``
node. If it produces a ``response`` (the normal case, given a configured
LLM), the graph ends there. If it doesn't -- no LLM configured, or the
LLM/tool-calling loop failed at runtime -- the graph falls through to the
original deterministic ``detect_intent`` -> ``plan`` -> ``dispatch`` ->
``respond`` pipeline, exactly as it worked before the Business Analyst
Agent was introduced.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from etsy_seller_copilot.agents.nodes import (
    business_analyst_node,
    detect_intent_node,
    dispatch_node,
    plan_node,
    respond_node,
)
from etsy_seller_copilot.agents.state import AgentState


def _route_after_business_analyst(state: AgentState) -> str:
    """END if the Business Analyst Agent already answered; otherwise fall
    through to the deterministic pipeline."""
    return END if state.get("response") is not None else "detect_intent"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("business_analyst", business_analyst_node)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "business_analyst")
    graph.add_conditional_edges(
        "business_analyst",
        _route_after_business_analyst,
        {END: END, "detect_intent": "detect_intent"},
    )
    graph.add_edge("detect_intent", "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
