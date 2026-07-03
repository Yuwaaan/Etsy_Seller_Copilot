"""LLM-driven Business Analyst Agent.

Given a natural-language business question, this runs a tool-calling loop:
the LLM decides which analytics tools to call and with what arguments, reads
their results, and either calls more tools or writes a final answer. No part
of *which* tools to call, in what order, or how many times is hardcoded here
-- that reasoning is entirely the LLM's job. The tools themselves
(``agents.analyst_tools``) are deterministic computations only, so every
number in the final answer is traceable back to a real analytics function.
"""

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from etsy_seller_copilot.agents.analyst_tools import build_analyst_tools
from etsy_seller_copilot.config.llm import get_chat_model

# Business investigation tools already bundle everything a task needs in one
# call, so most questions resolve in 1-2 rounds; this is bounded lower than
# before (was 6, when the model had to chain individual metric calls) but
# still generous enough for a question that legitimately needs two
# investigations, while stopping a confused model from looping forever.
MAX_TOOL_ITERATIONS = 4

SYSTEM_PROMPT = """\
You are an experienced Etsy business analyst helping a shop owner understand \
their sales data. You have tools that query their actual order data -- \
always call a tool to get real numbers before making any factual claim; \
never invent figures.

Your tools work at two levels:
- Direct lookups (get_total_revenue, get_number_of_orders, \
get_average_order_value, get_available_columns) for simple, single-number \
factual questions.
- Business investigation tools (investigate_sales_decline, \
business_health_check, analyze_customer_behavior, \
analyze_product_performance, analyze_revenue_trends, \
detect_business_anomalies) for open-ended or diagnostic questions. Each one \
already gathers every metric relevant to that investigation in a single \
call. Pick the investigation that matches the question and call it once -- \
do not call individual metrics one at a time when an investigation tool \
already covers the question; you will rarely need more than one or two tool \
calls in total.

For diagnostic or open-ended questions (e.g. "why are sales down", "is \
anything wrong", "how am I doing", "give me a health check"), structure your \
final answer with these sections:
- **Executive Summary**: 1-2 sentence overview.
- **Key Findings**: what the data shows, with specific numbers.
- **Possible Causes**: plausible explanations for what you found.
- **Recommendations**: concrete, prioritized next steps.

For simple, direct factual questions (e.g. "what's my total revenue"), just \
answer directly and briefly -- don't force the full structure onto a \
one-line answer.

If a question needs data this shop's Sold Orders export doesn't contain \
(e.g. conversion rate, traffic, ad spend, views), say so plainly, explain \
what's missing, and suggest what the seller should upload instead. Never \
respond with "I don't understand" or "I couldn't determine a plan" -- always \
reason your way to the closest useful answer from what's available, even if \
that answer is "here's what I'd need to answer that precisely."
"""


class BusinessAnalystUnavailableError(Exception):
    """Raised when no LLM is configured to run the Business Analyst Agent."""


def run_business_analyst(question: str, orders: pd.DataFrame, llm: BaseChatModel | None = None) -> str:
    """Answer a business question by letting the LLM plan and call analytics tools.

    Raises ``BusinessAnalystUnavailableError`` if no LLM is configured. The
    caller (``agents.nodes.business_analyst_node``) is expected to catch
    this -- and any runtime failure from the LLM call itself -- and fall
    back to the deterministic engine rather than let the request fail.
    """
    llm = llm if llm is not None else get_chat_model()
    if llm is None:
        raise BusinessAnalystUnavailableError("No LLM provider is configured.")

    tools = build_analyst_tools(orders)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    for _ in range(MAX_TOOL_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not _has_tool_calls(ai_message):
            return str(ai_message.content)

        for call in ai_message.tool_calls:
            messages.append(_invoke_tool(tools_by_name, call))

    # Ran out of tool-call rounds -- ask the model to wrap up with whatever
    # it has gathered so far, rather than silently truncating the answer.
    messages.append(HumanMessage(content="Summarize your findings and give your best answer now."))
    final_message = llm_with_tools.invoke(messages)
    return str(final_message.content)


def _has_tool_calls(message: AIMessage) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _invoke_tool(tools_by_name: dict[str, BaseTool], call: dict) -> ToolMessage:
    tool = tools_by_name.get(call["name"])
    if tool is None:
        content = f"Unknown tool: {call['name']}"
    else:
        try:
            content = str(tool.invoke(call["args"]))
        except Exception as exc:
            # Feed the failure back to the LLM as a tool result instead of
            # crashing the loop -- it's often something the model can reason
            # about (e.g. a missing column) and explain to the user.
            content = f"Tool '{call['name']}' failed: {exc}"
    return ToolMessage(content=content, tool_call_id=call["id"])
