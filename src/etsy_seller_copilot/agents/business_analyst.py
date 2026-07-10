"""LLM-driven Business Analyst Agent.

Given a natural-language business question, this runs a plan -> act -> reflect
loop: the LLM first reasons about the question with no tools available (so it
can't reach for one instead of thinking), then decides which analytics tools
to call and with what arguments, reflects on each result, and either
continues investigating or writes a final answer. No part of *which* tools to
call, in what order, or how many times is hardcoded here -- that reasoning is
entirely the LLM's job. The tools themselves (``agents.analyst_tools``) are
deterministic computations only, so every number in the final answer is
traceable back to a real analytics function.
"""

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from etsy_seller_copilot.agents.analyst_tools import build_analyst_tools
from etsy_seller_copilot.config.llm import get_chat_model

# A genuine investigation needs room for: gather evidence, reflect on whether
# it's sufficient, follow up if not -- often more than one round for an
# open-ended question. Bounded well above the common case so real multi-step
# investigation isn't cut short, while still stopping a confused model from
# looping forever.
MAX_TOOL_ITERATIONS = 6

PLANNING_PROMPT = """\
Before using any tools, think through this question like an analyst would: \
in 2-4 sentences, explain what the seller is really asking and what \
specific information you'd need to answer it well. Do not call any tools \
yet -- just reason about the question."""

REFLECTION_NUDGE = """\
Given the evidence so far, do you have enough to answer confidently? If \
yes, write your final answer now instead of calling another tool. If not, \
briefly say what's still unclear or missing, then continue investigating \
with another tool call."""

SYSTEM_PROMPT = """\
You are an experienced Etsy business analyst helping a shop owner understand \
their sales data.

Work the way a real analyst would:
1. First, think about what the question is actually asking and what \
information would answer it well -- before touching any tool.
2. Decide what evidence you need, and which of your tools would provide it.
3. Call a tool to gather that evidence.
4. After each tool result, reflect: do you have enough evidence now, or is \
something still unclear? Only call another tool if the reflection genuinely \
calls for it -- don't call a tool without a specific reason tied to what \
you're still missing.
5. Once you have enough evidence, write your final answer.

Every factual or numerical claim in your final answer must be grounded in a \
tool result -- never invent a figure, even one that sounds plausible. \
Reasoning about the question, forming hypotheses, and deciding what to \
check next does not require a tool call; only stating a number or fact \
about this shop's data does.

Your tools work at two levels:
- Direct lookups (get_total_revenue, get_number_of_orders, \
get_average_order_value, get_available_columns) for simple, single-number \
factual questions.
- Business investigation tools (investigate_sales_decline, \
business_health_check, analyze_customer_behavior, \
analyze_product_performance, analyze_revenue_trends, \
detect_business_anomalies) for open-ended or diagnostic questions. Each one \
gathers several related metrics in a single call. For a genuinely \
open-ended question, it's normal and expected to call more than one \
investigation tool, or follow up with another after reflecting that the \
first result didn't fully answer it -- don't force a diagnostic question \
into a single tool call if the evidence doesn't support a confident answer \
yet.

For diagnostic or open-ended questions (e.g. "why are sales down", "is \
anything wrong", "how am I doing", "give me a health check"), structure your \
final answer with these sections:
- **Executive Summary**: 1-2 sentence overview.
- **Key Findings**: what the data shows, with specific numbers.
- **Possible Causes**: plausible explanations for what you found.
- **Recommendations**: concrete, prioritized next steps.

For simple, direct factual questions (e.g. "what's my total revenue"), just \
answer directly and briefly -- don't force the full structure onto a \
one-line answer, and don't force multi-step investigation onto a question \
that only needs one number.

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

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    # Planning turn: no tools are bound yet, so the model can't reach for one
    # instead of reasoning -- it must respond in prose. That plan becomes
    # part of the context for every tool-calling turn that follows.
    messages.append(HumanMessage(content=PLANNING_PROMPT))
    plan_message = llm.invoke(messages)
    messages.append(plan_message)

    llm_with_tools = llm.bind_tools(tools)

    for i in range(MAX_TOOL_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not _has_tool_calls(ai_message):
            return str(ai_message.content)

        for call in ai_message.tool_calls:
            messages.append(_invoke_tool(tools_by_name, call))

        # Prompt an explicit stop/continue judgment after every round except
        # the last, where the forced-summary message below takes over.
        if i < MAX_TOOL_ITERATIONS - 1:
            messages.append(HumanMessage(content=REFLECTION_NUDGE))

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
