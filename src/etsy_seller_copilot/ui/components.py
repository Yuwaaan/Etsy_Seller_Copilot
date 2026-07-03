"""Reusable Streamlit UI components.

Everything here is UI plumbing/presentation: it calls into the existing
data, analytics, and agent layers rather than reimplementing any of them.
"""

import pandas as pd
import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from streamlit.runtime.uploaded_file_manager import UploadedFile

from etsy_seller_copilot.agents.health_report import ShopHealthReport, generate_shop_health_report
from etsy_seller_copilot.analytics.metrics import (
    average_order_value,
    number_of_orders,
    repeat_customer_rate,
    top_selling_listings,
    total_revenue,
)
from etsy_seller_copilot.data.utils import FriendlyColumnError, clean_orders


def render_file_uploader() -> UploadedFile | None:
    return st.file_uploader("Upload your Etsy Sold Orders CSV export", type="csv")


def load_uploaded_orders(uploaded_file: UploadedFile) -> tuple[pd.DataFrame, list[str]]:
    """Parse an uploaded CSV and run it through the cleaning/mapping pipeline.

    Returns the cleaned DataFrame plus a list of warnings (e.g. dropped rows)
    to surface to the user. Raises ``FriendlyColumnError`` with a message
    that's safe to show directly in the UI if the file can't be used at all.
    """
    try:
        raw = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise FriendlyColumnError(
            "This CSV file appears to be empty. Please upload a non-empty "
            "Etsy Sold Orders export."
        ) from exc
    except pd.errors.ParserError as exc:
        raise FriendlyColumnError(
            "This file couldn't be read as a CSV. Please make sure you're "
            "uploading the Sold Orders CSV export from Etsy."
        ) from exc

    return clean_orders(raw)


def render_data_preview(orders: pd.DataFrame) -> None:
    """Show the first few rows of the cleaned data, so sellers can sanity-check it."""
    st.subheader("Data Preview")
    st.caption(f"Showing the first 10 of {len(orders)} cleaned row(s).")
    st.dataframe(orders.head(10), use_container_width=True)


def render_metrics(orders: pd.DataFrame) -> None:
    """Render the four headline metrics as Streamlit metric tiles, with explanations."""
    columns = st.columns(4)

    columns[0].metric("Total Revenue", f"${total_revenue(orders):,.2f}")
    columns[0].caption("Sum of price × quantity across every order line item.")

    columns[1].metric("Total Orders", f"{number_of_orders(orders)}")
    columns[1].caption("Number of distinct orders (an order with multiple items counts once).")

    columns[2].metric("Average Order Value", f"${average_order_value(orders):,.2f}")
    columns[2].caption("Total revenue divided by number of orders.")

    try:
        rate = repeat_customer_rate(orders)
        columns[3].metric("Repeat Customer Rate", f"{rate:.1%}")
        columns[3].caption("Share of buyers who placed more than one order.")
    except ValueError:
        # Some Etsy export variants omit the "Buyer" column this needs.
        columns[3].metric("Repeat Customer Rate", "N/A")
        columns[3].caption("Needs a 'Buyer' column, which wasn't found in this file.")


def render_top_products(orders: pd.DataFrame, top_n: int = 10) -> None:
    """Render a table of the best-selling products by revenue."""
    st.subheader("Top Selling Products")
    st.caption("Products ranked by total revenue (price × quantity), highest first.")

    table = top_selling_listings(orders, top_n=top_n).rename(
        columns={"Item Name": "Product", "quantity_sold": "Quantity Sold", "revenue": "Revenue"}
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner="Analyzing your shop...")
def _cached_health_report(orders: pd.DataFrame) -> ShopHealthReport:
    # Cached (keyed on the data itself) so this doesn't recompute -- and
    # doesn't re-call the LLM -- on every Streamlit rerun, e.g. every chat message.
    return generate_shop_health_report(orders)


def render_ai_insights(orders: pd.DataFrame) -> None:
    """Render the AI Insights panel: health score, trends, risks, and recommendations.

    The score itself is always computed deterministically from real metrics
    (see ``analytics.health``) whether or not an LLM is configured -- only
    the executive summary and recommendations are AI-written when available.
    """
    st.subheader("AI Insights")
    report = _cached_health_report(orders)

    if report.ai_generated:
        st.caption("Health score is computed from your metrics; summary and recommendations are AI-written.")
    else:
        st.caption(
            "Health score and summary are computed directly from your metrics. Add "
            "OPENAI_API_KEY or ANTHROPIC_API_KEY to .env for AI-written summaries and "
            "richer recommendations."
        )

    score_emoji = "🟢" if report.score >= 70 else "🟡" if report.score >= 40 else "🔴"
    score_col, summary_col = st.columns([1, 3])
    score_col.metric("Business Health Score", f"{score_emoji} {report.score}/100")
    summary_col.write(report.executive_summary)

    with st.expander("Why this score?"):
        for line in report.score_breakdown:
            st.write(f"- {line}")

    improving_col, declining_col, risks_col = st.columns(3)
    with improving_col:
        st.markdown("**📈 What's Improving**")
        _render_bullet_list(report.improving, "Nothing notably improving right now.")
    with declining_col:
        st.markdown("**📉 What's Declining**")
        _render_bullet_list(report.declining, "Nothing notably declining right now.")
    with risks_col:
        st.markdown("**⚠️ Risks**")
        _render_bullet_list(report.risks, "No specific risks flagged.")

    st.markdown("**💡 Recommended Actions**")
    _render_bullet_list(report.recommendations, "No specific recommendations right now.")


def _render_bullet_list(items: tuple[str, ...], empty_message: str) -> None:
    if not items:
        st.caption(empty_message)
        return
    for item in items:
        st.write(f"- {item}")


def render_chat(graph: CompiledStateGraph, orders: pd.DataFrame) -> None:
    """Render chat history and handle new questions via the LangGraph agent."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.caption(
        'Ask naturally: "why are sales down?", "give me a health check", '
        '"what should I focus on next?" -- or simple ones like "total revenue".'
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input('Ask anything, e.g. "why are sales down?" or "total revenue"...')
    if question is None:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    final_state = graph.invoke({"question": question, "orders": orders})
    answer = final_state["response"]

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
