"""Streamlit dashboard for EtsySellerCopilot.

Wires together the data, analytics, and agent layers: upload a Sold Orders
CSV (or load bundled sample data), see headline metrics, top products, and a
sales-trend chart, then ask free-form questions through the LangGraph agent.
"""

import subprocess

import streamlit as st

from etsy_seller_copilot.agents.graph import build_graph
from etsy_seller_copilot.analytics.metrics import daily_sales_trend, revenue_by_month
from etsy_seller_copilot.data.loaders import load_sample_orders
from etsy_seller_copilot.data.utils import FriendlyColumnError
from etsy_seller_copilot.ui.charts import daily_sales_chart, monthly_revenue_chart
from etsy_seller_copilot.ui.components import (
    load_uploaded_orders,
    render_ai_insights,
    render_chat,
    render_data_preview,
    render_file_uploader,
    render_metrics,
    render_top_products,
)


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_resource
def get_git_commit_hash() -> str:
    """Return the short git commit hash of the running deployment, for verifying
    which version is live. TEMPORARY: remove once deployment verification is done."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _load_orders() -> tuple:
    """Return (orders, warnings) from the uploaded file or sample data, or None."""
    uploaded_file = render_file_uploader()

    if uploaded_file is not None:
        st.session_state.use_sample_data = False
        return load_uploaded_orders(uploaded_file)

    st.info("No file uploaded yet. Upload an Etsy Sold Orders CSV export above.")
    if st.session_state.get("use_sample_data"):
        return load_sample_orders(), []

    if st.button("Load sample Etsy order data"):
        st.session_state.use_sample_data = True
        st.rerun()

    return None


def main() -> None:
    st.set_page_config(page_title="EtsySellerCopilot", layout="wide")
    st.sidebar.caption(f"Version: {get_git_commit_hash()}")
    st.title("EtsySellerCopilot")
    st.write(
        "Upload your Etsy **Sold Orders** CSV export (Shop Manager > Settings > "
        "Options > Download Data) to get instant insights about your shop."
    )

    try:
        result = _load_orders()
    except FriendlyColumnError as exc:
        st.error(str(exc))
        return

    if result is None:
        return
    orders, warnings = result

    for warning in warnings:
        st.warning(warning)

    render_data_preview(orders)
    st.divider()
    render_metrics(orders)
    st.divider()
    render_ai_insights(orders)
    st.divider()
    render_top_products(orders)
    st.divider()

    st.subheader("Sales Trend")
    if "Sale Date" in orders.columns and orders["Sale Date"].notna().any():
        trend_choice = st.radio("Group by", ["Monthly", "Daily"], horizontal=True)
        if trend_choice == "Monthly":
            st.caption("Total revenue for each calendar month.")
            st.plotly_chart(monthly_revenue_chart(revenue_by_month(orders)), use_container_width=True)
        else:
            st.caption("Total revenue for each calendar day.")
            st.plotly_chart(daily_sales_chart(daily_sales_trend(orders)), use_container_width=True)
    else:
        st.info("Add a 'Sale Date' (or similar) column to your CSV to see sales trends over time.")

    st.divider()
    st.subheader("Ask a question")
    render_chat(get_graph(), orders)


if __name__ == "__main__":
    main()
