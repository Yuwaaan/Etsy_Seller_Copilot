"""Plotly chart construction for the Streamlit UI.

Charts are built directly from the analytics layer's output — no analytics
logic is reimplemented here, only presentation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_chart(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No revenue data yet",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return figure


def _revenue_trend_chart(revenue_by_period: pd.Series, *, period_label: str, title: str) -> go.Figure:
    if revenue_by_period.empty:
        return _empty_chart(title)

    chart_data = revenue_by_period.rename("revenue").reset_index()
    chart_data.columns = [period_label, "revenue"]
    chart_data[period_label] = chart_data[period_label].astype(str)

    figure = px.bar(
        chart_data,
        x=period_label,
        y="revenue",
        title=title,
        labels={period_label: period_label.capitalize(), "revenue": "Revenue ($)"},
    )
    # Without this, Plotly auto-detects "YYYY-MM"/"YYYY-MM-DD" labels as a
    # continuous date axis; with few (or one) bars that produces a nonsensical
    # sub-second tick range instead of one discrete bar per period.
    figure.update_xaxes(type="category")
    return figure


def monthly_revenue_chart(revenue_by_month: pd.Series) -> go.Figure:
    """Build a bar chart of revenue per calendar month.

    ``revenue_by_month`` is the Series returned by
    ``analytics.metrics.revenue_by_month`` (a monthly ``PeriodIndex`` mapped
    to revenue).
    """
    return _revenue_trend_chart(revenue_by_month, period_label="month", title="Revenue by Month")


def daily_sales_chart(daily_sales_trend: pd.Series) -> go.Figure:
    """Build a bar chart of revenue per calendar day.

    ``daily_sales_trend`` is the Series returned by
    ``analytics.metrics.daily_sales_trend`` (a daily ``PeriodIndex`` mapped
    to revenue).
    """
    return _revenue_trend_chart(daily_sales_trend, period_label="day", title="Revenue by Day")
