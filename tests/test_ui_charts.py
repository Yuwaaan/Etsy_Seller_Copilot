import pandas as pd
import plotly.graph_objects as go

from etsy_seller_copilot.analytics.metrics import revenue_by_month
from etsy_seller_copilot.ui.charts import monthly_revenue_chart

SAMPLE_ORDERS = pd.DataFrame(
    [
        {"Order ID": 1, "Sale Date": "2026-01-05", "Item Name": "Mug", "Quantity": 2, "Price": 10.0},
        {"Order ID": 2, "Sale Date": "2026-02-10", "Item Name": "Mug", "Quantity": 1, "Price": 10.0},
    ]
)


def test_monthly_revenue_chart_plots_each_month() -> None:
    series = revenue_by_month(SAMPLE_ORDERS)

    figure = monthly_revenue_chart(series)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == ["2026-01", "2026-02"]
    assert list(figure.data[0].y) == [20.0, 10.0]


def test_monthly_revenue_chart_handles_empty_data() -> None:
    empty_series = revenue_by_month(SAMPLE_ORDERS.iloc[0:0])

    figure = monthly_revenue_chart(empty_series)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 0


def test_monthly_revenue_chart_uses_categorical_axis_for_single_month() -> None:
    # A single month/day label like "2026-01" looks like a date to Plotly's
    # auto-detection, which (with only one bar) produces a nonsensical
    # sub-second tick range unless the axis is forced to be categorical.
    single_month_orders = SAMPLE_ORDERS[SAMPLE_ORDERS["Order ID"] == 1]
    series = revenue_by_month(single_month_orders)

    figure = monthly_revenue_chart(series)

    assert figure.layout.xaxis.type == "category"
