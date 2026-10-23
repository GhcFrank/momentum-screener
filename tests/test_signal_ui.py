"""Local figure contracts for the historical signal research chart."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from momentum_screener.signal_ui import build_price_volume_figure
from momentum_screener.signal_ui_data import clip_price_window


def test_price_volume_chart_has_matched_dates_and_one_bottom_slider():
    # Sunday has no price row: both markers must still use its calendar date.
    signal_date = date(2026, 6, 14)
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-14", "2025-06-16", "2026-06-15"]),
            "adjusted_close": [10.0, 12.0, 15.0],
            "volume": [0.0, np.nan, 1_234_567.0],
        }
    )
    window = clip_price_window(
        signal_date, prices["date"].min().date(), prices["date"].max().date()
    )
    fig = build_price_volume_figure(prices, signal_date, window)
    price, volume = fig.data

    assert (price.type, price.xaxis, price.yaxis) == ("scatter", "x", "y")
    assert (price.name, price.mode, price.line.color) == (
        "Adjusted Close",
        "lines",
        "red",
    )
    assert price.hovertemplate == (
        "%{x|%Y-%m-%d}<br>Adjusted Close: %{y:.2f}<extra></extra>"
    )
    assert (volume.type, volume.xaxis, volume.yaxis, volume.name) == (
        "bar",
        "x2",
        "y2",
        "Volume",
    )
    np.testing.assert_array_equal(price.x, volume.x)
    np.testing.assert_allclose(price.y, prices["adjusted_close"])
    np.testing.assert_allclose(volume.y, [0.0, np.nan, 1_234_567.0], equal_nan=True)
    assert "%{y:,.0f}" in volume.hovertemplate
    assert fig.layout.yaxis2.rangemode == "tozero"
    assert fig.layout.yaxis.matches is None
    assert fig.layout.yaxis2.matches is None

    assert fig.layout.xaxis.matches == "x2"
    assert fig.layout.xaxis2.matches is None
    for axis in (fig.layout.xaxis, fig.layout.xaxis2):
        assert axis.type == "date"
        assert axis.range == (date(2025, 6, 14), signal_date)
        assert (axis.minallowed, axis.maxallowed) == (window.start, window.end)
    assert fig.layout.xaxis.showticklabels is False
    assert fig.layout.xaxis2.showticklabels is True
    assert fig.layout.xaxis.rangeslider.visible is False
    assert fig.layout.xaxis2.rangeslider.visible is True
    assert fig.layout.xaxis2.rangeslider.range == (window.start, window.end)
    assert fig.layout.xaxis2.title.text == "Date"

    assert len(fig.layout.shapes) == 2
    for shape, xref, yref in zip(
        fig.layout.shapes, ("x", "x2"), ("y domain", "y2 domain"), strict=True
    ):
        assert (shape.xref, shape.yref) == (xref, yref)
        assert shape.x0 == shape.x1 == signal_date.isoformat()
        assert (shape.y0, shape.y1) == (0, 1)
        assert (shape.line.color, shape.line.dash, shape.line.width) == (
            "blue",
            "dash",
            2,
        )
    assert fig.layout.height == 680
    assert fig.layout.dragmode == "zoom"


@pytest.mark.parametrize(
    "dates",
    [
        ["2025-01-01", "2025-02-01"],  # No span in the default viewport.
        ["2026-06-15"],  # Retain the existing single-day autorange behavior.
    ],
)
def test_price_volume_chart_preserves_fallback_with_all_volume_missing(dates):
    prices = pd.DataFrame(
        {"date": pd.to_datetime(dates), "adjusted_close": 10.0, "volume": np.nan}
    )
    signal_date = date(2026, 6, 15)
    window = clip_price_window(
        signal_date, prices["date"].min().date(), prices["date"].max().date()
    )
    assert window.viewport_fallback

    fig = build_price_volume_figure(prices, signal_date, window)

    assert len(fig.data[0].x) == len(dates)
    assert np.isnan(fig.data[1].y).all()
    assert len(fig.layout.shapes) == 2
    assert fig.layout.xaxis2.rangeslider.range == (window.start, window.end)
    expected_range = (window.start, window.end) if len(dates) > 1 else None
    assert fig.layout.xaxis.range == fig.layout.xaxis2.range == expected_range
    fig.to_json()  # Missing bars serialize safely for Streamlit/Plotly.
