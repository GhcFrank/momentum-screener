from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from momentum_screener.technical_features import (
    add_adjusted_ohlc,
    highest_value,
    lowest_value,
    moving_average,
    rolling_count,
    safe_ratio,
)


def test_adjusted_ohlc_uses_one_adj_close_over_close_factor() -> None:
    source = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            "adj_close": [95.0],
        }
    )

    result = add_adjusted_ohlc(source).iloc[0]

    assert result["adjust_factor"] == pytest.approx(0.95)
    assert result["adj_open"] == pytest.approx(95.0)
    assert result["adj_high"] == pytest.approx(104.5)
    assert result["adj_low"] == pytest.approx(85.5)
    assert result["adj_close"] == pytest.approx(95.0)
    assert bool(result["adjusted_ohlc_valid"])
    assert source.iloc[0]["adj_close"] == 95.0


def test_adjusted_ohlc_factor_one_does_not_double_adjust() -> None:
    source = pd.DataFrame(
        {
            "open": [99.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            "adj_close": [100.0],
        }
    )

    result = add_adjusted_ohlc(source).iloc[0]

    assert result["adjust_factor"] == 1.0
    assert result["adj_open"] == 99.0
    assert result["adj_high"] == 110.0
    assert result["adj_low"] == 90.0
    assert result["adj_close"] == 100.0


@pytest.mark.parametrize(
    ("close", "adj_close"),
    [
        (0.0, 10.0),
        (-1.0, 10.0),
        (float("nan"), 10.0),
        (10.0, float("nan")),
        (10.0, float("inf")),
    ],
)
def test_adjusted_ohlc_invalid_factor_fails_closed_without_inf(
    close: float,
    adj_close: float,
) -> None:
    source = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [close],
            "adj_close": [adj_close],
        }
    )

    result = add_adjusted_ohlc(source).iloc[0]

    assert not bool(result["adjusted_ohlc_valid"])
    assert pd.isna(result["adjust_factor"])
    assert all(
        pd.isna(result[column])
        for column in ("adj_open", "adj_high", "adj_low", "adj_close")
    )


def test_adjusted_ohlc_multiplication_overflow_fails_closed_without_inf() -> None:
    source = pd.DataFrame(
        {
            "open": [1e308],
            "high": [1e308],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1e308],
        }
    )

    result = add_adjusted_ohlc(source).iloc[0]

    assert not bool(result["adjusted_ohlc_valid"])
    assert all(
        pd.isna(result[column])
        for column in ("adj_open", "adj_high", "adj_low", "adj_close")
    )


def test_rolling_helpers_are_inclusive_and_require_complete_row_windows() -> None:
    values = pd.Series([5.0, 2.0, 9.0, 4.0])
    condition = pd.Series([True, False, True, True])

    ma = moving_average(values, 3)
    hhv = highest_value(values, 3)
    llv = lowest_value(values, 3)
    count = rolling_count(condition, 3)

    assert ma.iloc[:2].isna().all()
    assert hhv.iloc[:2].isna().all()
    assert llv.iloc[:2].isna().all()
    assert count.iloc[:2].isna().all()
    assert ma.iloc[2] == pytest.approx(16.0 / 3.0)
    assert hhv.iloc[2] == 9.0
    assert llv.iloc[2] == 2.0
    assert count.iloc[2] == 2.0
    assert count.iloc[3] == 2.0


def test_safe_ratio_requires_positive_finite_denominator() -> None:
    result = safe_ratio(
        pd.Series([9.0, 9.0, np.inf, 9.0]),
        pd.Series([3.0, 0.0, 3.0, np.nan]),
    )

    assert result.iloc[0] == 3.0
    assert result.iloc[1:].isna().all()
