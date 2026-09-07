from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from momentum_screener.technical_features import (
    add_adjusted_ohlc,
    bars_since_highest,
    highest_value,
    lowest_since_anchor,
    lowest_value,
    moving_average,
    rolling_count,
    rolling_every,
    safe_ratio,
    value_at_offset,
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


def test_every_requires_full_true_window_and_treats_na_as_false() -> None:
    condition = pd.Series([True, True, True, pd.NA, True, True, True], dtype="boolean")
    assert rolling_every(condition, 3).tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3, 4, 5, 6], 0), ([1, 2, 3, 4, 6, 5], 1), ([100, 90, 100, 90, 90, 95], 3)],
)
def test_hhvbars_selects_nearest_tied_high(values: list[float], expected: int) -> None:
    result = bars_since_highest(pd.Series(values), 6)
    assert result.iloc[:-1].isna().all()
    assert result.iloc[-1] == expected


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_hhvbars_rejects_incomplete_or_invalid_window(invalid: float) -> None:
    result = bars_since_highest(pd.Series([1.0, invalid, 2.0, 3.0, 4.0]), 3)
    assert result.iloc[:4].isna().all()
    assert result.iloc[-1] == 0


def test_lowest_since_anchor_excludes_anchor_and_earlier_lows() -> None:
    values = pd.Series([1.0, 2.0, 75.0, 80.0, 75.0, 90.0])
    anchors = pd.Series([0, 0, 1, 2, 3, 4])
    result = lowest_since_anchor(values, anchors)
    assert result.iloc[-1].tolist() == [75.0, 1.0]
    assert result.iloc[0].tolist() == [1.0, 0.0]
    assert result.iloc[1].tolist() == [2.0, 0.0]


def test_dynamic_helpers_reject_future_fractional_or_out_of_range_offsets() -> None:
    values = pd.Series([10.0] * 6)
    offsets = pd.Series([1.0, -1.0, 1.5, np.nan, np.inf, 0.0])
    ref = value_at_offset(values, offsets)
    low = lowest_since_anchor(values, offsets)
    assert ref.iloc[:5].isna().all()
    assert low.iloc[:5].isna().all().all()
    assert ref.iloc[-1] == 10.0
    assert low.iloc[-1].tolist() == [10.0, 0.0]


def test_lowest_since_anchor_never_skips_missing_path_values() -> None:
    result = lowest_since_anchor(pd.Series([1.0, np.nan, 10.0]), pd.Series([0, 1, 2]))
    assert result.iloc[-1].isna().all()


# The helpers share _validate_window; cover each entry point and each
# validation branch once instead of crossing every helper with every value.
@pytest.mark.parametrize(
    ("helper", "window"),
    [
        (moving_average, 0),
        (rolling_count, -1),
        (rolling_every, True),
        (bars_since_highest, 2.5),
    ],
)
def test_rolling_helpers_reject_invalid_windows(helper, window) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        helper(pd.Series([1, 2, 3]), window)
