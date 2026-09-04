"""Shared, point-in-time-safe technical feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

ADJUSTED_OHLC_COLUMNS = ("adj_open", "adj_high", "adj_low", "adj_close")
_SOURCE_OHLC_COLUMNS = ("open", "high", "low", "close", "adj_close")


def add_adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with split/dividend-consistent adjusted OHLC columns.

    The row adjustment factor is ``adj_close / close``.  A row is valid only
    when all source OHLC values and the factor are finite and strictly
    positive.  Invalid rows receive ``NaN`` in every adjusted OHLC field; data
    is never filled from another session.
    """

    missing = [column for column in _SOURCE_OHLC_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Adjusted OHLC input is missing columns: {missing}")

    result = frame.copy()
    source = {
        column: pd.to_numeric(result[column], errors="coerce").astype("float64")
        for column in _SOURCE_OHLC_COLUMNS
    }
    close = source["close"]
    persisted_adj_close = source["adj_close"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        factor = persisted_adj_close.div(close)

    valid = pd.Series(True, index=result.index, dtype="bool")
    for values in source.values():
        valid &= values.gt(0) & np.isfinite(values)
    valid &= factor.gt(0) & np.isfinite(factor)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        adjusted = {
            "adj_open": source["open"].mul(factor),
            "adj_high": source["high"].mul(factor),
            "adj_low": source["low"].mul(factor),
            "adj_close": persisted_adj_close,
        }
    for values in adjusted.values():
        valid &= values.gt(0) & np.isfinite(values)

    result["adjust_factor"] = factor.where(valid)
    for column, values in adjusted.items():
        result[column] = values.where(valid)
    result["adjusted_ohlc_valid"] = valid
    return result


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Return a trailing simple average requiring one complete row window."""

    _validate_window(window)
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.rolling(window=window, min_periods=window).mean()


def highest_value(series: pd.Series, window: int) -> pd.Series:
    """Return an inclusive trailing maximum requiring a complete row window."""

    _validate_window(window)
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.rolling(window=window, min_periods=window).max()


def lowest_value(series: pd.Series, window: int) -> pd.Series:
    """Return an inclusive trailing minimum requiring a complete row window."""

    _validate_window(window)
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.rolling(window=window, min_periods=window).min()


def rolling_count(condition: pd.Series, window: int) -> pd.Series:
    """Count true values in an inclusive trailing complete row window."""

    _validate_window(window)
    values = condition.fillna(False).astype("bool").astype("int64")
    return values.rolling(window=window, min_periods=window).sum()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide finite values only when the denominator is finite and positive."""

    top = pd.to_numeric(numerator, errors="coerce").astype("float64")
    bottom = pd.to_numeric(denominator, errors="coerce").astype("float64")
    valid = (
        top.notna()
        & bottom.notna()
        & np.isfinite(top)
        & np.isfinite(bottom)
        & bottom.gt(0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = top.div(bottom)
    return ratio.where(valid & np.isfinite(ratio))


def _validate_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("rolling window must be a positive integer")
