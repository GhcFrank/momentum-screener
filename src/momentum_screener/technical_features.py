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


def rolling_every(condition: pd.Series, window: int) -> pd.Series:
    """True only for a complete trailing window of True ticker rows.

    Missing values count as False, like rolling_count; warmup returns False.
    Missing rows are never skipped or filled from another session.
    """

    return rolling_count(condition, window).eq(window)


def bars_since_highest(series: pd.Series, window: int) -> pd.Series:
    """Return bars since the most recent maximum in a full trailing window.

    Today is 0, yesterday is 1. Tied maxima always select the occurrence
    closest to today (e.g. equal highs at t-5 and t-3 return 3). Incomplete
    windows or windows containing NaN/nonfinite values return NaN.
    """

    _validate_window(window)
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.where(np.isfinite(values))
    return values.rolling(window, min_periods=window).apply(
        lambda rows: float(np.argmax(rows[::-1])), raw=True
    )


def value_at_offset(series: pd.Series, bars_ago: pd.Series) -> pd.Series:
    """Return a dynamic backward REF; invalid or future offsets yield NaN."""

    if not series.index.equals(bars_ago.index):
        raise ValueError("series and bars_ago must have identical indexes")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    offsets = pd.to_numeric(bars_ago, errors="coerce").to_numpy(dtype="float64")
    positions = np.arange(len(values))
    valid = np.isfinite(offsets) & (offsets >= 0) & (offsets <= positions)
    valid &= offsets == np.floor(offsets)
    result = np.full(len(values), np.nan)
    result[valid] = values[positions[valid] - offsets[valid].astype("int64")]
    result[~np.isfinite(result)] = np.nan
    return pd.Series(result, index=series.index, dtype="float64")


def lowest_since_anchor(
    series: pd.Series, bars_since_anchor: pd.Series
) -> pd.DataFrame:
    """Return lowest_value and bars_since_low strictly AFTER a dynamic anchor.

    For an anchor n > 0 bars ago, use rows t-n+1 through t, excluding the
    anchor's own low. For n == 0 use today's value and offset 0. Equal lows
    select the most recent occurrence. Invalid offsets, incomplete paths or
    any nonfinite value on the path yield NaN in both output columns.
    """

    if not series.index.equals(bars_since_anchor.index):
        raise ValueError("series and bars_since_anchor must have identical indexes")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    anchors = pd.to_numeric(bars_since_anchor, errors="coerce").to_numpy(
        dtype="float64"
    )
    lows = np.full(len(values), np.nan)
    offsets = np.full(len(values), np.nan)
    for position, anchor in enumerate(anchors):
        if (
            not np.isfinite(anchor)
            or anchor < 0
            or anchor > position
            or anchor != int(anchor)
        ):
            continue
        start = position - max(int(anchor), 1) + 1
        path = values[start : position + 1]
        if not np.isfinite(path).all():
            continue
        offset = int(np.argmin(path[::-1]))
        lows[position] = values[position - offset]
        offsets[position] = offset
    return pd.DataFrame(
        {"lowest_value": lows, "bars_since_low": offsets}, index=series.index
    )


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
