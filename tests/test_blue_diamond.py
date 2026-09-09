from datetime import date

import numpy as np
import pandas as pd
import pytest

from momentum_screener.blue_diamond import (
    DEFAULT_CONFIG,
    calculate_blue_diamond_features,
)
from momentum_screener.strategy_data import resolve_strategy_sessions


def price_frame() -> pd.DataFrame:
    closes = 50 + np.arange(280) * 0.2
    for position in (-2, -1):
        index = len(closes) + position
        closes[index] = closes[index - 19 : index].mean() * 0.999
    # Raw prices are twice adjusted prices: turnover must retain raw close.
    return pd.DataFrame(
        {
            "date": resolve_strategy_sessions(date(2026, 9, 4), len(closes)),
            "ticker": "PASS",
            "open": closes * 2,
            "high": (closes + 0.2) * 2,
            "low": (closes - 0.2) * 2,
            "close": closes * 2,
            "adj_close": closes,
            "volume": 100.0,
            "market_cap": 1_000_000.0,
            "rps20": 98.0,
            "rps50": 95.0,
        }
    )


def set_adjusted_prices(frame, index, close, high=None, low=None):
    frame.loc[index, ["open", "close"]] = close * 2
    frame.loc[index, "adj_close"] = close
    frame.loc[index, "high"] = (close if high is None else high) * 2
    frame.loc[index, "low"] = (close if low is None else low) * 2


def test_formula_strict_boundaries_adjustment_and_every_qualifying_day():
    frame = price_frame()
    original = frame.copy(deep=True)
    features = calculate_blue_diamond_features(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert features["signal"].tail(2).tolist() == [True, True]
    row = features.iloc[-1]
    assert row["status"] == "ok"
    assert row["adj_close"] < row["ma20"]  # NearMA20 has no lower bound.
    assert row["adj_high"] == frame.iloc[-1]["high"] / 2
    assert row["dollar_volume"] == frame.iloc[-1]["close"] * 100
    assert row["turnover"] == row["dollar_volume"] / 1_000_000
    assert row["close_below_ma20_count"] == 2
    assert row["close_below_ma10_count"] == 2
    assert row["low_below_ma20_count"] == 2
    assert DEFAULT_CONFIG.required_price_rows == 250
    assert not features.iloc[248]["history_sufficient"]
    assert features.iloc[249]["history_sufficient"]

    for field, expected_ratio, condition in (
        ("turnover", 0.10, "normal_turnover"),
        ("price_position_250", 0.80, "near_250_high"),
        ("ma20_distance", 1.005, "near_ma20"),
    ):
        changed = frame.copy()
        if field == "turnover":
            changed.loc[changed.index[-1], "market_cap"] = row["dollar_volume"] / 0.10
        elif field == "price_position_250":
            for column in ("open", "high", "low", "close", "adj_close"):
                changed[column] *= 80 / row["adj_close"]
            set_adjusted_prices(changed, changed.index[-1], 80)
            set_adjusted_prices(changed, changed.index[-100], 100)
        else:
            # Exactly C=201 / MA20=200, avoiding a synthetic near-boundary
            # value caused by solving backwards with floating-point inputs.
            set_adjusted_prices(changed, changed.index[-20:], 200)
            set_adjusted_prices(changed, changed.index[-2], 199)
            set_adjusted_prices(changed, changed.index[-1], 201)
        current = calculate_blue_diamond_features(changed).iloc[-1]
        assert current[field] == expected_ratio
        assert not current[condition] and not current["signal"], field

    # The OR branches still require BOTH RPS scores to be available.
    for rps20, rps50, expected in (
        (50, 98, True),
        (98, 50, True),
        (93, 97, True),
        (92.9, 97, False),
        (-1, 99, False),
        (99, np.nan, False),
    ):
        current = calculate_blue_diamond_features(
            frame.assign(rps20=rps20, rps50=rps50)
        ).iloc[-1]
        assert bool(current["extreme_rps"]) is expected
        assert bool(current["signal"]) is expected

    future = frame.tail(1).assign(
        date=date(2026, 9, 8),
        close=1e6,
        adj_close=1e6,
        high=1e6,
        low=1,
        market_cap=1,
        rps20=0,
    )
    extended = calculate_blue_diamond_features(
        pd.concat([frame, future], ignore_index=True)
    )
    pd.testing.assert_frame_equal(features, extended.iloc[:-1].reset_index(drop=True))


def test_dynamic_pullback_uses_past_series_latest_high_and_missing_caps_fail_closed():
    frame = price_frame()
    set_adjusted_prices(frame, frame.index[-30:], 95, high=96, low=94)
    # An older, higher anchor ages out today. Past pullback values in the
    # current post-high window still remember it, despite today's shallow path.
    set_adjusted_prices(frame, frame.index[-21], 195, high=200, low=190)
    for index in frame.index[[-11, -10]]:
        set_adjusted_prices(frame, index, 95, high=100, low=1)
    features = calculate_blue_diamond_features(frame)
    row = features.iloc[-1]
    assert row["days_since_high_20"] == 9  # Latest tied high.
    assert row["recent_high"] == 100
    assert row["recent_low"] == 94  # Anchor's low=1 is excluded.
    assert row["days_since_low"] == 0  # Latest tied low.
    assert row["pullback"] == pytest.approx(0.06)
    assert features["pullback"].iloc[-9:-1].gt(0.25).any()
    assert not row["no_deep_pullback_since_high"]
    assert not row["controlled_pullback"] and not row["signal"]

    set_adjusted_prices(frame, frame.index[-1], 95, high=120, low=50)
    today_high = calculate_blue_diamond_features(frame).iloc[-1]
    assert today_high["days_since_high_20"] == today_high["days_since_low"] == 0
    assert today_high["no_deep_pullback_since_high"]  # Empty dynamic COUNT.
    assert today_high["pullback"] == pytest.approx(70 / 120)  # Not forced to zero.
    assert not today_high["controlled_pullback"]

    for column, value in (
        ("market_cap", np.nan),
        ("market_cap", 0),
        ("volume", np.inf),
        ("close", np.nan),
    ):
        missing = price_frame()
        missing.loc[missing.index[-1], column] = value
        current = calculate_blue_diamond_features(missing).iloc[-1]
        assert not current["market_cap_available"]
        assert not current["normal_turnover"] and not current["signal"]
        assert current["status"] == (
            "price_unavailable" if column == "close" else "market_cap_unavailable"
        )
