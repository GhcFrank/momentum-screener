from datetime import date

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from momentum_screener.daily_watch_3 import (
    DEFAULT_CONFIG,
    calculate_daily_watch_3_core_features,
    calculate_daily_watch_3_features,
)
from momentum_screener.strategy_data import resolve_strategy_sessions


def price_frame(module: str) -> pd.DataFrame:
    count = DEFAULT_CONFIG.required_price_rows
    adjusted_close = np.full(count, 50.0)
    adjusted_high = np.full(count, 100.0)
    if module == "breakout":
        # The first and third rows of the final five-day window set their own
        # rolling 250-day close highs. Today's close is below the later high.
        adjusted_close[-5] = 60.0
        adjusted_close[-3] = 65.0
        adjusted_close[-1] = 55.0
        rps50, rps120, rps250 = 50.0, 96.0, 50.0
    elif module == "near_85":
        adjusted_close[-20] = 100.0
        adjusted_close[-1] = 85.0
        rps50, rps120, rps250 = 50.0, 97.0, 50.0
    elif module == "near_70":
        adjusted_close[-20] = 100.0
        adjusted_close[-1] = 70.0
        rps50, rps120, rps250 = 50.0, 98.0, 50.0
    else:
        raise AssertionError(module)

    raw_close = adjusted_close * 2
    volume = np.full(count, 1_000.0)
    return pd.DataFrame(
        {
            "date": resolve_strategy_sessions(date(2026, 9, 4), count),
            "ticker": "PASS",
            "open": raw_close,
            "high": adjusted_high * 2,
            "low": np.maximum(adjusted_close - 1, 1) * 2,
            "close": raw_close,
            "adj_close": adjusted_close,
            "volume": volume,
            "market_cap": raw_close * volume / 0.10,
            "rps50": rps50,
            "rps120": rps120,
            "rps250": rps250,
        }
    )


def test_three_price_position_rps_modules_use_close_high_and_intraday_high() -> None:
    cases = (
        ("breakout", "high_breakout_ok", 0.55),
        ("near_85", "near_high_85_ok", 0.85),
        ("near_70", "near_high_70_ok", 0.70),
    )
    module_columns = {"high_breakout_ok", "near_high_85_ok", "near_high_70_ok"}
    for module, expected_module, expected_position in cases:
        features = calculate_daily_watch_3_features(price_frame(module))
        row = features.iloc[-1]
        assert row[expected_module]
        assert not row[list(module_columns - {expected_module})].any()
        assert row["core_signal"] and row["signal"]
        assert row["high_250_intraday"] == 100.0
        assert row["price_position_250_high"] == pytest.approx(expected_position)
        if module == "breakout":
            assert row["high_close_250"] == 65.0
        else:
            assert row["high_close_250"] == 100.0


def test_core_ignores_market_cap_proxy_while_formal_signal_fails_closed() -> None:
    boundary = price_frame("near_85")
    current = boundary.index[-1]
    boundary.loc[current, "market_cap"] = (
        boundary.loc[current, "close"] * boundary.loc[current, "volume"] / 0.20
    )
    formal = calculate_daily_watch_3_features(boundary).iloc[-1]
    core = calculate_daily_watch_3_core_features(boundary).iloc[-1]
    assert formal["turnover_market_cap_proxy"] == pytest.approx(0.20)
    assert formal["dollar_volume"] == (
        boundary.loc[current, "close"] * boundary.loc[current, "volume"]
    )
    assert formal["core_signal"] and not formal["normal_turnover"]
    assert not formal["signal"] and core["signal"]
    assert "turnover_rate" not in formal.index

    missing = boundary.copy()
    missing.loc[current, "market_cap"] = np.nan
    formal_missing = calculate_daily_watch_3_features(missing).iloc[-1]
    core_missing = calculate_daily_watch_3_core_features(missing).iloc[-1]
    assert not formal_missing["market_cap_available"]
    assert formal_missing["core_signal"] and not formal_missing["signal"]
    assert formal_missing["status"] == "market_cap_unavailable"
    assert core_missing["signal"] and core_missing["status"] == "ok"

    missing_rps = boundary.copy()
    missing_rps.loc[current, "rps50"] = np.nan
    unavailable = calculate_daily_watch_3_core_features(missing_rps).iloc[-1]
    assert not unavailable["rps_available"]
    assert not unavailable["core_signal"] and not unavailable["signal"]


def test_recent_high_uses_each_days_rolling_250_high_with_254_row_warmup() -> None:
    features = calculate_daily_watch_3_features(price_frame("breakout"))
    earlier_high = features.iloc[-5]
    later_high = features.iloc[-3]
    current = features.iloc[-1]

    assert DEFAULT_CONFIG.required_price_rows == 254
    assert not features.iloc[-2]["history_sufficient"]
    assert current["history_sufficient"]
    assert earlier_high["is_250d_close_high"]
    assert later_high["is_250d_close_high"]
    assert current["high_close_250"] > earlier_high["adj_close"]
    assert not current["is_250d_close_high"]
    assert current["recent_250d_close_high"]
