from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.trend_reacceleration as strategy
from momentum_screener.daily_screening_notification import render_daily_screening_email
from momentum_screener.monthly_reversal import SCREEN_COLUMNS as MONTHLY_COLUMNS
from momentum_screener.rps import InvalidRpsSessionError
from momentum_screener.storage_manifest import PRICE_SCHEMA
from momentum_screener.strategy_data import resolve_strategy_sessions
from momentum_screener.trend_reacceleration import (
    DEFAULT_CONFIG,
    SCREEN_COLUMNS,
    TrendReaccelerationConfig,
    TrendReaccelerationTickerNotFoundError,
    calculate_trend_reacceleration_features,
    calculate_trend_reacceleration_history,
    evaluate_trend_reacceleration,
    screen_trend_reacceleration,
)


def price_frame(length: int = 320, *, rising: bool = False) -> pd.DataFrame:
    closes = 50 + np.arange(length) * 0.2 if rising else np.full(length, 100.0)
    return pd.DataFrame(
        {
            "date": resolve_strategy_sessions(date(2026, 9, 3), length),
            "ticker": "PASS",
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "adj_close": closes,
            "volume": 100,
            "rps120": 95.0,
            "rps250": 95.0,
        }
    )


def set_closes(
    frame: pd.DataFrame, positions: slice | list[int], values: object
) -> None:
    for column in ("open", "high", "low", "close", "adj_close"):
        frame.loc[frame.index[positions], column] = values


@pytest.mark.parametrize(("total", "expected"), [(185.0, False), (185.01, True)])
def test_momentum_strict_boundary(total: float, expected: bool) -> None:
    frame = price_frame(rising=True)
    frame["rps120"] = 95.0
    frame["rps250"] = total - 95
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["rps_sum"] == total
    assert bool(row["momentum_ok"]) is expected
    assert bool(row["signal"]) is expected


@pytest.mark.parametrize(("days", "expected"), [(25, True), (24, False)])
def test_long_ma_counts_require_at_least_25_of_30(days: int, expected: bool) -> None:
    frame = price_frame()
    set_closes(frame, slice(-days, None), np.linspace(110, 120, days))
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    for ma in (200, 250):
        assert row[f"above_ma{ma}_count"] == days
        assert bool(row[f"above_ma{ma}_30"]) is expected
    assert bool(row["trend_ok"]) is expected


@pytest.mark.parametrize(("days", "expected"), [(3, True), (2, False)])
def test_short_ma_count_boundary(days: int, expected: bool) -> None:
    frame = price_frame()
    set_closes(frame, slice(-days, None), 110.0)
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["above_ma10_count"] == days
    assert bool(row["above_ma10_4"]) is expected


def test_short_term_or_accepts_ma20_when_ma10_count_fails() -> None:
    frame = price_frame(rising=True)
    set_closes(frame, slice(-30, None), 120.0)
    set_closes(frame, slice(-10, None), 130.0)
    set_closes(frame, slice(-3, None), 128.0)
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert not bool(row["above_ma10_4"])
    assert bool(row["above_ma20_4"])
    assert bool(row["trend_ok"])


def drawdown_frame(low: float = 75.0) -> pd.DataFrame:
    frame = price_frame()
    set_closes(frame, slice(None), 90.0)
    frame["high"] = 95.0
    frame["low"] = 85.0
    frame.loc[frame.index[-6], "high"] = 100.0
    frame.loc[frame.index[-7], "low"] = 1.0  # Before the high.
    frame.loc[frame.index[-6], "low"] = 2.0  # On the high, also excluded.
    frame.loc[frame.index[-3], "low"] = low
    return frame


@pytest.mark.parametrize(("low", "expected"), [(75.0, True), (74.99, False)])
def test_drawdown_uses_entire_post_high_path_and_inclusive_25_percent(
    low: float,
    expected: bool,
) -> None:
    row = calculate_trend_reacceleration_features(drawdown_frame(low)).iloc[-1]
    assert row["days_since_20d_high"] == 5
    assert row["recent_high"] == 100
    assert row["recent_low"] == low
    assert row["days_since_low"] == 2
    assert row["drawdown"] == pytest.approx((100 - low) / 100)
    assert bool(row["drawdown_ok"]) is expected
    assert bool(row["no_deep_drawdown_since_high"]) is expected


def test_today_high_defines_zero_drawdown_even_with_large_intraday_range() -> None:
    frame = drawdown_frame()
    frame.loc[frame.index[-1], ["high", "low"]] = [200.0, 1.0]
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["days_since_20d_high"] == 0
    assert row["days_since_low"] == 0
    assert row["recent_low"] == 1
    assert row["drawdown"] == 0
    assert bool(row["drawdown_ok"])


def test_strategy_tied_high_uses_latest_anchor_and_excludes_earlier_trough() -> None:
    frame = drawdown_frame(low=85)
    frame.loc[frame.index[-4], "high"] = 100.0
    frame.loc[frame.index[-5], "low"] = 10.0
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["days_since_20d_high"] == 3
    assert row["recent_low"] == 85.0
    assert row["drawdown"] == pytest.approx(0.15)


@pytest.mark.parametrize(("close", "expected"), [(80.0, False), (80.01, True)])
def test_annual_high_is_highest_adjusted_close_with_strict_ratio(
    close: float, expected: bool
) -> None:
    frame = price_frame()
    frame.loc[frame.index[-100], "high"] = 1000.0
    set_closes(frame, [-1], close)
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["annual_high_ratio"] == close / 100
    assert bool(row["near_250d_high"]) is expected


def test_ma_confirmation_allows_equality_but_current_rising_is_strict() -> None:
    row = calculate_trend_reacceleration_features(price_frame()).iloc[-1]
    assert bool(row["ma20_non_decreasing_5"])
    assert bool(row["ma10_above_ma20_5"])
    assert bool(row["ma10_above_ma20"])
    assert not bool(row["ma10_rising"])
    assert not bool(row["ma20_rising"])
    assert not bool(row["reacceleration_ok"])


def test_ma_confirmation_passes_flat_then_rising_but_fails_one_down_day() -> None:
    frame = price_frame()
    set_closes(frame, [-1], 110.0)
    passed = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert bool(passed["ma20_non_decreasing_5"])
    assert bool(passed["ma10_above_ma20_5"])
    assert bool(passed["reacceleration_ok"])
    set_closes(frame, [-3], 90.0)
    failed = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert bool(failed["ma20_rising"])
    assert not bool(failed["ma20_non_decreasing_5"])
    assert not bool(failed["ma10_above_ma20_5"])
    assert not bool(failed["reacceleration_ok"])


@pytest.mark.parametrize(
    ("length", "expected"), [(278, False), (279, True), (300, True)]
)
def test_exact_warmup_derivation(length: int, expected: bool) -> None:
    assert DEFAULT_CONFIG.required_price_rows == 279
    row = calculate_trend_reacceleration_features(
        price_frame(length, rising=True)
    ).iloc[-1]
    assert bool(row["history_sufficient"]) is expected
    assert bool(row["setup"]) is expected
    assert bool(row["signal"]) is expected
    assert row["status"] == ("evaluated" if expected else "insufficient_history")


def test_split_raw_gap_does_not_create_false_drawdown_or_double_adjustment() -> None:
    adjusted = price_frame(rising=True)
    # Highest adjusted high is five days before today, on the pre-split side.
    adjusted.loc[adjusted.index[-6], "high"] = 120.0
    split = adjusted.copy()
    split.loc[split.index[:-4], ["open", "high", "low", "close"]] *= 2
    assert split.iloc[-4]["close"] / split.iloc[-5]["close"] < 0.51
    expected = calculate_trend_reacceleration_features(adjusted)
    actual = calculate_trend_reacceleration_features(split)
    pd.testing.assert_frame_equal(
        actual.loc[:, SCREEN_COLUMNS], expected.loc[:, SCREEN_COLUMNS]
    )
    assert bool(actual.iloc[-1]["drawdown_ok"])
    assert bool(actual.iloc[-1]["signal"])


def test_appending_future_rows_does_not_change_any_past_feature() -> None:
    frame = price_frame(350, rising=True)
    frame.loc[frame.index[290], "high"] = 200.0
    before = calculate_trend_reacceleration_features(frame.iloc[:320])
    set_closes(frame, slice(320, None), np.linspace(1, 10000, 30))
    after = calculate_trend_reacceleration_features(frame)
    pd.testing.assert_frame_equal(before, after.iloc[:320])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 101.0])
def test_invalid_rps_cannot_signal_even_with_custom_low_threshold(
    invalid: float,
) -> None:
    frame = price_frame(rising=True)
    frame["rps250"] = invalid
    row = calculate_trend_reacceleration_features(
        frame, config=replace(DEFAULT_CONFIG, rps_sum_threshold=0)
    ).iloc[-1]
    assert not bool(row["momentum_ok"])
    assert not bool(row["signal"])
    assert row["status"] == "rps_unavailable"


def test_nan_ohlc_history_fails_closed_and_current_invalid_is_explained() -> None:
    frame = price_frame(rising=True)
    frame.loc[frame.index[-25], "close"] = np.nan
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["status"] == "insufficient_history"
    assert not bool(row["signal"])
    frame.loc[frame.index[-1], "low"] = 0.0
    row = calculate_trend_reacceleration_features(frame).iloc[-1]
    assert row["status"] == "invalid_adjusted_ohlc"
    assert not bool(row["signal"])


def test_custom_config_controls_thresholds_counts_windows_and_warmup() -> None:
    config = replace(
        DEFAULT_CONFIG,
        ma_long_count_window=60,
        ma_long_min_days=55,
        short_count_window=5,
        short_min_days=4,
        recent_high_lookback=10,
        year_high_lookback=300,
        ma_trend_confirm_days=7,
        rps_sum_threshold=195,
        max_drawdown=0.1,
        year_high_ratio_min=0.95,
    )
    assert config.required_price_rows == 309
    result = calculate_trend_reacceleration_features(
        price_frame(320, rising=True), config=config
    )
    assert not bool(result.iloc[307]["history_sufficient"])
    assert bool(result.iloc[308]["history_sufficient"])
    assert result.iloc[-1]["above_ma250_count"] == 60
    assert result.iloc[-1]["above_ma10_count"] == 5
    assert not bool(result.iloc[-1]["momentum_ok"])
    shallow = calculate_trend_reacceleration_features(
        drawdown_frame(85), config=config
    ).iloc[-1]
    assert not bool(shallow["drawdown_ok"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ma_long_count_window": 0},
        {"ma_long_min_days": 31},
        {"short_min_days": 5},
        {"recent_high_lookback": True},
        {"year_high_lookback": 2.5},
        {"max_drawdown": np.nan},
        {"year_high_ratio_min": 1.01},
        {"rps_sum_threshold": -1},
    ],
)
def test_invalid_config_fails_clearly(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrendReaccelerationConfig(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def trend_dataset(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    frame = price_frame(330, rising=True)
    frames = [
        frame,
        frame.assign(ticker="FAIL", rps120=10, rps250=10),
        frame.iloc[-278:].assign(ticker="SHORT"),
        frame.iloc[:-1].assign(ticker="MISSING"),
    ]
    prices = pd.concat(frames, ignore_index=True)
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker\nPASS\nFAIL\nSHORT\nMISSING\n", encoding="utf-8")
    root = tmp_path / "prices"
    for year, rows in prices.groupby(prices["date"].map(lambda value: value.year)):
        path = root / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(
                rows.sort_values(["date", "ticker"]),
                schema=PRICE_SCHEMA,
                preserve_index=False,
            ),
            path,
        )
    rps = pd.concat(
        [
            frame.assign(
                ticker=ticker,
                rps120=(10 if ticker == "FAIL" else 95),
                rps250=(10 if ticker == "FAIL" else 95),
            )
            for ticker in ("PASS", "FAIL", "SHORT", "MISSING")
        ]
    )[["date", "ticker", "rps120", "rps250"]].reset_index(drop=True)
    return root, universe, prices, rps


def test_history_evaluate_screen_and_daily_render_keep_three_consecutive_signals(
    trend_dataset: tuple[Path, Path, pd.DataFrame, pd.DataFrame],
) -> None:
    root, universe, prices, rps = trend_dataset
    dates = sorted(prices["date"].unique())[-3:]
    kwargs = {
        "prices_root": root,
        "universe_path": universe,
        "rps_root": None,
        "rps_snapshots": rps,
    }
    history = calculate_trend_reacceleration_history(
        "pass", dates[0], dates[-1], **kwargs
    )
    assert history["signal"].tolist() == [True, True, True]
    assert history["setup"].equals(history["signal"].rename("setup"))
    for session in dates:
        screen = screen_trend_reacceleration(session, **kwargs)
        assert "PASS" in screen["ticker"].tolist()
        assert set(SCREEN_COLUMNS) == set(screen.columns)
        for ticker in ("PASS", "FAIL", "SHORT", "MISSING"):
            explanation = evaluate_trend_reacceleration(ticker, session, **kwargs)
            assert bool(explanation["signal"]) == (ticker in screen["ticker"].tolist())
        rendered = render_daily_screening_email(
            as_of_date=session,
            monthly_reversal_rows=pd.DataFrame(columns=MONTHLY_COLUMNS),
            trend_reacceleration_rows=screen,
        )
        assert "PASS" in rendered.text_body
        assert "PASS" in rendered.html_body
    assert (
        evaluate_trend_reacceleration("SHORT", dates[-1], **kwargs)["status"]
        == "insufficient_history"
    )
    assert (
        evaluate_trend_reacceleration("MISSING", dates[-1], **kwargs)["status"]
        == "price_unavailable"
    )


def test_empty_momentum_prefilter_avoids_loading_technical_price_history(
    monkeypatch: pytest.MonkeyPatch,
    trend_dataset: tuple[Path, Path, pd.DataFrame, pd.DataFrame],
) -> None:
    root, universe, prices, rps = trend_dataset
    rps.loc[:, ["rps120", "rps250"]] = 0.0

    def unexpected(**kwargs: object) -> None:
        raise AssertionError("no candidates must not load technical history")

    monkeypatch.setattr(strategy, "load_strategy_price_history", unexpected)
    result = screen_trend_reacceleration(
        max(prices["date"]),
        prices_root=root,
        universe_path=universe,
        rps_root=None,
        rps_snapshots=rps,
    )
    assert result.empty
    assert tuple(result.columns) == SCREEN_COLUMNS
    assert result.attrs["momentum_candidate_count"] == 0


def test_api_rejects_unknown_ticker_and_non_session_date(trend_dataset) -> None:
    root, universe, prices, rps = trend_dataset
    kwargs = {
        "prices_root": root,
        "universe_path": universe,
        "rps_root": None,
        "rps_snapshots": rps,
    }
    with pytest.raises(TrendReaccelerationTickerNotFoundError):
        evaluate_trend_reacceleration("UNKNOWN", max(prices["date"]), **kwargs)
    with pytest.raises(InvalidRpsSessionError):
        screen_trend_reacceleration("2026-09-05", **kwargs)


def test_feature_input_is_not_mutated_and_rejects_duplicate_dates() -> None:
    frame = price_frame()
    original = frame.copy(deep=True)
    calculate_trend_reacceleration_features(frame)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="duplicate"):
        calculate_trend_reacceleration_features(pd.concat([frame, frame.iloc[[-1]]]))
