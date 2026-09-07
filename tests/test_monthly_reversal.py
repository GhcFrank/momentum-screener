from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.strategy_data as strategy_data_module
from momentum_screener.monthly_reversal import (
    MONTHLY_REVERSAL_REQUIRED_SIGNAL_ROWS,
    calculate_monthly_reversal_features,
    calculate_monthly_reversal_signal,
    calculate_yxfz,
    evaluate_monthly_reversal,
    screen_monthly_reversal,
)
from momentum_screener.rps import RPS_CALENDAR_NAME
from momentum_screener.storage_manifest import PRICE_SCHEMA


def _price_frame(length: int = 300, *, close: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=length, freq="B").date
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": 100,
            "rps50": -1.0,
            "rps120": -1.0,
        }
    )


def test_fyx1_and_fyx130_use_their_exact_threshold_boundaries() -> None:
    frame = _price_frame()
    frame.loc[frame.index[-1], ["rps50", "rps120"]] = [87.0, 90.0]

    boundary = calculate_monthly_reversal_features(frame).iloc[-1]

    assert not bool(boundary["fyx11"])
    assert not bool(boundary["fyx12"])
    assert not bool(boundary["fyx1"])
    assert bool(boundary["fyx130"])

    frame.loc[frame.index[-1], "rps50"] = 87.01
    assert bool(calculate_monthly_reversal_features(frame).iloc[-1]["fyx11"])
    frame.loc[frame.index[-1], ["rps50", "rps120"]] = [0.0, 90.01]
    assert bool(calculate_monthly_reversal_features(frame).iloc[-1]["fyx12"])


def test_fyx13_requires_high_rps_and_current_70_row_highest_close() -> None:
    frame = _price_frame()
    frame.loc[frame.index[-1], "rps50"] = 90.0
    passing = calculate_monthly_reversal_features(frame).iloc[-1]
    assert bool(passing["fyx131"])
    assert bool(passing["fyx13"])

    frame.loc[frame.index[-2], ["open", "high", "low", "close", "adj_close"]] = 101.0
    failing = calculate_monthly_reversal_features(frame).iloc[-1]
    assert not bool(failing["fyx131"])
    assert not bool(failing["fyx13"])


def test_fyx2_or_paths_can_trigger_independently() -> None:
    fyx21_frame = _price_frame()
    fyx21_frame.loc[fyx21_frame.index[-1], "rps50"] = 100.0
    fyx21_frame.loc[fyx21_frame.index[-150], "low"] = 70.0
    fyx21 = calculate_monthly_reversal_features(fyx21_frame).iloc[-1]
    assert bool(fyx21["fyx21"])
    assert not bool(fyx21["fyx22"])
    assert not bool(fyx21["fyx23"])
    assert bool(fyx21["fyx2"])

    fyx22_frame = _price_frame()
    fyx22_frame.loc[fyx22_frame.index[-1], "rps50"] = 100.0
    fyx22_frame.loc[fyx22_frame.index[-40], "low"] = 60.0
    fyx22 = calculate_monthly_reversal_features(fyx22_frame).iloc[-1]
    assert not bool(fyx22["fyx21"])
    assert bool(fyx22["fyx22"])
    assert not bool(fyx22["fyx23"])
    assert bool(fyx22["fyx2"])

    fyx23_frame = _price_frame()
    fyx23_frame.loc[fyx23_frame.index[-50] :, "low"] = 80.0
    fyx23_frame.loc[fyx23_frame.index[-20] :, "low"] = 90.0
    fyx23_frame.loc[fyx23_frame.index[-10] :, "low"] = 95.0
    fyx23 = calculate_monthly_reversal_features(fyx23_frame).iloc[-1]
    assert not bool(fyx23["fyx21"])
    assert not bool(fyx23["fyx22"])
    assert bool(fyx23["fyx23"])
    assert bool(fyx23["fyx2"])

    all_false = calculate_monthly_reversal_features(_price_frame()).iloc[-1]
    assert not bool(all_false["fyx2"])


def test_fyx31_counts_each_days_own_nh80_in_last_ten_rows() -> None:
    frame = _price_frame()
    frame.loc[frame.index[-20] :, "high"] = 90.0
    frame.loc[frame.index[-5], "high"] = 110.0

    result = calculate_monthly_reversal_features(frame)

    assert bool(result.iloc[-5]["nh80"])
    assert not bool(result.iloc[-1]["nh80"])
    assert bool(result.iloc[-1]["fyx31"])


def test_fyx31_waits_until_all_ten_nh80_values_are_evaluable() -> None:
    result = calculate_monthly_reversal_features(_price_frame(89))

    assert not bool(result.iloc[79]["fyx31"])
    assert not bool(result.iloc[87]["fyx31"])
    assert bool(result.iloc[88]["fyx31"])


@pytest.mark.parametrize("new_high_field", ["close", "high"])
def test_fyx32_accepts_close_or_high_50_row_new_high(new_high_field: str) -> None:
    frame = _price_frame()
    frame.loc[frame.index[-2], ["open", "high", "low", "close", "adj_close"]] = 110.0
    frame.loc[frame.index[-1], "rps50"] = 90.0
    if new_high_field == "close":
        frame.loc[frame.index[-1], ["open", "close", "adj_close"]] = 120.0
        frame.loc[frame.index[-1], "high"] = 120.0
        frame.loc[frame.index[-1], "low"] = 100.0
    else:
        frame.loc[frame.index[-1], "high"] = 120.0

    result = calculate_monthly_reversal_features(frame).iloc[-1]

    assert bool(result["fyx32"])


def test_fyx4_uses_strict_close_and_ratio_comparisons() -> None:
    equal_close = calculate_monthly_reversal_features(_price_frame()).iloc[-1]
    assert equal_close["adj_close"] == equal_close["ma20"]
    assert not bool(equal_close["fyx4"])

    rising = _price_frame()
    rising.loc[rising.index[-1], ["open", "high", "low", "close", "adj_close"]] = 110.0
    result = calculate_monthly_reversal_features(rising).iloc[-1]
    assert result["adj_close"] > result["ma20"]
    assert result["adj_close"] > result["ma200"]
    assert result["ma120"] / result["ma200"] > 0.9
    assert bool(result["fyx4"])


@pytest.mark.parametrize(
    ("above_days", "expected_count", "expected_fyx51"),
    [(2, 2.0, False), (3, 3.0, True), (45, 45.0, False)],
)
def test_fyx51_count_boundaries(
    above_days: int,
    expected_count: float,
    expected_fyx51: bool,
) -> None:
    frame = _price_frame()
    columns = ["open", "high", "low", "close", "adj_close"]
    frame.loc[frame.index[-above_days] :, columns] = 200.0

    result = calculate_monthly_reversal_features(frame).iloc[-1]

    assert result["aa200"] == expected_count
    assert bool(result["fyx51"]) is expected_fyx51


def test_fyx52_uses_each_days_own_ma200_and_low_count() -> None:
    frame = _price_frame()
    columns = ["open", "high", "low", "close", "adj_close"]
    frame.loc[frame.index[-3] :, columns] = 200.0
    frame.loc[frame.index[-10], "low"] = 90.0

    result = calculate_monthly_reversal_features(frame)
    current = result.iloc[-1]

    expected_nn200 = result["adj_close"].gt(result["ma200"])
    expected_lnn200 = result["adj_low"].lt(result["ma200"])
    pd.testing.assert_series_equal(result["nn200"], expected_nn200, check_names=False)
    pd.testing.assert_series_equal(result["lnn200"], expected_lnn200, check_names=False)
    assert current["aa200"] == 3.0
    assert current["laa200"] == 1.0
    assert bool(current["fyx52"])
    assert bool(current["fyx5"])


@pytest.mark.parametrize(
    ("high", "threshold_column", "expected"),
    [(150.0, "fyx61", False), (159.0, "fyx62", True), (160.0, "fyx62", False)],
)
def test_fyx6_platform_width_uses_strict_ratio_thresholds(
    high: float,
    threshold_column: str,
    expected: bool,
) -> None:
    frame = _price_frame(close=120.0)
    frame["high"] = high
    frame["low"] = 100.0

    result = calculate_monthly_reversal_features(frame).iloc[-1]

    assert bool(result["fyx601"])
    assert bool(result["fyx602"])
    assert bool(result[threshold_column]) is expected


def test_fyx6_ma_ref_equality_satisfies_greater_or_equal() -> None:
    result = calculate_monthly_reversal_features(_price_frame()).iloc[-1]

    assert bool(result["fyx601"])
    assert bool(result["fyx602"])


def test_fyx7_uses_strict_boundaries_and_fyx72_depends_on_fyx13() -> None:
    ratio_85 = _price_frame(close=80.0)
    ratio_85.loc[ratio_85.index[-50], "high"] = 100.0
    ratio_85.loc[ratio_85.index[-5] :, "high"] = 85.0
    at_85 = calculate_monthly_reversal_features(ratio_85).iloc[-1]
    assert at_85["hhv_h_5"] / at_85["hhv_h_120"] == pytest.approx(0.85)
    assert not bool(at_85["fyx71"])
    assert not bool(at_85["fyx72"])

    ratio_80 = _price_frame(close=75.0)
    ratio_80.loc[ratio_80.index[-50], "high"] = 100.0
    ratio_80.loc[ratio_80.index[-5] :, "high"] = 80.0
    ratio_80.loc[ratio_80.index[-1], "rps50"] = 100.0
    at_80 = calculate_monthly_reversal_features(ratio_80).iloc[-1]
    assert at_80["hhv_h_5"] / at_80["hhv_h_120"] == pytest.approx(0.80)
    assert not bool(at_80["fyx72"])

    ratio_90 = _price_frame(close=90.0)
    ratio_90["high"] = 100.0
    at_90 = calculate_monthly_reversal_features(ratio_90).iloc[-1]
    assert at_90["adj_close"] / at_90["hhv_h_10"] == pytest.approx(0.90)
    assert not bool(at_90["fyx73"])
    assert not bool(at_90["fyx7"])


def test_yxfz_is_exactly_all_seven_fyx_modules() -> None:
    all_true = pd.DataFrame({f"fyx{index}": [True] for index in range(1, 8)})
    assert bool(calculate_yxfz(all_true).iloc[0])

    for failed_index in range(1, 8):
        values = all_true.copy()
        values.loc[0, f"fyx{failed_index}"] = False
        assert not bool(calculate_yxfz(values).iloc[0])


@pytest.mark.parametrize(
    ("prior_true_offset", "expected"),
    [(None, True), (10, False), (14, False), (15, True)],
)
def test_barssincen_style_signal_exact_15_row_semantics(
    prior_true_offset: int | None,
    expected: bool,
) -> None:
    yxfz = pd.Series(False, index=range(16))
    yxfz.iloc[-1] = True
    if prior_true_offset is not None:
        yxfz.iloc[-1 - prior_true_offset] = True

    signal = calculate_monthly_reversal_signal(yxfz)

    assert bool(signal.iloc[-1]) is expected


def test_signal_is_false_when_current_yxfz_is_false() -> None:
    assert not bool(calculate_monthly_reversal_signal(pd.Series([False] * 20)).iloc[-1])


def test_signal_requires_fourteen_actual_prior_rows() -> None:
    only_thirteen_prior = pd.Series([False] * 13 + [True])

    assert not bool(calculate_monthly_reversal_signal(only_thirteen_prior).iloc[-1])


@pytest.mark.parametrize("length", [100, 200, 249])
def test_insufficient_history_cannot_produce_yxfz_or_signal(length: int) -> None:
    frame = _price_frame(length)
    frame["rps50"] = 100.0
    frame["rps120"] = 100.0

    result = calculate_monthly_reversal_features(frame).iloc[-1]

    assert not bool(result["history_sufficient"])
    assert not bool(result["yxfz"])
    assert not bool(result["signal"])
    assert result["status"] == "insufficient_history"


def test_signal_history_requires_250_plus_14_ticker_rows() -> None:
    result = calculate_monthly_reversal_features(
        _price_frame(MONTHLY_REVERSAL_REQUIRED_SIGNAL_ROWS)
    )

    assert not bool(result.iloc[-2]["signal_history_sufficient"])
    assert bool(result.iloc[-1]["signal_history_sufficient"])


def test_future_price_changes_do_not_change_past_features_or_signals() -> None:
    frame = _price_frame(320)
    frame["rps50"] = 100.0
    cutoff = 299
    before = calculate_monthly_reversal_features(frame.iloc[: cutoff + 1]).iloc[-1]

    future_columns = ["open", "high", "low", "close", "adj_close"]
    frame.loc[frame.index[cutoff + 1] :, future_columns] = 10_000.0
    after = calculate_monthly_reversal_features(frame).iloc[cutoff]

    checked = [
        "ma20",
        "ma120",
        "ma200",
        "ma250",
        "hhv_h_120",
        "llv_l_200",
        *[f"fyx{index}" for index in range(1, 8)],
        "yxfz",
        "signal",
    ]
    for column in checked:
        if isinstance(before[column], (bool, np.bool_)):
            assert bool(before[column]) is bool(after[column])
        else:
            assert before[column] == pytest.approx(after[column])


def _write_universe(path: Path, tickers: tuple[str, ...]) -> None:
    rows = "".join(
        f"{ticker},{ticker} Inc.,{1000 - index},{index + 1}\n"
        for index, ticker in enumerate(tickers)
    )
    path.write_text(
        "ticker,company_name,market_cap,market_cap_rank\n" + rows,
        encoding="utf-8",
    )


def _write_strategy_prices(
    prices_root: Path,
    sessions: tuple[date, ...],
    tickers: tuple[str, ...],
) -> None:
    records: list[dict[str, object]] = []
    for ticker in tickers:
        closes = np.full(len(sessions), 100.0)
        if ticker == "PASS":
            closes[-4:] = [101.0, 102.0, 103.0, 110.0]
        for index, (session, close) in enumerate(zip(sessions, closes, strict=True)):
            low = 80.0 if ticker == "PASS" and index == len(sessions) - 40 else close
            records.append(
                {
                    "date": session,
                    "ticker": ticker,
                    "open": close,
                    "high": close + (1.0 if ticker == "PASS" else 0.0),
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": 100,
                }
            )
    frame = pd.DataFrame(records)
    for year, rows in frame.groupby(frame["date"].map(lambda value: value.year)):
        rows = rows.sort_values(["date", "ticker"], kind="mergesort")
        path = prices_root / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(rows, schema=PRICE_SCHEMA, preserve_index=False),
            path,
            compression="zstd",
        )


@pytest.fixture
def strategy_dataset(tmp_path: Path) -> tuple[Path, Path, date, pd.DataFrame]:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    as_of = date(2026, 8, 31)
    as_of_session = calendar.date_to_session(pd.Timestamp(as_of), direction="none")
    end_index = int(calendar.sessions.get_loc(as_of_session))
    sessions = tuple(
        value.date() for value in calendar.sessions[end_index - 349 : end_index + 1]
    )
    universe = ("PASS", "FAIL")
    universe_path = tmp_path / "universe.csv"
    prices_root = tmp_path / "prices"
    _write_universe(universe_path, universe)
    _write_strategy_prices(prices_root, sessions, universe)

    signal_sessions = sessions[-15:]
    rps_rows = pd.DataFrame(
        [
            {
                "date": session,
                "ticker": ticker,
                "rps50": 100.0 if ticker == "PASS" and session == as_of else -1.0,
                "rps120": -1.0,
            }
            for session in signal_sessions
            for ticker in universe
        ]
    )
    return universe_path, prices_root, as_of, rps_rows


def test_screen_and_evaluate_share_formula_and_reuse_injected_rps(
    strategy_dataset: tuple[Path, Path, date, pd.DataFrame],
) -> None:
    universe_path, prices_root, as_of, rps_rows = strategy_dataset
    rps_rows = rps_rows.set_index("ticker", drop=False)

    screen = screen_monthly_reversal(
        as_of,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_snapshots=rps_rows,
    )
    explanation = evaluate_monthly_reversal(
        "PASS",
        as_of,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_snapshots=rps_rows,
    )

    assert screen["ticker"].tolist() == ["PASS"]
    assert bool(explanation["yxfz"])
    assert bool(explanation["signal"])
    assert all(bool(explanation[f"fyx{index}"]) for index in range(1, 8))
    assert screen.attrs["universe_count"] == 2
    assert screen.attrs["fyx1_candidate_count"] == 1
    assert screen.attrs["yxfz_count"] == 1
    assert screen.attrs["signal_count"] == 1
    assert screen.attrs["rps_snapshot_count"] == 15
    assert screen.attrs["fyx1_prefilter_used"] is True

    raw = screen_monthly_reversal(
        as_of,
        signal_only=False,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_snapshots=rps_rows,
    )
    assert raw["ticker"].tolist() == ["PASS"]


def test_missing_rps_is_explainable_and_cannot_satisfy_fyx1(
    strategy_dataset: tuple[Path, Path, date, pd.DataFrame],
) -> None:
    universe_path, prices_root, as_of, rps_rows = strategy_dataset
    rps_rows.loc[rps_rows["ticker"].eq("PASS"), ["rps50", "rps120"]] = -1.0

    explanation = evaluate_monthly_reversal(
        "PASS",
        as_of,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_snapshots=rps_rows,
    )

    assert not bool(explanation["rps50_available"])
    assert not bool(explanation["rps120_available"])
    assert not bool(explanation["fyx1"])
    assert not bool(explanation["fyx130"])
    assert explanation["status"] == "rps_unavailable"


def _minimal_rps_rows(
    sessions: tuple[date, ...],
    tickers: tuple[str, ...],
    *,
    value: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": session,
                "ticker": ticker,
                "rps50": value,
                "rps120": value,
            }
            for session in sessions
            for ticker in tickers
        ]
    )


def test_get_rps_rows_prefers_persisted_history_and_injected_current_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = (date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31))
    universe = ("AAA", "BBB")
    rps_root = tmp_path / "rps"
    rps_root.mkdir()
    (rps_root / "manifest.json").write_text("{}", encoding="utf-8")
    persisted = _minimal_rps_rows(sessions, universe, value=40.0)
    injected = _minimal_rps_rows((sessions[-1],), universe, value=90.0)

    monkeypatch.setattr(
        strategy_data_module,
        "read_rps_history",
        lambda **_kwargs: persisted,
    )

    def unexpected_calculation(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("complete persisted/injected RPS must not be recalculated")

    monkeypatch.setattr(
        strategy_data_module,
        "calculate_rps_snapshots",
        unexpected_calculation,
    )

    result = strategy_data_module.load_or_calculate_rps(
        sessions,
        lookbacks=(50, 120),
        prices_root=tmp_path / "prices",
        universe_path=tmp_path / "universe.csv",
        price_rows=pd.DataFrame(),
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=injected.set_index("ticker", drop=False),
    )

    assert len(result) == len(sessions) * len(universe)
    assert set(result.loc[result["date"].eq(sessions[-1]), "rps50"]) == {90.0}
    assert set(result.loc[result["date"].ne(sessions[-1]), "rps50"]) == {40.0}


def test_get_rps_rows_calculates_only_missing_historical_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = (date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31))
    universe = ("AAA", "BBB")
    rps_root = tmp_path / "rps"
    rps_root.mkdir()
    (rps_root / "manifest.json").write_text("{}", encoding="utf-8")
    persisted = _minimal_rps_rows((sessions[0],), universe, value=40.0)
    injected = _minimal_rps_rows((sessions[-1],), universe, value=90.0)
    calculated_sessions: list[tuple[date, ...]] = []

    monkeypatch.setattr(
        strategy_data_module,
        "read_rps_history",
        lambda **_kwargs: persisted,
    )

    def calculate_missing(
        requested: tuple[date, ...],
        **_kwargs: object,
    ) -> pd.DataFrame:
        calculated_sessions.append(requested)
        return _minimal_rps_rows(requested, universe, value=60.0)

    monkeypatch.setattr(
        strategy_data_module,
        "calculate_rps_snapshots",
        calculate_missing,
    )

    result = strategy_data_module.load_or_calculate_rps(
        sessions,
        lookbacks=(50, 120),
        prices_root=tmp_path / "prices",
        universe_path=tmp_path / "universe.csv",
        price_rows=pd.DataFrame(),
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=injected,
    )

    assert calculated_sessions == [(sessions[1],)]
    assert len(result) == len(sessions) * len(universe)
    assert set(result.loc[result["date"].eq(sessions[1]), "rps50"]) == {60.0}
