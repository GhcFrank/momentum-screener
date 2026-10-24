from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import momentum_screener.strategy_data as data
from momentum_screener import local_price_data
from momentum_screener.rps import calculate_rps_snapshots
from momentum_screener.rps_storage import (
    RPS_DATA_COLUMNS,
    RpsStorageError,
    persist_rps_snapshot,
)


def rps_rows(sessions: tuple[date, ...], *, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": session, "ticker": ticker, **{f"rps{n}": 95.0 for n in lookbacks}}
            for session in sessions
            for ticker in ("AAA", "BBB")
        ]
    )


@pytest.mark.parametrize("lookbacks", [(20, 50), (120, 250), (50, 120, 250), (10, 75)])
def test_complete_injected_horizons_skip_read_and_calculation(
    lookbacks: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = (date(2026, 9, 2), date(2026, 9, 3))
    rows = rps_rows(sessions, lookbacks=lookbacks)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("complete injection must not read or calculate RPS")

    monkeypatch.setattr(data, "read_rps_history", unexpected)
    monkeypatch.setattr(data, "calculate_rps_snapshots", unexpected)
    root = tmp_path / "rps"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    result = data.load_or_calculate_rps(
        sessions,
        lookbacks=lookbacks,
        universe=("AAA", "BBB"),
        rps_root=root,
        rps_snapshots=rows,
    )
    pd.testing.assert_frame_equal(result, rows)


def test_partial_horizon_fallback_preserves_supplied_values_and_full_market_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = date(2026, 9, 3)
    rows = rps_rows((session,), lookbacks=(120,))
    rows["rps120"] = 42.0
    calls: list[object] = []

    def calculate(sessions: tuple[date, ...], **kwargs: object) -> pd.DataFrame:
        calls.append((sessions, kwargs))
        assert kwargs["universe"] == ("AAA", "BBB")
        assert kwargs["result_tickers"] == ("AAA",)
        return rps_rows(sessions, lookbacks=(120, 250))

    monkeypatch.setattr(data, "calculate_rps_snapshots", calculate)
    result = data.load_or_calculate_rps(
        (session,),
        lookbacks=(120, 250),
        universe=("AAA", "BBB"),
        rps_root=None,
        rps_snapshots=rows,
        result_tickers=("AAA",),
    )
    assert len(calls) == 1
    assert result["ticker"].tolist() == ["AAA"]
    assert result["rps120"].tolist() == [42.0]
    assert result["rps250"].tolist() == [95.0]


@pytest.mark.parametrize("value", [-1.0, np.nan])
def test_explicit_unavailable_results_do_not_trigger_fallback(
    value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = date(2026, 9, 3)
    rows = rps_rows((session,), lookbacks=(50, 120))
    rows.loc[:, ["rps50", "rps120"]] = value

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit unavailable values must remain unavailable")

    monkeypatch.setattr(data, "calculate_rps_snapshots", unexpected)
    result = data.load_or_calculate_rps(
        (session,),
        lookbacks=(50, 120),
        universe=("AAA", "BBB"),
        rps_root=None,
        rps_snapshots=rows,
    )
    pd.testing.assert_frame_equal(result, rows)


def test_real_fallback_is_persistable_and_stored_snapshot_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = ("AAA", "BBB")
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("ticker\nAAA\nBBB\n", encoding="utf-8")
    sessions = data.resolve_strategy_sessions(date(2026, 9, 3), 320)
    prices = pd.DataFrame(
        [
            {
                "date": session,
                "ticker": ticker,
                "adj_close": 10 + index * (0.1 if ticker == "AAA" else 0.01),
            }
            for index, session in enumerate(sessions)
            for ticker in universe
        ]
    )
    expected = calculate_rps_snapshots(
        (sessions[-1],), price_rows=prices, universe=universe
    )
    result = data.load_or_calculate_rps(
        (sessions[-1],), price_rows=prices, universe=universe, rps_root=None
    )
    pd.testing.assert_frame_equal(
        result.loc[:, RPS_DATA_COLUMNS],
        expected.rename(columns={"as_of_date": "date"}).loc[:, RPS_DATA_COLUMNS],
    )
    root = tmp_path / "rps"
    persisted = persist_rps_snapshot(result, root=root, universe_path=universe_path)
    assert persisted["rows_persisted"] == 2

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("persisted session must not be calculated again")

    monkeypatch.setattr(data, "calculate_rps_snapshots", unexpected)
    restored = data.load_or_calculate_rps(
        (sessions[-1],), universe_path=universe_path, rps_root=root
    )
    pd.testing.assert_frame_equal(
        result.loc[:, RPS_DATA_COLUMNS],
        restored.loc[:, RPS_DATA_COLUMNS],
        check_dtype=False,
    )
    subset = data.load_or_calculate_rps(
        (sessions[-1],), lookbacks=(20, 50), universe_path=universe_path, rps_root=root
    )
    pd.testing.assert_frame_equal(
        subset[["rps20", "rps50"]], restored[["rps20", "rps50"]]
    )


def test_price_rps_join_never_forward_fills_and_rejects_duplicate_keys() -> None:
    sessions = (date(2026, 9, 2), date(2026, 9, 3))
    prices = pd.DataFrame(
        {"date": sessions, "ticker": "AAA", "close": [10.0, 11.0], "rps250": 999.0}
    )
    rows = rps_rows((sessions[0],), lookbacks=(120, 250))
    merged = data.merge_prices_and_rps(prices, rows, lookbacks=(120, 250))
    assert merged.iloc[0]["rps250"] == 95
    assert pd.isna(merged.iloc[1]["rps250"])
    with pytest.raises(pd.errors.MergeError):
        data.merge_prices_and_rps(prices, pd.concat([rows, rows]), lookbacks=(120, 250))


def test_nonempty_rps_root_without_manifest_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "orphan.parquet").write_bytes(b"invalid")
    with pytest.raises(RpsStorageError, match="manifest"):
        data.load_or_calculate_rps(
            (date(2026, 9, 3),), universe=("AAA",), rps_root=tmp_path
        )


def test_sessions_count_is_inclusive_and_rejects_invalid_counts() -> None:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 3), 3)
    assert sessions == (date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3))
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            data.resolve_strategy_sessions(date(2026, 9, 3), count)


def test_signal_volumes_use_prior_sessions_and_preserve_selection() -> None:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 8), 23)
    volumes = np.arange(23, dtype="float64")
    volumes[5] = 0  # A valid zero belongs in both historical averages.
    volumes[-1] = 50_000_000  # Signal-day volume must not enter its own average.
    prices = pd.DataFrame({"date": sessions, "ticker": "AAA", "volume": volumes})
    prices = pd.concat(
        [
            prices,
            pd.DataFrame(
                {
                    "date": [date(2026, 9, 7), date(2026, 9, 9)],
                    "ticker": "AAA",
                    "volume": 99_000_000_000,
                }
            ),
        ],
        ignore_index=True,
    )  # Labor Day and future observations are excluded by the trading calendar.
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp(sessions[-1]), pd.Timestamp(sessions[-2])] * 2,
            "ticker": ["aaa", "AAA"] * 2,
            "strategy": ["monthly", "trend", "blue", "watch"],
            "signal": [True, True, False, True],
            "volume": -100,
        },
        index=[8, 3, 8, 5],
    )
    signals.attrs["strategy_version"] = "unchanged"
    original = signals.copy()
    result = data.load_signal_volumes(signals, price_rows=prices)
    assert len(result) == 2
    by_day = result.set_index("date")
    assert by_day.loc[sessions[-1], "volume"] == 50_000_000
    assert by_day.loc[sessions[-1], "avg_volume_20"] == pytest.approx(
        volumes[2:22].mean()
    )
    assert by_day.loc[sessions[-2], "volume"] == volumes[-2]
    assert by_day.loc[sessions[-2], "avg_volume_20"] == pytest.approx(
        volumes[1:21].mean()
    )
    assert result["avg_volume_50"].isna().all()
    enriched = data.enrich_signal_volume(signals, result)
    pd.testing.assert_frame_equal(
        enriched.loc[:, original.columns.drop("volume")],
        original.drop(columns="volume"),
    )
    pd.testing.assert_frame_equal(signals, original)
    assert enriched.attrs == original.attrs
    assert enriched["volume"].tolist() == [50_000_000, volumes[-2]] * 2
    assert enriched["avg_volume_20"].iloc[0] == enriched["avg_volume_20"].iloc[2]
    assert enriched["avg_volume_50"].isna().all()
    missing_signal = data.load_signal_volumes(
        signals.iloc[:1], price_rows=prices.loc[prices["date"].ne(sessions[-1])]
    )
    assert pd.isna(missing_signal["volume"].item())
    assert missing_signal["avg_volume_20"].item() == pytest.approx(volumes[2:22].mean())


@pytest.mark.parametrize(
    "unavailable", ["short", "missing_day", "nan", "negative", "inf"]
)
def test_signal_volume_window_never_skips_missing_or_invalid_history(
    unavailable: str,
) -> None:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 8), 52)
    prices = pd.DataFrame({"date": sessions, "ticker": "AAA", "volume": 100.0})
    if unavailable == "short":
        prices = prices.iloc[32:].copy()  # Nineteen historical sessions plus signal.
    elif unavailable == "missing_day":
        prices = prices.drop(index=40)  # An older valid day cannot replace this day.
    else:
        prices.loc[40, "volume"] = {
            "nan": np.nan,
            "negative": -1,
            "inf": np.inf,
        }[unavailable]
    prices.loc[prices["date"].eq(sessions[-1]), "volume"] = 0
    keys = pd.DataFrame({"date": [sessions[-1]], "ticker": ["AAA"]})
    result = data.load_signal_volumes(keys, price_rows=prices)
    assert result["volume"].item() == 0
    assert pd.isna(result["avg_volume_20"].item())
    assert pd.isna(result["avg_volume_50"].item())


def test_volume_averages_share_calendar_and_exclude_current_day() -> None:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 9), 52)
    volumes = pd.Series(np.arange(52, dtype="float64"), index=pd.Index(sessions))
    volumes.iloc[10] = 0
    volumes.iloc[50:] = [50_000_000, 90_000_000]
    signal_date = sessions[50]
    # Reversed input and the holiday preserve their positions in the output;
    # neither observation order nor an off-calendar row changes the window.
    volumes.loc[date(2026, 9, 7)] = 999_000_000
    volumes = volumes.iloc[::-1]
    result = data.compute_volume_averages(volumes)
    assert result.index.equals(volumes.index)
    assert result.loc[date(2026, 9, 7)].isna().all()
    assert pd.isna(
        result.loc[sessions[49], "avg_volume_50"]
    )  # Forty-nine historical sessions are insufficient.
    assert result.loc[signal_date, "avg_volume_50"] == pytest.approx(
        volumes.reindex(sessions[:50]).mean()
    )
    assert result.loc[signal_date, "avg_volume_20"] == pytest.approx(
        volumes.reindex(sessions[30:50]).mean()
    )
    keys = pd.DataFrame({"date": [signal_date], "ticker": ["AAA"]})
    prices = (
        volumes.rename("volume").rename_axis("date").reset_index().assign(ticker="AAA")
    )
    email = data.load_signal_volumes(keys, price_rows=prices)
    assert email["volume"].item() == 50_000_000
    for column in ("avg_volume_20", "avg_volume_50"):
        assert email[column].item() == result.loc[signal_date, column]
    missing_older_day = data.compute_volume_averages(volumes.drop(index=sessions[10]))
    assert pd.isna(missing_older_day.loc[signal_date, "avg_volume_50"])
    assert (
        missing_older_day.loc[signal_date, "avg_volume_20"]
        == result.loc[signal_date, "avg_volume_20"]
    )
    without_signal = data.load_signal_volumes(
        keys, price_rows=prices.loc[prices["date"].ne(signal_date)]
    )
    assert pd.isna(without_signal["volume"].item())
    assert (
        without_signal["avg_volume_50"].item()
        == result.loc[signal_date, "avg_volume_50"]
    )


def test_signal_volume_reads_one_projected_batch_for_selected_tickers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 8), 51)
    prices = pd.DataFrame(
        [
            {"date": day, "ticker": ticker, "volume": 200, "adj_close": 9.0}
            for day in (*sessions, date(2026, 9, 9))
            for ticker in ("AAA", "BBB", "UNSELECTED")
        ]
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    path = tmp_path / "prices.parquet"
    pq.write_table(pa.Table.from_pandas(prices, preserve_index=False), path)
    sources = tuple(local_price_data.LocalFile.inspect(p) for p in (manifest, path))
    files = Mock(return_value=sources)
    read = Mock(wraps=local_price_data.pq.read_table)
    monkeypatch.setattr(data, "price_files_in_range", files)
    monkeypatch.setattr(local_price_data.pq, "read_table", read)
    keys = pd.DataFrame({"date": sessions[-1], "ticker": ["AAA", "BBB", "aaa", "BBB"]})
    result = data.load_signal_volumes(keys, prices_root=tmp_path)
    assert result["ticker"].tolist() == ["AAA", "BBB"]
    assert result["volume"].tolist() == [200, 200]
    assert result["avg_volume_20"].tolist() == [200, 200]
    assert result["avg_volume_50"].tolist() == [200, 200]
    files.assert_called_once_with(sessions[0], sessions[-1], prices_root=tmp_path)
    assert read.call_count == 1
    assert read.call_args.kwargs == {
        "columns": ["date", "ticker", "volume"],
        "filters": [
            ("ticker", "in", ["AAA", "BBB"]),
            ("date", ">=", sessions[0]),
            ("date", "<=", sessions[-1]),
        ],
    }
    files.reset_mock()
    read.reset_mock()
    empty = data.load_signal_volumes(keys.iloc[:0], prices_root=tmp_path)
    assert empty.columns.tolist() == [
        "date",
        "ticker",
        "volume",
        "avg_volume_20",
        "avg_volume_50",
    ]
    files.assert_not_called()
    read.assert_not_called()
