from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.strategy_data as data
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
