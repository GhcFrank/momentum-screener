from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.rps_storage as rps_storage_module
from momentum_screener.rps import (
    INVALID_RPS,
    RPS_CALENDAR_NAME,
    calculate_rps_snapshot,
)
from momentum_screener.rps_storage import (
    RPS_DATA_COLUMNS,
    RPS_SCHEMA_VERSION,
    backfill_rps_history,
    calculate_rps_history,
    load_rps_manifest,
    persist_rps_snapshot,
    read_rps_history,
    read_rps_snapshot,
    read_stock_rps_history,
    validate_rps_dataset,
)
from momentum_screener.storage_manifest import PRICE_SCHEMA


def _write_universe(path: Path, tickers: tuple[str, ...]) -> None:
    rows = "".join(
        f"{ticker},{ticker} Inc.,{1000 - index},{index + 1}\n"
        for index, ticker in enumerate(tickers)
    )
    path.write_text(
        "ticker,company_name,market_cap,market_cap_rank\n" + rows,
        encoding="utf-8",
    )


def _snapshot(
    session: date,
    tickers: tuple[str, ...],
    *,
    rps50: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    count = len(tickers)
    values50 = rps50 or tuple(float(index) for index in range(count))
    return pd.DataFrame(
        {
            "ticker": tickers,
            "as_of_date": session,
            "rps50": values50,
            "rps120": [10.0 + index for index in range(count)],
            "rps250": [20.0 + index for index in range(count)],
            "return_50": [np.nan, *[0.1] * (count - 1)],
            "return_120": [0.2] * count,
            "return_250": [0.3] * count,
            "rps50_base_date": date(2026, 6, 19),
            "rps120_base_date": date(2026, 3, 10),
            "rps250_base_date": date(2025, 9, 2),
        }
    )


def test_write_and_read_snapshot_preserves_all_default_horizons_and_invalids(
    tmp_path: Path,
) -> None:
    tickers = ("AAA", "BBB")
    universe_path = tmp_path / "universe.csv"
    root = tmp_path / "rps"
    _write_universe(universe_path, tickers)
    snapshot = _snapshot(
        date(2026, 8, 31), tickers, rps50=(INVALID_RPS, 100.0)
    ).set_index("ticker", drop=False)

    result = persist_rps_snapshot(
        snapshot,
        root=root,
        universe_path=universe_path,
    )
    restored = read_rps_snapshot(date(2026, 8, 31), root=root)
    manifest = validate_rps_dataset(root, universe_path=universe_path)

    assert result["rows_persisted"] == 2
    assert tuple(restored.columns) == RPS_DATA_COLUMNS
    assert restored["ticker"].tolist() == ["AAA", "BBB"]
    assert restored["rps50"].tolist() == [INVALID_RPS, 100.0]
    assert restored["rps120"].tolist() == [10.0, 11.0]
    assert restored["rps250"].tolist() == [20.0, 21.0]
    assert pd.isna(restored.loc[0, "return_50"])
    assert manifest["schema_version"] == RPS_SCHEMA_VERSION
    assert manifest["lookbacks"] == [50, 120, 250]
    assert manifest["price_field"] == "adj_close"
    assert manifest["universe_ticker_count"] == 2


def test_repeated_same_session_upsert_is_idempotent_and_replaces_values(
    tmp_path: Path,
) -> None:
    tickers = ("AAA", "BBB")
    universe_path = tmp_path / "universe.csv"
    root = tmp_path / "rps"
    _write_universe(universe_path, tickers)
    session = date(2026, 8, 31)

    persist_rps_snapshot(
        _snapshot(session, tickers), root=root, universe_path=universe_path
    )
    persist_rps_snapshot(
        _snapshot(session, tickers, rps50=(99.0, 98.0)),
        root=root,
        universe_path=universe_path,
    )

    restored = read_rps_snapshot(session, root=root)
    assert len(restored) == 2
    assert not bool(restored.duplicated(["date", "ticker"]).any())
    assert restored["rps50"].tolist() == [99.0, 98.0]
    assert load_rps_manifest(root)["total_row_count"] == 2


def test_history_read_apis_filter_dates_and_tickers(tmp_path: Path) -> None:
    tickers = ("AAA", "BBB")
    universe_path = tmp_path / "universe.csv"
    root = tmp_path / "rps"
    _write_universe(universe_path, tickers)
    first = date(2026, 8, 31)
    second = date(2026, 9, 1)
    persist_rps_snapshot(
        _snapshot(first, tickers), root=root, universe_path=universe_path
    )
    persist_rps_snapshot(
        _snapshot(second, tickers), root=root, universe_path=universe_path
    )

    one_stock = read_stock_rps_history("aaa", root=root)
    one_day = read_rps_history(start_date=second, end_date=second, root=root)

    assert one_stock["ticker"].tolist() == ["AAA", "AAA"]
    assert one_stock["date"].tolist() == [first, second]
    assert one_day["date"].tolist() == [second, second]


def _write_prices(
    root: Path,
    sessions: tuple[date, ...],
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        for ticker_index, ticker in enumerate(tickers):
            adjusted = 50.0 + ticker_index * 10.0 + session_index * (ticker_index + 1)
            records.append(
                {
                    "date": session,
                    "ticker": ticker,
                    "open": adjusted * 2.0,
                    "high": adjusted * 2.0,
                    "low": adjusted * 2.0,
                    "close": adjusted * 2.0,
                    "adj_close": adjusted,
                    "volume": 100,
                }
            )
    frame = pd.DataFrame(records).sort_values(
        ["date", "ticker"], kind="mergesort", ignore_index=True
    )
    for year, rows in frame.groupby(frame["date"].map(lambda value: value.year)):
        path = root / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(rows, schema=PRICE_SCHEMA, preserve_index=False),
            path,
            compression="zstd",
        )
    return frame.loc[:, ["date", "ticker", "adj_close"]]


def test_vectorized_history_matches_on_demand_snapshots() -> None:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    end_session = calendar.date_to_session(
        pd.Timestamp(date(2026, 8, 31)), direction="none"
    )
    end_index = int(calendar.sessions.get_loc(end_session))
    sessions = tuple(
        value.date() for value in calendar.sessions[end_index - 279 : end_index + 1]
    )
    tickers = ("AAA", "BBB", "CCC", "DDD")

    # Use a TemporaryDirectory rather than pytest's fixture so both APIs consume
    # exactly the same physical rows within this regression.
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        prices_root = base / "prices"
        universe_path = base / "universe.csv"
        _write_universe(universe_path, tickers)
        slim_prices = _write_prices(prices_root, sessions, tickers)
        requested = (sessions[-2], sessions[-1])
        historical = calculate_rps_history(
            slim_prices,
            universe=tickers,
            start_date=requested[0],
            end_date=requested[1],
        )

        for session in requested:
            on_demand = calculate_rps_snapshot(
                session,
                prices_root=prices_root,
                universe_path=universe_path,
            ).rename(columns={"as_of_date": "date"})
            actual = historical.loc[historical["date"].eq(session)].reset_index(
                drop=True
            )
            expected = on_demand.reset_index(drop=True)
            for column in (
                "rps50",
                "rps120",
                "rps250",
                "return_50",
                "return_120",
                "return_250",
            ):
                np.testing.assert_allclose(
                    actual[column], expected[column], equal_nan=True, rtol=0, atol=1e-12
                )


def test_backfill_persists_values_matching_on_demand_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    end_session = calendar.date_to_session(
        pd.Timestamp(date(2026, 8, 31)), direction="none"
    )
    end_index = int(calendar.sessions.get_loc(end_session))
    sessions = tuple(
        value.date() for value in calendar.sessions[end_index - 279 : end_index + 1]
    )
    tickers = ("AAA", "BBB", "CCC", "DDD")
    prices_root = tmp_path / "prices"
    rps_root = tmp_path / "rps"
    universe_path = tmp_path / "universe.csv"
    _write_universe(universe_path, tickers)
    _write_prices(prices_root, sessions, tickers)
    monkeypatch.setattr(
        rps_storage_module,
        "load_price_manifest",
        lambda _path: {
            "actual_min_date": sessions[0].isoformat(),
            "latest_session": sessions[-1].isoformat(),
        },
    )

    result = backfill_rps_history(
        start_date=sessions[-2],
        end_date=sessions[-1],
        prices_root=prices_root,
        root=rps_root,
        universe_path=universe_path,
    )

    assert result["rows_persisted"] == 2 * len(tickers)
    for session in sessions[-2:]:
        persisted = read_rps_snapshot(session, root=rps_root)
        on_demand = calculate_rps_snapshot(
            session,
            prices_root=prices_root,
            universe_path=universe_path,
        ).reset_index(drop=True)
        for column in (
            "rps50",
            "rps120",
            "rps250",
            "return_50",
            "return_120",
            "return_250",
        ):
            np.testing.assert_allclose(
                persisted[column],
                on_demand[column],
                equal_nan=True,
                rtol=0,
                atol=1e-12,
            )
