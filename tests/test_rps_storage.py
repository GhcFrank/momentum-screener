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
    RPS_LOOKBACKS,
    calculate_rps_snapshot,
)
from momentum_screener.rps_storage import (
    RPS_DATA_COLUMNS,
    RPS_SCHEMA_VERSION,
    backfill_rps_history,
    calculate_rps_history,
    load_rps_manifest,
    migrate_rps_history,
    persist_rps_snapshot,
    read_rps_history,
    read_rps_snapshot,
    read_stock_rps_history,
    rps_columns,
    rps_schema,
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
            "rps20": [30.0 + index for index in range(count)],
            "rps50": values50,
            "rps120": [10.0 + index for index in range(count)],
            "rps250": [20.0 + index for index in range(count)],
            "return_20": [0.05] * count,
            "return_50": [np.nan, *[0.1] * (count - 1)],
            "return_120": [0.2] * count,
            "return_250": [0.3] * count,
            "rps20_base_date": date(2026, 8, 3),
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
    assert restored["rps20"].tolist() == [30.0, 31.0]
    assert restored["return_20"].eq(0.05).all()
    assert restored["rps20_base_date"].eq(date(2026, 8, 3)).all()
    assert restored["rps50"].tolist() == [INVALID_RPS, 100.0]
    assert restored["rps120"].tolist() == [10.0, 11.0]
    assert restored["rps250"].tolist() == [20.0, 21.0]
    assert pd.isna(restored.loc[0, "return_50"])
    assert manifest["schema_version"] == RPS_SCHEMA_VERSION
    assert manifest["lookbacks"] == [20, 50, 120, 250]
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
                "rps20",
                "rps50",
                "rps120",
                "rps250",
                "return_20",
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


def test_v1_migration_preserves_old_values_and_is_repeatable(tmp_path, monkeypatch):
    from momentum_screener.prices import universe_sha256
    from momentum_screener.storage_manifest import (
        build_asset_record,
        write_json_atomically,
    )

    root = tmp_path / "rps"
    prices_root = tmp_path / "prices"
    universe_path = tmp_path / "universe.csv"
    tickers = ("AAA", "BBB")
    _write_universe(universe_path, tickers)
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    end = calendar.sessions.get_loc(pd.Timestamp("2026-08-31"))
    sessions = tuple(value.date() for value in calendar.sessions[end - 279 : end + 1])
    prices = _write_prices(prices_root, sessions, tickers)
    legacy_horizons = (50, 120, 250)
    old = calculate_rps_history(
        prices,
        universe=tickers,
        start_date=sessions[-2],
        end_date=sessions[-1],
        lookbacks=legacy_horizons,
    )
    old.loc[0, "rps50"] = 42.0  # Preserve a distinct historical observation exactly.
    path = root / "daily" / "year=2026" / "rps.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(
            old, schema=rps_schema(legacy_horizons), preserve_index=False
        ),
        path,
    )
    previous = rps_storage_module._manifest_for_partitions(
        universe=tickers, partitions={2026: (path, len(old))}
    )
    previous.update(schema_version="rps_v1", lookbacks=list(legacy_horizons))
    write_json_atomically(root / "manifest.json", previous)
    with pytest.raises(rps_storage_module.RpsStorageError, match="migration required"):
        load_rps_manifest(root)
    price_assets = {
        str(year): build_asset_record(
            prices_root / f"daily/year={year}/prices.parquet",
            asset_name=f"prices-year-{year}.parquet",
            local_path=f"daily/year={year}/prices.parquet",
        )
        for year in {session.year for session in sessions}
    }
    monkeypatch.setattr(
        rps_storage_module,
        "load_price_manifest",
        lambda _: {
            "universe_sha256": universe_sha256(tickers),
            "latest_session": str(sessions[-1]),
            "requested_start": str(sessions[0]),
            "actual_min_date": str(sessions[0]),
            "assets": price_assets,
        },
    )
    price_path = prices_root / f"daily/year={sessions[-1].year}/prices.parquet"
    committed_prices = price_path.read_bytes()
    price_path.write_bytes(committed_prices[:-1])
    with pytest.raises(
        rps_storage_module.RpsStorageError, match="complete committed price"
    ):
        migrate_rps_history(
            root=root, prices_root=prices_root, universe_path=universe_path
        )
    assert load_rps_manifest(root, allow_legacy=True)["schema_version"] == "rps_v1"
    price_path.write_bytes(committed_prices)
    result = migrate_rps_history(
        root=root, prices_root=prices_root, universe_path=universe_path
    )
    restored = read_rps_history(root=root)
    pd.testing.assert_frame_equal(restored.loc[:, rps_columns(legacy_horizons)], old)
    expected = calculate_rps_history(
        prices,
        universe=tickers,
        start_date=sessions[-2],
        end_date=sessions[-1],
        lookbacks=(20,),
    )
    pd.testing.assert_frame_equal(restored.loc[:, rps_columns((20,))], expected)
    assert Path(result["archive"]).joinpath("manifest.json").is_file()
    before = path.read_bytes()
    assert (
        migrate_rps_history(
            root=root, prices_root=prices_root, universe_path=universe_path
        )["status"]
        == "already_current"
    )
    assert path.read_bytes() == before
    assert load_rps_manifest(root)["lookbacks"] == list(RPS_LOOKBACKS)
