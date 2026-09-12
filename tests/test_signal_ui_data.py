"""Small local fixtures for research data, without testing Streamlit rendering."""

import json
import os
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from momentum_screener import signal_ui_data
from momentum_screener.rps_storage import RpsStorageError, persist_rps_snapshot
from momentum_screener.signal_ui_data import (
    LocalFile,
    build_ticker_rps_table,
    clip_price_window,
    combine_signals,
    discover_signal_csvs,
    filter_tickers_by_strategies,
    load_rps_for_session,
    load_turnover_for_session,
    local_market_cap_files,
    local_price_files,
    local_rps_files,
    maximum_price_window,
    read_local_prices,
    read_signal_csv,
)
from momentum_screener.storage_manifest import (
    PRICE_SCHEMA,
    SCHEMA_VERSION,
    build_asset_record,
)


def test_discover_single_csv_path(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("session,ticker\n2026-06-15,AAPL\n")

    files, warnings = discover_signal_csvs(["", f"  {path}  ", " \t"])

    assert files == [path.resolve()]
    assert warnings == []


def test_discover_directory_csvs_in_stable_order(tmp_path):
    # Create in reverse filename order; discovery must sort rather than depend
    # on filesystem enumeration order, and accept uppercase CSV extensions.
    for name in ("notes.txt", "b.CSV", "a.csv"):
        (tmp_path / name).write_text("")

    files, warnings = discover_signal_csvs([str(tmp_path)])

    assert files == [tmp_path / "a.csv", tmp_path / "b.CSV"]
    assert warnings == []


def test_discover_does_not_recurse_into_subdirectories(tmp_path):
    root = tmp_path / "root.csv"
    root.write_text("")
    for name in ("archive", "nested.csv"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "old.csv").write_text("")

    files, warnings = discover_signal_csvs([str(tmp_path)])

    assert files == [root]
    assert warnings == []


def test_discover_deduplicates_resolved_paths_and_preserves_input_order(
    tmp_path, monkeypatch
):
    first = tmp_path / "b.csv"
    second = tmp_path / "a.csv"
    first.write_text("")
    second.write_text("")
    (tmp_path / "alias.csv").symlink_to(first)
    monkeypatch.chdir(tmp_path)

    files, warnings = discover_signal_csvs([str(first), str(tmp_path), "./b.csv"])

    assert files == [first, second]
    assert warnings == []


def test_discover_bad_inputs_do_not_block_valid_csv(tmp_path):
    valid = tmp_path / "valid.csv"
    valid.write_text("")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("")
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing"

    files, warnings = discover_signal_csvs(
        [str(missing), str(valid), str(unsupported), str(empty)]
    )

    assert files == [valid]
    assert warnings == [
        f"Path does not exist: {missing}",
        f"Unsupported file type: {unsupported}",
        f"No CSV files found in: {empty}",
    ]


def test_csv_normalization_and_invalid_rows(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text(
        "session,ticker,strategy_id,strategy_version,score\n"
        "2026-06-15, brk.b ,monthly_reversal,6.2,12\n"
        "2026-06-16, brk-a ,trend_reacceleration,2,13\n"
        "2026-06-16,NA,trend_reacceleration,2,14\n"
        "not-a-date,AAPL,monthly_reversal,6.2,15\n"
        "2026-06-16, ,monthly_reversal,6.2,16\n"
    )
    frame, warnings = read_signal_csv(LocalFile.inspect(path))
    assert frame["session"].dt.date.tolist() == [
        date(2026, 6, 15),
        date(2026, 6, 16),
        date(2026, 6, 16),
    ]
    assert frame["ticker"].tolist() == ["BRK.B", "BRK-A", "NA"]
    assert frame["strategy_id"].tolist() == [
        "monthly_reversal",
        "trend_reacceleration",
        "trend_reacceleration",
    ]
    assert frame["strategy_version"].tolist() == ["6.2", "2", "2"]
    assert frame["score"].tolist() == [12, 13, 14]
    assert frame["source_file"].eq(str(path)).all()
    assert len(warnings) == 2
    assert "invalid dates" in warnings[0]
    assert "empty tickers" in warnings[1]


def test_date_and_strategy_filename_fallback(tmp_path):
    path = tmp_path / "custom_strategy.csv"
    path.write_text("date,ticker\n2026-06-15,nvda\n")
    frame, warnings = read_signal_csv(LocalFile.inspect(path))
    assert frame.loc[0, "session"] == pd.Timestamp("2026-06-15")
    assert frame.loc[0, "strategy_id"] == "custom_strategy"
    assert frame["strategy_version"].isna().all()
    assert warnings == []


@pytest.mark.parametrize("contents", ["ticker\nAAPL\n", "session\n2026-06-15\n"])
def test_invalid_schema(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match="requires 'session' .* and 'ticker'"):
        read_signal_csv(LocalFile.inspect(path))


@pytest.mark.parametrize(
    "contents", ["", "session,ticker\n", "session,ticker\ninvalid,AAPL\n"]
)
def test_empty_or_entirely_invalid_csv(tmp_path, contents):
    path = tmp_path / "empty.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_signal_csv(LocalFile.inspect(path))


def test_merge_diagnostics_duplicates_and_reload_identity(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "session,ticker,strategy_id,score\n"
        "2026-06-15,AAPL,example,1\n2026-06-15,AAPL,example,1\n"
    )
    second.write_text(
        "date,ticker,strategy_id,score,extra\n"
        "2026-06-15,AAPL,example,2,diagnostic\n2026-06-15,NVDA,other,3,other\n"
    )
    original = LocalFile.inspect(first)
    frame1, _ = read_signal_csv(original)
    exact, exact_warnings = combine_signals([frame1])
    assert len(exact) == 1
    assert not any("conflicting" in warning for warning in exact_warnings)
    frame2, _ = read_signal_csv(LocalFile.inspect(second))
    combined, warnings = combine_signals([frame1, frame2])
    assert len(combined) == 2
    assert combined.loc[0, "score"] == 1
    assert pd.isna(combined.loc[0, "extra"])
    assert combined.loc[1, "extra"] == "other"
    assert "1 duplicate signal key(s) have conflicting" in warnings[0]
    assert "first row" in warnings[0]
    first.write_text("session,ticker\n2026-06-17,META\n")
    os.utime(first, ns=(original.mtime_ns + 1_000_000, original.mtime_ns + 1_000_000))
    updated = LocalFile.inspect(first)
    assert updated != original
    with pytest.raises(ValueError, match="changed while loading"):
        read_signal_csv(original)
    reloaded, _ = read_signal_csv(updated)
    assert reloaded.loc[0, "ticker"] == "META"


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (["A"], ["META", "NVDA"]),
        (["A", "B"], ["NVDA"]),
        (["A", "D"], []),
        (["A", "B", "C"], ["NVDA"]),
        ([], []),
        (["A", "missing"], []),
    ],
)
def test_filter_tickers_by_strategy_intersection(selected, expected):
    rows = [
        ("A", "NVDA"),
        ("A", "META"),
        ("A", "META"),
        ("B", "NVDA"),
        ("B", "APP"),
        ("C", "NVDA"),
        ("C", "MSFT"),
        ("D", "APP"),
    ]
    signals = pd.DataFrame(rows, columns=["strategy_id", "ticker"])
    signals["session"] = pd.Timestamp("2026-06-15")
    signals.loc[len(signals)] = ["A", "APP", pd.Timestamp("2026-06-16")]

    assert (
        filter_tickers_by_strategies(signals, date(2026, 6, 15), selected) == expected
    )
    assert filter_tickers_by_strategies(signals, date(2026, 6, 14), selected) == []


def test_ticker_rps_table_joins_snapshot_and_retains_missing_values():
    snapshot = pd.DataFrame(
        {
            "ticker": ["NVDA", "META"],
            "rps20": [98.0, 93.0],
            "rps50": [90.0, 88.0],
            "rps120": [95.0, 92.0],
            "rps250": [97.0, 94.0],
        }
    )
    table = build_ticker_rps_table(["NVDA", "META", "NVDA"], snapshot)
    assert table.to_dict("list") == {
        "Ticker": ["META", "NVDA"],
        "RPS20": [93.0, 98.0],
        "RPS50": [88.0, 90.0],
        "RPS120": [92.0, 95.0],
        "RPS250": [94.0, 97.0],
    }

    snapshot.loc[0, "rps50"] = -1  # Storage's INVALID_RPS sentinel is not a score.
    snapshot.loc[0, "rps20"] = -1
    snapshot.loc[1, "rps20"] = 101
    snapshot.loc[1, "rps120"] = float("nan")
    missing = build_ticker_rps_table(
        ["NVDA", "META", "UNKNOWN"], snapshot.drop(columns="rps250")
    ).set_index("Ticker")
    assert missing.index.tolist() == ["META", "NVDA", "UNKNOWN"]
    assert missing["RPS20"].isna().all()
    assert pd.isna(missing.loc["NVDA", "RPS50"])
    assert missing.loc["NVDA", "RPS120"] == 95.0
    assert missing.loc["META", "RPS50"] == 88.0
    assert pd.isna(missing.loc["META", "RPS120"])
    assert missing["RPS250"].isna().all()
    assert missing.loc["UNKNOWN"].isna().all()
    empty = build_ticker_rps_table(["NVDA"], pd.DataFrame())
    assert empty["Ticker"].tolist() == ["NVDA"]
    assert empty.drop(columns="Ticker").isna().all().all()


def test_load_local_rps_snapshot_once_and_refresh_after_replacement(
    tmp_path, monkeypatch
):
    session = date(2026, 6, 15)
    root = tmp_path / "rps"
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker\nNVDA\nMETA\n")
    snapshot = pd.DataFrame(
        {
            "ticker": ["NVDA", "META"],
            "as_of_date": session,
            "rps20": [89.0, 87.0],
            "rps50": [90.0, 88.0],
            "rps120": [95.0, 92.0],
            "rps250": [97.0, 94.0],
            **{f"return_{days}": 0.1 for days in (20, 50, 120, 250)},
            **{f"rps{days}_base_date": date(2025, 1, 2) for days in (20, 50, 120, 250)},
        }
    )
    persist_rps_snapshot(snapshot, root=root, universe_path=universe)
    files = local_rps_files(session, root)
    original_reader = signal_ui_data.read_rps_snapshot
    calls = []

    def read_snapshot(as_of_date, *, root):
        calls.append(as_of_date)
        return original_reader(as_of_date, root=root)

    monkeypatch.setattr(signal_ui_data, "read_rps_snapshot", read_snapshot)
    restored = load_rps_for_session(session, files, root)
    assert calls == [session]
    assert build_ticker_rps_table(["NVDA", "META"], restored)["RPS50"].tolist() == [
        88.0,
        90.0,
    ]
    assert load_rps_for_session(date(2026, 6, 16), files, root).empty

    snapshot.loc[0, "rps50"] = 99.0
    persist_rps_snapshot(snapshot, root=root, universe_path=universe)
    updated = local_rps_files(session, root)
    assert updated != files
    with pytest.raises(RpsStorageError, match="changed while loading"):
        load_rps_for_session(session, files, root)
    refreshed = load_rps_for_session(session, updated, root)
    assert build_ticker_rps_table(["NVDA"], refreshed).loc[0, "RPS50"] == 99.0
    with pytest.raises(FileNotFoundError):
        local_rps_files(session, tmp_path / "missing")


def local_turnover_file_inputs(tmp_path):
    paths = [
        tmp_path / "prices-manifest.json",
        tmp_path / "prices.parquet",
        tmp_path / "market-cap-manifest.json",
        tmp_path / "market-cap.parquet",
    ]
    for path in paths:
        path.write_text(path.name)
    files = [LocalFile.inspect(path) for path in paths]
    return tuple(files[:2]), tuple(files[2:])


def test_session_turnover_uses_raw_close_exact_cap_and_ticker_alignment(
    tmp_path, monkeypatch
):
    session = date(2026, 6, 15)
    price_files, market_cap_files = local_turnover_file_inputs(tmp_path)
    price_calls = []
    cap_calls = []

    def read_prices(tickers, start_date, end_date, files, *, columns):
        price_calls.append((tickers, start_date, end_date, files, columns))
        return pd.DataFrame(
            {
                "date": [session, session, session],
                "ticker": ["AAA", "MISSING", "BBB"],
                "close": [100.0, 50.0, 200.0],
                "adj_close": [1.0, 2.0, 3.0],
                "volume": [100.0, 100.0, 100.0],
            }
        )

    def read_caps(*, tickers, start_date, end_date, root):
        cap_calls.append((tickers, start_date, end_date, root))
        return pd.DataFrame(
            {
                "date": [session, session],
                "ticker": ["BBB", "AAA"],
                "market_cap": [1_000_000, 100_000],
            }
        )

    monkeypatch.setattr(signal_ui_data, "read_local_price_rows", read_prices)
    monkeypatch.setattr(signal_ui_data, "read_market_cap", read_caps)
    cap_root = tmp_path / "caps"
    result = load_turnover_for_session(
        session,
        ["BBB", "MISSING", "AAA", "BBB"],
        price_files,
        market_cap_files,
        cap_root,
    )

    assert result["ticker"].tolist() == ["BBB", "MISSING", "AAA"]
    assert result.loc[0, "turnover"] == pytest.approx(0.02)
    assert pd.isna(result.loc[1, "turnover"])
    assert result.loc[2, "turnover"] == pytest.approx(0.10)
    assert price_calls == [
        (
            ["BBB", "MISSING", "AAA"],
            session,
            session,
            (price_files[1],),
            ("close", "volume"),
        )
    ]
    assert cap_calls == [(["BBB", "MISSING", "AAA"], session, session, cap_root)]


def test_session_turnover_keeps_all_tickers_when_snapshot_date_is_absent(
    tmp_path, monkeypatch
):
    session = date(2025, 1, 2)
    price_files, market_cap_files = local_turnover_file_inputs(tmp_path)
    monkeypatch.setattr(
        signal_ui_data,
        "read_local_price_rows",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "date": [session, session],
                "ticker": ["AAA", "BBB"],
                "close": [10.0, 20.0],
                "volume": [100.0, 200.0],
            }
        ),
    )
    monkeypatch.setattr(
        signal_ui_data,
        "read_market_cap",
        lambda **kwargs: pd.DataFrame(columns=["date", "ticker", "market_cap"]),
    )

    result = load_turnover_for_session(
        session, ["AAA", "BBB"], price_files, market_cap_files, tmp_path
    )
    assert result["ticker"].tolist() == ["AAA", "BBB"]
    assert result["turnover"].isna().all()


def test_market_cap_cache_files_change_after_local_release_replacement(
    tmp_path, monkeypatch
):
    session = date(2026, 6, 15)
    root = tmp_path / "market-cap"
    manifest_path = root / "manifest.json"
    partition_path = root / "daily/year=2026/market_cap.parquet"
    partition_path.parent.mkdir(parents=True)
    manifest_path.write_text("old")
    partition_path.write_text("old partition")
    monkeypatch.setattr(
        signal_ui_data,
        "load_market_cap_manifest",
        lambda root: {
            "assets": {"2026": {"local_path": "daily/year=2026/market_cap.parquet"}}
        },
    )

    original = local_market_cap_files(session, root)
    manifest_path.write_text("new manifest payload")
    partition_path.write_text("new partition payload")
    updated = local_market_cap_files(session, root)
    assert updated != original


def test_price_window_clipping_and_calendar_offsets():
    signal_date = date(2026, 6, 15)
    assert maximum_price_window(signal_date) == (date(2024, 6, 15), date(2027, 6, 15))
    window = clip_price_window(signal_date, date(2020, 1, 1), date(2026, 9, 4))
    assert (window.start, window.end) == (date(2024, 6, 15), date(2026, 9, 4))
    assert (window.viewport_start, window.viewport_end) == (
        date(2026, 3, 15),
        date(2026, 7, 15),
    )
    clipped = clip_price_window(signal_date, date(2026, 4, 1), date(2026, 6, 30))
    assert (clipped.viewport_start, clipped.viewport_end) == (
        date(2026, 4, 1),
        date(2026, 6, 30),
    )
    assert not clipped.viewport_fallback
    assert maximum_price_window(date(2024, 2, 29)) == (
        date(2022, 2, 28),
        date(2025, 2, 28),
    )
    month_end = clip_price_window(
        date(2024, 3, 31), date(2020, 1, 1), date(2025, 12, 31)
    )
    assert (month_end.viewport_start, month_end.viewport_end) == (
        date(2023, 12, 31),
        date(2024, 4, 30),
    )
    assert clip_price_window(signal_date, date(2020, 1, 1), date(2023, 1, 1)) is None
    sparse = clip_price_window(signal_date, date(2025, 1, 1), date(2025, 2, 1))
    assert sparse.viewport_fallback
    assert (sparse.viewport_start, sparse.viewport_end) == (sparse.start, sparse.end)


def test_local_manifest_prices_use_adjusted_close_and_exact_bounds(tmp_path):
    # Full real storage schema, including deliberately different raw Close.
    dates = [
        date(2024, 6, 14),
        date(2024, 6, 15),
        date(2026, 6, 15),
        date(2027, 6, 15),
        date(2027, 6, 16),
    ]
    records = [
        {
            "date": day,
            "ticker": ticker,
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 100.0,
            "adj_close": 50.0 + index,
            "volume": 100,
        }
        for index, day in enumerate(dates)
        for ticker in ("AAPL", "MSFT")
    ]
    table = pa.Table.from_pylist(records, schema=PRICE_SCHEMA)
    assets, counts = {}, {}
    for year in (2024, 2026, 2027):
        path = tmp_path / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True)
        partition = table.filter(
            pa.array([row["date"].year == year for row in records])
        )
        pq.write_table(partition, path)
        assets[str(year)] = build_asset_record(
            path,
            asset_name=f"prices-year-{year}.parquet",
            local_path=str(path.relative_to(tmp_path)),
        )
        counts[str(year)] = partition.num_rows
    coverage = tmp_path / "ticker_coverage.csv"
    coverage.write_text(
        "ticker,status,first_date,last_date,row_count,attempt_count,last_error\n"
    )
    assets["ticker_coverage"] = build_asset_record(
        coverage, asset_name="prices-ticker-coverage.csv", local_path=coverage.name
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "completed": True,
                "source": "yahoo_finance_via_yfinance",
                "universe_sha256": "a" * 64,
                "universe_ticker_count": 2,
                "requested_start": "2024-01-01",
                "latest_session": "2027-06-16",
                "actual_min_date": "2024-06-14",
                "actual_max_date": "2027-06-16",
                "last_successful_update_utc": "2027-06-17T00:00:00+00:00",
                "total_row_count": len(records),
                "partition_row_counts": counts,
                "assets": assets,
            }
        )
    )
    signal_date = date(2026, 6, 15)
    files = local_price_files(signal_date, tmp_path)
    result = read_local_prices("AAPL", signal_date, files)
    assert result["date"].dt.date.tolist() == [
        date(2024, 6, 15),
        signal_date,
        date(2027, 6, 15),
    ]
    assert result["adjusted_close"].tolist() == [51.0, 52.0, 53.0]
    assert read_local_prices("UNKNOWN", signal_date, files).empty
    assert read_local_prices("AAPL", signal_date, ()).empty
    assert local_price_files(date(2032, 1, 1), tmp_path) == ()
