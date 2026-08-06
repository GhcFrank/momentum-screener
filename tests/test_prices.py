from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.prices as prices_module
import momentum_screener.release_storage as release_module
import momentum_screener.storage_manifest as manifest_module
from momentum_screener.prices import (
    COVERAGE_COLUMNS,
    DEFAULT_START,
    FAILURE_COLUMNS,
    PRICE_COLUMNS,
    PRICE_SCHEMA,
    BackfillIncompleteError,
    DataConflictError,
    DataValidationError,
    PriceBackfillError,
    PriceUpdateError,
    StagingError,
    UniverseReadError,
    build_batches,
    build_run_key,
    calculate_end_exclusive,
    calculate_refresh_start,
    determine_target_session,
    download_batch,
    execute_batch_with_retries,
    expected_active_tickers,
    index_release_assets,
    isolate_incompatible_staging,
    load_or_create_run_state,
    load_universe,
    main,
    normalize_download_frame,
    read_affected_partitions,
    rotate_price_output_to_legacy,
    run_backfill,
    run_update,
    universe_sha256,
    upsert_refresh_window,
    validate_backfill_dataset,
    validate_complete_dataset,
    validate_target_coverage,
    write_batch_atomically,
    write_year_partitions,
)

START = date(2016, 1, 1)
END = date(2021, 1, 1)
FIXED_NOW = datetime(2020, 12, 31, 17, 0, tzinfo=UTC)


def ordinary_frame(
    dates: Sequence[str] = ("2020-01-02",),
    *,
    close: Sequence[object] = (10.0,),
    adj_close: Sequence[object] = (9.0,),
    volume: Sequence[object] = (100,),
    extra: bool = False,
) -> pd.DataFrame:
    values: dict[str, Sequence[object]] = {
        "Close": close,
        "Adj Close": adj_close,
        "Volume": volume,
    }
    if extra:
        values["Repaired?"] = [True] * len(dates)
        values["Open"] = close
    return pd.DataFrame(values, index=pd.to_datetime(list(dates)))


def multi_frame(
    ticker_values: dict[str, tuple[float, float, int]],
    *,
    field_first: bool = False,
    dates: Sequence[str] = ("2020-01-02",),
    include_extra: bool = False,
) -> pd.DataFrame:
    columns: list[tuple[str, str]] = []
    values: list[list[object]] = []
    for ticker, (close, adjusted, volume) in ticker_values.items():
        for field, value in (
            ("Close", close),
            ("Adj Close", adjusted),
            ("Volume", volume),
        ):
            columns.append((field, ticker) if field_first else (ticker, field))
            values.append([value] * len(dates))
        if include_extra:
            columns.append(
                ("Repaired?", ticker) if field_first else (ticker, "Repaired?")
            )
            values.append([True] * len(dates))
    matrix = list(zip(*values, strict=True))
    return pd.DataFrame(
        matrix,
        index=pd.to_datetime(list(dates)),
        columns=pd.MultiIndex.from_tuples(columns),
    )


def write_universe(path: Path, tickers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["ticker", "company_name"])
        writer.writeheader()
        for ticker in tickers:
            writer.writerow({"ticker": ticker, "company_name": f"{ticker} Inc."})


def response_for_requested(tickers: Sequence[str]) -> pd.DataFrame:
    values = {
        ticker: (100.0 + index, 90.0 + index, 1_000 + index)
        for index, ticker in enumerate(tickers)
    }
    return multi_frame(values)


def test_load_universe_normalizes_deduplicates_and_preserves_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "universe.csv"
    path.write_text(
        "ticker,company_name\n aapl ,Apple\nMSFT,Microsoft\nAAPL,Again\n,Blank\nbrk.b,Berkshire\n",
        encoding="utf-8",
    )

    tickers = load_universe(path)

    assert tickers == ("AAPL", "MSFT", "BRK-B")
    assert universe_sha256(tickers) == universe_sha256(tickers)
    assert universe_sha256(tickers) != universe_sha256(tuple(reversed(tickers)))


def test_load_universe_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(UniverseReadError, match="does not exist"):
        load_universe(tmp_path / "missing.csv")


def test_load_universe_missing_ticker_column_or_invalid_ticker_fails(
    tmp_path: Path,
) -> None:
    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("symbol\nAAPL\n", encoding="utf-8")
    with pytest.raises(UniverseReadError, match="missing required 'ticker'"):
        load_universe(missing_column)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("ticker\nBAD/TICKER\n", encoding="utf-8")
    with pytest.raises(UniverseReadError, match="Invalid ticker"):
        load_universe(invalid)


def test_default_start_and_new_york_end_exclusive() -> None:
    assert DEFAULT_START == date(2016, 1, 1)
    before_midnight_new_york = datetime(2026, 2, 1, 4, 30, tzinfo=UTC)
    assert calculate_end_exclusive(before_midnight_new_york) == date(2026, 2, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate_end_exclusive(datetime(2026, 1, 1))  # noqa: DTZ001


def test_build_batches_is_deterministic_and_bounded() -> None:
    assert build_batches(("A", "B", "C", "D", "E"), 2) == (
        ("A", "B"),
        ("C", "D"),
        ("E",),
    )
    with pytest.raises(ValueError, match="between 1 and 250"):
        build_batches(("A",), 0)
    with pytest.raises(ValueError, match="between 1 and 250"):
        build_batches(("A",), 251)


def test_download_batch_sets_every_required_yfinance_parameter() -> None:
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return ordinary_frame()

    result = download_batch(
        ("AAA",),
        start_date=START,
        end_exclusive=END,
        timeout=12.5,
        download_func=fake_download,
    )

    assert result.successful_tickers == frozenset({"AAA"})
    assert calls == [
        {
            "tickers": ["AAA"],
            "start": "2016-01-01",
            "end": "2021-01-01",
            "interval": "1d",
            "group_by": "ticker",
            "auto_adjust": False,
            "actions": False,
            "repair": True,
            "keepna": False,
            "threads": True,
            "progress": False,
            "timeout": 12.5,
            "multi_level_index": True,
            "prepost": False,
            "rounding": False,
        }
    ]


def test_yfinance_repair_warning_does_not_override_valid_daily_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def warning_then_valid_download(**kwargs: object) -> pd.DataFrame:
        release_module.LOGGER.warning(
            "AAUC: price-reconstruct auxiliary 1h request is too old"
        )
        return ordinary_frame(dates=("2026-08-06",))

    result = download_batch(
        ("AAUC",),
        start_date=date(2026, 8, 1),
        end_exclusive=date(2026, 8, 7),
        download_func=warning_then_valid_download,
    )

    assert "price-reconstruct" in caplog.text
    assert result.successful_tickers == frozenset({"AAUC"})
    assert result.failed_tickers == {}
    assert result.rows["date"].tolist() == [date(2026, 8, 6)]


def test_pandas3_writable_numpy_compatibility_is_scoped() -> None:
    original_to_numpy = pd.Series.to_numpy
    writable_inside: list[bool] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        writable_inside.append(pd.Series([1.0]).to_numpy().flags.writeable)
        return ordinary_frame()

    result = download_batch(
        ("AAA",),
        start_date=START,
        end_exclusive=END,
        download_func=fake_download,
    )

    assert result.successful_tickers == frozenset({"AAA"})
    assert writable_inside == [True]
    assert pd.Series.to_numpy is original_to_numpy


@pytest.mark.parametrize("field_first", [False, True])
def test_multiindex_both_level_orders_are_normalized(field_first: bool) -> None:
    frame = multi_frame(
        {"AAA": (10.0, 9.0, 100), "BBB": (20.0, 18.0, 200)},
        field_first=field_first,
        include_extra=True,
    )

    result = normalize_download_frame(
        frame,
        ("AAA", "BBB"),
        start_date=START,
        end_exclusive=END,
    )

    assert tuple(result.rows.columns) == PRICE_COLUMNS
    assert result.successful_tickers == frozenset({"AAA", "BBB"})
    assert result.rows.to_dict("records") == [
        {
            "date": date(2020, 1, 2),
            "ticker": "AAA",
            "close": 10.0,
            "adj_close": 9.0,
            "volume": 100,
        },
        {
            "date": date(2020, 1, 2),
            "ticker": "BBB",
            "close": 20.0,
            "adj_close": 18.0,
            "volume": 200,
        },
    ]


def test_single_ticker_ordinary_columns_and_extra_fields() -> None:
    result = normalize_download_frame(
        ordinary_frame(extra=True),
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )

    assert result.successful_tickers == frozenset({"AAA"})
    assert tuple(result.rows.columns) == PRICE_COLUMNS
    assert "Repaired?" not in result.rows.columns


def test_partial_batch_and_empty_frame_are_no_data() -> None:
    partial = normalize_download_frame(
        multi_frame({"AAA": (10.0, 9.0, 100)}),
        ("AAA", "BBB"),
        start_date=START,
        end_exclusive=END,
    )
    empty = normalize_download_frame(
        pd.DataFrame(),
        ("AAA", "BBB"),
        start_date=START,
        end_exclusive=END,
    )

    assert partial.successful_tickers == frozenset({"AAA"})
    assert partial.no_data_tickers == frozenset({"BBB"})
    assert empty.no_data_tickers == frozenset({"AAA", "BBB"})


def test_2017_listing_is_success_without_synthetic_2016_rows() -> None:
    result = normalize_download_frame(
        ordinary_frame(dates=("2017-06-15",)),
        ("NEW",),
        start_date=START,
        end_exclusive=date(2020, 1, 1),
    )

    assert result.successful_tickers == frozenset({"NEW"})
    assert result.rows["date"].tolist() == [date(2017, 6, 15)]
    assert len(result.rows) == 1


def test_missing_required_field_is_failed() -> None:
    frame = ordinary_frame().drop(columns=["Adj Close"])
    result = normalize_download_frame(
        frame,
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )

    assert result.successful_tickers == frozenset()
    assert result.failed_tickers["AAA"].error_type == "MissingFields"


def test_price_volume_and_range_cleaning_counts_invalid_rows() -> None:
    frame = ordinary_frame(
        dates=(
            "2009-12-31",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
        ),
        close=(10.0, 0.0, 10.0, 10.0, 10.0, 10.0),
        adj_close=(9.0, 9.0, -1.0, 9.0, 9.0, 9.0),
        volume=(100, 100, 100, -1, 1.5, 0),
    )

    result = normalize_download_frame(
        frame,
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )

    assert result.invalid_rows_removed == 5
    assert result.rows.to_dict("records") == [
        {
            "date": date(2020, 1, 6),
            "ticker": "AAA",
            "close": 10.0,
            "adj_close": 9.0,
            "volume": 0,
        }
    ]
    assert result.rows["volume"].dtype.name == "int64"


def test_identical_duplicate_is_removed_and_conflict_fails() -> None:
    identical = ordinary_frame(
        dates=("2020-01-02", "2020-01-02"),
        close=(10.0, 10.0),
        adj_close=(9.0, 9.0),
        volume=(100, 100),
    )
    result = normalize_download_frame(
        identical,
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )
    assert len(result.rows) == 1
    assert result.duplicate_rows_removed == 1

    conflict = identical.copy()
    conflict.iloc[1, conflict.columns.get_loc("Close")] = 11.0
    with pytest.raises(DataConflictError, match="Conflicting duplicate"):
        normalize_download_frame(
            conflict,
            ("AAA",),
            start_date=START,
            end_exclusive=END,
        )


def test_partial_ticker_triggers_small_then_single_retry() -> None:
    calls: list[tuple[str, ...]] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        tickers = tuple(
            str(value) for value in cast(Sequence[object], kwargs["tickers"])
        )
        calls.append(tickers)
        if tickers == ("AAA", "BBB"):
            return multi_frame({"AAA": (10.0, 9.0, 100)})
        if calls.count(("BBB",)) == 1:
            return pd.DataFrame()
        return ordinary_frame(close=(20.0,), adj_close=(18.0,), volume=(200,))

    execution = execute_batch_with_retries(
        ("AAA", "BBB"),
        start_date=START,
        end_exclusive=END,
        max_retries=1,
        pause_seconds=0,
        download_func=fake_download,
        sleep_func=lambda _: None,
    )

    assert calls == [("AAA", "BBB"), ("BBB",), ("BBB",)]
    assert execution.statuses == {"AAA": "success", "BBB": "success"}
    assert execution.attempt_counts == {"AAA": 1, "BBB": 3}
    assert set(execution.rows["ticker"]) == {"AAA", "BBB"}


def test_retries_are_finite_and_no_data_is_terminal() -> None:
    calls = 0

    def empty_download(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame()

    execution = execute_batch_with_retries(
        ("AAA",),
        start_date=START,
        end_exclusive=END,
        max_retries=2,
        pause_seconds=0,
        download_func=empty_download,
        sleep_func=lambda _: None,
    )

    assert calls == 4
    assert execution.attempt_counts["AAA"] == 4
    assert execution.statuses["AAA"] == "no_data"


def test_network_exceptions_become_failed_with_bounded_attempts() -> None:
    calls = 0

    def failing_download(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise TimeoutError("temporary timeout")

    execution = execute_batch_with_retries(
        ("AAA",),
        start_date=START,
        end_exclusive=END,
        max_retries=1,
        pause_seconds=0,
        download_func=failing_download,
        sleep_func=lambda _: None,
    )

    assert calls == 3
    assert execution.statuses["AAA"] == "failed"
    assert execution.errors["AAA"].error_type == "TimeoutError"


def test_staging_parquet_schema_sorting_and_zstd(tmp_path: Path) -> None:
    normalized = normalize_download_frame(
        ordinary_frame(
            dates=("2020-01-03", "2020-01-02"),
            close=(11.0, 10.0),
            adj_close=(10.0, 9.0),
            volume=(101, 100),
        ),
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )
    path = tmp_path / "batch_0000.parquet"

    write_batch_atomically(
        path,
        normalized.rows,
        start_date=START,
        end_exclusive=END,
        allowed_tickers=("AAA",),
    )

    table = pq.read_table(path)
    assert table.schema.equals(PRICE_SCHEMA)
    assert table.column_names == list(PRICE_COLUMNS)
    assert table["date"].type == pa.date32()
    assert table["volume"].type == pa.int64()
    assert table["date"].to_pylist() == [date(2020, 1, 2), date(2020, 1, 3)]
    metadata = pq.ParquetFile(path).metadata
    assert all(
        metadata.row_group(group).column(column).compression == "ZSTD"
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def test_year_partitions_have_one_file_correct_year_and_counts(tmp_path: Path) -> None:
    frame = ordinary_frame(
        dates=("2019-12-31", "2020-01-02"),
        close=(10.0, 11.0),
        adj_close=(9.0, 10.0),
        volume=(100, 101),
    )
    normalized = normalize_download_frame(
        frame,
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )
    table = pa.Table.from_pandas(
        normalized.rows,
        schema=PRICE_SCHEMA,
        preserve_index=False,
    )

    counts = write_year_partitions(table, tmp_path)
    validate_complete_dataset(
        tmp_path,
        start_date=START,
        end_exclusive=END,
        tickers=("AAA",),
        expected_partition_counts=counts,
        expected_total_rows=2,
    )

    assert counts == {"2019": 1, "2020": 1}
    paths = sorted(tmp_path.glob("daily/year=*/prices.parquet"))
    assert [path.parent.name for path in paths] == ["year=2019", "year=2020"]
    for path in paths:
        table = pq.read_table(path)
        year = int(path.parent.name.split("=")[1])
        assert set(pc.year(table["date"]).to_pylist()) == {year}
        assert table.column_names == list(PRICE_COLUMNS)


def test_run_backfill_publishes_manifest_coverage_and_empty_failures(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "processed" / "prices"
    staging = tmp_path / "staging"
    write_universe(universe, ("AAA", "BBB", "CCC"))
    calls: list[tuple[str, ...]] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[object], kwargs["tickers"])
        )
        calls.append(requested)
        return response_for_requested(requested)

    manifest = run_backfill(
        universe_path=universe,
        output_root=output,
        staging_root=staging,
        batch_size=2,
        max_retries=1,
        pause_seconds=0,
        now=FIXED_NOW,
        download_func=fake_download,
        sleep_func=lambda _: None,
    )

    assert calls == [("AAA", "BBB"), ("CCC",)]
    assert manifest["completed"] is True
    assert manifest["requested_start"] == "2016-01-01"
    assert manifest["universe_sha256"] == universe_sha256(("AAA", "BBB", "CCC"))
    assert manifest["universe_ticker_count"] == 3
    assert manifest["successful_ticker_count"] == 3
    assert manifest["no_data_ticker_count"] == 0
    assert manifest["failed_ticker_count"] == 0
    assert manifest["total_row_count"] == 3
    assert sum(manifest["partition_row_counts"].values()) == 3
    assert all(int(year) >= 2016 for year in manifest["partition_row_counts"])
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest

    with (output / "ticker_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as coverage_file:
        coverage_reader = csv.DictReader(coverage_file)
        coverage = list(coverage_reader)
    assert tuple(coverage_reader.fieldnames or ()) == COVERAGE_COLUMNS
    assert [row["ticker"] for row in coverage] == ["AAA", "BBB", "CCC"]
    assert {row["status"] for row in coverage} == {"success"}
    assert {row["first_date"] for row in coverage} == {"2020-01-02"}

    with (output / "download_failures.csv").open(
        encoding="utf-8", newline=""
    ) as failures_file:
        failure_reader = csv.DictReader(failures_file)
        failures = list(failure_reader)
    assert tuple(failure_reader.fieldnames or ()) == FAILURE_COLUMNS
    assert failures == []


def test_no_data_blocks_publish_and_writes_staging_reports(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "prices"
    staging = tmp_path / "staging"
    write_universe(universe, ("AAA",))

    with pytest.raises(BackfillIncompleteError, match="1 no_data"):
        run_backfill(
            universe_path=universe,
            output_root=output,
            staging_root=staging,
            batch_size=1,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            download_func=lambda **kwargs: pd.DataFrame(),
            sleep_func=lambda _: None,
        )

    assert not output.exists()
    run_dir = next(staging.iterdir())
    assert (run_dir / "run_state.json").is_file()
    incomplete = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert incomplete["completed"] is False
    assert incomplete["no_data_ticker_count"] == 1
    with (run_dir / "download_failures.csv").open(
        encoding="utf-8", newline=""
    ) as failure_file:
        failures = list(csv.DictReader(failure_file))
    assert failures[0]["status"] == "no_data"


def test_allow_no_data_publishes_but_failed_never_publishes(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    write_universe(universe, ("AAA", "BBB"))
    output = tmp_path / "allowed"

    def partial_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[object], kwargs["tickers"])
        )
        if "AAA" in requested:
            return multi_frame({"AAA": (10.0, 9.0, 100)})
        return pd.DataFrame()

    manifest = run_backfill(
        universe_path=universe,
        output_root=output,
        staging_root=tmp_path / "staging_allowed",
        batch_size=2,
        max_retries=0,
        pause_seconds=0,
        now=FIXED_NOW,
        allow_no_data=True,
        download_func=partial_download,
        sleep_func=lambda _: None,
    )
    assert manifest["successful_ticker_count"] == 1
    assert manifest["no_data_ticker_count"] == 1
    assert output.is_dir()

    failed_output = tmp_path / "failed"

    def failed_download(**kwargs: object) -> pd.DataFrame:
        raise ConnectionError("offline")

    with pytest.raises(BackfillIncompleteError, match="failed tickers"):
        run_backfill(
            universe_path=universe,
            output_root=failed_output,
            staging_root=tmp_path / "staging_failed",
            batch_size=2,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            allow_no_data=True,
            download_func=failed_download,
            sleep_func=lambda _: None,
        )
    assert not failed_output.exists()


def test_existing_final_output_refuses_before_download(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "prices"
    write_universe(universe, ("AAA",))
    output.mkdir()
    (output / "sentinel.txt").write_text("keep", encoding="utf-8")
    called = False

    def fake_download(**kwargs: object) -> pd.DataFrame:
        nonlocal called
        called = True
        return ordinary_frame()

    with pytest.raises(PriceBackfillError, match="refusing to overwrite"):
        run_backfill(
            universe_path=universe,
            output_root=output,
            staging_root=tmp_path / "staging",
            now=FIXED_NOW,
            download_func=fake_download,
        )
    assert called is False


def test_existing_empty_final_output_can_be_atomically_published(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "prices"
    output.mkdir()
    write_universe(universe, ("AAA",))

    manifest = run_backfill(
        universe_path=universe,
        output_root=output,
        staging_root=tmp_path / "staging",
        max_retries=0,
        pause_seconds=0,
        now=FIXED_NOW,
        download_func=lambda **kwargs: ordinary_frame(),
        sleep_func=lambda _: None,
    )

    assert manifest["completed"] is True
    assert (output / "manifest.json").is_file()


def test_resume_reuses_completed_batch_without_downloading(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "prices"
    staging = tmp_path / "staging"
    write_universe(universe, ("AAA",))
    calls = 0

    def empty_download(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame()

    with pytest.raises(BackfillIncompleteError):
        run_backfill(
            universe_path=universe,
            output_root=output,
            staging_root=staging,
            batch_size=1,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            download_func=empty_download,
            sleep_func=lambda _: None,
        )
    assert calls == 2

    def must_not_download(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("completed staging batch was downloaded again")

    manifest = run_backfill(
        universe_path=universe,
        output_root=output,
        staging_root=staging,
        batch_size=1,
        max_retries=0,
        pause_seconds=0,
        now=FIXED_NOW,
        allow_no_data=True,
        download_func=must_not_download,
        sleep_func=lambda _: None,
    )
    assert manifest["completed"] is True
    assert manifest["no_data_ticker_count"] == 1


def test_changed_universe_gets_different_run_key_and_staging(tmp_path: Path) -> None:
    first = build_run_key(("AAA",), START, END)
    second = build_run_key(("BBB",), START, END)
    assert first != second

    universe = tmp_path / "universe.csv"
    staging = tmp_path / "staging"
    write_universe(universe, ("AAA",))
    with pytest.raises(BackfillIncompleteError):
        run_backfill(
            universe_path=universe,
            output_root=tmp_path / "first_output",
            staging_root=staging,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            download_func=lambda **kwargs: pd.DataFrame(),
            sleep_func=lambda _: None,
        )
    write_universe(universe, ("BBB",))
    with pytest.raises(BackfillIncompleteError):
        run_backfill(
            universe_path=universe,
            output_root=tmp_path / "second_output",
            staging_root=staging,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            download_func=lambda **kwargs: pd.DataFrame(),
            sleep_func=lambda _: None,
        )
    assert len(list(staging.iterdir())) == 2


def test_incompatible_staging_is_isolated_and_exact_match_is_retained(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.csv"
    staging = tmp_path / ".staging"
    tickers = ("AAA", "BBB")
    write_universe(universe, tickers)
    run_key = build_run_key(tickers, START, END)
    matching = staging / run_key
    load_or_create_run_state(
        matching,
        run_key=run_key,
        universe_path=universe,
        universe_hash=universe_sha256(tickers),
        tickers=tickers,
        start_date=START,
        end_exclusive=END,
        batch_size=2,
        resume=True,
    )
    mismatching = staging / "old-run"
    mismatching.mkdir(parents=True)
    (mismatching / "run_state.json").write_text(
        json.dumps(
            {
                "run_key": "old-run",
                "universe_sha256": universe_sha256(("OLD",)),
                "start": "2014-01-01",
            }
        ),
        encoding="utf-8",
    )
    timestamp = datetime(2026, 8, 6, 1, 30, tzinfo=UTC)

    legacy, compatible = isolate_incompatible_staging(
        staging,
        universe_path=universe,
        tickers=tickers,
        start_date=START,
        end_exclusive=END,
        batch_size=2,
        now=timestamp,
    )

    assert legacy == tmp_path / "staging_legacy_20260805_213000"
    assert compatible == (matching,)
    assert matching.is_dir()
    assert not mismatching.exists()
    assert (legacy / "old-run" / "run_state.json").is_file()


def test_rotate_price_output_is_atomic_retained_and_collision_safe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prices"
    output.mkdir()
    (output / "manifest.json").write_text('{"keep": true}\n', encoding="utf-8")
    timestamp = datetime(2026, 8, 6, 1, 30, tzinfo=UTC)

    legacy = rotate_price_output_to_legacy(output, now=timestamp)

    assert legacy == tmp_path / "prices_legacy_20260805_213000"
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (legacy / "manifest.json").read_text(encoding="utf-8") == (
        '{"keep": true}\n'
    )

    (output / "new.txt").write_text("new", encoding="utf-8")
    with pytest.raises(PriceBackfillError, match="already exists"):
        rotate_price_output_to_legacy(output, now=timestamp)
    assert (output / "new.txt").is_file()
    assert (legacy / "manifest.json").is_file()


def test_rotate_price_output_rename_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "prices"
    output.mkdir()
    (output / "sentinel.txt").write_text("keep", encoding="utf-8")
    timestamp = datetime(2026, 8, 6, 1, 30, tzinfo=UTC)
    original_rename = Path.rename

    def fail_target_rename(path: Path, target: Path) -> Path:
        if path == output:
            raise OSError("simulated rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_target_rename)

    with pytest.raises(OSError, match="simulated rename failure"):
        rotate_price_output_to_legacy(output, now=timestamp)

    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "prices_legacy_20260805_213000").exists()


def test_corrupt_staging_batch_is_not_silently_reused(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    staging = tmp_path / "staging"
    write_universe(universe, ("AAA",))
    with pytest.raises(BackfillIncompleteError):
        run_backfill(
            universe_path=universe,
            output_root=tmp_path / "output",
            staging_root=staging,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            download_func=lambda **kwargs: pd.DataFrame(),
            sleep_func=lambda _: None,
        )
    run_dir = next(staging.iterdir())
    (run_dir / "batch_0000.parquet").write_bytes(b"not parquet")

    with pytest.raises(StagingError, match="corrupt or incompatible"):
        run_backfill(
            universe_path=universe,
            output_root=tmp_path / "output",
            staging_root=staging,
            max_retries=0,
            pause_seconds=0,
            now=FIXED_NOW,
            allow_no_data=True,
            download_func=lambda **kwargs: ordinary_frame(),
            sleep_func=lambda _: None,
        )


def test_cli_success_failure_and_argument_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def capture_backfill(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {}

    monkeypatch.setattr(prices_module, "run_backfill", capture_backfill)
    assert main(["backfill", "--pause-seconds", "0", "--batch-size", "1"]) == 0
    assert calls[-1]["start_date"] == date(2016, 1, 1)
    assert main(["backfill", "--start", "2014-03-04"]) == 0
    assert calls[-1]["start_date"] == date(2014, 3, 4)

    def fail_backfill(**kwargs: object) -> dict[str, Any]:
        raise PriceBackfillError("expected failure")

    monkeypatch.setattr(prices_module, "run_backfill", fail_backfill)
    assert main(["backfill"]) == 1
    with pytest.raises(SystemExit):
        main(["backfill", "--batch-size", "0"])


def test_backfill_help_shows_default_start(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["backfill", "--help"])

    assert exit_info.value.code == 0
    assert (
        "first requested calendar date (default: 2016-01-01)" in capsys.readouterr().out
    )


def test_update_help_has_no_release_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["update", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--repository" not in output
    assert "--release-tag" not in output


def test_completed_dataset_acceptance_checks_target_coverage(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.csv"
    output = tmp_path / "prices"
    write_universe(universe, ("AAA", "BBB"))

    def staggered_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[object], kwargs["tickers"])
        )
        assert len(requested) == 1
        row_date = "2020-01-02" if requested[0] == "AAA" else "2020-01-01"
        return ordinary_frame(dates=(row_date,))

    run_backfill(
        universe_path=universe,
        output_root=output,
        staging_root=tmp_path / "staging",
        batch_size=1,
        max_retries=0,
        pause_seconds=0,
        now=FIXED_NOW,
        download_func=staggered_download,
        sleep_func=lambda _: None,
    )

    with pytest.raises(DataValidationError, match="coverage 0.5000"):
        validate_backfill_dataset(
            output,
            universe_path=universe,
            target_session=date(2020, 1, 2),
            minimum_target_coverage=0.97,
        )

    report = validate_backfill_dataset(
        output,
        universe_path=universe,
        target_session=date(2020, 1, 2),
        minimum_target_coverage=0.5,
    )
    assert report["target_session_ticker_count"] == 1
    assert report["target_session_coverage_ratio"] == 0.5
    assert report["duplicate_key_count"] == 0
    assert report["null_count"] == 0
    assert report["failed_ticker_count"] == 0


def test_validate_complete_dataset_rejects_wrong_year_partition(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "daily" / "year=2019"
    partition.mkdir(parents=True)
    normalized = normalize_download_frame(
        ordinary_frame(dates=("2020-01-02",)),
        ("AAA",),
        start_date=START,
        end_exclusive=END,
    )
    table = pa.Table.from_pandas(
        normalized.rows, schema=PRICE_SCHEMA, preserve_index=False
    )
    pq.write_table(table, partition / "prices.parquet", compression="zstd")

    with pytest.raises(DataValidationError, match="contains years"):
        validate_complete_dataset(
            tmp_path,
            start_date=START,
            end_exclusive=END,
            tickers=("AAA",),
            expected_partition_counts={"2019": 1},
            expected_total_rows=1,
        )


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 8, 4, 22, 0, tzinfo=UTC), date(2026, 8, 4)),
        (datetime(2026, 8, 4, 20, 30, tzinfo=UTC), date(2026, 8, 3)),
        (datetime(2026, 8, 8, 16, 0, tzinfo=UTC), date(2026, 8, 7)),
        (datetime(2026, 7, 4, 16, 0, tzinfo=UTC), date(2026, 7, 2)),
        (datetime(2025, 3, 10, 21, 45, tzinfo=UTC), date(2025, 3, 10)),
    ],
)
def test_determine_target_session_handles_close_weekend_holiday_and_dst(
    now: datetime, expected: date
) -> None:
    assert determine_target_session(now=now) == expected


def test_determine_target_session_uses_half_day_close_and_override() -> None:
    assert determine_target_session(
        now=datetime(2025, 11, 28, 19, 0, tzinfo=UTC)
    ) == date(2025, 11, 26)
    assert determine_target_session(
        now=datetime(2025, 11, 28, 20, 0, tzinfo=UTC)
    ) == date(2025, 11, 28)
    assert determine_target_session(
        now=datetime(2025, 11, 28, 17, 0, tzinfo=UTC),
        target_date=date(2025, 11, 28),
        allow_partial_session=True,
    ) == date(2025, 11, 28)
    with pytest.raises(PriceUpdateError, match="not an XNYS session"):
        determine_target_session(
            now=datetime(2025, 11, 29, 20, 0, tzinfo=UTC),
            target_date=date(2025, 11, 29),
        )
    with pytest.raises(PriceUpdateError, match="has not completed"):
        determine_target_session(
            now=datetime(2025, 11, 28, 19, 0, tzinfo=UTC),
            target_date=date(2025, 11, 28),
        )


def canonical_price_rows(
    records: Sequence[tuple[date, str, float, float, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [record[0] for record in records],
            "ticker": pd.Series([record[1] for record in records], dtype="string"),
            "close": pd.Series([record[2] for record in records], dtype="float64"),
            "adj_close": pd.Series([record[3] for record in records], dtype="float64"),
            "volume": pd.Series([record[4] for record in records], dtype="int64"),
        },
        columns=PRICE_COLUMNS,
    ).sort_values(["date", "ticker"], ignore_index=True)


def test_refresh_window_default_clamps_and_crosses_year() -> None:
    target = date(2026, 1, 5)
    assert calculate_refresh_start(date(2010, 1, 4), target) == date(2024, 7, 4)
    assert calculate_refresh_start(date(2025, 12, 31), target) == date(2025, 12, 31)


def test_upsert_inserts_replaces_preserves_missing_and_is_idempotent() -> None:
    old = canonical_price_rows(
        [
            (date(2025, 12, 31), "AAA", 10, 9, 100),
            (date(2026, 1, 2), "AAA", 11, 10, 101),
            (date(2026, 1, 2), "BBB", 20, 19, 200),
        ]
    )
    new = canonical_price_rows(
        [
            (date(2026, 1, 2), "AAA", 12, 11.5, 111),
            (date(2026, 1, 5), "AAA", 13, 12, 112),
        ]
    )
    merged = upsert_refresh_window(
        old,
        new,
        refresh_start=date(2025, 12, 20),
        target_session=date(2026, 1, 5),
        tickers=("AAA", "BBB"),
    )
    assert len(merged) == 4
    revised = merged.loc[
        (merged["date"] == date(2026, 1, 2)) & (merged["ticker"] == "AAA")
    ].iloc[0]
    assert revised["adj_close"] == 11.5
    assert bool(
        ((merged["date"] == date(2026, 1, 2)) & (merged["ticker"] == "BBB")).any()
    )
    rerun = upsert_refresh_window(
        merged,
        new,
        refresh_start=date(2025, 12, 20),
        target_session=date(2026, 1, 5),
        tickers=("AAA", "BBB"),
    )
    assert rerun.equals(merged)
    assert not rerun.duplicated(["date", "ticker"]).any()


def test_expected_active_uses_prior_ten_sessions_and_coverage_gate() -> None:
    old = canonical_price_rows(
        [
            (date(2026, 1, 2), "AAA", 10, 9, 100),
            (date(2025, 12, 31), "BBB", 20, 19, 200),
            (date(2025, 1, 2), "OLD", 30, 29, 300),
        ]
    )
    expected = expected_active_tickers(
        old,
        universe=("AAA", "BBB", "OLD"),
        target_session=date(2026, 1, 5),
    )
    assert expected == frozenset({"AAA", "BBB"})
    merged = canonical_price_rows([(date(2026, 1, 5), "AAA", 11, 10, 101)])
    with pytest.raises(PriceUpdateError, match="coverage"):
        validate_target_coverage(
            merged,
            expected_active=expected,
            target_session=date(2026, 1, 5),
            minimum_ratio=0.97,
        )
    ratio, missing = validate_target_coverage(
        merged,
        expected_active=expected,
        target_session=date(2026, 1, 5),
        minimum_ratio=0.97,
        allow_partial_session=True,
    )
    assert ratio == 0.5
    assert missing == ("BBB",)


def test_read_affected_partitions_reads_only_requested_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for year in (2024, 2025, 2026):
        rows = canonical_price_rows([(date(year, 1, 2), "AAA", 10, 9, 100)])
        partition = tmp_path / "daily" / f"year={year}"
        partition.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pandas(rows, schema=PRICE_SCHEMA, preserve_index=False),
            partition / "prices.parquet",
            compression="zstd",
        )
    real_read = prices_module.pq.read_table
    paths: list[str] = []

    def observed_read(path: Path) -> pa.Table:
        paths.append(str(path))
        return real_read(path)

    monkeypatch.setattr(prices_module.pq, "read_table", observed_read)
    rows = read_affected_partitions(tmp_path, (2025, 2026), tickers=("AAA",))
    assert len(rows) == 2
    assert all("year=2024" not in path for path in paths)
    assert {value.year for value in rows["date"]} == {2025, 2026}


def write_incremental_fixture(root: Path) -> tuple[Path, Path]:
    universe = root / "universe.csv"
    prices_root = root / "prices"
    write_universe(universe, ("AAA", "BBB"))
    rows = canonical_price_rows(
        [
            (date(2025, 12, 31), "AAA", 10, 9, 100),
            (date(2025, 12, 31), "BBB", 20, 19, 200),
            (date(2026, 1, 2), "AAA", 11, 10, 101),
            (date(2026, 1, 2), "BBB", 21, 20, 201),
        ]
    )
    counts: dict[str, int] = {}
    for year in (2025, 2026):
        year_rows = rows.loc[rows["date"].map(lambda value: value.year).eq(year)]
        path = prices_root / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pandas(year_rows, schema=PRICE_SCHEMA, preserve_index=False),
            path,
            compression="zstd",
        )
        counts[str(year)] = len(year_rows)
    (prices_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "daily_prices_v1",
                "source": "yahoo_finance_via_yfinance",
                "requested_start": "2016-01-01",
                "actual_min_date": "2025-12-31",
                "actual_max_date": "2026-01-02",
                "universe_sha256": universe_sha256(("AAA", "BBB")),
                "universe_ticker_count": 2,
                "successful_ticker_count": 2,
                "no_data_ticker_count": 0,
                "failed_ticker_count": 0,
                "partition_row_counts": counts,
                "total_row_count": 4,
                "completed": True,
            }
        ),
        encoding="utf-8",
    )
    with (prices_root / "ticker_coverage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        for ticker in ("AAA", "BBB"):
            writer.writerow(
                {
                    "ticker": ticker,
                    "status": "success",
                    "first_date": "2025-12-31",
                    "last_date": "2026-01-02",
                    "row_count": 2,
                    "attempt_count": 1,
                    "last_error": "",
                }
            )
    return universe, prices_root


def test_run_update_writes_partitions_manifest_coverage_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    replacement_order: list[str] = []
    real_replace = prices_module.replace_files_transactionally

    def record_replacement_order(
        root: Path,
        staging_root: Path,
        relative_paths: Sequence[str],
        **kwargs: object,
    ) -> None:
        replacement_order.extend(relative_paths)
        real_replace(root, staging_root, relative_paths, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        prices_module, "replace_files_transactionally", record_replacement_order
    )

    def fake_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[Any], kwargs["tickers"])
        )
        return multi_frame(
            {ticker: (30.0, 29.0, 300) for ticker in requested},
            dates=("2026-01-05",),
        )

    result = run_update(
        universe_path=universe,
        prices_root=prices_root,
        refresh_calendar_days=10,
        batch_size=2,
        max_retries=0,
        pause_seconds=0,
        target_date=date(2026, 1, 5),
        now=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
        download_func=fake_download,
        sleep_func=lambda _: None,
    )
    assert result["status"] == "updated"
    assert result["local_update_success"] is True
    assert result["changed_partition_years"] == [2025, 2026]
    assert result["changed_local_assets"][-1] == "manifest.json"
    manifest = json.loads((prices_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest_session"] == "2026-01-05"
    assert manifest["requested_end_exclusive"] == "2026-01-06"
    assert manifest["requested_start"] == "2016-01-01"
    assert manifest["universe_sha256"] == universe_sha256(("AAA", "BBB"))
    assert manifest["last_update_refresh_start"] == "2025-12-26"
    assert manifest["last_update_target_session"] == "2026-01-05"
    assert manifest["last_update_target_coverage_ratio"] == 1.0
    assert isinstance(manifest["last_update_run_id"], str)
    assert manifest["total_row_count"] == 6
    assert set(manifest["assets"]) == {
        "2025",
        "2026",
        "ticker_coverage",
        "update_missing_tickers",
        "update_report",
    }
    assert manifest["assets"]["2026"]["asset_name"] == "prices-year-2026.parquet"
    updated = pq.read_table(prices_root / "daily/year=2026/prices.parquet").to_pandas()
    assert len(updated) == 4
    assert not updated.duplicated(["date", "ticker"]).any()
    report = json.loads(
        (prices_root / "update_report.json").read_text(encoding="utf-8")
    )
    assert report["target_session_coverage_ratio"] == 1.0
    assert report["local_update_success"] is True
    with (prices_root / "ticker_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as coverage_file:
        coverage = list(csv.DictReader(coverage_file))
    assert {row["row_count"] for row in coverage} == {"3"}
    assert {row["last_date"] for row in coverage} == {"2026-01-05"}
    assert replacement_order[-1] == "manifest.json"
    assert not (prices_root / "release_publish_plan.json").exists()


def test_run_update_without_github_configuration_is_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("MOMENTUM_SCREENER_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("local update must not use GitHub release storage")

    monkeypatch.setattr(release_module, "resolve_repository", forbidden)
    monkeypatch.setattr(release_module, "resolve_github_token", forbidden)
    monkeypatch.setattr(release_module, "build_publish_plan", forbidden)
    monkeypatch.setattr(release_module, "GitHubClient", forbidden)

    def fake_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[Any], kwargs["tickers"])
        )
        return multi_frame(
            {ticker: (30.0, 29.0, 300) for ticker in requested},
            dates=("2026-01-05",),
        )

    result = run_update(
        universe_path=universe,
        prices_root=prices_root,
        refresh_calendar_days=10,
        batch_size=2,
        max_retries=0,
        pause_seconds=0,
        target_date=date(2026, 1, 5),
        now=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
        download_func=fake_download,
        sleep_func=lambda _: None,
    )

    assert result["status"] == "updated"
    assert result["local_update_success"] is True
    assert not (prices_root / "release_publish_plan.json").exists()


def test_run_update_noop_and_dry_run_never_download(
    tmp_path: Path,
) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)

    def forbidden_download(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("dry-run/no-op must not access Yahoo")

    no_op = run_update(
        universe_path=universe,
        prices_root=prices_root,
        target_date=date(2026, 1, 2),
        allow_partial_session=True,
        download_func=forbidden_download,
    )
    assert no_op["status"] == "no_op"
    dry_run = run_update(
        universe_path=universe,
        prices_root=prices_root,
        target_date=date(2026, 1, 5),
        allow_partial_session=True,
        dry_run=True,
        download_func=forbidden_download,
    )
    assert dry_run["status"] == "dry_run"
    assert not (prices_root / "update_report.json").exists()

    clamped = run_update(
        universe_path=universe,
        prices_root=prices_root,
        refresh_calendar_days=5000,
        target_date=date(2026, 1, 5),
        allow_partial_session=True,
        dry_run=True,
        download_func=forbidden_download,
    )
    assert clamped["refresh_start"] == "2016-01-01"
    assert min(clamped["affected_years"]) == 2016


def test_run_update_does_not_rewrite_unaffected_partition(tmp_path: Path) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    old_rows = canonical_price_rows(
        [
            (date(2024, 1, 2), "AAA", 8, 7, 80),
            (date(2024, 1, 2), "BBB", 18, 17, 180),
        ]
    )
    old_path = prices_root / "daily/year=2024/prices.parquet"
    old_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(old_rows, schema=PRICE_SCHEMA, preserve_index=False),
        old_path,
        compression="zstd",
    )
    manifest = json.loads((prices_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["actual_min_date"] = "2024-01-02"
    manifest["partition_row_counts"]["2024"] = 2
    manifest["total_row_count"] = 6
    manifest["universe_sha256"] = universe_sha256(("AAA", "BBB"))
    for row in csv.DictReader(
        (prices_root / "ticker_coverage.csv").read_text(encoding="utf-8").splitlines()
    ):
        assert row["row_count"] == "2"
    with (prices_root / "ticker_coverage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        for ticker in ("AAA", "BBB"):
            writer.writerow(
                {
                    "ticker": ticker,
                    "status": "success",
                    "first_date": "2024-01-02",
                    "last_date": "2026-01-02",
                    "row_count": 3,
                    "attempt_count": 1,
                    "last_error": "",
                }
            )
    manifest = index_release_assets(prices_root, manifest)
    (prices_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = old_path.read_bytes()

    def fake_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[Any], kwargs["tickers"])
        )
        return multi_frame(
            {ticker: (30.0, 29.0, 300) for ticker in requested},
            dates=("2026-01-05",),
        )

    run_update(
        universe_path=universe,
        prices_root=prices_root,
        refresh_calendar_days=10,
        batch_size=2,
        max_retries=0,
        pause_seconds=0,
        target_date=date(2026, 1, 5),
        now=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
        download_func=fake_download,
        sleep_func=lambda _: None,
    )
    assert old_path.read_bytes() == before
    updated_manifest = json.loads(
        (prices_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert updated_manifest["assets"]["2024"] == manifest["assets"]["2024"]


def test_run_update_identity_mismatch_fails_before_yahoo(tmp_path: Path) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    manifest_path = prices_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["universe_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_download(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("identity mismatch must fail before Yahoo")

    with pytest.raises(PriceUpdateError, match="different static Universe"):
        run_update(
            universe_path=universe,
            prices_root=prices_root,
            target_date=date(2026, 1, 5),
            allow_partial_session=True,
            download_func=forbidden_download,
        )


def test_run_update_failure_keeps_existing_files(tmp_path: Path) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    before = (prices_root / "manifest.json").read_bytes()

    def failing_download(**kwargs: object) -> pd.DataFrame:
        raise TimeoutError("offline")

    with pytest.raises(PriceUpdateError, match="unresolved failures"):
        run_update(
            universe_path=universe,
            prices_root=prices_root,
            refresh_calendar_days=10,
            max_retries=0,
            pause_seconds=0,
            target_date=date(2026, 1, 5),
            now=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
            download_func=failing_download,
            sleep_func=lambda _: None,
        )
    assert (prices_root / "manifest.json").read_bytes() == before
    assert not (prices_root / "update_report.json").exists()
    diagnostic_reports = list(
        (prices_root / ".update_diagnostics").glob("update-*/update_report.json")
    )
    assert len(diagnostic_reports) == 1
    assert (
        json.loads(diagnostic_reports[0].read_text(encoding="utf-8"))["failure_reason"]
        == "unresolved_download_failure"
    )


def test_run_update_replacement_failure_rolls_back_every_formal_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe, prices_root = write_incremental_fixture(tmp_path)
    tracked = (
        prices_root / "daily/year=2025/prices.parquet",
        prices_root / "daily/year=2026/prices.parquet",
        prices_root / "ticker_coverage.csv",
        prices_root / "manifest.json",
    )
    before = {path: path.read_bytes() for path in tracked}
    real_transaction = prices_module.replace_files_transactionally
    real_os_replace = manifest_module.os.replace

    def fail_during_transaction(
        root: Path,
        staging_root: Path,
        relative_paths: Sequence[str],
        **kwargs: object,
    ) -> None:
        def fail_on_staged_coverage(source: object, destination: object) -> None:
            source_path = Path(source)  # type: ignore[arg-type]
            if (
                ".update_staging" in source_path.parts
                and source_path.name == "ticker_coverage.csv"
            ):
                raise OSError("simulated local replacement failure")
            real_os_replace(source, destination)  # type: ignore[arg-type]

        monkeypatch.setattr(manifest_module.os, "replace", fail_on_staged_coverage)
        try:
            real_transaction(
                root,
                staging_root,
                relative_paths,
                **kwargs,  # type: ignore[arg-type]
            )
        finally:
            monkeypatch.setattr(manifest_module.os, "replace", real_os_replace)

    monkeypatch.setattr(
        prices_module, "replace_files_transactionally", fail_during_transaction
    )

    def fake_download(**kwargs: object) -> pd.DataFrame:
        requested = tuple(
            str(value) for value in cast(Sequence[Any], kwargs["tickers"])
        )
        return multi_frame(
            {ticker: (30.0, 29.0, 300) for ticker in requested},
            dates=("2026-01-05",),
        )

    with pytest.raises(OSError, match="simulated local replacement failure"):
        run_update(
            universe_path=universe,
            prices_root=prices_root,
            refresh_calendar_days=10,
            batch_size=2,
            max_retries=0,
            pause_seconds=0,
            target_date=date(2026, 1, 5),
            now=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
            download_func=fake_download,
            sleep_func=lambda _: None,
        )

    assert {path: path.read_bytes() for path in tracked} == before
    assert not (prices_root / "update_report.json").exists()
    assert not (prices_root / "update_missing_tickers.csv").exists()


def test_update_cli_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prices_module, "run_update", lambda **kwargs: {"status": "dry_run"}
    )
    result_path = tmp_path / "result.json"
    assert main(["update", "--dry-run", "--result-json", str(result_path)]) == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"status": "dry_run"}

    def fail_update(**kwargs: object) -> dict[str, Any]:
        raise PriceUpdateError("expected")

    monkeypatch.setattr(prices_module, "run_update", fail_update)
    assert main(["update"]) == 1
