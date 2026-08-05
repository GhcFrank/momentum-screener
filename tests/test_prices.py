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
    StagingError,
    UniverseReadError,
    build_batches,
    build_run_key,
    calculate_end_exclusive,
    download_batch,
    execute_batch_with_retries,
    load_universe,
    main,
    normalize_download_frame,
    run_backfill,
    universe_sha256,
    validate_complete_dataset,
    write_batch_atomically,
    write_year_partitions,
)

START = date(2010, 1, 1)
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
    assert DEFAULT_START == date(2010, 1, 1)
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
            "start": "2010-01-01",
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


def test_later_listing_is_success_without_synthetic_earlier_rows() -> None:
    result = normalize_download_frame(
        ordinary_frame(dates=("2018-06-15",)),
        ("NEW",),
        start_date=START,
        end_exclusive=date(2020, 1, 1),
    )

    assert result.successful_tickers == frozenset({"NEW"})
    assert result.rows["date"].tolist() == [date(2018, 6, 15)]
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
    assert manifest["universe_ticker_count"] == 3
    assert manifest["successful_ticker_count"] == 3
    assert manifest["no_data_ticker_count"] == 0
    assert manifest["failed_ticker_count"] == 0
    assert manifest["total_row_count"] == 3
    assert sum(manifest["partition_row_counts"].values()) == 3
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
    monkeypatch.setattr(prices_module, "run_backfill", lambda **kwargs: {})
    assert main(["backfill", "--pause-seconds", "0", "--batch-size", "1"]) == 0

    def fail_backfill(**kwargs: object) -> dict[str, Any]:
        raise PriceBackfillError("expected failure")

    monkeypatch.setattr(prices_module, "run_backfill", fail_backfill)
    assert main(["backfill"]) == 1
    with pytest.raises(SystemExit):
        main(["backfill", "--batch-size", "0"])


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
