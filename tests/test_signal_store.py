from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.signal_store as store
import momentum_screener.storage_manifest as transactions

GENERATED = datetime(2026, 9, 3, 22, tzinfo=UTC)


def batch(
    strategy: str,
    days: dict[date, tuple[str, ...]],
    *,
    version: str = "1.0",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = {
        "strategy_id": strategy,
        "strategy_version": version,
        "generated_at": GENERATED,
        "config_hash": "a" * 64,
        "config_json": "{}",
        "universe_hash": "b" * 64,
        "input_fingerprint": None,
        "data_mode": store.DATA_MODE,
    }
    coverage = pd.DataFrame(
        [
            dict(session=day, status="complete", signal_count=len(tickers), **metadata)
            for day, tickers in days.items()
        ]
    )
    records = [
        dict(
            session=day,
            ticker=ticker,
            signal=True,
            diagnostic_value=12.5,
            diagnostic_flag=True,
            unavailable=float("nan"),
            **metadata,
        )
        for day, tickers in days.items()
        for ticker in tickers
    ]
    signals = pd.DataFrame(
        records,
        columns=[
            *metadata,
            "session",
            "ticker",
            "signal",
            "diagnostic_value",
            "diagnostic_flag",
            "unavailable",
        ],
    )
    signals["signal"] = signals["signal"].astype("bool")
    signals["diagnostic_flag"] = signals["diagnostic_flag"].astype("bool")
    for column in ("diagnostic_value", "unavailable"):
        signals[column] = signals[column].astype("float64")
    return signals, coverage


def persist(
    root: Path,
    strategy: str,
    days: dict[date, tuple[str, ...]],
    *,
    version: str = "1.0",
) -> None:
    rows, coverage = batch(strategy, days, version=version)
    store.replace_signal_range({strategy: rows}, coverage, root=root)


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*.parquet")
    }


def test_queries_keep_zero_coverage_and_full_typed_diagnostics(tmp_path: Path) -> None:
    first, second = date(2026, 9, 2), date(2026, 9, 3)
    monthly, mc = batch(
        "monthly_reversal", {first: ("AAA",), second: ()}, version="6.2"
    )
    monthly["fyx1"] = True
    trend, tc = batch("trend_reacceleration", {first: ("AAA", "BBB"), second: ("BBB",)})
    trend["drawdown"] = 0.2
    result = store.replace_signal_range(
        {"monthly_reversal": monthly, "trend_reacceleration": trend},
        pd.concat([mc, tc]),
        root=tmp_path,
    )
    assert result["coverage_rows"] == 4
    assert result["signal_rows"] == 4
    calendar = store.get_signal_calendar(first, second, root=tmp_path)
    assert calendar["signal_count"].tolist() == [1, 2, 0, 1]
    assert calendar["status"].tolist() == ["complete"] * 4
    assert (
        len(
            store.get_signal_calendar(
                first, second, strategy_id="monthly_reversal", root=tmp_path
            )
        )
        == 2
    )
    matches = store.get_signals_for_date(first, root=tmp_path)
    assert matches["ticker"].tolist() == ["AAA", "AAA", "BBB"]
    assert (
        "drawdown" not in matches
    )  # Common projection; detail retains strategy columns.
    detail = store.get_signal_detail(
        first, "trend_reacceleration", "aaa", root=tmp_path
    )
    assert detail is not None
    assert detail["drawdown"] == 0.2
    assert bool(detail["diagnostic_flag"])
    assert pd.isna(detail["unavailable"])
    assert detail["generated_at"] == GENERATED
    assert detail["config_hash"] == "a" * 64
    assert detail["data_mode"] == "retrospective_latest_data"
    assert (
        store.get_signal_detail(second, "monthly_reversal", "AAA", root=tmp_path)
        is None
    )
    table = pq.read_table(tmp_path / "trend_reacceleration/2026.parquet")
    assert table.schema.field("session").type == pa.date32()
    assert table.schema.field("diagnostic_flag").type == pa.bool_()
    assert table.schema.field("diagnostic_value").type == pa.float64()
    assert (
        pq.ParquetFile(tmp_path / "coverage.parquet")
        .metadata.row_group(0)
        .column(0)
        .compression
        == "ZSTD"
    )


def test_zero_only_store_and_uncomputed_date_are_distinct(tmp_path: Path) -> None:
    day = date(2026, 9, 3)
    persist(tmp_path, "trend_reacceleration", {day: ()})
    calendar = store.get_signal_calendar("2026-09-01", day, root=tmp_path)
    assert calendar["session"].tolist() == [day]
    assert calendar["signal_count"].tolist() == [0]
    assert store.get_signals_for_date(day, root=tmp_path).empty
    assert store.get_signals_for_date("2026-09-02", root=tmp_path).empty
    assert store.read_strategy_signals("trend_reacceleration", root=tmp_path).empty


def test_rerun_is_idempotent_and_changed_results_remove_old_matches(
    tmp_path: Path,
) -> None:
    first, second = date(2026, 9, 2), date(2026, 9, 3)
    persist(tmp_path, "trend_reacceleration", {first: ("AAA",), second: ("BBB",)})
    before = tree_bytes(tmp_path)
    persist(tmp_path, "trend_reacceleration", {first: ("AAA",), second: ("BBB",)})
    assert tree_bytes(tmp_path) == before
    persist(tmp_path, "trend_reacceleration", {first: ("CCC",)})
    assert store.get_signals_for_date(first, root=tmp_path)["ticker"].tolist() == [
        "CCC"
    ]
    assert store.get_signals_for_date(second, root=tmp_path)["ticker"].tolist() == [
        "BBB"
    ]
    persist(tmp_path, "trend_reacceleration", {first: ()})
    assert (
        store.get_signal_detail(first, "trend_reacceleration", "CCC", root=tmp_path)
        is None
    )
    assert store.get_signal_calendar(first, first, root=tmp_path)[
        "signal_count"
    ].tolist() == [0]


def test_replacement_preserves_other_years_strategies_and_replaces_version(
    tmp_path: Path,
) -> None:
    old, day = date(2025, 12, 31), date(2026, 9, 3)
    persist(tmp_path, "trend_reacceleration", {old: ("OLD",), day: ("AAA",)})
    persist(tmp_path, "monthly_reversal", {day: ("MONTHLY",)}, version="6.2")
    preserved = {
        name: (tmp_path / name).read_bytes()
        for name in (
            "trend_reacceleration/2025.parquet",
            "monthly_reversal/2026.parquet",
        )
    }
    persist(tmp_path, "trend_reacceleration", {day: ("NEW",)}, version="2.0")
    for name, contents in preserved.items():
        assert (tmp_path / name).read_bytes() == contents
    detail = store.get_signal_detail(day, "trend_reacceleration", "NEW", root=tmp_path)
    assert detail is not None and detail["strategy_version"] == "2.0"
    calendar = store.get_signal_calendar(old, day, root=tmp_path)
    assert not calendar.duplicated(["session", "strategy_id"]).any()


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "metadata",
        "false_signal",
        "duplicate",
        "coverage_duplicate",
        "outside",
        "unsafe_id",
    ],
)
def test_invalid_batch_is_rejected_before_store_mutation(
    tmp_path: Path, fault: str
) -> None:
    rows, coverage = batch("trend_reacceleration", {date(2026, 9, 3): ("AAA",)})
    if fault == "count":
        coverage["signal_count"] = 0
    elif fault == "metadata":
        rows["strategy_version"] = "wrong"
    elif fault == "false_signal":
        rows["signal"] = False
    elif fault == "duplicate":
        rows = pd.concat([rows, rows])
    elif fault == "coverage_duplicate":
        coverage = pd.concat([coverage, coverage])
    elif fault == "outside":
        rows["session"] = date(2026, 9, 2)
    elif fault == "unsafe_id":
        rows["strategy_id"] = coverage["strategy_id"] = "../escape"
    root = tmp_path / "store"
    with pytest.raises(store.SignalStoreError):
        store.replace_signal_range({"trend_reacceleration": rows}, coverage, root=root)
    assert not root.exists()


@pytest.mark.parametrize("failure", ["staging", "coverage_install", "verify"])
def test_failed_replace_restores_all_results_and_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    day = date(2026, 9, 3)
    persist(tmp_path, "trend_reacceleration", {day: ("OLD",)})
    before = tree_bytes(tmp_path)
    rows, coverage = batch("trend_reacceleration", {day: ("NEW",)})
    if failure == "staging":
        real_write = store.pq.write_table

        def fail_write(table: pa.Table, where: Path, **kwargs: Any) -> None:
            if Path(where).name == "coverage.parquet":
                raise OSError("staging failed")
            real_write(table, where, **kwargs)

        monkeypatch.setattr(store.pq, "write_table", fail_write)
    elif failure == "coverage_install":
        real_replace = transactions.os.replace

        def fail_replace(source: Path, destination: Path) -> None:
            if (
                Path(source).parent.name.startswith(".signal-staging-")
                and Path(destination) == tmp_path / "coverage.parquet"
            ):
                raise OSError("coverage install failed")
            real_replace(source, destination)

        monkeypatch.setattr(transactions.os, "replace", fail_replace)
    else:
        real_transaction = store.replace_files_transactionally

        def fail_verify(*args: Any, **kwargs: Any) -> None:
            def validation() -> None:
                raise OSError("verification failed")

            kwargs["validate_after"] = validation
            real_transaction(*args, **kwargs)

        monkeypatch.setattr(store, "replace_files_transactionally", fail_verify)
    with pytest.raises(OSError):
        store.replace_signal_range(
            {"trend_reacceleration": rows}, coverage, root=tmp_path
        )
    assert tree_bytes(tmp_path) == before
    assert store.get_signals_for_date(day, root=tmp_path)["ticker"].tolist() == ["OLD"]
    assert not (tmp_path / ".replacement-in-progress").exists()


def test_coverage_installs_last_and_reads_refuse_mid_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = store.replace_files_transactionally
    seen: list[str] = []

    def transaction(root: Path, staging: Path, paths: list[str], **kwargs: Any) -> None:
        seen.extend(paths)
        with pytest.raises(store.SignalStoreError, match="in progress"):
            store.get_signal_calendar("2026-09-01", "2026-09-03", root=root)
        real(root, staging, paths, **kwargs)

    monkeypatch.setattr(store, "replace_files_transactionally", transaction)
    persist(tmp_path, "trend_reacceleration", {date(2026, 9, 3): ("AAA",)})
    assert seen == ["trend_reacceleration/2026.parquet", "coverage.parquet"]


def test_interrupted_store_and_missing_partition_fail_closed(tmp_path: Path) -> None:
    day = date(2026, 9, 3)
    persist(tmp_path, "trend_reacceleration", {day: ("AAA",)})
    marker = tmp_path / ".replacement-in-progress"
    marker.write_text("interrupted", encoding="utf-8")
    with pytest.raises(store.SignalStoreError, match="interrupted"):
        store.get_signals_for_date(day, root=tmp_path)
    with pytest.raises(store.SignalStoreError, match="interrupted"):
        persist(tmp_path, "trend_reacceleration", {day: ()})
    marker.unlink()
    (tmp_path / "trend_reacceleration/2026.parquet").unlink()
    with pytest.raises(store.SignalStoreError, match="Unable to read"):
        store.get_signal_detail(day, "trend_reacceleration", "AAA", root=tmp_path)


def test_empty_store_queries_do_not_create_files(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    assert store.get_signal_calendar("2026-01-01", "2026-09-03", root=root).empty
    assert store.get_signals_for_date("2026-09-03", root=root).empty
    assert (
        store.get_signal_detail("2026-09-03", "unknown_strategy", "AAA", root=root)
        is None
    )
    assert not root.exists()
