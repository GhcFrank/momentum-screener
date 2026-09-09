"""Parquet persistence and queries for rebuildable historical signal results.

No strategy or RPS computation belongs here. Coverage is the committed index
of evaluated sessions, including sessions with zero matching stocks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from momentum_screener.storage_manifest import (
    calculate_sha256,
    remove_owned_tree,
    replace_files_transactionally,
)
from momentum_screener.universe import normalize_ticker

LOGGER = logging.getLogger(__name__)
DEFAULT_SIGNAL_ROOT: Final[Path] = Path("data/signals")
SIGNAL_SCHEMA_VERSION: Final[str] = "historical_signals_v1"
DATA_MODE: Final[str] = "retrospective_latest_data"
COVERAGE_NAME: Final[str] = "coverage.parquet"
_WRITE_MARKER: Final[str] = ".replacement-in-progress"
_SCHEMA_METADATA = {b"signal_store_schema": SIGNAL_SCHEMA_VERSION.encode()}
_COMMON_FIELDS = (
    pa.field("session", pa.date32(), nullable=False),
    pa.field("strategy_id", pa.string(), nullable=False),
    pa.field("strategy_version", pa.string(), nullable=False),
    pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("config_hash", pa.string()),
    pa.field("config_json", pa.string(), nullable=False),
    pa.field("universe_hash", pa.string()),
    pa.field("input_fingerprint", pa.string()),
    pa.field("data_mode", pa.string(), nullable=False),
)
COMMON_COLUMNS: Final[tuple[str, ...]] = tuple(field.name for field in _COMMON_FIELDS)
COVERAGE_SCHEMA = pa.schema(
    [
        *_COMMON_FIELDS,
        pa.field("status", pa.string(), nullable=False),
        pa.field("signal_count", pa.int64(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)
SIGNAL_COMMON_SCHEMA = pa.schema(
    [
        *_COMMON_FIELDS,
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("signal", pa.bool_(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)


class SignalStoreError(RuntimeError):
    """The derived store cannot be read or replaced safely."""


def _date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise SignalStoreError("session must be a date or YYYY-MM-DD string")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SignalStoreError("session must use YYYY-MM-DD") from exc


def _strategy_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise SignalStoreError(f"Invalid strategy_id: {value!r}")
    return value


def _empty(*, coverage: bool = False) -> pd.DataFrame:
    schema = COVERAGE_SCHEMA if coverage else SIGNAL_COMMON_SCHEMA
    return schema.empty_table().to_pandas()


def _guard(root: Path) -> None:
    if (root / _WRITE_MARKER).exists():
        raise SignalStoreError(
            f"Signal store replacement is in progress or was interrupted: {root}; "
            "recover the prior files or rebuild this derived store before reading"
        )


def _normalize(rows: pd.DataFrame, *, coverage: bool) -> pa.Table:
    """Validate public fields while preserving typed strategy-specific columns."""

    base = COVERAGE_SCHEMA if coverage else SIGNAL_COMMON_SCHEMA
    missing = sorted(set(base.names).difference(rows.columns))
    if missing:
        raise SignalStoreError(f"Signal store rows are missing columns: {missing}")
    if rows.columns.duplicated().any():
        raise SignalStoreError("Signal store rows contain duplicate columns")
    frame = rows.copy().reset_index(drop=True)
    try:
        dates = pd.to_datetime(frame["session"], errors="raise")
        if dates.isna().any():
            raise ValueError("null session")
        frame["session"] = dates.dt.date
        frame["generated_at"] = pd.to_datetime(
            frame["generated_at"], utc=True, errors="raise"
        )
        for name in ("strategy_id", "strategy_version", "config_json", "data_mode"):
            if (
                not frame[name]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
            ):
                raise ValueError(f"{name} must be a non-empty string")
        for value in frame["strategy_id"].unique():
            _strategy_id(value)
        if not frame["data_mode"].eq(DATA_MODE).all():
            raise ValueError(f"data_mode must be {DATA_MODE}")
        for value in frame["config_json"].unique():
            if not isinstance(json.loads(value), dict):
                raise TypeError("config_json must encode an object")
        for name in ("config_hash", "universe_hash"):
            valid = frame[name].isna() | frame[name].map(
                lambda value: (
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                )
            ).astype("bool")
            if not valid.all():
                raise ValueError(f"{name} must be a SHA-256 hex digest or null")
        keys = ["session", "strategy_id"]
        if coverage:
            if not frame["status"].eq("complete").all():
                raise ValueError("Phase 1 coverage status must be complete")
            counts = pd.to_numeric(frame["signal_count"], errors="raise")
            if not (counts.notna() & counts.ge(0) & counts.eq(counts.round())).all():
                raise ValueError("signal_count must be a non-negative integer")
            frame["signal_count"] = counts.astype("int64")
        else:
            if (
                not pd.api.types.is_bool_dtype(frame["signal"])
                or not frame["signal"].fillna(False).all()
            ):
                raise ValueError("Signal results must contain only Boolean signal=True")
            frame["ticker"] = frame["ticker"].map(normalize_ticker)
            if frame["ticker"].isna().any():
                raise ValueError("ticker must be valid")
            keys.append("ticker")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate signal store keys: {keys}")
        frame = frame.sort_values(keys, kind="mergesort", ignore_index=True)
        diagnostics = frame.drop(columns=base.names)
        inferred = pa.Schema.from_pandas(diagnostics, preserve_index=False)
        if any(pa.types.is_nested(field.type) for field in inferred):
            raise ValueError("Diagnostics must be scalar typed columns")
        schema = pa.schema([*base, *inferred], metadata=_SCHEMA_METADATA)
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        table.validate(full=True)
        for field in base:
            if not field.nullable and table[field.name].null_count:
                raise ValueError(f"{field.name} must not contain null values")
        return table.replace_schema_metadata(_SCHEMA_METADATA)
    except (ValueError, TypeError, pa.ArrowException) as exc:
        raise SignalStoreError(f"Invalid signal store rows: {exc}") from exc


def _read(path: Path, *, coverage: bool = False) -> pd.DataFrame:
    try:
        table = pq.read_table(path)
        if (table.schema.metadata or {}).get(
            b"signal_store_schema"
        ) != SIGNAL_SCHEMA_VERSION.encode():
            raise SignalStoreError(f"Unsupported signal store schema: {path}")
        base = COVERAGE_SCHEMA if coverage else SIGNAL_COMMON_SCHEMA
        if any(table.schema.field(field.name) != field for field in base):
            raise SignalStoreError(f"Invalid signal store common schema: {path}")
        normalized = _normalize(table.to_pandas(), coverage=coverage)
        if not normalized.equals(table, check_metadata=False):
            raise SignalStoreError(f"Signal partition is not normalized/sorted: {path}")
        return normalized.to_pandas()
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        raise SignalStoreError(
            f"Unable to read signal store partition {path}: {exc}"
        ) from exc


def _coverage(root: Path) -> pd.DataFrame:
    path = root / COVERAGE_NAME
    if path.is_file():
        return _read(path, coverage=True)
    if any(root.glob("*/*.parquet")):
        raise SignalStoreError(f"Signal partitions exist without coverage: {root}")
    return _empty(coverage=True)


def _validate_matches(signals: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Require every match to belong to coverage with identical run metadata."""

    keys = ["session", "strategy_id"]
    if not signals.empty:
        merged = signals.merge(
            coverage,
            on=keys,
            how="left",
            suffixes=("", "_coverage"),
            validate="many_to_one",
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise SignalStoreError(
                "Signals contain sessions/strategies outside coverage"
            )
        for column in COMMON_COLUMNS[2:]:
            left, right = merged[column], merged[f"{column}_coverage"]
            if not (left.eq(right) | (left.isna() & right.isna())).all():
                raise SignalStoreError(
                    f"Signal metadata differs from coverage: {column}"
                )
    counts = signals.groupby(keys).size()
    expected = coverage.set_index(keys)["signal_count"]
    if not counts.reindex(expected.index, fill_value=0).eq(expected).all():
        raise SignalStoreError("Coverage signal_count differs from actual signal rows")


def _partition(root: Path, strategy_id: str, year: int) -> Path:
    return root / _strategy_id(strategy_id) / f"{year}.parquet"


def _read_strategy(
    root: Path,
    strategy_id: str,
    coverage: pd.DataFrame,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    selected = coverage.loc[coverage["strategy_id"].eq(strategy_id)]
    selected = _slice_dates(selected, start, end)
    frames: list[pd.DataFrame] = []
    for year in sorted({value.year for value in selected["session"]}):
        rows = _read(_partition(root, strategy_id, year))
        if (
            not rows["strategy_id"].eq(strategy_id).all()
            or not rows["session"].map(lambda value: value.year).eq(year).all()
        ):
            raise SignalStoreError("Signal partition contains a wrong strategy or year")
        yearly_coverage = coverage.loc[
            coverage["strategy_id"].eq(strategy_id)
            & coverage["session"].map(lambda value: value.year).eq(year)
        ]
        _validate_matches(rows, yearly_coverage)
        frames.append(_slice_dates(rows, start, end))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def _slice_dates(
    rows: pd.DataFrame, start: date | None, end: date | None
) -> pd.DataFrame:
    if start is not None and end is not None and start > end:
        raise SignalStoreError("start_date cannot be after end_date")
    result = rows
    if start is not None:
        result = result.loc[result["session"].ge(start)]
    if end is not None:
        result = result.loc[result["session"].le(end)]
    return result.reset_index(drop=True)


def get_signal_calendar(
    start_date: date | str,
    end_date: date | str,
    *,
    strategy_id: str | None = None,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> pd.DataFrame:
    """Return coverage, including complete/zero rows; uncomputed dates are absent."""

    _guard(root)
    rows = _slice_dates(_coverage(root), _date(start_date), _date(end_date))
    if strategy_id is not None:
        rows = rows.loc[rows["strategy_id"].eq(_strategy_id(strategy_id))]
    _guard(root)
    return rows.reset_index(drop=True)


def read_strategy_signals(
    strategy_id: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    *,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> pd.DataFrame:
    """Read one strategy's full typed diagnostics over an inclusive interval."""

    _guard(root)
    result = _read_strategy(
        root,
        _strategy_id(strategy_id),
        _coverage(root),
        None if start_date is None else _date(start_date),
        None if end_date is None else _date(end_date),
    )
    _guard(root)
    return result


def export_signal_csv(
    destination: Path,
    start_date: date | str,
    end_date: date | str,
    *,
    strategies: Sequence[str] | None = None,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> int:
    """Export committed signals with all diagnostics through one shared CSV path.

    Replace only the explicitly named CSV atomically; no signals are fabricated
    for complete/zero sessions. The Parquet coverage remains authoritative.
    """
    destination = Path(destination).expanduser()
    if destination.suffix.lower() != ".csv":
        raise SignalStoreError("Signal export destination must be a .csv file")
    coverage = get_signal_calendar(start_date, end_date, root=root)
    ids = (
        tuple(dict.fromkeys(strategies))
        if strategies is not None
        else tuple(coverage["strategy_id"].unique())
    )
    frames = [
        read_strategy_signals(value, start_date, end_date, root=root) for value in ids
    ]
    rows = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty()
    rows = rows.sort_values(["session", "strategy_id", "ticker"], ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".signal-csv-", dir=destination.parent
    ) as temp:
        staged = Path(temp) / "signals.csv"
        rows.to_csv(staged, index=False)
        os.replace(staged, destination)
    return len(rows)


def get_signals_for_date(
    session: date | str,
    *,
    strategy_id: str | None = None,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> pd.DataFrame:
    """Return common fields and tickers for one date, across selected strategies."""

    requested = _date(session)
    _guard(root)
    coverage = _coverage(root)
    ids = (
        [_strategy_id(strategy_id)]
        if strategy_id is not None
        else sorted(
            coverage.loc[coverage["session"].eq(requested), "strategy_id"].unique()
        )
    )
    frames = [
        _read_strategy(root, value, coverage, requested, requested).loc[
            :, SIGNAL_COMMON_SCHEMA.names
        ]
        for value in ids
    ]
    _guard(root)
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def get_signal_detail(
    session: date | str,
    strategy_id: str,
    ticker: str,
    *,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> pd.Series | None:
    """Return common metadata plus every diagnostic; a missing match is None."""

    normalized = normalize_ticker(ticker)
    if normalized is None:
        raise SignalStoreError(f"Invalid ticker: {ticker!r}")
    rows = read_strategy_signals(strategy_id, session, session, root=root)
    selected = rows.loc[rows["ticker"].eq(normalized)]
    return None if selected.empty else selected.iloc[0].copy()


def replace_signal_range(
    signals_by_strategy: Mapping[str, pd.DataFrame],
    coverage: pd.DataFrame,
    *,
    root: Path = DEFAULT_SIGNAL_ROOT,
) -> dict[str, Any]:
    """Replace exact session/strategy keys, including old matches becoming false.

    Coverage defines the replacement set. All versions/configurations of those
    session/strategy pairs are replaced; unrelated dates and strategies remain.
    Stage and verify results first, install coverage last, roll back on errors.
    A durable marker prevents treating an interrupted replacement as complete.
    """

    incoming_coverage = _normalize(coverage, coverage=True).to_pandas()
    if incoming_coverage.empty:
        raise SignalStoreError("Replacement coverage cannot be empty")
    ids = set(incoming_coverage["strategy_id"])
    if set(signals_by_strategy) != ids:
        raise SignalStoreError("Signals mapping must match coverage strategies exactly")
    incoming: dict[str, pd.DataFrame] = {}
    for strategy_id, rows in signals_by_strategy.items():
        normalized = _normalize(rows, coverage=False).to_pandas()
        if not normalized["strategy_id"].eq(strategy_id).all():
            raise SignalStoreError("Signals mapping contains another strategy")
        _validate_matches(
            normalized,
            incoming_coverage.loc[incoming_coverage["strategy_id"].eq(strategy_id)],
        )
        incoming[strategy_id] = normalized

    root.mkdir(parents=True, exist_ok=True)
    marker = root / _WRITE_MARKER
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write("Signal replacement in progress; do not consume coverage.\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise SignalStoreError(
            f"Another or interrupted signal replacement exists: {root}"
        ) from exc

    before: dict[str, str | None] = {}
    mutation_started = False
    complete = False
    backup = root / f".signal-backup-{uuid.uuid4().hex}"
    try:
        previous_coverage = _coverage(root)
        replace_keys = pd.MultiIndex.from_frame(
            incoming_coverage[["session", "strategy_id"]]
        )
        old_keys = pd.MultiIndex.from_frame(
            previous_coverage[["session", "strategy_id"]]
        )
        combined_coverage = pd.concat(
            [previous_coverage.loc[~old_keys.isin(replace_keys)], incoming_coverage],
            ignore_index=True,
        )
        tables: dict[str, pa.Table] = {}
        for strategy_id in sorted(ids):
            selected_coverage = incoming_coverage.loc[
                incoming_coverage["strategy_id"].eq(strategy_id)
            ]
            for year in sorted({value.year for value in selected_coverage["session"]}):
                sessions = set(
                    selected_coverage.loc[
                        selected_coverage["session"]
                        .map(lambda value: value.year)
                        .eq(year),
                        "session",
                    ]
                )
                path = _partition(root, strategy_id, year)
                previous = _read_strategy(
                    root,
                    strategy_id,
                    previous_coverage,
                    date(year, 1, 1),
                    date(year, 12, 31),
                )
                if (
                    path.exists()
                    and previous_coverage.loc[
                        previous_coverage["strategy_id"].eq(strategy_id)
                        & previous_coverage["session"]
                        .map(lambda value: value.year)
                        .eq(year)
                    ].empty
                ):
                    raise SignalStoreError(f"Unindexed signal partition: {path}")
                new_rows = incoming[strategy_id].loc[
                    incoming[strategy_id]["session"].isin(sessions)
                ]
                retained = previous.loc[~previous["session"].isin(sessions)]
                combined = (
                    pd.concat([retained, new_rows], ignore_index=True)
                    if not retained.empty
                    else new_rows
                )
                table = _normalize(combined, coverage=False)
                yearly = combined_coverage.loc[
                    combined_coverage["strategy_id"].eq(strategy_id)
                    & combined_coverage["session"]
                    .map(lambda value: value.year)
                    .eq(year)
                ]
                _validate_matches(table.to_pandas(), yearly)
                tables[f"{strategy_id}/{year}.parquet"] = table
        tables[COVERAGE_NAME] = _normalize(combined_coverage, coverage=True)
        before = {
            name: calculate_sha256(root / name) if (root / name).is_file() else None
            for name in tables
        }
        with tempfile.TemporaryDirectory(prefix=".signal-staging-", dir=root) as temp:
            staging = Path(temp)
            for name, table in tables.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, destination, compression="zstd")
                _read(destination, coverage=name == COVERAGE_NAME)
            expected_hashes = {
                name: calculate_sha256(staging / name) for name in tables
            }

            def validate_installation() -> None:
                for name, digest in expected_hashes.items():
                    if calculate_sha256(root / name) != digest:
                        raise SignalStoreError(
                            f"Installed signal partition hash mismatch: {name}"
                        )

            mutation_started = True
            replace_files_transactionally(
                root,
                staging,
                list(tables),
                backup_root=backup,
                validate_after=validate_installation,
            )
            complete = True
        LOGGER.info(
            "Replaced historical signals: coverage_rows=%d signals=%d partitions=%d",
            len(incoming_coverage),
            int(incoming_coverage["signal_count"].sum()),
            len(tables) - 1,
        )
    finally:
        restored = (
            not mutation_started
            or complete
            or all(
                (not (root / name).exists())
                if digest is None
                else (
                    (root / name).is_file() and calculate_sha256(root / name) == digest
                )
                for name, digest in before.items()
            )
        )
        if restored:
            remove_owned_tree(backup, parent=root, prefix=".signal-backup-")
            marker.unlink(missing_ok=True)
    return {
        "coverage_rows": len(incoming_coverage),
        "signal_rows": int(incoming_coverage["signal_count"].sum()),
        "success": True,
    }
