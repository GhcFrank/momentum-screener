"""Derived consecutive-session counts for committed signal facts."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from momentum_screener.signal_store import (
    DEFAULT_SIGNAL_ROOT,
    get_signal_calendar,
    read_strategy_signals,
)
from momentum_screener.storage_manifest import (
    calculate_sha256,
    remove_owned_tree,
    replace_files_transactionally,
)
from momentum_screener.strategy_data import (
    resolve_strategy_sessions,
    sessions_in_range,
)
from momentum_screener.universe import normalize_ticker

LOGGER = logging.getLogger(__name__)
STREAK_SCHEMA_VERSION: Final[str] = "signal_streak_v1"
STREAK_DIRECTORY: Final[str] = "_streaks"
INITIAL_CONTEXT_SESSIONS: Final[int] = 64
MAX_CONTEXT_SESSIONS: Final[int] = 2048
_WRITE_MARKER = ".replacement-in-progress"
_SCHEMA_METADATA = {b"signal_streak_schema": STREAK_SCHEMA_VERSION.encode()}
STREAK_SCHEMA = pa.schema(
    [
        pa.field("session", pa.date32(), nullable=False),
        pa.field("strategy_id", pa.string(), nullable=False),
        pa.field("strategy_version", pa.string(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("signal_streak", pa.int64()),
    ],
    metadata=_SCHEMA_METADATA,
)
STREAK_COLUMNS: Final[tuple[str, ...]] = tuple(STREAK_SCHEMA.names)


class SignalStreakError(RuntimeError):
    """Signal streak data cannot be calculated or persisted safely."""


class SignalStreakContextUnavailable(RuntimeError):
    """Current-definition historical context is legitimately unavailable."""


@dataclass(frozen=True, slots=True)
class SignalIdentity:
    """Fields that determine whether two signal facts share one definition."""

    strategy_id: str
    strategy_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.strategy_id) is None:
            raise ValueError(f"Invalid strategy_id: {self.strategy_id!r}")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version must be non-empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.config_hash) is None:
            raise ValueError("config_hash must be a SHA-256 hex digest")


ContextRecompute = Callable[
    [date, date, Sequence[str]],
    tuple[Mapping[str, pd.DataFrame], pd.DataFrame],
]


def default_streak_root(signal_root: Path = DEFAULT_SIGNAL_ROOT) -> Path:
    """Keep deletable streak data inside, but distinct from, the signal store."""

    return Path(signal_root) / STREAK_DIRECTORY


def _empty() -> pd.DataFrame:
    return STREAK_SCHEMA.empty_table().to_pandas()


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise SignalStreakError("session must be a date or YYYY-MM-DD string")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SignalStreakError("session must use YYYY-MM-DD") from exc


def _partition(root: Path, strategy_id: str, year: int) -> Path:
    return root / strategy_id / f"{year}.parquet"


def _guard(root: Path) -> None:
    if (root / _WRITE_MARKER).exists():
        raise SignalStreakError(
            f"Streak replacement is in progress or was interrupted: {root}"
        )


def _normalize(rows: pd.DataFrame) -> pa.Table:
    missing = sorted(set(STREAK_COLUMNS).difference(rows.columns))
    if missing:
        raise SignalStreakError(f"Streak rows are missing columns: {missing}")
    frame = rows.loc[:, STREAK_COLUMNS].copy().reset_index(drop=True)
    try:
        values = pd.to_datetime(frame["session"], errors="raise")
        if values.isna().any():
            raise ValueError("null session")
        frame["session"] = values.dt.date
        for value in frame["strategy_id"].unique():
            if re.fullmatch(r"[a-z][a-z0-9_]*", str(value)) is None:
                raise ValueError(f"invalid strategy_id: {value!r}")
        if (
            not frame["strategy_version"]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError("strategy_version must be non-empty")
        if (
            not frame["config_hash"]
            .map(
                lambda value: (
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                )
            )
            .all()
        ):
            raise ValueError("config_hash must be a SHA-256 hex digest")
        frame["ticker"] = frame["ticker"].map(normalize_ticker).astype("string")
        if frame["ticker"].isna().any():
            raise ValueError("ticker must be valid")
        streak = pd.to_numeric(frame["signal_streak"], errors="coerce")
        valid = streak.isna() | (streak.ge(1) & streak.eq(streak.round()))
        if not valid.all():
            raise ValueError("signal_streak must be a positive integer or null")
        frame["signal_streak"] = streak.astype("Int64")
        keys = ["session", "strategy_id", "ticker"]
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate streak keys: {keys}")
        frame = frame.sort_values(keys, kind="mergesort", ignore_index=True)
        table = pa.Table.from_pandas(frame, schema=STREAK_SCHEMA, preserve_index=False)
        table.validate(full=True)
        return table.replace_schema_metadata(_SCHEMA_METADATA)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise SignalStreakError(f"Invalid streak rows: {exc}") from exc


def _read_partition(path: Path) -> pd.DataFrame:
    try:
        table = pq.read_table(path)
        if not table.schema.equals(STREAK_SCHEMA, check_metadata=True):
            raise SignalStreakError(f"Unexpected streak schema: {path}")
        normalized = _normalize(table.to_pandas())
        if not normalized.equals(table, check_metadata=False):
            raise SignalStreakError(f"Streak partition is not normalized: {path}")
        return normalized.to_pandas()
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        raise SignalStreakError(
            f"Unable to read streak partition {path}: {exc}"
        ) from exc


def read_signal_streaks(
    strategy_id: str,
    start_date: date | str,
    end_date: date | str,
    *,
    signal_root: Path = DEFAULT_SIGNAL_ROOT,
    streak_root: Path | None = None,
) -> pd.DataFrame:
    """Read one strategy's derived rows over an inclusive natural-date range."""

    start, end = _as_date(start_date), _as_date(end_date)
    if start > end:
        raise SignalStreakError("start_date cannot be after end_date")
    identity_check = SignalIdentity(strategy_id, "read", "0" * 64)
    root = streak_root or default_streak_root(signal_root)
    _guard(root)
    frames = []
    for year in range(start.year, end.year + 1):
        path = _partition(root, identity_check.strategy_id, year)
        if path.is_file():
            rows = _read_partition(path)
            rows = rows.loc[rows["session"].between(start, end)]
            frames.append(rows)
    result = pd.concat(frames, ignore_index=True) if frames else _empty()
    _guard(root)
    return result


def _replace_streak_range(
    rows_by_strategy: Mapping[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    signal_root: Path,
    streak_root: Path | None,
) -> dict[str, Any]:
    root = streak_root or default_streak_root(signal_root)
    incoming = {
        strategy_id: _normalize(rows).to_pandas()
        for strategy_id, rows in rows_by_strategy.items()
    }
    for strategy_id, rows in incoming.items():
        if not rows.empty and not rows["strategy_id"].eq(strategy_id).all():
            raise SignalStreakError("Streak mapping contains another strategy")
        if not rows.empty and not rows["session"].between(start, end).all():
            raise SignalStreakError(
                "Incoming streak rows are outside replacement range"
            )
    root.mkdir(parents=True, exist_ok=True)
    marker = root / _WRITE_MARKER
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write("Streak replacement in progress; do not consume rows.\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise SignalStreakError(
            f"Interrupted streak replacement exists: {root}"
        ) from exc

    tables: dict[str, pa.Table] = {}
    before: dict[str, str | None] = {}
    mutation_started = False
    complete = False
    backup = root / f".streak-backup-{uuid.uuid4().hex}"
    try:
        for strategy_id, rows in incoming.items():
            for year in range(start.year, end.year + 1):
                path = _partition(root, strategy_id, year)
                previous = _read_partition(path) if path.is_file() else _empty()
                retained = previous.loc[
                    ~previous["session"].between(
                        max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
                    )
                ]
                selected = rows.loc[
                    rows["session"].map(lambda value: value.year).eq(year)
                ]
                combined = pd.concat([retained, selected], ignore_index=True)
                name = f"{strategy_id}/{year}.parquet"
                tables[name] = _normalize(combined)
        before = {
            name: calculate_sha256(root / name) if (root / name).is_file() else None
            for name in tables
        }
        with tempfile.TemporaryDirectory(prefix=".streak-staging-", dir=root) as temp:
            staging = Path(temp)
            for name, table in tables.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, destination, compression="zstd")
                _read_partition(destination)
            hashes = {name: calculate_sha256(staging / name) for name in tables}

            def validate_installation() -> None:
                for name, digest in hashes.items():
                    if calculate_sha256(root / name) != digest:
                        raise SignalStreakError(
                            f"Installed streak partition hash mismatch: {name}"
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
    finally:
        restored = (
            not mutation_started
            or complete
            or all(
                (not (root / name).exists())
                if digest is None
                else (root / name).is_file() and calculate_sha256(root / name) == digest
                for name, digest in before.items()
            )
        )
        if restored:
            remove_owned_tree(backup, parent=root, prefix=".streak-backup-")
            marker.unlink(missing_ok=True)
    return {
        "success": True,
        "strategy_count": len(incoming),
        "streak_rows": sum(len(rows) for rows in incoming.values()),
    }


def _identity_from_coverage(strategy_id: str, coverage: pd.DataFrame) -> SignalIdentity:
    unique = coverage.loc[
        coverage["strategy_id"].eq(strategy_id),
        ["strategy_version", "config_hash"],
    ].drop_duplicates()
    if len(unique) != 1:
        raise SignalStreakError(
            f"Streak refresh requires one signal definition for {strategy_id}"
        )
    row = unique.iloc[0]
    return SignalIdentity(
        strategy_id=strategy_id,
        strategy_version=str(row["strategy_version"]),
        config_hash=str(row["config_hash"]),
    )


def _matches_identity(row: pd.Series, identity: SignalIdentity) -> bool:
    return bool(
        row["strategy_id"] == identity.strategy_id
        and row["strategy_version"] == identity.strategy_version
        and row["config_hash"] == identity.config_hash
    )


def _signal_sets(rows: pd.DataFrame) -> dict[date, set[str]]:
    if rows.empty:
        return {}
    return {
        session: set(group["ticker"].astype(str))
        for session, group in rows.groupby("session", sort=False)
    }


def _walk_context(
    identity: SignalIdentity,
    tickers: set[str],
    expected_sessions: Sequence[date],
    coverage: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[dict[str, int], set[str], bool]:
    """Walk backward until false, returning whether the left edge was trusted."""

    unresolved = set(tickers)
    counts = {ticker: 0 for ticker in tickers}
    resolved: dict[str, int] = {}
    true_by_session = _signal_sets(signals)
    coverage_by_session = {
        session: group for session, group in coverage.groupby("session", sort=False)
    }
    for session in reversed(expected_sessions):
        rows = coverage_by_session.get(session)
        if (
            rows is None
            or len(rows) != 1
            or not _matches_identity(rows.iloc[0], identity)
        ):
            return resolved, unresolved, False
        present = true_by_session.get(session, set())
        for ticker in tuple(unresolved):
            if ticker in present:
                counts[ticker] += 1
            else:
                resolved[ticker] = counts[ticker]
                unresolved.remove(ticker)
        if not unresolved:
            return resolved, unresolved, True
    return resolved, unresolved, True


def _prior_sessions(start: date, count: int) -> tuple[date, ...]:
    sessions = resolve_strategy_sessions(start, count + 1)
    if not sessions or sessions[-1] != start:
        raise SignalStreakError(f"Streak start is not an XNYS session: {start}")
    return sessions[:-1]


def _stored_prior_counts(
    identity: SignalIdentity,
    start: date,
    tickers: set[str],
    *,
    signal_root: Path,
) -> tuple[dict[str, int], set[str]]:
    if not tickers:
        return {}, set()
    window = INITIAL_CONTEXT_SESSIONS
    while True:
        expected = _prior_sessions(start, window)
        coverage = get_signal_calendar(
            expected[0],
            expected[-1],
            strategy_id=identity.strategy_id,
            root=signal_root,
        )
        signals = read_strategy_signals(
            identity.strategy_id, expected[0], expected[-1], root=signal_root
        )
        resolved, unresolved, trusted_edge = _walk_context(
            identity, tickers, expected, coverage, signals
        )
        if not unresolved:
            return resolved, set()
        if not trusted_edge or window >= MAX_CONTEXT_SESSIONS:
            return resolved, unresolved
        window = min(window * 2, MAX_CONTEXT_SESSIONS)


def _recomputed_prior_counts(
    identities: Mapping[str, SignalIdentity],
    start: date,
    tickers_by_strategy: Mapping[str, set[str]],
    *,
    context_recompute: ContextRecompute | None,
) -> dict[str, dict[str, int | None]]:
    result: dict[str, dict[str, int | None]] = {
        strategy_id: {} for strategy_id in identities
    }
    pending = {
        strategy_id: set(tickers)
        for strategy_id, tickers in tickers_by_strategy.items()
        if tickers
    }
    if not pending or context_recompute is None:
        for strategy_id, tickers in pending.items():
            result[strategy_id].update({ticker: None for ticker in tickers})
        return result
    window = INITIAL_CONTEXT_SESSIONS
    while pending:
        expected = _prior_sessions(start, window)
        try:
            signals_by_strategy, coverage = context_recompute(
                expected[0], expected[-1], tuple(sorted(pending))
            )
        except SignalStreakContextUnavailable as exc:
            LOGGER.warning("Signal streak context unavailable: %s", exc)
            for strategy_id, tickers in pending.items():
                result[strategy_id].update({ticker: None for ticker in tickers})
            break
        next_pending: dict[str, set[str]] = {}
        for strategy_id, tickers in pending.items():
            identity = identities[strategy_id]
            resolved, unresolved, trusted_edge = _walk_context(
                identity,
                tickers,
                expected,
                coverage.loc[coverage["strategy_id"].eq(strategy_id)],
                signals_by_strategy.get(strategy_id, pd.DataFrame()),
            )
            result[strategy_id].update(resolved)
            if unresolved:
                if trusted_edge and window < MAX_CONTEXT_SESSIONS:
                    next_pending[strategy_id] = unresolved
                else:
                    result[strategy_id].update({ticker: None for ticker in unresolved})
        pending = next_pending
        window = min(window * 2, MAX_CONTEXT_SESSIONS)
    return result


def _calculate_streak_rows(
    signals_by_strategy: Mapping[str, pd.DataFrame],
    coverage: pd.DataFrame,
    identities: Mapping[str, SignalIdentity],
    start: date,
    end: date,
    *,
    signal_root: Path,
    context_recompute: ContextRecompute | None,
) -> dict[str, pd.DataFrame]:
    expected = sessions_in_range(start, end)
    normalized_signals: dict[str, pd.DataFrame] = {}
    start_tickers: dict[str, set[str]] = {}
    all_tickers: dict[str, set[str]] = {}
    for strategy_id, identity in identities.items():
        strategy_coverage = coverage.loc[coverage["strategy_id"].eq(strategy_id)]
        if set(strategy_coverage["session"]) != set(expected) or len(
            strategy_coverage
        ) != len(expected):
            raise SignalStreakError(
                f"Signal coverage is incomplete for {strategy_id} in {start}..{end}"
            )
        if not all(
            _matches_identity(row, identity) for _, row in strategy_coverage.iterrows()
        ):
            raise SignalStreakError(
                f"Signal coverage identity changed inside {strategy_id} range"
            )
        rows = signals_by_strategy.get(strategy_id, pd.DataFrame()).copy()
        if rows.empty:
            rows = pd.DataFrame(columns=["session", "ticker"])
        else:
            if "session" in rows:
                rows["session"] = pd.to_datetime(
                    rows["session"], errors="raise"
                ).dt.date
            elif "date" in rows:
                rows["session"] = pd.to_datetime(rows["date"], errors="raise").dt.date
            elif start == end:
                rows["session"] = start
            else:
                raise SignalStreakError(
                    "Signal rows require a session/date column for a date range"
                )
            if "signal" in rows:
                rows = rows.loc[rows["signal"].fillna(False).astype("bool")]
            rows["ticker"] = rows["ticker"].map(normalize_ticker)
            if rows["ticker"].isna().any():
                raise SignalStreakError("Current signal rows contain an invalid ticker")
            if rows.duplicated(["session", "ticker"]).any():
                raise SignalStreakError("Current signal rows contain duplicate keys")
        rows = rows.loc[rows["session"].isin(expected), ["session", "ticker"]]
        normalized_signals[strategy_id] = rows
        all_tickers[strategy_id] = set(rows["ticker"].astype(str))
        start_tickers[strategy_id] = set(
            rows.loc[rows["session"].eq(expected[0]), "ticker"].astype(str)
        )

    prior: dict[str, dict[str, int | None]] = {key: {} for key in identities}
    need_recompute: dict[str, set[str]] = {}
    for strategy_id, tickers in start_tickers.items():
        resolved, unresolved = _stored_prior_counts(
            identities[strategy_id], start, tickers, signal_root=signal_root
        )
        prior[strategy_id].update(resolved)
        if unresolved:
            need_recompute[strategy_id] = unresolved
    recomputed = _recomputed_prior_counts(
        {key: identities[key] for key in need_recompute},
        start,
        need_recompute,
        context_recompute=context_recompute,
    )
    for strategy_id, values in recomputed.items():
        prior[strategy_id].update(values)

    output: dict[str, pd.DataFrame] = {}
    for strategy_id, identity in identities.items():
        true_by_session = _signal_sets(normalized_signals[strategy_id])
        state: dict[str, int | None] = {
            ticker: prior[strategy_id].get(ticker, 0)
            for ticker in all_tickers[strategy_id]
        }
        records: list[tuple[object, ...]] = []
        for session in expected:
            present = true_by_session.get(session, set())
            for ticker in all_tickers[strategy_id]:
                if ticker not in present:
                    state[ticker] = 0
                    continue
                value = state[ticker]
                state[ticker] = None if value is None else value + 1
                records.append(
                    (
                        session,
                        identity.strategy_id,
                        identity.strategy_version,
                        identity.config_hash,
                        ticker,
                        state[ticker],
                    )
                )
        output[strategy_id] = _normalize(
            pd.DataFrame(records, columns=STREAK_COLUMNS)
        ).to_pandas()
    return output


def refresh_signal_streak_range(
    start_date: date | str,
    end_date: date | str,
    *,
    strategies: Sequence[str] | None = None,
    signal_root: Path = DEFAULT_SIGNAL_ROOT,
    streak_root: Path | None = None,
    context_recompute: ContextRecompute | None = None,
) -> dict[str, Any]:
    """Rebuild only ``start_date:end_date`` using trusted or recomputed context."""

    start, end = _as_date(start_date), _as_date(end_date)
    if start > end:
        raise SignalStreakError("start_date cannot be after end_date")
    coverage = get_signal_calendar(start, end, root=signal_root)
    ids = (
        tuple(dict.fromkeys(strategies))
        if strategies is not None
        else tuple(coverage["strategy_id"].unique())
    )
    if not ids:
        raise SignalStreakError("No signal coverage exists for streak refresh")
    identities = {
        strategy_id: _identity_from_coverage(strategy_id, coverage)
        for strategy_id in ids
    }
    signals = {
        strategy_id: read_strategy_signals(strategy_id, start, end, root=signal_root)
        for strategy_id in ids
    }
    rows = _calculate_streak_rows(
        signals,
        coverage,
        identities,
        start,
        end,
        signal_root=signal_root,
        context_recompute=context_recompute,
    )
    result = _replace_streak_range(
        rows,
        start,
        end,
        signal_root=signal_root,
        streak_root=streak_root,
    )
    LOGGER.info(
        "Refreshed signal streaks %s..%s strategies=%d rows=%d",
        start,
        end,
        len(ids),
        result["streak_rows"],
    )
    return {**result, "start_date": start.isoformat(), "end_date": end.isoformat()}


def enrich_signals_with_streak(
    rows: pd.DataFrame,
    *,
    signal_root: Path = DEFAULT_SIGNAL_ROOT,
    streak_root: Path | None = None,
) -> pd.DataFrame:
    """Left-join persisted streaks by full signal identity, preserving row order."""

    result = rows.drop(columns="signal_streak", errors="ignore").copy()
    original_attrs = rows.attrs.copy()
    if result.empty:
        result["signal_streak"] = pd.Series(dtype="Int64")
        result.attrs = original_attrs
        return result
    required = {
        "session",
        "strategy_id",
        "strategy_version",
        "config_hash",
        "ticker",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise SignalStreakError(f"Signal rows are missing identity columns: {missing}")
    start = pd.to_datetime(result["session"], errors="raise").dt.date.min()
    end = pd.to_datetime(result["session"], errors="raise").dt.date.max()
    frames = [
        read_signal_streaks(
            strategy_id,
            start,
            end,
            signal_root=signal_root,
            streak_root=streak_root,
        )
        for strategy_id in result["strategy_id"].unique()
    ]
    streaks = pd.concat(frames, ignore_index=True) if frames else _empty()
    keys = [
        "session",
        "strategy_id",
        "strategy_version",
        "config_hash",
        "ticker",
    ]
    result["session"] = pd.to_datetime(result["session"], errors="raise").dt.date
    result = result.merge(streaks, on=keys, how="left", validate="one_to_one")
    result["signal_streak"] = result["signal_streak"].astype("Int64")
    result.attrs = original_attrs
    return result


def resolve_current_signal_streaks(
    session: date,
    rows_by_strategy: Mapping[str, pd.DataFrame],
    identities: Mapping[str, SignalIdentity],
    *,
    signal_root: Path = DEFAULT_SIGNAL_ROOT,
    context_recompute: ContextRecompute | None = None,
) -> dict[str, pd.DataFrame]:
    """Attach current streaks without persisting daily email signal facts."""

    if set(rows_by_strategy) != set(identities):
        raise SignalStreakError(
            "Current signal rows and identities must name the same strategies"
        )
    if any(key != identity.strategy_id for key, identity in identities.items()):
        raise SignalStreakError(
            "Current signal identity key does not match strategy_id"
        )

    coverage = pd.DataFrame(
        [
            {
                "session": session,
                "strategy_id": identity.strategy_id,
                "strategy_version": identity.strategy_version,
                "config_hash": identity.config_hash,
            }
            for identity in identities.values()
        ]
    )
    streaks = _calculate_streak_rows(
        rows_by_strategy,
        coverage,
        identities,
        session,
        session,
        signal_root=signal_root,
        context_recompute=context_recompute,
    )
    result: dict[str, pd.DataFrame] = {}
    for strategy_id, rows in rows_by_strategy.items():
        enriched = rows.drop(columns="signal_streak", errors="ignore").copy()
        original_attrs = rows.attrs.copy()
        lookup = streaks[strategy_id].set_index("ticker")["signal_streak"]
        enriched["signal_streak"] = pd.Series(
            enriched["ticker"].map(normalize_ticker).map(lookup),
            index=enriched.index,
            dtype="Int64",
        )
        enriched.attrs = original_attrs
        result[strategy_id] = enriched
    return result
