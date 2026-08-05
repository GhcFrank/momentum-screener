"""Backfill validated daily Yahoo prices for the static stock universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

from momentum_screener.universe import normalize_ticker

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "daily_prices_v1"
SOURCE_NAME = "yahoo_finance_via_yfinance"
DEFAULT_UNIVERSE = Path("data/universe/universe.csv")
DEFAULT_START = date(2010, 1, 1)
DEFAULT_OUTPUT_ROOT = Path("data/processed/prices")
DEFAULT_STAGING_ROOT = Path("data/raw/yahoo_daily/.staging")
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 250
DEFAULT_MAX_RETRIES = 2
DEFAULT_PAUSE_SECONDS = 1.0
DEFAULT_TIMEOUT = 30.0
SMALL_RETRY_BATCH_SIZE = 10

PRICE_COLUMNS = ("date", "ticker", "close", "adj_close", "volume")
COVERAGE_COLUMNS = (
    "ticker",
    "status",
    "first_date",
    "last_date",
    "row_count",
    "attempt_count",
    "last_error",
)
FAILURE_COLUMNS = (
    "ticker",
    "status",
    "attempt_count",
    "error_type",
    "error_message",
)
PRICE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("adj_close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
    ]
)

DownloadFunction = Callable[..., object]
SleepFunction = Callable[[float], None]
_YFINANCE_PANDAS_COMPATIBILITY_LOCK = RLock()


class PriceBackfillError(RuntimeError):
    """Base error for a safe historical price backfill failure."""


class UniverseReadError(PriceBackfillError):
    """Raised when the static Universe cannot be read safely."""


class StagingError(PriceBackfillError):
    """Raised when staging state is missing, inconsistent, or corrupt."""


class DataValidationError(PriceBackfillError):
    """Raised when downloaded or persisted data violates required invariants."""


class DataConflictError(DataValidationError):
    """Raised for conflicting values at the same date and ticker."""


class BackfillIncompleteError(PriceBackfillError):
    """Raised when terminal ticker statuses prevent publication."""


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """A compact, report-safe error description."""

    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """The pure normalization result for one Yahoo response."""

    rows: pd.DataFrame
    successful_tickers: frozenset[str]
    no_data_tickers: frozenset[str]
    failed_tickers: Mapping[str, ErrorSummary]
    invalid_rows_removed: int
    duplicate_rows_removed: int


@dataclass(frozen=True, slots=True)
class DownloadAttempt:
    """One network request and its per-ticker interpretation."""

    rows: pd.DataFrame
    successful_tickers: frozenset[str]
    no_data_tickers: frozenset[str]
    failed_tickers: Mapping[str, ErrorSummary]
    invalid_rows_removed: int
    duplicate_rows_removed: int


@dataclass(frozen=True, slots=True)
class BatchExecution:
    """A fully retried logical main batch."""

    rows: pd.DataFrame
    statuses: Mapping[str, str]
    attempt_counts: Mapping[str, int]
    errors: Mapping[str, ErrorSummary]
    invalid_rows_removed: int
    duplicate_rows_removed: int


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="object"),
            "ticker": pd.Series(dtype="string"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
        },
        columns=PRICE_COLUMNS,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_universe(path: Path) -> tuple[str, ...]:
    """Load normalized, unique tickers in deterministic file order."""

    if not path.exists():
        raise UniverseReadError(f"Universe file does not exist: {path}")
    if not path.is_file():
        raise UniverseReadError(f"Universe path is not a file: {path}")

    try:
        with path.open(encoding="utf-8-sig", newline="") as universe_file:
            reader = csv.DictReader(universe_file)
            if reader.fieldnames is None:
                raise UniverseReadError(f"Universe file has no header: {path}")
            if "ticker" not in reader.fieldnames:
                raise UniverseReadError(
                    f"Universe file is missing required 'ticker' column: {path}"
                )

            tickers: list[str] = []
            seen: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                raw_ticker = row.get("ticker")
                if raw_ticker is None:
                    raise UniverseReadError(
                        f"Universe row {line_number} cannot be parsed safely: {path}"
                    )
                if not raw_ticker.strip():
                    continue
                ticker = normalize_ticker(str(raw_ticker))
                if ticker is None:
                    raise UniverseReadError(
                        f"Invalid ticker at {path}:{line_number}: {raw_ticker!r}"
                    )
                if ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)
    except UnicodeError as exc:
        raise UniverseReadError(
            f"Unable to decode Universe file {path}: {exc}"
        ) from exc
    except csv.Error as exc:
        raise UniverseReadError(f"Unable to parse Universe CSV {path}: {exc}") from exc

    if not tickers:
        raise UniverseReadError(f"Universe contains no valid tickers: {path}")
    return tuple(tickers)


def universe_sha256(tickers: Sequence[str]) -> str:
    """Hash the canonical ticker content used by this run."""

    canonical_content = "".join(f"{ticker}\n" for ticker in tickers).encode()
    return hashlib.sha256(canonical_content).hexdigest()


def build_batches(
    tickers: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[tuple[str, ...], ...]:
    """Split tickers into deterministic bounded batches."""

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    return tuple(
        tuple(tickers[offset : offset + batch_size])
        for offset in range(0, len(tickers), batch_size)
    )


def calculate_end_exclusive(now: datetime | None = None) -> date:
    """Return the next calendar date relative to America/New_York."""

    new_york = ZoneInfo("America/New_York")
    if now is None:
        current_new_york = datetime.now(new_york)
    else:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        current_new_york = now.astimezone(new_york)
    return current_new_york.date() + timedelta(days=1)


def build_run_key(
    tickers: Sequence[str],
    start_date: date,
    end_exclusive: date,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Build a stable key from all inputs that define downloaded content."""

    payload = {
        "schema_version": schema_version,
        "tickers": list(tickers),
        "start": start_date.isoformat(),
        "end_exclusive": end_exclusive.isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_field_name(value: object) -> str:
    return " ".join(str(value).strip().replace("_", " ").casefold().split())


def _resolve_download_columns(
    frame: pd.DataFrame, requested_tickers: Sequence[str]
) -> dict[str, dict[str, int]]:
    required_fields = {"close", "adj close", "volume"}
    requested_set = set(requested_tickers)
    resolved: dict[str, dict[str, int]] = {}

    if isinstance(frame.columns, pd.MultiIndex):
        for position, column in enumerate(frame.columns):
            parts = tuple(column) if isinstance(column, tuple) else (column,)
            field_parts = [
                _normalize_field_name(part)
                for part in parts
                if _normalize_field_name(part) in required_fields
            ]
            ticker_parts = []
            for part in parts:
                candidate = normalize_ticker(part)
                if candidate in requested_set:
                    ticker_parts.append(candidate)
            if len(field_parts) != 1 or len(ticker_parts) != 1:
                continue
            ticker = ticker_parts[0]
            field_name = field_parts[0]
            multi_ticker_fields = resolved.setdefault(ticker, {})
            if field_name in multi_ticker_fields:
                raise DataValidationError(
                    f"Yahoo response contains duplicate {field_name!r} columns "
                    f"for ticker {ticker}"
                )
            multi_ticker_fields[field_name] = position
        return resolved

    if len(requested_tickers) != 1:
        if len(frame.columns) == 0:
            return resolved
        raise DataValidationError(
            "Yahoo returned ordinary columns for a multi-ticker request; "
            "the successful ticker cannot be identified safely"
        )

    ticker = requested_tickers[0]
    ordinary_ticker_fields: dict[str, int] = {}
    for position, column in enumerate(frame.columns):
        field_name = _normalize_field_name(column)
        if field_name not in required_fields:
            continue
        if field_name in ordinary_ticker_fields:
            raise DataValidationError(
                f"Yahoo response contains duplicate {field_name!r} columns "
                f"for ticker {ticker}"
            )
        ordinary_ticker_fields[field_name] = position
    if ordinary_ticker_fields:
        resolved[ticker] = ordinary_ticker_fields
    return resolved


def _calendar_dates(index: pd.Index) -> pd.Series:
    values: list[date | None] = []
    for raw_value in index:
        try:
            timestamp = pd.Timestamp(raw_value)
        except (TypeError, ValueError, OverflowError):
            values.append(None)
            continue
        if pd.isna(timestamp):
            values.append(None)
            continue
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        values.append(timestamp.date())
    return pd.Series(values, dtype="object")


def _deduplicate_normalized_rows(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if rows.empty:
        return _empty_price_frame(), 0

    ordered = rows.loc[:, PRICE_COLUMNS].sort_values(
        ["date", "ticker"], kind="mergesort", ignore_index=True
    )
    duplicated = ordered.duplicated(["date", "ticker"], keep=False)
    if not bool(duplicated.any()):
        return ordered, 0

    duplicate_rows_removed = 0
    keep = pd.Series(True, index=ordered.index)
    for (row_date, ticker), group in ordered.loc[duplicated].groupby(
        ["date", "ticker"], sort=False, dropna=False
    ):
        values = group.loc[:, ["close", "adj_close", "volume"]]
        if any(values[column].nunique(dropna=False) != 1 for column in values):
            raise DataConflictError(
                f"Conflicting duplicate rows for ticker {ticker} on {row_date}"
            )
        duplicate_rows_removed += len(group) - 1
        keep.loc[group.index[1:]] = False

    deduplicated = ordered.loc[keep].reset_index(drop=True)
    return deduplicated, duplicate_rows_removed


def validate_normalized_rows(
    rows: pd.DataFrame,
    *,
    start_date: date,
    end_exclusive: date,
    allowed_tickers: Sequence[str] | None = None,
) -> None:
    """Validate the canonical long-table invariants for normalized rows."""

    if tuple(rows.columns) != PRICE_COLUMNS:
        raise DataValidationError(
            f"Expected columns {PRICE_COLUMNS}, found {tuple(rows.columns)}"
        )
    if rows.isna().any(axis=None):
        raise DataValidationError("Normalized price rows contain null values")
    if not pd.api.types.is_float_dtype(rows["close"].dtype):
        raise DataValidationError("close must use a floating-point dtype")
    if not pd.api.types.is_float_dtype(rows["adj_close"].dtype):
        raise DataValidationError("adj_close must use a floating-point dtype")
    if rows["volume"].dtype != np.dtype("int64"):
        raise DataValidationError("volume must use int64 dtype")
    if not bool((rows["close"] > 0).all()):
        raise DataValidationError("close must be positive")
    if not bool((rows["adj_close"] > 0).all()):
        raise DataValidationError("adj_close must be positive")
    if not bool((rows["volume"] >= 0).all()):
        raise DataValidationError("volume cannot be negative")
    if not rows.empty:
        if not rows["date"].map(lambda value: isinstance(value, date)).all():
            raise DataValidationError("date must contain calendar date values")
        if not bool(
            rows["date"].map(lambda value: start_date <= value < end_exclusive).all()
        ):
            raise DataValidationError(
                "Price rows fall outside the requested date range"
            )
        if (
            not rows["ticker"]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        ):
            raise DataValidationError("ticker must be a non-empty string")
    if allowed_tickers is not None:
        unexpected = set(rows["ticker"]) - set(allowed_tickers)
        if unexpected:
            raise DataValidationError(
                f"Rows contain unexpected tickers: {sorted(unexpected)[:10]}"
            )
    if rows.duplicated(["date", "ticker"]).any():
        raise DataValidationError("Normalized rows contain duplicate date+ticker keys")
    expected = rows.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)
    if not rows.reset_index(drop=True).equals(expected):
        raise DataValidationError("Normalized rows are not sorted by date and ticker")


def normalize_download_frame(
    frame: pd.DataFrame,
    requested_tickers: Sequence[str],
    *,
    start_date: date,
    end_exclusive: date,
) -> NormalizationResult:
    """Normalize Yahoo ordinary or MultiIndex columns into a strict long table."""

    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"yf.download returned {type(frame).__name__}, expected DataFrame"
        )
    normalized_requested = tuple(
        ticker
        for ticker in (normalize_ticker(value) for value in requested_tickers)
        if ticker is not None
    )
    if len(normalized_requested) != len(requested_tickers) or len(
        set(normalized_requested)
    ) != len(normalized_requested):
        raise ValueError("requested_tickers must be unique normalized tickers")
    if frame.empty:
        return NormalizationResult(
            rows=_empty_price_frame(),
            successful_tickers=frozenset(),
            no_data_tickers=frozenset(normalized_requested),
            failed_tickers={},
            invalid_rows_removed=0,
            duplicate_rows_removed=0,
        )

    resolved = _resolve_download_columns(frame, normalized_requested)
    calendar_dates = _calendar_dates(frame.index)
    normalized_frames: list[pd.DataFrame] = []
    no_data: set[str] = set()
    failed: dict[str, ErrorSummary] = {}
    invalid_rows_removed = 0
    duplicate_rows_removed = 0
    required_fields = {"close", "adj close", "volume"}

    for ticker in normalized_requested:
        fields = resolved.get(ticker)
        if not fields:
            no_data.add(ticker)
            continue
        missing_fields = sorted(required_fields - set(fields))
        if missing_fields:
            failed[ticker] = ErrorSummary(
                "MissingFields",
                f"Yahoo response is missing fields: {', '.join(missing_fields)}",
            )
            continue

        close_values = pd.to_numeric(
            frame.iloc[:, fields["close"]].reset_index(drop=True), errors="coerce"
        )
        adjusted_values = pd.to_numeric(
            frame.iloc[:, fields["adj close"]].reset_index(drop=True), errors="coerce"
        )
        volume_values = pd.to_numeric(
            frame.iloc[:, fields["volume"]].reset_index(drop=True), errors="coerce"
        )

        close_array = close_values.to_numpy(dtype="float64", na_value=np.nan)
        adjusted_array = adjusted_values.to_numpy(dtype="float64", na_value=np.nan)
        volume_array = volume_values.to_numpy(dtype="float64", na_value=np.nan)
        finite_volume = np.isfinite(volume_array)
        integral_volume = finite_volume & np.isclose(
            volume_array,
            np.rint(volume_array),
            rtol=0.0,
            atol=1e-9,
        )
        in_date_range = calendar_dates.map(
            lambda value: value is not None and start_date <= value < end_exclusive
        ).to_numpy(dtype="bool")
        valid = (
            in_date_range
            & np.isfinite(close_array)
            & (close_array > 0)
            & np.isfinite(adjusted_array)
            & (adjusted_array > 0)
            & integral_volume
            & (volume_array >= 0)
        )
        ticker_invalid_count = int((~valid).sum())
        invalid_rows_removed += ticker_invalid_count
        if not bool(valid.any()):
            failed[ticker] = ErrorSummary(
                "InvalidData",
                "Yahoo returned rows, but none passed date, price, and volume validation",
            )
            continue

        ticker_frame = pd.DataFrame(
            {
                "date": calendar_dates.loc[valid].reset_index(drop=True),
                "ticker": pd.Series([ticker] * int(valid.sum()), dtype="string"),
                "close": close_array[valid].astype("float64"),
                "adj_close": adjusted_array[valid].astype("float64"),
                "volume": np.rint(volume_array[valid]).astype("int64"),
            },
            columns=PRICE_COLUMNS,
        )
        ticker_frame, removed = _deduplicate_normalized_rows(ticker_frame)
        duplicate_rows_removed += removed
        normalized_frames.append(ticker_frame)

    if normalized_frames:
        rows = pd.concat(normalized_frames, ignore_index=True)
        rows, removed = _deduplicate_normalized_rows(rows)
        duplicate_rows_removed += removed
    else:
        rows = _empty_price_frame()

    successful_tickers = frozenset(str(value) for value in rows["ticker"].unique())
    no_data -= successful_tickers
    for ticker in successful_tickers:
        failed.pop(ticker, None)
    validate_normalized_rows(
        rows,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=normalized_requested,
    )
    return NormalizationResult(
        rows=rows,
        successful_tickers=successful_tickers,
        no_data_tickers=frozenset(no_data),
        failed_tickers=failed,
        invalid_rows_removed=invalid_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
    )


def _safe_error_message(error: BaseException | str, limit: int = 500) -> str:
    raw_message = str(error)
    compact = " ".join(raw_message.split())
    compact = re.sub(
        r"([?&](?:crumb|cookie|token|key)=)[^&\s]+",
        r"\1<redacted>",
        compact,
        flags=re.IGNORECASE,
    )
    return compact[:limit] or "unspecified error"


@contextmanager
def _writable_series_numpy_for_yfinance() -> Any:
    """Restore the writable-array behavior yfinance repair expects on Pandas 3."""

    if int(pd.__version__.split(".", maxsplit=1)[0]) < 3:
        yield
        return

    with _YFINANCE_PANDAS_COMPATIBILITY_LOCK:
        original_to_numpy = pd.Series.to_numpy

        def writable_to_numpy(
            series: pd.Series, *args: object, **kwargs: object
        ) -> Any:
            result = original_to_numpy(series, *args, **kwargs)
            if isinstance(result, np.ndarray) and not result.flags.writeable:
                return result.copy()
            return result

        pd.Series.to_numpy = writable_to_numpy  # type: ignore[method-assign]
        try:
            yield
        finally:
            pd.Series.to_numpy = original_to_numpy  # type: ignore[method-assign]


def download_batch(
    tickers: Sequence[str],
    *,
    start_date: date,
    end_exclusive: date,
    timeout: float = DEFAULT_TIMEOUT,
    download_func: DownloadFunction | None = None,
) -> DownloadAttempt:
    """Download one batch exactly once with all required yfinance parameters."""

    if not tickers:
        raise ValueError("download_batch requires at least one ticker")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    call_download = download_func or yf.download
    try:
        with _writable_series_numpy_for_yfinance():
            response = call_download(
                tickers=list(tickers),
                start=start_date.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                repair=True,
                keepna=False,
                threads=True,
                progress=False,
                timeout=timeout,
                multi_level_index=True,
                prepost=False,
                rounding=False,
            )
        normalized = normalize_download_frame(
            response,
            tickers,
            start_date=start_date,
            end_exclusive=end_exclusive,
        )
    except Exception as exc:  # noqa: BLE001 - third-party downloader exceptions vary
        summary = ErrorSummary(type(exc).__name__, _safe_error_message(exc))
        return DownloadAttempt(
            rows=_empty_price_frame(),
            successful_tickers=frozenset(),
            no_data_tickers=frozenset(),
            failed_tickers={ticker: summary for ticker in tickers},
            invalid_rows_removed=0,
            duplicate_rows_removed=0,
        )

    return DownloadAttempt(
        rows=normalized.rows,
        successful_tickers=normalized.successful_tickers,
        no_data_tickers=normalized.no_data_tickers,
        failed_tickers=normalized.failed_tickers,
        invalid_rows_removed=normalized.invalid_rows_removed,
        duplicate_rows_removed=normalized.duplicate_rows_removed,
    )


def _merge_successful_rows(
    existing: pd.DataFrame, new_rows: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    return _deduplicate_normalized_rows(combined)


def execute_batch_with_retries(
    tickers: Sequence[str],
    *,
    start_date: date,
    end_exclusive: date,
    max_retries: int = DEFAULT_MAX_RETRIES,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT,
    existing_rows: pd.DataFrame | None = None,
    existing_statuses: Mapping[str, str] | None = None,
    existing_attempt_counts: Mapping[str, int] | None = None,
    existing_errors: Mapping[str, ErrorSummary] | None = None,
    download_func: DownloadFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> BatchExecution:
    """Run main, small-batch, then bounded single-ticker retry stages."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")

    rows = existing_rows.copy() if existing_rows is not None else _empty_price_frame()
    statuses = {ticker: "pending" for ticker in tickers}
    statuses.update(existing_statuses or {})
    attempt_counts = {ticker: 0 for ticker in tickers}
    attempt_counts.update(existing_attempt_counts or {})
    errors = dict(existing_errors or {})
    saw_no_data = {ticker: statuses.get(ticker) == "no_data" for ticker in tickers}
    saw_failure = {ticker: statuses.get(ticker) == "failed" for ticker in tickers}
    invalid_rows_removed = 0
    duplicate_rows_removed = 0

    successful_in_existing = {str(value) for value in rows["ticker"].unique()}
    for ticker in successful_in_existing:
        statuses[ticker] = "success"
        errors.pop(ticker, None)

    def request(request_tickers: Sequence[str]) -> None:
        nonlocal rows, invalid_rows_removed, duplicate_rows_removed
        if not request_tickers:
            return
        for ticker in request_tickers:
            attempt_counts[ticker] += 1
        attempt = download_batch(
            request_tickers,
            start_date=start_date,
            end_exclusive=end_exclusive,
            timeout=timeout,
            download_func=download_func,
        )
        invalid_rows_removed += attempt.invalid_rows_removed
        duplicate_rows_removed += attempt.duplicate_rows_removed
        try:
            rows, removed = _merge_successful_rows(rows, attempt.rows)
            duplicate_rows_removed += removed
        except DataConflictError as exc:
            summary = ErrorSummary(type(exc).__name__, _safe_error_message(exc))
            for ticker in request_tickers:
                if statuses.get(ticker) != "success":
                    saw_failure[ticker] = True
                    statuses[ticker] = "failed"
                    errors[ticker] = summary
            return

        for ticker in attempt.successful_tickers:
            statuses[ticker] = "success"
            errors.pop(ticker, None)
        for ticker in attempt.no_data_tickers:
            if statuses.get(ticker) != "success":
                saw_no_data[ticker] = True
                statuses[ticker] = "no_data"
                errors[ticker] = ErrorSummary(
                    "NoData", "Yahoo returned no valid historical rows"
                )
        for ticker, summary in attempt.failed_tickers.items():
            if statuses.get(ticker) != "success":
                saw_failure[ticker] = True
                statuses[ticker] = "failed"
                errors[ticker] = summary

    pending = [ticker for ticker in tickers if statuses[ticker] != "success"]
    if pending:
        request(pending)

    pending = [ticker for ticker in tickers if statuses[ticker] != "success"]
    for retry_batch in build_batches(pending, SMALL_RETRY_BATCH_SIZE):
        if pause_seconds:
            sleep_func(pause_seconds)
        request(retry_batch)

    pending = [ticker for ticker in tickers if statuses[ticker] != "success"]
    for ticker in pending:
        for retry_number in range(max_retries):
            if statuses[ticker] == "success":
                break
            delay = pause_seconds * (2**retry_number)
            if delay:
                sleep_func(delay)
            request((ticker,))

    row_tickers = {str(value) for value in rows["ticker"].unique()}
    for ticker in tickers:
        if ticker in row_tickers:
            statuses[ticker] = "success"
            errors.pop(ticker, None)
        elif saw_failure[ticker]:
            statuses[ticker] = "failed"
            errors.setdefault(
                ticker, ErrorSummary("DownloadError", "Download could not be validated")
            )
        elif saw_no_data[ticker]:
            statuses[ticker] = "no_data"
            errors[ticker] = ErrorSummary(
                "NoData", "Yahoo returned no valid historical rows after retries"
            )
        else:
            statuses[ticker] = "failed"
            errors[ticker] = ErrorSummary(
                "DownloadError", "Download ended without a determinate result"
            )

    rows, removed = _deduplicate_normalized_rows(rows)
    duplicate_rows_removed += removed
    validate_normalized_rows(
        rows,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=tickers,
    )
    return BatchExecution(
        rows=rows,
        statuses=statuses,
        attempt_counts=attempt_counts,
        errors=errors,
        invalid_rows_removed=invalid_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
    )


def _dataframe_to_arrow(rows: pd.DataFrame) -> pa.Table:
    validate_columns = rows.loc[:, PRICE_COLUMNS].copy()
    return pa.Table.from_pandas(
        validate_columns,
        schema=PRICE_SCHEMA,
        preserve_index=False,
        safe=True,
    )


def _validate_arrow_table(
    table: pa.Table,
    *,
    start_date: date,
    end_exclusive: date,
    allowed_tickers: Sequence[str] | None = None,
) -> None:
    if table.schema != PRICE_SCHEMA:
        raise DataValidationError(
            f"Unexpected Arrow schema: expected {PRICE_SCHEMA}, found {table.schema}"
        )
    if any(column.null_count for column in table.columns):
        raise DataValidationError("Arrow price table contains null values")
    if table.num_rows == 0:
        return
    if pc.any(pc.less_equal(table["close"], pa.scalar(0.0))).as_py():
        raise DataValidationError("Arrow table contains non-positive close values")
    if pc.any(pc.less_equal(table["adj_close"], pa.scalar(0.0))).as_py():
        raise DataValidationError("Arrow table contains non-positive adj_close values")
    if pc.any(pc.less(table["volume"], pa.scalar(0, pa.int64()))).as_py():
        raise DataValidationError("Arrow table contains negative volume values")
    minimum_date = pc.min(table["date"]).as_py()
    maximum_date = pc.max(table["date"]).as_py()
    if minimum_date < start_date or maximum_date >= end_exclusive:
        raise DataValidationError("Arrow table contains out-of-range dates")
    if allowed_tickers is not None:
        actual_tickers = set(pc.unique(table["ticker"]).to_pylist())
        unexpected = actual_tickers - set(allowed_tickers)
        if unexpected:
            raise DataValidationError(
                f"Arrow table contains unexpected tickers: {sorted(unexpected)[:10]}"
            )


def _arrow_has_duplicate_keys(table: pa.Table) -> bool:
    if table.num_rows < 2:
        return False
    dates = table["date"].combine_chunks()
    tickers = table["ticker"].combine_chunks()
    duplicate_adjacent = pc.and_(
        pc.equal(dates.slice(1), dates.slice(0, table.num_rows - 1)),
        pc.equal(tickers.slice(1), tickers.slice(0, table.num_rows - 1)),
    )
    return bool(pc.any(duplicate_adjacent).as_py())


def write_batch_atomically(
    path: Path,
    rows: pd.DataFrame,
    *,
    start_date: date,
    end_exclusive: date,
    allowed_tickers: Sequence[str],
) -> None:
    """Write and revalidate one staging Parquet through an atomic replace."""

    validate_normalized_rows(
        rows,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=allowed_tickers,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _dataframe_to_arrow(rows)
    _validate_arrow_table(
        table,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=allowed_tickers,
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        pq.write_table(table, temp_path, compression="zstd")
        staged = pq.read_table(temp_path)
        _validate_arrow_table(
            staged,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=allowed_tickers,
        )
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_staging_batch(
    path: Path,
    *,
    start_date: date,
    end_exclusive: date,
    allowed_tickers: Sequence[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise StagingError(f"Staging batch file is missing: {path}")
    try:
        table = pq.read_table(path)
        _validate_arrow_table(
            table,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=allowed_tickers,
        )
        rows = table.to_pandas()
        rows["date"] = rows["date"].map(
            lambda value: value.date() if isinstance(value, pd.Timestamp) else value
        )
        rows["ticker"] = rows["ticker"].astype("string")
        rows["close"] = rows["close"].astype("float64")
        rows["adj_close"] = rows["adj_close"].astype("float64")
        rows["volume"] = rows["volume"].astype("int64")
        rows = rows.loc[:, PRICE_COLUMNS].sort_values(
            ["date", "ticker"], kind="mergesort", ignore_index=True
        )
        validate_normalized_rows(
            rows,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=allowed_tickers,
        )
        return rows
    except (OSError, pa.ArrowException, DataValidationError) as exc:
        raise StagingError(
            f"Staging batch is corrupt or incompatible: {path}: {exc}"
        ) from exc


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        with temp_path.open(encoding="utf-8") as input_file:
            json.load(input_file)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_or_create_run_state(
    staging_dir: Path,
    *,
    run_key: str,
    universe_path: Path,
    universe_hash: str,
    tickers: Sequence[str],
    start_date: date,
    end_exclusive: date,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    """Load a matching state file or atomically create a new one."""

    state_path = staging_dir / "run_state.json"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_key": run_key,
        "universe_path": str(universe_path),
        "universe_sha256": universe_hash,
        "tickers": list(tickers),
        "start": start_date.isoformat(),
        "end_exclusive": end_exclusive.isoformat(),
        "batch_size": batch_size,
    }

    if state_path.exists():
        if not resume:
            raise StagingError(
                f"Staging already exists and --no-resume was requested: {staging_dir}"
            )
        try:
            loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StagingError(
                f"Unable to read staging state {state_path}: {exc}"
            ) from exc
        if not isinstance(loaded_state, dict):
            raise StagingError(f"Staging state must be a JSON object: {state_path}")
        mismatches = [
            key for key, value in expected.items() if loaded_state.get(key) != value
        ]
        if mismatches:
            raise StagingError(
                f"Staging state does not match current run for fields: {mismatches}"
            )
        if not isinstance(loaded_state.get("batches"), dict):
            raise StagingError(
                f"Staging state has invalid batches mapping: {state_path}"
            )
        return loaded_state

    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise StagingError(
            f"Non-empty staging directory has no valid run_state.json: {staging_dir}"
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    now = _utc_now_iso()
    new_state: dict[str, Any] = {
        **expected,
        "completed_batches": [],
        "batches": {},
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    _write_json_atomically(state_path, new_state)
    return new_state


def _error_from_state(value: object) -> ErrorSummary:
    if not isinstance(value, Mapping):
        return ErrorSummary("DownloadError", "Invalid error entry in staging state")
    return ErrorSummary(
        str(value.get("error_type") or "DownloadError"),
        _safe_error_message(str(value.get("message") or "unspecified error")),
    )


def _batch_state_payload(
    *,
    tickers: Sequence[str],
    execution: BatchExecution,
    completed: bool,
) -> dict[str, Any]:
    return {
        "tickers": list(tickers),
        "completed": completed,
        "statuses": dict(execution.statuses),
        "attempt_counts": dict(execution.attempt_counts),
        "errors": {
            ticker: {
                "error_type": summary.error_type,
                "message": summary.message,
            }
            for ticker, summary in execution.errors.items()
            if execution.statuses.get(ticker) != "success"
        },
        "row_count": len(execution.rows),
        "invalid_rows_removed": execution.invalid_rows_removed,
        "duplicate_rows_removed": execution.duplicate_rows_removed,
        "updated_at_utc": _utc_now_iso(),
    }


def _process_logical_batch(
    batch_index: int,
    tickers: Sequence[str],
    *,
    state: dict[str, Any],
    staging_dir: Path,
    start_date: date,
    end_exclusive: date,
    max_retries: int,
    pause_seconds: float,
    timeout: float,
    download_func: DownloadFunction | None,
    sleep_func: SleepFunction,
) -> BatchExecution:
    key = f"{batch_index:04d}"
    batch_path = staging_dir / f"batch_{key}.parquet"
    raw_entry = state["batches"].get(key)
    existing_rows = _empty_price_frame()
    statuses: dict[str, str] = {}
    attempt_counts: dict[str, int] = {}
    errors: dict[str, ErrorSummary] = {}
    prior_invalid = 0
    prior_duplicates = 0

    if raw_entry is not None:
        if not isinstance(raw_entry, Mapping):
            raise StagingError(f"Invalid state entry for batch {key}")
        if raw_entry.get("tickers") != list(tickers):
            raise StagingError(f"Ticker list mismatch for staged batch {key}")
        existing_rows = _read_staging_batch(
            batch_path,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=tickers,
        )
        raw_statuses = raw_entry.get("statuses")
        raw_attempts = raw_entry.get("attempt_counts")
        raw_errors = raw_entry.get("errors")
        if not isinstance(raw_statuses, Mapping) or not isinstance(
            raw_attempts, Mapping
        ):
            raise StagingError(f"Invalid status metadata for staged batch {key}")
        statuses = {
            ticker: str(raw_statuses.get(ticker, "pending")) for ticker in tickers
        }
        try:
            attempt_counts = {
                ticker: int(raw_attempts.get(ticker, 0)) for ticker in tickers
            }
        except (TypeError, ValueError) as exc:
            raise StagingError(
                f"Invalid attempt counts for staged batch {key}"
            ) from exc
        if isinstance(raw_errors, Mapping):
            errors = {
                ticker: _error_from_state(raw_errors[ticker])
                for ticker in tickers
                if ticker in raw_errors
            }
        prior_invalid = int(raw_entry.get("invalid_rows_removed", 0))
        prior_duplicates = int(raw_entry.get("duplicate_rows_removed", 0))

        data_tickers = {str(value) for value in existing_rows["ticker"].unique()}
        state_successes = {
            ticker for ticker, status in statuses.items() if status == "success"
        }
        if data_tickers != state_successes:
            raise StagingError(
                f"Staging data/status mismatch for batch {key}: "
                f"data={sorted(data_tickers)}, state={sorted(state_successes)}"
            )
        if bool(raw_entry.get("completed")):
            if any(status == "failed" for status in statuses.values()):
                raise StagingError(f"Completed batch {key} contains failed tickers")
            LOGGER.info(
                "Reusing completed batch %s (%d tickers, %d rows)",
                key,
                len(tickers),
                len(existing_rows),
            )
            return BatchExecution(
                rows=existing_rows,
                statuses=statuses,
                attempt_counts=attempt_counts,
                errors=errors,
                invalid_rows_removed=prior_invalid,
                duplicate_rows_removed=prior_duplicates,
            )

    execution = execute_batch_with_retries(
        tickers,
        start_date=start_date,
        end_exclusive=end_exclusive,
        max_retries=max_retries,
        pause_seconds=pause_seconds,
        timeout=timeout,
        existing_rows=existing_rows,
        existing_statuses=statuses,
        existing_attempt_counts=attempt_counts,
        existing_errors=errors,
        download_func=download_func,
        sleep_func=sleep_func,
    )
    execution = BatchExecution(
        rows=execution.rows,
        statuses=execution.statuses,
        attempt_counts=execution.attempt_counts,
        errors=execution.errors,
        invalid_rows_removed=prior_invalid + execution.invalid_rows_removed,
        duplicate_rows_removed=prior_duplicates + execution.duplicate_rows_removed,
    )
    completed = not any(status == "failed" for status in execution.statuses.values())
    write_batch_atomically(
        batch_path,
        execution.rows,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=tickers,
    )
    state["batches"][key] = _batch_state_payload(
        tickers=tickers,
        execution=execution,
        completed=completed,
    )
    completed_batches = {int(value) for value in state["completed_batches"]}
    if completed:
        completed_batches.add(batch_index)
    else:
        completed_batches.discard(batch_index)
    state["completed_batches"] = sorted(completed_batches)
    state["updated_at_utc"] = _utc_now_iso()
    _write_json_atomically(staging_dir / "run_state.json", state)
    LOGGER.info(
        "Staged batch %s: %d rows, success=%d, no_data=%d, failed=%d",
        key,
        len(execution.rows),
        sum(status == "success" for status in execution.statuses.values()),
        sum(status == "no_data" for status in execution.statuses.values()),
        sum(status == "failed" for status in execution.statuses.values()),
    )
    return execution


def combine_staging_batches(
    batch_paths: Sequence[Path],
    *,
    start_date: date,
    end_exclusive: date,
    tickers: Sequence[str],
) -> tuple[pa.Table, int]:
    """Combine validated staging batches and enforce global key uniqueness."""

    tables: list[pa.Table] = []
    for path in batch_paths:
        if not path.is_file():
            raise StagingError(f"Required staging batch is missing: {path}")
        try:
            table = pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            raise StagingError(f"Unable to read staging batch {path}: {exc}") from exc
        _validate_arrow_table(
            table,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=tickers,
        )
        tables.append(table)

    if not tables:
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in PRICE_SCHEMA],
            schema=PRICE_SCHEMA,
        ), 0
    combined = pa.concat_tables(tables)
    if combined.num_rows:
        combined = combined.sort_by([("date", "ascending"), ("ticker", "ascending")])

    duplicate_rows_removed = 0
    if _arrow_has_duplicate_keys(combined):
        rows = combined.to_pandas()
        rows["date"] = rows["date"].map(
            lambda value: value.date() if isinstance(value, pd.Timestamp) else value
        )
        rows["ticker"] = rows["ticker"].astype("string")
        rows["close"] = rows["close"].astype("float64")
        rows["adj_close"] = rows["adj_close"].astype("float64")
        rows["volume"] = rows["volume"].astype("int64")
        rows, duplicate_rows_removed = _deduplicate_normalized_rows(rows)
        validate_normalized_rows(
            rows,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=tickers,
        )
        result = _dataframe_to_arrow(rows)
    else:
        result = combined
    _validate_arrow_table(
        result,
        start_date=start_date,
        end_exclusive=end_exclusive,
        allowed_tickers=tickers,
    )
    return result, duplicate_rows_removed


def _coverage_rows(
    tickers: Sequence[str],
    table: pa.Table,
    state: Mapping[str, Any],
) -> list[dict[str, object]]:
    attempts: dict[str, int] = {ticker: 0 for ticker in tickers}
    statuses: dict[str, str] = {ticker: "failed" for ticker in tickers}
    errors: dict[str, ErrorSummary] = {}
    for raw_entry in state["batches"].values():
        entry = raw_entry
        for ticker, count in entry["attempt_counts"].items():
            attempts[ticker] = int(count)
        for ticker, status in entry["statuses"].items():
            statuses[ticker] = str(status)
        for ticker, value in entry.get("errors", {}).items():
            errors[ticker] = _error_from_state(value)

    aggregates: dict[str, tuple[date, date, int]] = {}
    if table.num_rows:
        grouped = (
            table.select(["ticker", "date"])
            .group_by("ticker")
            .aggregate([("date", "min"), ("date", "max"), ("date", "count")])
        )
        for row in grouped.to_pylist():
            aggregates[str(row["ticker"])] = (
                row["date_min"],
                row["date_max"],
                int(row["date_count"]),
            )

    coverage: list[dict[str, object]] = []
    for ticker in sorted(tickers):
        aggregate = aggregates.get(ticker)
        if aggregate is not None:
            first_date, last_date, row_count = aggregate
            status = "success"
            last_error = ""
        else:
            first_date = None
            last_date = None
            row_count = 0
            status = statuses[ticker]
            last_error = errors.get(
                ticker, ErrorSummary("DownloadError", "No terminal status recorded")
            ).message
        coverage.append(
            {
                "ticker": ticker,
                "status": status,
                "first_date": first_date.isoformat() if first_date else "",
                "last_date": last_date.isoformat() if last_date else "",
                "row_count": row_count,
                "attempt_count": attempts[ticker],
                "last_error": last_error,
            }
        )
    return coverage


def _failure_rows(
    coverage: Sequence[Mapping[str, object]], state: Mapping[str, Any]
) -> list[dict[str, object]]:
    error_lookup: dict[str, ErrorSummary] = {}
    for entry in state["batches"].values():
        for ticker, value in entry.get("errors", {}).items():
            error_lookup[ticker] = _error_from_state(value)

    failures: list[dict[str, object]] = []
    for row in coverage:
        status = str(row["status"])
        if status == "success":
            continue
        ticker = str(row["ticker"])
        default_type = "NoData" if status == "no_data" else "DownloadError"
        summary = error_lookup.get(
            ticker, ErrorSummary(default_type, str(row["last_error"]))
        )
        failures.append(
            {
                "ticker": ticker,
                "status": status,
                "attempt_count": int(str(row["attempt_count"])),
                "error_type": summary.error_type,
                "error_message": summary.message,
            }
        )
    return failures


def _write_csv_atomically(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _manifest(
    *,
    universe_path: Path,
    universe_hash: str,
    tickers: Sequence[str],
    start_date: date,
    end_exclusive: date,
    table: pa.Table,
    coverage: Sequence[Mapping[str, object]],
    batch_size: int,
    state: Mapping[str, Any],
    partition_row_counts: Mapping[str, int],
    final_duplicate_rows_removed: int,
    completed: bool,
) -> dict[str, Any]:
    successful = sum(row["status"] == "success" for row in coverage)
    no_data = sum(row["status"] == "no_data" for row in coverage)
    failed = sum(row["status"] == "failed" for row in coverage)
    invalid_removed = sum(
        int(entry.get("invalid_rows_removed", 0)) for entry in state["batches"].values()
    )
    duplicate_removed = final_duplicate_rows_removed + sum(
        int(entry.get("duplicate_rows_removed", 0))
        for entry in state["batches"].values()
    )
    actual_min = pc.min(table["date"]).as_py() if table.num_rows else None
    actual_max = pc.max(table["date"]).as_py() if table.num_rows else None
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "yfinance_version": yf.__version__,
        "generated_at_utc": _utc_now_iso(),
        "universe_path": str(universe_path),
        "universe_sha256": universe_hash,
        "universe_ticker_count": len(tickers),
        "requested_start": start_date.isoformat(),
        "requested_end_exclusive": end_exclusive.isoformat(),
        "actual_min_date": actual_min.isoformat() if actual_min else None,
        "actual_max_date": actual_max.isoformat() if actual_max else None,
        "successful_ticker_count": successful,
        "no_data_ticker_count": no_data,
        "failed_ticker_count": failed,
        "total_row_count": table.num_rows,
        "duplicate_rows_removed": duplicate_removed,
        "invalid_rows_removed": invalid_removed,
        "partition_row_counts": dict(partition_row_counts),
        "batch_size": batch_size,
        "completed": completed,
    }


def write_year_partitions(table: pa.Table, output_root: Path) -> dict[str, int]:
    """Write one zstd Parquet file per year below an unpublished temp root."""

    daily_root = output_root / "daily"
    daily_root.mkdir(parents=True, exist_ok=True)
    if table.num_rows == 0:
        return {}
    years = sorted(
        {int(value) for value in pc.unique(pc.year(table["date"])).to_pylist()}
    )
    partition_counts: dict[str, int] = {}
    year_values = pc.year(table["date"])
    for year in years:
        partition = table.filter(pc.equal(year_values, pa.scalar(year, pa.int64())))
        partition = partition.sort_by([("date", "ascending"), ("ticker", "ascending")])
        partition_dir = daily_root / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=False)
        pq.write_table(
            partition,
            partition_dir / "prices.parquet",
            compression="zstd",
        )
        partition_counts[str(year)] = partition.num_rows
    return partition_counts


def validate_complete_dataset(
    output_root: Path,
    *,
    start_date: date,
    end_exclusive: date,
    tickers: Sequence[str],
    expected_partition_counts: Mapping[str, int],
    expected_total_rows: int,
) -> None:
    """Reopen every final partition and validate schema, sorting, year, and counts."""

    partition_paths = sorted((output_root / "daily").glob("year=*/prices.parquet"))
    actual_counts: dict[str, int] = {}
    total_rows = 0
    seen_tickers: set[str] = set()
    for path in partition_paths:
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError as exc:
            raise DataValidationError(f"Invalid year partition path: {path}") from exc
        table = pq.read_table(path)
        _validate_arrow_table(
            table,
            start_date=start_date,
            end_exclusive=end_exclusive,
            allowed_tickers=tickers,
        )
        if table.num_rows:
            row_years = {
                int(value) for value in pc.unique(pc.year(table["date"])).to_pylist()
            }
            if row_years != {year}:
                raise DataValidationError(
                    f"Partition {path} contains years {sorted(row_years)}"
                )
            sorted_table = table.sort_by(
                [("date", "ascending"), ("ticker", "ascending")]
            )
            if not table.equals(sorted_table):
                raise DataValidationError(f"Partition is not sorted: {path}")
            if _arrow_has_duplicate_keys(table):
                raise DataValidationError(
                    f"Partition contains duplicate date+ticker keys: {path}"
                )
            seen_tickers.update(pc.unique(table["ticker"]).to_pylist())
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.metadata.num_row_groups):
            for column in range(parquet_file.metadata.num_columns):
                compression = (
                    parquet_file.metadata.row_group(row_group)
                    .column(column)
                    .compression
                )
                if compression.upper() != "ZSTD":
                    raise DataValidationError(
                        f"Partition {path} is not fully zstd compressed"
                    )
        actual_counts[str(year)] = table.num_rows
        total_rows += table.num_rows
    if actual_counts != dict(expected_partition_counts):
        raise DataValidationError(
            f"Partition counts mismatch: expected {expected_partition_counts}, "
            f"found {actual_counts}"
        )
    if total_rows != expected_total_rows:
        raise DataValidationError(
            f"Partition total {total_rows} does not match expected {expected_total_rows}"
        )
    if not seen_tickers.issubset(set(tickers)):
        raise DataValidationError("Final partitions contain non-Universe tickers")


def _remove_owned_temp_output(path: Path, output_root: Path) -> None:
    expected_parent = output_root.parent.resolve()
    if path.parent.resolve() != expected_parent:
        raise DataValidationError(f"Refusing to clean unexpected temp path: {path}")
    expected_prefix = f".{output_root.name}.tmp-"
    if not path.name.startswith(expected_prefix):
        raise DataValidationError(f"Refusing to clean unexpected temp path: {path}")
    shutil.rmtree(path)


def run_backfill(
    *,
    universe_path: Path = DEFAULT_UNIVERSE,
    start_date: date = DEFAULT_START,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    staging_root: Path = DEFAULT_STAGING_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT,
    resume: bool = True,
    allow_no_data: bool = False,
    now: datetime | None = None,
    download_func: DownloadFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> dict[str, Any]:
    """Run a resumable full-Universe backfill and safely publish the dataset."""

    if output_root.exists():
        raise PriceBackfillError(
            f"Final output already exists; refusing to overwrite: {output_root}"
        )
    if start_date < DEFAULT_START:
        LOGGER.info("Using requested start date before the default: %s", start_date)
    end_exclusive = calculate_end_exclusive(now)
    if start_date >= end_exclusive:
        raise ValueError(f"start date {start_date} must be before end {end_exclusive}")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    tickers = load_universe(universe_path)
    batches = build_batches(tickers, batch_size)
    content_hash = universe_sha256(tickers)
    run_key = build_run_key(tickers, start_date, end_exclusive)
    staging_dir = staging_root / run_key
    state = load_or_create_run_state(
        staging_dir,
        run_key=run_key,
        universe_path=universe_path,
        universe_hash=content_hash,
        tickers=tickers,
        start_date=start_date,
        end_exclusive=end_exclusive,
        batch_size=batch_size,
        resume=resume,
    )

    LOGGER.info(
        "Starting price backfill: %d tickers in %d batches, %s through %s exclusive, staging=%s",
        len(tickers),
        len(batches),
        start_date,
        end_exclusive,
        staging_dir,
    )
    executions: list[BatchExecution] = []
    for batch_index, batch in enumerate(batches):
        execution = _process_logical_batch(
            batch_index,
            batch,
            state=state,
            staging_dir=staging_dir,
            start_date=start_date,
            end_exclusive=end_exclusive,
            max_retries=max_retries,
            pause_seconds=pause_seconds,
            timeout=timeout,
            download_func=download_func,
            sleep_func=sleep_func,
        )
        executions.append(execution)
        if batch_index + 1 < len(batches) and pause_seconds:
            sleep_func(pause_seconds)

    batch_paths = tuple(
        staging_dir / f"batch_{index:04d}.parquet" for index in range(len(batches))
    )
    table, final_duplicates_removed = combine_staging_batches(
        batch_paths,
        start_date=start_date,
        end_exclusive=end_exclusive,
        tickers=tickers,
    )
    coverage = _coverage_rows(tickers, table, state)
    failures = _failure_rows(coverage, state)
    status_counts = {
        status: sum(row["status"] == status for row in coverage)
        for status in ("success", "no_data", "failed")
    }
    incomplete_manifest = _manifest(
        universe_path=universe_path,
        universe_hash=content_hash,
        tickers=tickers,
        start_date=start_date,
        end_exclusive=end_exclusive,
        table=table,
        coverage=coverage,
        batch_size=batch_size,
        state=state,
        partition_row_counts={},
        final_duplicate_rows_removed=final_duplicates_removed,
        completed=False,
    )
    _write_csv_atomically(
        staging_dir / "ticker_coverage.csv",
        fieldnames=COVERAGE_COLUMNS,
        rows=coverage,
    )
    _write_csv_atomically(
        staging_dir / "download_failures.csv",
        fieldnames=FAILURE_COLUMNS,
        rows=failures,
    )
    _write_json_atomically(staging_dir / "manifest.json", incomplete_manifest)

    if status_counts["failed"]:
        raise BackfillIncompleteError(
            f"Backfill has {status_counts['failed']} failed tickers; "
            f"reports and resumable data remain at {staging_dir}"
        )
    if status_counts["no_data"] and not allow_no_data:
        raise BackfillIncompleteError(
            f"Backfill has {status_counts['no_data']} no_data tickers; "
            f"rerun with --allow-no-data to publish, or inspect {staging_dir}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    try:
        partition_counts = write_year_partitions(table, temp_output)
        completed_manifest = _manifest(
            universe_path=universe_path,
            universe_hash=content_hash,
            tickers=tickers,
            start_date=start_date,
            end_exclusive=end_exclusive,
            table=table,
            coverage=coverage,
            batch_size=batch_size,
            state=state,
            partition_row_counts=partition_counts,
            final_duplicate_rows_removed=final_duplicates_removed,
            completed=True,
        )
        _write_csv_atomically(
            temp_output / "ticker_coverage.csv",
            fieldnames=COVERAGE_COLUMNS,
            rows=coverage,
        )
        _write_csv_atomically(
            temp_output / "download_failures.csv",
            fieldnames=FAILURE_COLUMNS,
            rows=failures,
        )
        _write_json_atomically(temp_output / "manifest.json", completed_manifest)
        validate_complete_dataset(
            temp_output,
            start_date=start_date,
            end_exclusive=end_exclusive,
            tickers=tickers,
            expected_partition_counts=partition_counts,
            expected_total_rows=table.num_rows,
        )
        if sum(partition_counts.values()) != completed_manifest["total_row_count"]:
            raise DataValidationError(
                "Manifest total row count does not equal partition row counts"
            )
        if output_root.exists():
            raise PriceBackfillError(
                f"Final output appeared during build; refusing to overwrite: {output_root}"
            )
        os.replace(temp_output, output_root)
        temp_output = Path()
    finally:
        if temp_output != Path() and temp_output.exists():
            _remove_owned_temp_output(temp_output, output_root)

    LOGGER.info(
        "Published %d rows for %d successful tickers to %s",
        table.num_rows,
        status_counts["success"],
        output_root,
    )
    return completed_manifest


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an ISO date in YYYY-MM-DD format"
        ) from exc


def _bounded_batch_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= MAX_BATCH_SIZE:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_BATCH_SIZE}")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.prices",
        description="Backfill validated daily Yahoo prices for the static Universe.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    backfill = subparsers.add_parser(
        "backfill", help="download and publish full history"
    )
    backfill.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    backfill.add_argument("--start", type=_parse_date, default=DEFAULT_START)
    backfill.add_argument(
        "--batch-size", type=_bounded_batch_size, default=DEFAULT_BATCH_SIZE
    )
    backfill.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    backfill.add_argument(
        "--max-retries", type=_nonnegative_integer, default=DEFAULT_MAX_RETRIES
    )
    backfill.add_argument(
        "--pause-seconds", type=_nonnegative_float, default=DEFAULT_PAUSE_SECONDS
    )
    backfill.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT)
    backfill.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse validated staging batches (default: enabled)",
    )
    backfill.add_argument(
        "--allow-no-data",
        action="store_true",
        help="allow publication when fully retried tickers have no Yahoo data",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the historical price backfill."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "backfill":
        try:
            run_backfill(
                universe_path=args.universe,
                start_date=args.start,
                output_root=args.output_root,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                pause_seconds=args.pause_seconds,
                timeout=args.timeout,
                resume=args.resume,
                allow_no_data=args.allow_no_data,
            )
        except (PriceBackfillError, OSError, ValueError) as exc:
            LOGGER.error("Price backfill failed: %s", exc)
            return 1
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
