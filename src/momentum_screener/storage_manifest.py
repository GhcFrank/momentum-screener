"""Validation and hashing helpers for the release-backed price manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from momentum_screener.dataset_config import DATASET_SCHEMA_VERSION

SCHEMA_VERSION = DATASET_SCHEMA_VERSION
MANIFEST_ASSET_NAME = "prices-manifest.json"
TICKER_COVERAGE_ASSET_NAME = "prices-ticker-coverage.csv"
DOWNLOAD_FAILURES_ASSET_NAME = "prices-download-failures.csv"
UPDATE_REPORT_ASSET_NAME = "prices-update-report.json"
UPDATE_MISSING_ASSET_NAME = "prices-update-missing-tickers.csv"
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
UPDATE_MISSING_COLUMNS = (
    "ticker",
    "expected_active",
    "download_status",
    "last_error",
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

_SHA256_LENGTH = 64
_AUXILIARY_ASSETS = {
    "ticker_coverage": (TICKER_COVERAGE_ASSET_NAME, "ticker_coverage.csv"),
    "download_failures": (DOWNLOAD_FAILURES_ASSET_NAME, "download_failures.csv"),
    "update_missing_tickers": (
        UPDATE_MISSING_ASSET_NAME,
        "update_missing_tickers.csv",
    ),
    "update_report": (UPDATE_REPORT_ASSET_NAME, "update_report.json"),
}


class ManifestError(RuntimeError):
    """Raised when a storage manifest or managed asset is unsafe or invalid."""


def calculate_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a lowercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_safe_relative_path(value: object) -> PurePosixPath:
    """Return a safe POSIX relative path or raise ``ManifestError``."""

    if not isinstance(value, str) or not value.strip():
        raise ManifestError("Asset local_path must be a non-empty string")
    if "\\" in value:
        raise ManifestError(f"Asset local_path cannot contain backslashes: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ManifestError(f"Asset local_path cannot be absolute: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"Asset local_path is unsafe: {value!r}")
    return path


def resolve_local_asset_path(root: Path, relative_path: object) -> Path:
    """Map a validated manifest path beneath ``root`` without traversal."""

    safe_path = validate_safe_relative_path(relative_path)
    resolved_root = root.resolve()
    candidate = root.joinpath(*safe_path.parts)
    resolved_candidate = candidate.resolve(strict=False)
    if (
        resolved_candidate != resolved_root
        and resolved_root not in resolved_candidate.parents
    ):
        raise ManifestError(f"Asset path escapes prices root: {relative_path!r}")
    return candidate


def _validate_asset_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("Asset asset_name must be a non-empty string")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ManifestError(f"Asset name must not contain a path: {value!r}")
    return value


def _validate_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise ManifestError("Asset sha256 must be a string")
    normalized = value.casefold()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ManifestError(f"Asset sha256 is invalid: {value!r}")
    return normalized


def validate_asset_mapping(key: str, value: object) -> dict[str, str | int]:
    """Validate and normalize one manifest asset mapping."""

    if not isinstance(value, Mapping):
        raise ManifestError(f"Asset mapping {key!r} must be an object")
    asset_name = _validate_asset_name(value.get("asset_name"))
    local_path = str(validate_safe_relative_path(value.get("local_path")))
    size_bytes = value.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ManifestError(f"Asset {key!r} size_bytes must be a non-negative integer")
    sha256 = _validate_sha256(value.get("sha256"))
    if key.isdigit():
        expected_asset = f"prices-year-{key}.parquet"
        expected_path = f"daily/year={key}/prices.parquet"
        if asset_name != expected_asset or local_path != expected_path:
            raise ManifestError(
                f"Year asset {key!r} must map {expected_asset!r} to {expected_path!r}"
            )
    elif key in _AUXILIARY_ASSETS:
        expected_asset, expected_path = _AUXILIARY_ASSETS[key]
        if asset_name != expected_asset or local_path != expected_path:
            raise ManifestError(
                f"Asset {key!r} must map {expected_asset!r} to {expected_path!r}"
            )
    elif key != "asset":
        raise ManifestError(f"Unsupported manifest asset key: {key!r}")
    return {
        "asset_name": asset_name,
        "local_path": local_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def validate_manifest(
    payload: object,
    *,
    require_completed: bool = True,
    require_assets: bool = True,
) -> dict[str, Any]:
    """Validate the supported manifest schema and every managed asset."""

    if not isinstance(payload, Mapping):
        raise ManifestError("Manifest must be a JSON object")
    manifest = dict(payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema_version: {manifest.get('schema_version')!r}"
        )
    if require_completed and manifest.get("completed") is not True:
        raise ManifestError("Manifest completed must be true")

    raw_assets = manifest.get("assets")
    if raw_assets is None and not require_assets:
        manifest["assets"] = {}
        return manifest
    if not isinstance(raw_assets, Mapping) or not raw_assets:
        raise ManifestError("Manifest assets must be a non-empty object")

    normalized_assets: dict[str, dict[str, str | int]] = {}
    asset_names: set[str] = set()
    local_paths: set[str] = set()
    for raw_key, raw_value in raw_assets.items():
        key = str(raw_key)
        asset = validate_asset_mapping(key, raw_value)
        name = str(asset["asset_name"])
        local_path = str(asset["local_path"])
        if name in asset_names:
            raise ManifestError(f"Duplicate manifest asset_name: {name}")
        if local_path in local_paths:
            raise ManifestError(f"Duplicate manifest local_path: {local_path}")
        asset_names.add(name)
        local_paths.add(local_path)
        normalized_assets[key] = asset
    manifest["assets"] = normalized_assets

    if require_assets:
        if manifest.get("source") != "yahoo_finance_via_yfinance":
            raise ManifestError(
                f"Manifest source is invalid: {manifest.get('source')!r}"
            )
        manifest["universe_sha256"] = _validate_sha256(manifest.get("universe_sha256"))
        universe_ticker_count = manifest.get("universe_ticker_count")
        if (
            isinstance(universe_ticker_count, bool)
            or not isinstance(universe_ticker_count, int)
            or universe_ticker_count <= 0
        ):
            raise ManifestError(
                "Manifest universe_ticker_count must be a positive integer"
            )
        requested_start = manifest.get("requested_start")
        if not isinstance(requested_start, str):
            raise ManifestError("Manifest requested_start must be an ISO date string")
        try:
            parsed_requested_start = date.fromisoformat(requested_start)
        except ValueError as exc:
            raise ManifestError(
                f"Manifest requested_start is invalid: {requested_start!r}"
            ) from exc
        latest_session = manifest.get("latest_session")
        if not isinstance(latest_session, str):
            raise ManifestError("Manifest latest_session must be an ISO date string")
        try:
            date.fromisoformat(latest_session)
        except ValueError as exc:
            raise ManifestError(
                f"Manifest latest_session is invalid: {latest_session!r}"
            ) from exc
        actual_min = manifest.get("actual_min_date")
        actual_max = manifest.get("actual_max_date")
        if not isinstance(actual_min, str) or not isinstance(actual_max, str):
            raise ManifestError(
                "Manifest actual_min_date and actual_max_date must be ISO dates"
            )
        try:
            parsed_min = date.fromisoformat(actual_min)
            parsed_max = date.fromisoformat(actual_max)
        except ValueError as exc:
            raise ManifestError("Manifest actual date bounds are invalid") from exc
        if (
            parsed_min < parsed_requested_start
            or parsed_min > parsed_max
            or actual_max != latest_session
        ):
            raise ManifestError(
                "Manifest date bounds must not precede requested_start, must be "
                "ordered, and latest_session must equal actual_max_date"
            )
        last_update = manifest.get("last_successful_update_utc")
        if not isinstance(last_update, str):
            raise ManifestError(
                "Manifest last_successful_update_utc must be an ISO timestamp"
            )
        try:
            parsed_update = datetime.fromisoformat(last_update)
        except ValueError as exc:
            raise ManifestError(
                f"Manifest last_successful_update_utc is invalid: {last_update!r}"
            ) from exc
        if parsed_update.utcoffset() is None:
            raise ManifestError(
                "Manifest last_successful_update_utc must include a UTC offset"
            )
        total_rows = manifest.get("total_row_count")
        if (
            isinstance(total_rows, bool)
            or not isinstance(total_rows, int)
            or total_rows < 0
        ):
            raise ManifestError(
                "Manifest total_row_count must be a non-negative integer"
            )
        raw_counts = manifest.get("partition_row_counts")
        if not isinstance(raw_counts, Mapping) or not raw_counts:
            raise ManifestError(
                "Manifest partition_row_counts must be a non-empty object"
            )
        counts: dict[str, int] = {}
        for raw_year, raw_count in raw_counts.items():
            year = str(raw_year)
            if (
                not year.isdigit()
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                raise ManifestError(
                    f"Manifest partition row count is invalid for {year!r}"
                )
            counts[year] = raw_count
        year_assets = {key for key in normalized_assets if key.isdigit()}
        if year_assets != set(counts):
            raise ManifestError(
                "Manifest year assets do not match partition_row_counts"
            )
        if sum(counts.values()) != total_rows:
            raise ManifestError(
                "Manifest total_row_count does not equal partition row counts"
            )
        if "ticker_coverage" not in normalized_assets:
            raise ManifestError("Manifest must manage ticker_coverage")
        if any(int(year) < parsed_requested_start.year for year in counts):
            raise ManifestError("Manifest cannot manage a year before requested_start")
        manifest["partition_row_counts"] = counts
    return manifest


def load_manifest(
    path: Path,
    *,
    require_completed: bool = True,
    require_assets: bool = True,
) -> dict[str, Any]:
    """Load and validate a JSON manifest from disk."""

    if not path.is_file():
        raise ManifestError(f"Manifest file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Unable to parse manifest {path}: {exc}") from exc
    return validate_manifest(
        payload,
        require_completed=require_completed,
        require_assets=require_assets,
    )


def build_asset_record(
    path: Path,
    *,
    asset_name: str,
    local_path: str,
) -> dict[str, str | int]:
    """Build a validated manifest mapping from an existing local file."""

    if not path.is_file():
        raise ManifestError(f"Managed asset file does not exist: {path}")
    record = {
        "asset_name": asset_name,
        "local_path": local_path,
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }
    return validate_asset_mapping("asset", record)


def validate_asset_size_and_hash(path: Path, asset: Mapping[str, object]) -> None:
    """Verify an asset against its manifest size and digest."""

    if not path.is_file():
        raise ManifestError(f"Managed asset is missing: {path}")
    expected_size = asset.get("size_bytes")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ManifestError(
            f"Asset size mismatch for {path}: expected {expected_size}, found {actual_size}"
        )
    expected_hash = asset.get("sha256")
    actual_hash = calculate_sha256(path)
    if actual_hash != expected_hash:
        raise ManifestError(
            f"Asset SHA-256 mismatch for {path}: expected {expected_hash}, found {actual_hash}"
        )


def _validate_parquet(path: Path, year: int) -> None:
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise ManifestError(f"Unable to read Parquet asset {path}: {exc}") from exc
    if not table.schema.equals(PRICE_SCHEMA, check_metadata=False):
        raise ManifestError(f"Parquet schema mismatch for {path}: {table.schema}")
    if table.num_rows:
        if any(column.null_count for column in table.columns):
            raise ManifestError(f"Parquet asset contains null values: {path}")
        if bool(pc.any(pc.invert(pc.is_finite(table["close"]))).as_py()):
            raise ManifestError(f"Parquet asset contains non-finite close: {path}")
        if bool(pc.any(pc.invert(pc.is_finite(table["adj_close"]))).as_py()):
            raise ManifestError(f"Parquet asset contains non-finite adj_close: {path}")
        if bool(pc.any(pc.less_equal(table["close"], 0)).as_py()):
            raise ManifestError(f"Parquet asset contains non-positive close: {path}")
        if bool(pc.any(pc.less_equal(table["adj_close"], 0)).as_py()):
            raise ManifestError(
                f"Parquet asset contains non-positive adj_close: {path}"
            )
        if bool(pc.any(pc.less(table["volume"], 0)).as_py()):
            raise ManifestError(f"Parquet asset contains negative volume: {path}")
        years = {int(value) for value in pc.unique(pc.year(table["date"])).to_pylist()}
        if years != {year}:
            raise ManifestError(
                f"Parquet asset {path} contains years {sorted(years)}, expected {year}"
            )
        sorted_table = table.sort_by([("date", "ascending"), ("ticker", "ascending")])
        if not table.equals(sorted_table):
            raise ManifestError(f"Parquet asset is not sorted: {path}")
        dates = table["date"].to_pylist()
        tickers = table["ticker"].to_pylist()
        if any(
            dates[index] == dates[index - 1] and tickers[index] == tickers[index - 1]
            for index in range(1, table.num_rows)
        ):
            raise ManifestError(f"Parquet asset contains duplicate keys: {path}")
    metadata = pq.ParquetFile(path).metadata
    for row_group in range(metadata.num_row_groups):
        for column in range(metadata.num_columns):
            if (
                metadata.row_group(row_group).column(column).compression.upper()
                != "ZSTD"
            ):
                raise ManifestError(
                    f"Parquet asset is not fully zstd compressed: {path}"
                )


def _validate_csv_header(path: Path, expected: Sequence[str]) -> None:
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file)
            header = tuple(next(reader, ()))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"Unable to read CSV asset {path}: {exc}") from exc
    if header != tuple(expected):
        raise ManifestError(
            f"CSV header mismatch for {path}: expected {tuple(expected)}, found {header}"
        )


def validate_managed_asset(
    path: Path,
    *,
    key: str,
    asset: Mapping[str, object],
) -> None:
    """Validate size/hash plus the format implied by a manifest asset key."""

    validate_asset_size_and_hash(path, asset)
    if key.isdigit():
        _validate_parquet(path, int(key))
    elif key == "ticker_coverage":
        _validate_csv_header(path, COVERAGE_COLUMNS)
    elif key == "download_failures":
        _validate_csv_header(path, FAILURE_COLUMNS)
    elif key == "update_missing_tickers":
        _validate_csv_header(path, UPDATE_MISSING_COLUMNS)
    elif key == "update_report":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Unable to parse update report {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ManifestError(f"Update report must be a JSON object: {path}")


def write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    """Write validated JSON via fsync and same-directory atomic replacement."""

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
        json.loads(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def remove_owned_tree(path: Path, *, parent: Path, prefix: str) -> None:
    """Remove one validated tool-owned tree without following symlinks."""

    resolved_parent = parent.resolve()
    if path.parent.resolve() != resolved_parent or not path.name.startswith(prefix):
        raise ManifestError(f"Refusing to remove unowned temporary directory: {path}")
    if not path.exists():
        return
    for child in sorted(
        path.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def replace_files_transactionally(
    root: Path,
    staging_root: Path,
    relative_paths: Sequence[str],
    *,
    backup_root: Path,
    validate_after: Callable[[], None] | None = None,
) -> None:
    """Replace staged files in order and restore every prior file on failure."""

    normalized = [str(validate_safe_relative_path(value)) for value in relative_paths]
    if len(set(normalized)) != len(normalized):
        raise ManifestError("Transactional replacement contains duplicate paths")
    root.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=False)
    installed: list[tuple[Path, Path | None]] = []
    try:
        for relative_path in normalized:
            source = resolve_local_asset_path(staging_root, relative_path)
            destination = resolve_local_asset_path(root, relative_path)
            if not source.is_file():
                raise ManifestError(f"Staged asset is missing: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if destination.exists():
                if not destination.is_file():
                    raise ManifestError(
                        f"Asset destination is not a file: {destination}"
                    )
                backup = resolve_local_asset_path(backup_root, relative_path)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            try:
                os.replace(source, destination)
            except Exception:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            installed.append((destination, backup))
        if validate_after is not None:
            validate_after()
    except Exception:
        for destination, backup in reversed(installed):
            destination.unlink(missing_ok=True)
            if backup is not None and backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise
