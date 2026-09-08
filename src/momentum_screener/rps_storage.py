"""Validated Parquet persistence and vectorized backfill for RPS history."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from momentum_screener.dataset_config import DEFAULT_BACKFILL_START
from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PRICES_ROOT,
)
from momentum_screener.prices import (
    DEFAULT_UNIVERSE,
    load_universe,
    universe_sha256,
)
from momentum_screener.rps import (
    INVALID_RPS,
    RPS_CALENDAR_NAME,
    RPS_LOOKBACKS,
    RPS_PRICE_FIELD,
    _normalize_lookbacks,
    calculate_rps_snapshot,
)
from momentum_screener.storage_manifest import (
    PRICE_SCHEMA,
    ManifestError,
    calculate_sha256,
    remove_owned_tree,
    replace_files_transactionally,
    resolve_local_asset_path,
    validate_asset_size_and_hash,
    write_json_atomically,
)
from momentum_screener.storage_manifest import (
    load_manifest as load_price_manifest,
)
from momentum_screener.universe import normalize_ticker

LOGGER = logging.getLogger(__name__)

RPS_SCHEMA_VERSION: Final[str] = "rps_v2"
LEGACY_RPS_LOOKBACKS: Final[tuple[int, ...]] = (50, 120, 250)
DEFAULT_RPS_ROOT: Final[Path] = Path("data/processed/rps")
RPS_MANIFEST_NAME: Final[str] = "manifest.json"


def rps_columns(lookbacks: Sequence[int] = RPS_LOOKBACKS) -> tuple[str, ...]:
    horizons = _normalize_lookbacks(lookbacks)
    return (
        "date",
        "ticker",
        *(f"rps{n}" for n in horizons),
        *(f"return_{n}" for n in horizons),
        *(f"rps{n}_base_date" for n in horizons),
    )


def rps_schema(lookbacks: Sequence[int] = RPS_LOOKBACKS) -> pa.Schema:
    horizons = _normalize_lookbacks(lookbacks)
    return pa.schema(
        [
            pa.field("date", pa.date32(), nullable=False),
            pa.field("ticker", pa.string(), nullable=False),
            *(pa.field(f"rps{n}", pa.float64(), nullable=False) for n in horizons),
            *(pa.field(f"return_{n}", pa.float64(), nullable=True) for n in horizons),
            *(
                pa.field(f"rps{n}_base_date", pa.date32(), nullable=False)
                for n in horizons
            ),
        ]
    )


RPS_DATA_COLUMNS: Final[tuple[str, ...]] = rps_columns()
RPS_SCHEMA: Final[pa.Schema] = rps_schema()


class RpsStorageError(RuntimeError):
    """Raised when the RPS dataset cannot be read or changed safely."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _partition_path(root: Path, year: int) -> Path:
    return root / "daily" / f"year={year}" / "rps.parquet"


def _asset_name(year: int) -> str:
    return f"rps-year-{year}.parquet"


def _normalize_date_series(values: pd.Series, *, name: str) -> pd.Series:
    normalized = pd.to_datetime(values, errors="coerce")
    if bool(normalized.isna().any()):
        raise RpsStorageError(f"RPS {name} contains invalid dates")
    if normalized.dt.tz is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.dt.date


def normalize_rps_rows(
    rows: pd.DataFrame, *, lookbacks: Sequence[int] = RPS_LOOKBACKS
) -> pd.DataFrame:
    """Normalize either an on-demand snapshot or persisted-shape RPS rows."""

    columns = rps_columns(lookbacks)
    source = rows.reset_index(drop=True)
    date_column = "as_of_date" if "as_of_date" in source else "date"
    required = {date_column, *columns[1:]}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RpsStorageError(f"RPS rows are missing columns: {missing}")
    result = source.loc[:, [date_column, *columns[1:]]].copy()
    result = result.rename(columns={date_column: "date"})
    result["date"] = _normalize_date_series(result["date"], name="date")
    result["ticker"] = result["ticker"].astype("string")
    if bool(result["ticker"].isna().any()) or bool(result["ticker"].eq("").any()):
        raise RpsStorageError("RPS ticker must be a non-empty string")
    for column in (f"rps{n}" for n in lookbacks):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
        valid = result[column].eq(INVALID_RPS) | result[column].between(0, 100)
        if not bool(
            (result[column].notna() & np.isfinite(result[column]) & valid).all()
        ):
            raise RpsStorageError(
                f"{column} must be finite INVALID_RPS or within 0..100"
            )
    for column in (f"return_{n}" for n in lookbacks):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
        if not bool((result[column].isna() | np.isfinite(result[column])).all()):
            raise RpsStorageError(f"{column} must be finite or null")
    for column in (f"rps{n}_base_date" for n in lookbacks):
        result[column] = _normalize_date_series(result[column], name=column)
    result = result.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)
    if bool(result.duplicated(["date", "ticker"]).any()):
        raise RpsStorageError("RPS rows contain duplicate date/ticker keys")
    return result.loc[:, columns]


def _write_rps_parquet(path: Path, rows: pd.DataFrame) -> None:
    normalized = normalize_rps_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(normalized, schema=RPS_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    validate_rps_partition(path, expected_year=int(normalized.iloc[0]["date"].year))


def validate_rps_partition(
    path: Path, *, expected_year: int, lookbacks: Sequence[int] = RPS_LOOKBACKS
) -> int:
    """Validate schema, values, ordering, key uniqueness, and partition year."""

    if not path.is_file():
        raise RpsStorageError(f"RPS partition does not exist: {path}")
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise RpsStorageError(f"Unable to read RPS partition {path}: {exc}") from exc
    if not table.schema.equals(rps_schema(lookbacks), check_metadata=False):
        raise RpsStorageError(f"RPS partition has unexpected schema: {path}")
    if table.num_rows == 0:
        raise RpsStorageError(f"RPS partition cannot be empty: {path}")
    years = {int(value) for value in pc.unique(pc.year(table["date"])).to_pylist()}
    if years != {expected_year}:
        raise RpsStorageError(f"RPS partition {path} contains years {sorted(years)}")
    normalized = normalize_rps_rows(table.to_pandas(), lookbacks=lookbacks)
    original = table.to_pandas().loc[:, rps_columns(lookbacks)]
    original["ticker"] = original["ticker"].astype("string")
    for column in ("date", *(f"rps{n}_base_date" for n in lookbacks)):
        original[column] = original[column].map(
            lambda value: value.date() if isinstance(value, pd.Timestamp) else value
        )
    if not original.reset_index(drop=True).equals(normalized):
        raise RpsStorageError(f"RPS partition is not deterministically sorted: {path}")
    return len(normalized)


def _asset_record(path: Path, year: int) -> dict[str, str | int]:
    return {
        "asset_name": _asset_name(year),
        "local_path": f"daily/year={year}/rps.parquet",
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }


def validate_rps_manifest(
    payload: object, *, allow_legacy: bool = False
) -> dict[str, Any]:
    """Validate the identity and partition index for the RPS dataset."""

    if not isinstance(payload, Mapping):
        raise RpsStorageError("RPS manifest must be a JSON object")
    manifest = dict(payload)
    legacy = manifest.get("schema_version") == "rps_v1"
    if legacy and not allow_legacy:
        raise RpsStorageError(
            "RPS migration required: rps_v1 lacks RPS20. Pull with --allow-legacy, "
            "run python -m momentum_screener.rps_storage migrate, then publish "
            "with --allow-migration. Daily jobs never migrate history."
        )
    expected = {
        "schema_version": "rps_v1" if legacy else RPS_SCHEMA_VERSION,
        "lookbacks": list(LEGACY_RPS_LOOKBACKS if legacy else RPS_LOOKBACKS),
        "price_field": RPS_PRICE_FIELD,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RpsStorageError(
                f"RPS manifest {key} must be {expected_value!r}, "
                f"found {manifest.get(key)!r}"
            )
    if manifest.get("completed") is not True:
        raise RpsStorageError("RPS manifest completed must be true")
    universe_hash = manifest.get("universe_sha256")
    if (
        not isinstance(universe_hash, str)
        or len(universe_hash) != 64
        or any(character not in "0123456789abcdef" for character in universe_hash)
    ):
        raise RpsStorageError("RPS manifest universe_sha256 is invalid")
    ticker_count = manifest.get("universe_ticker_count")
    if (
        isinstance(ticker_count, bool)
        or not isinstance(ticker_count, int)
        or ticker_count <= 0
    ):
        raise RpsStorageError("RPS manifest universe_ticker_count is invalid")
    for key in ("actual_min_date", "latest_session"):
        date_value = manifest.get(key)
        if not isinstance(date_value, str):
            raise RpsStorageError(f"RPS manifest {key} must be an ISO date")
        try:
            date.fromisoformat(date_value)
        except ValueError as exc:
            raise RpsStorageError(f"RPS manifest {key} is invalid") from exc
    if manifest["actual_min_date"] > manifest["latest_session"]:
        raise RpsStorageError("RPS manifest date range is reversed")
    counts = manifest.get("partition_row_counts")
    assets = manifest.get("assets")
    if not isinstance(counts, Mapping) or not counts:
        raise RpsStorageError("RPS manifest partition_row_counts must be non-empty")
    if not isinstance(assets, Mapping) or set(assets) != set(counts):
        raise RpsStorageError("RPS manifest assets must match partition years")
    normalized_counts: dict[str, int] = {}
    normalized_assets: dict[str, dict[str, str | int]] = {}
    for raw_year, raw_count in counts.items():
        year = str(raw_year)
        if not (year.isdigit() and len(year) == 4):
            raise RpsStorageError(f"RPS manifest partition year is invalid: {year}")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise RpsStorageError(f"RPS manifest row count is invalid for {year}")
        raw_asset = assets.get(raw_year)
        if not isinstance(raw_asset, Mapping):
            raise RpsStorageError(f"RPS manifest asset is invalid for {year}")
        expected_asset = {
            "asset_name": _asset_name(int(year)),
            "local_path": f"daily/year={year}/rps.parquet",
        }
        for key, value in expected_asset.items():
            if raw_asset.get(key) != value:
                raise RpsStorageError(f"RPS manifest asset {year} has invalid {key}")
        size = raw_asset.get("size_bytes")
        digest = raw_asset.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RpsStorageError(f"RPS manifest asset {year} has invalid size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RpsStorageError(f"RPS manifest asset {year} has invalid sha256")
        normalized_counts[year] = raw_count
        normalized_assets[year] = {
            **expected_asset,
            "size_bytes": size,
            "sha256": digest,
        }
    total = manifest.get("total_row_count")
    if total != sum(normalized_counts.values()):
        raise RpsStorageError("RPS manifest total_row_count is inconsistent")
    manifest["partition_row_counts"] = normalized_counts
    manifest["assets"] = normalized_assets
    return manifest


def load_rps_manifest(
    root: Path = DEFAULT_RPS_ROOT, *, allow_legacy: bool = False
) -> dict[str, Any]:
    """Load and validate the local RPS manifest."""

    path = root / RPS_MANIFEST_NAME
    if not path.is_file():
        raise RpsStorageError(f"RPS manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RpsStorageError(f"Unable to parse RPS manifest {path}: {exc}") from exc
    return validate_rps_manifest(payload, allow_legacy=allow_legacy)


def validate_rps_dataset(
    root: Path = DEFAULT_RPS_ROOT,
    *,
    universe_path: Path = DEFAULT_UNIVERSE,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Fully validate every local RPS partition against its manifest."""

    manifest = load_rps_manifest(root, allow_legacy=allow_legacy)
    universe = load_universe(universe_path)
    if manifest["universe_sha256"] != universe_sha256(universe):
        raise RpsStorageError("RPS manifest Universe hash does not match")
    if manifest["universe_ticker_count"] != len(universe):
        raise RpsStorageError("RPS manifest Universe ticker count does not match")
    total = 0
    observed_min: date | None = None
    observed_max: date | None = None
    expected_tickers = set(universe)
    for year, expected_count in manifest["partition_row_counts"].items():
        path = _partition_path(root, int(year))
        asset = manifest["assets"][year]
        if path.stat().st_size != asset["size_bytes"]:
            raise RpsStorageError(f"RPS asset size mismatch for {year}")
        if calculate_sha256(path) != asset["sha256"]:
            raise RpsStorageError(f"RPS asset hash mismatch for {year}")
        count = validate_rps_partition(
            path, expected_year=int(year), lookbacks=manifest["lookbacks"]
        )
        if count != expected_count:
            raise RpsStorageError(f"RPS partition row count mismatch for {year}")
        rows = pq.read_table(path, columns=["date", "ticker"]).to_pandas()
        for _, daily in rows.groupby("date", sort=False):
            if len(daily) != len(universe) or set(daily["ticker"]) != expected_tickers:
                raise RpsStorageError(
                    f"RPS session {daily.iloc[0]['date']} is not a complete Universe"
                )
        part_min = min(rows["date"])
        part_max = max(rows["date"])
        observed_min = part_min if observed_min is None else min(observed_min, part_min)
        observed_max = part_max if observed_max is None else max(observed_max, part_max)
        total += count
    if total != manifest["total_row_count"]:
        raise RpsStorageError("RPS validated total differs from manifest")
    if observed_min is None or observed_max is None:
        raise RpsStorageError("RPS dataset contains no rows")
    if observed_min.isoformat() != manifest["actual_min_date"]:
        raise RpsStorageError("RPS actual_min_date differs from partitions")
    if observed_max.isoformat() != manifest["latest_session"]:
        raise RpsStorageError("RPS latest_session differs from partitions")
    return manifest


def _manifest_for_partitions(
    *,
    universe: tuple[str, ...],
    partitions: Mapping[int, tuple[Path, int]],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    years = sorted(partitions)
    dates_by_year = {
        year: pq.read_table(path, columns=["date"])["date"].to_pylist()
        for year, (path, _) in partitions.items()
    }
    actual_min = min(min(values) for values in dates_by_year.values())
    latest = max(max(values) for values in dates_by_year.values())
    now = _utc_now_iso()
    return {
        "schema_version": RPS_SCHEMA_VERSION,
        "completed": True,
        "created_at_utc": (
            previous.get("created_at_utc", now) if previous is not None else now
        ),
        "updated_at_utc": now,
        "actual_min_date": actual_min.isoformat(),
        "latest_session": latest.isoformat(),
        "universe_sha256": universe_sha256(universe),
        "universe_ticker_count": len(universe),
        "lookbacks": list(RPS_LOOKBACKS),
        "price_field": RPS_PRICE_FIELD,
        "partition_row_counts": {str(year): partitions[year][1] for year in years},
        "total_row_count": sum(partitions[year][1] for year in years),
        "assets": {
            str(year): _asset_record(partitions[year][0], year) for year in years
        },
    }


def persist_rps_snapshot(
    snapshot: pd.DataFrame,
    *,
    root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> dict[str, Any]:
    """Idempotently upsert one complete-Universe RPS snapshot."""

    rows = normalize_rps_rows(snapshot)
    dates = rows["date"].unique().tolist()
    if len(dates) != 1:
        raise RpsStorageError("Daily RPS persistence requires exactly one session")
    session = dates[0]
    universe = load_universe(universe_path)
    if len(rows) != len(universe) or set(rows["ticker"]) != set(universe):
        raise RpsStorageError(
            "RPS snapshot must contain the complete Universe exactly once"
        )

    previous: dict[str, Any] | None = None
    existing_partitions: dict[int, tuple[Path, int]] = {}
    manifest_path = root / RPS_MANIFEST_NAME
    if manifest_path.exists():
        previous = load_rps_manifest(root)
        if previous["universe_sha256"] != universe_sha256(universe):
            raise RpsStorageError("Existing RPS dataset uses a different Universe")
        for year, count in previous["partition_row_counts"].items():
            existing_partitions[int(year)] = (_partition_path(root, int(year)), count)
    elif root.exists() and any(root.iterdir()):
        raise RpsStorageError("Non-empty RPS root has no valid manifest")

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rps-staging-", dir=root.parent) as temp:
        staging = Path(temp)
        target = _partition_path(staging, session.year)
        existing_path = _partition_path(root, session.year)
        if existing_path.is_file():
            existing = pq.read_table(existing_path).to_pandas()
            existing["date"] = existing["date"].map(
                lambda value: value.date() if isinstance(value, pd.Timestamp) else value
            )
            existing = existing.loc[~existing["date"].eq(session)]
            combined = pd.concat([existing, rows], ignore_index=True)
        else:
            combined = rows
        _write_rps_parquet(target, combined)
        existing_partitions[session.year] = (target, len(combined))
        manifest = _manifest_for_partitions(
            universe=universe,
            partitions=existing_partitions,
            previous=previous,
        )
        validate_rps_manifest(manifest)
        write_json_atomically(staging / RPS_MANIFEST_NAME, manifest)

        backup = root.parent / f".rps-backup-{uuid.uuid4().hex}"

        replace_files_transactionally(
            root,
            staging,
            [f"daily/year={session.year}/rps.parquet", RPS_MANIFEST_NAME],
            backup_root=backup,
            validate_after=lambda: _validate_changed_partition(
                root, manifest, session.year
            ),
        )
        remove_owned_tree(backup, parent=root.parent, prefix=".rps-backup-")
    return {
        "schema_version": RPS_SCHEMA_VERSION,
        "session": session.isoformat(),
        "rows_persisted": len(rows),
        "partition_row_count": manifest["partition_row_counts"][str(session.year)],
        "total_row_count": manifest["total_row_count"],
        "latest_session": manifest["latest_session"],
        "success": True,
    }


def _validate_changed_partition(
    root: Path, manifest: Mapping[str, Any], year: int
) -> None:
    stored = load_rps_manifest(root)
    if stored != manifest:
        raise RpsStorageError("Stored RPS manifest differs after replacement")
    path = _partition_path(root, year)
    count = validate_rps_partition(path, expected_year=year)
    if count != manifest["partition_row_counts"][str(year)]:
        raise RpsStorageError("Changed RPS partition row count differs from manifest")
    asset = manifest["assets"][str(year)]
    if (
        path.stat().st_size != asset["size_bytes"]
        or calculate_sha256(path) != asset["sha256"]
    ):
        raise RpsStorageError("Changed RPS partition asset metadata differs")


def read_rps_history(
    *,
    tickers: Sequence[str] | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    root: Path = DEFAULT_RPS_ROOT,
    allow_legacy: bool = False,
) -> pd.DataFrame:
    """Read RPS history without exposing its physical yearly layout."""

    manifest = load_rps_manifest(root, allow_legacy=allow_legacy)
    start = _coerce_optional_date(start_date)
    end = _coerce_optional_date(end_date)
    if start is not None and end is not None and start > end:
        raise RpsStorageError("start_date cannot be after end_date")
    selected_tickers = tuple(tickers) if tickers is not None else None
    years = [
        int(year)
        for year in manifest["partition_row_counts"]
        if (start is None or int(year) >= start.year)
        and (end is None or int(year) <= end.year)
    ]
    frames: list[pd.DataFrame] = []
    for year in sorted(years):
        path = _partition_path(root, year)
        validate_rps_partition(
            path, expected_year=year, lookbacks=manifest["lookbacks"]
        )
        frame = pq.read_table(path).to_pandas()
        frame["date"] = frame["date"].map(
            lambda value: value.date() if isinstance(value, pd.Timestamp) else value
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=rps_columns(manifest["lookbacks"]))
    result = pd.concat(frames, ignore_index=True)
    if start is not None:
        result = result.loc[result["date"].ge(start)]
    if end is not None:
        result = result.loc[result["date"].le(end)]
    if selected_tickers is not None:
        result = result.loc[result["ticker"].isin(selected_tickers)]
    return normalize_rps_rows(result, lookbacks=manifest["lookbacks"]).reset_index(
        drop=True
    )


def read_rps_snapshot(
    as_of_date: date | str,
    *,
    root: Path = DEFAULT_RPS_ROOT,
) -> pd.DataFrame:
    """Read one persisted all-market RPS snapshot."""

    requested = _coerce_date(as_of_date)
    return read_rps_history(start_date=requested, end_date=requested, root=root)


def read_stock_rps_history(
    ticker: str,
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    root: Path = DEFAULT_RPS_ROOT,
) -> pd.DataFrame:
    """Read one normalized ticker's persisted RPS history."""

    normalized = normalize_ticker(ticker)
    if normalized is None:
        raise RpsStorageError(f"Invalid ticker for RPS history: {ticker!r}")
    return read_rps_history(
        tickers=(normalized,),
        start_date=start_date,
        end_date=end_date,
        root=root,
    )


def _coerce_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise RpsStorageError("RPS date must be a date or ISO string")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise RpsStorageError("RPS date must use YYYY-MM-DD") from exc


def _coerce_optional_date(value: date | str | None) -> date | None:
    return None if value is None else _coerce_date(value)


def calculate_rps_history(
    price_rows: pd.DataFrame,
    *,
    universe: tuple[str, ...],
    start_date: date,
    end_date: date,
    lookbacks: Sequence[int] = RPS_LOOKBACKS,
) -> pd.DataFrame:
    """Vectorize historical RPS while preserving on-demand ranking semantics."""

    lookbacks = _normalize_lookbacks(lookbacks)
    required = {"date", "ticker", RPS_PRICE_FIELD}
    missing = sorted(required.difference(price_rows.columns))
    if missing:
        raise RpsStorageError(f"Historical RPS prices are missing columns: {missing}")
    if start_date > end_date:
        raise RpsStorageError("start_date cannot be after end_date")
    prices = price_rows.loc[:, ["date", "ticker", RPS_PRICE_FIELD]].copy()
    prices["date"] = _normalize_date_series(prices["date"], name="price date")
    if bool(prices.duplicated(["date", "ticker"]).any()):
        raise RpsStorageError("Historical prices contain duplicate date/ticker keys")
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(min(prices["date"])), pd.Timestamp(end_date)
    )
    session_dates = pd.Index([value.date() for value in sessions], name="date")
    panel = prices.pivot(index="date", columns="ticker", values=RPS_PRICE_FIELD)
    panel = panel.reindex(index=session_dates, columns=universe).astype("float64")
    metrics = _wide_rps_metrics(panel, lookbacks=lookbacks)
    output_dates = tuple(
        value for value in session_dates if start_date <= value <= end_date
    )
    return _history_frame_from_wide(
        output_dates,
        universe=universe,
        metrics=metrics,
        calendar=calendar,
        lookbacks=lookbacks,
    )


def _wide_rps_metrics(
    panel: pd.DataFrame, *, lookbacks: Sequence[int] = RPS_LOOKBACKS
) -> dict[str, pd.DataFrame]:
    metrics: dict[str, pd.DataFrame] = {}
    current_valid = panel.notna() & np.isfinite(panel) & panel.gt(0)
    for lookback in lookbacks:
        base = panel.shift(lookback)
        valid = current_valid & base.notna() & np.isfinite(base) & base.gt(0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            returns = panel.div(base).sub(1.0)
        valid &= np.isfinite(returns)
        returns = returns.where(valid)
        ranks = returns.rank(axis="columns", method="average", ascending=True)
        counts = valid.sum(axis="columns")
        denominator = counts.sub(1).replace(0, np.nan)
        rps = ranks.sub(1.0).div(denominator, axis="index").mul(100.0)
        rps = rps.where(valid, INVALID_RPS)
        single = counts.eq(1)
        if bool(single.any()):
            rps.loc[single] = rps.loc[single].where(~valid.loc[single], 100.0)
        metrics[f"return_{lookback}"] = returns
        metrics[f"rps{lookback}"] = rps.astype("float64")
    return metrics


def _history_frame_from_wide(
    output_dates: Sequence[date],
    *,
    universe: tuple[str, ...],
    metrics: Mapping[str, pd.DataFrame],
    calendar: Any,
    lookbacks: Sequence[int] = RPS_LOOKBACKS,
) -> pd.DataFrame:
    dates = tuple(output_dates)
    count = len(universe)
    result = pd.DataFrame(
        {
            "date": np.repeat(np.asarray(dates, dtype="object"), count),
            "ticker": np.tile(np.asarray(universe, dtype="object"), len(dates)),
        }
    )
    for lookback in lookbacks:
        result[f"rps{lookback}"] = (
            metrics[f"rps{lookback}"].loc[list(dates)].to_numpy().ravel()
        )
    for lookback in lookbacks:
        result[f"return_{lookback}"] = (
            metrics[f"return_{lookback}"].loc[list(dates)].to_numpy().ravel()
        )
    for lookback in lookbacks:
        base_dates: list[date] = []
        for session_date in dates:
            index = int(calendar.sessions.get_loc(pd.Timestamp(session_date)))
            if index < lookback:
                raise RpsStorageError(
                    f"XNYS calendar cannot resolve {lookback}-session base for {session_date}"
                )
            base_dates.append(calendar.sessions[index - lookback].date())
        result[f"rps{lookback}_base_date"] = np.repeat(
            np.asarray(base_dates, dtype="object"), count
        )
    return normalize_rps_rows(result, lookbacks=lookbacks)


def _read_slim_price_history(
    *,
    prices_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = prices_root / "daily" / f"year={year}" / "prices.parquet"
        if not path.is_file():
            raise RpsStorageError(f"Required price partition is missing: {path}")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(PRICE_SCHEMA, check_metadata=False):
            raise RpsStorageError(f"Price partition has unexpected schema: {path}")
        frames.append(
            pq.read_table(path, columns=["date", "ticker", RPS_PRICE_FIELD]).to_pandas()
        )
    rows = pd.concat(frames, ignore_index=True)
    rows["date"] = rows["date"].map(
        lambda value: value.date() if isinstance(value, pd.Timestamp) else value
    )
    rows["ticker"] = rows["ticker"].astype("string")
    return rows


def backfill_rps_history(
    *,
    start_date: date = DEFAULT_BACKFILL_START,
    end_date: date | None = None,
    prices_root: Path = DEFAULT_PRICES_ROOT,
    root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> dict[str, Any]:
    """Build complete historical RPS from one in-memory price panel."""

    price_manifest = load_price_manifest(prices_root / "manifest.json")
    target_end = (
        date.fromisoformat(str(price_manifest["latest_session"]))
        if end_date is None
        else end_date
    )
    if start_date > target_end:
        raise RpsStorageError("Backfill start_date cannot be after end_date")
    universe = load_universe(universe_path)
    price_rows = _read_slim_price_history(
        prices_root=prices_root,
        start_year=int(str(price_manifest["actual_min_date"])[:4]),
        end_year=target_end.year,
    )
    history = calculate_rps_history(
        price_rows,
        universe=universe,
        start_date=start_date,
        end_date=target_end,
    )
    if history.empty:
        raise RpsStorageError("Historical RPS backfill produced no rows")

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rps-backfill-", dir=root.parent) as temp:
        staging = Path(temp)
        partitions: dict[int, tuple[Path, int]] = {}
        for year, rows in history.groupby(
            history["date"].map(lambda value: value.year)
        ):
            path = _partition_path(staging, int(year))
            _write_rps_parquet(path, rows)
            partitions[int(year)] = (path, len(rows))
        previous = (
            load_rps_manifest(root) if (root / RPS_MANIFEST_NAME).is_file() else None
        )
        manifest = _manifest_for_partitions(
            universe=universe,
            partitions=partitions,
            previous=previous,
        )
        validate_rps_manifest(manifest)
        write_json_atomically(staging / RPS_MANIFEST_NAME, manifest)
        relative_paths = [
            *(f"daily/year={year}/rps.parquet" for year in sorted(partitions)),
            RPS_MANIFEST_NAME,
        ]
        backup = root.parent / f".rps-backup-{uuid.uuid4().hex}"

        def validate_replacement() -> None:
            validate_rps_dataset(root, universe_path=universe_path)

        replace_files_transactionally(
            root,
            staging,
            relative_paths,
            backup_root=backup,
            validate_after=validate_replacement,
        )
        remove_owned_tree(backup, parent=root.parent, prefix=".rps-backup-")
    return {
        "schema_version": RPS_SCHEMA_VERSION,
        "start_date": manifest["actual_min_date"],
        "latest_session": manifest["latest_session"],
        "rows_persisted": manifest["total_row_count"],
        "partition_years": sorted(
            int(year) for year in manifest["partition_row_counts"]
        ),
        "success": True,
    }


def rps_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True).encode()
    ).hexdigest()


def migrate_rps_history(
    *,
    prices_root: Path = DEFAULT_PRICES_ROOT,
    root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> dict[str, Any]:
    """One-time local v1 migration: preserve old metrics and compute missing ones.

    Keep the original manifest/partitions in an archive returned to the caller.
    Re-running on v2 validates and does no writes. No network or publication.
    """

    previous = validate_rps_dataset(
        root, universe_path=universe_path, allow_legacy=True
    )
    if previous["schema_version"] == RPS_SCHEMA_VERSION:
        return {
            "success": True,
            "status": "already_current",
            "schema_version": RPS_SCHEMA_VERSION,
        }
    prices = load_price_manifest(prices_root / "manifest.json")
    universe = load_universe(universe_path)
    if prices["universe_sha256"] != universe_sha256(universe):
        raise RpsStorageError("Migration price Universe differs from RPS Universe")
    if (
        prices["latest_session"] < previous["latest_session"]
        or prices["requested_start"] > previous["actual_min_date"]
    ):
        raise RpsStorageError(
            "Migration requires complete local price history covering the old RPS dataset"
        )
    old = read_rps_history(root=root, allow_legacy=True)
    missing = tuple(n for n in RPS_LOOKBACKS if n not in previous["lookbacks"])
    # A runner may have only rolling update inputs despite a full-history manifest.
    # Require each source partition to match the committed price dataset exactly.
    for year in range(
        int(prices["actual_min_date"][:4]), int(previous["latest_session"][:4]) + 1
    ):
        asset = prices["assets"].get(str(year))
        if asset is None:
            raise RpsStorageError(f"Migration requires price asset metadata for {year}")
        try:
            validate_asset_size_and_hash(
                resolve_local_asset_path(prices_root, asset["local_path"]), asset
            )
        except ManifestError as exc:
            raise RpsStorageError(
                f"Migration requires complete committed price history: {exc}"
            ) from exc
    price_rows = _read_slim_price_history(
        prices_root=prices_root,
        start_year=int(prices["actual_min_date"][:4]),
        end_year=int(previous["latest_session"][:4]),
    )
    added = calculate_rps_history(
        price_rows,
        universe=universe,
        start_date=date.fromisoformat(previous["actual_min_date"]),
        end_date=date.fromisoformat(previous["latest_session"]),
        lookbacks=missing,
    )
    merged = old.merge(added, on=["date", "ticker"], how="left", validate="one_to_one")
    merged = normalize_rps_rows(merged)
    # This is also a guard against accidentally recalculating the old horizons.
    if not merged.loc[:, rps_columns(previous["lookbacks"])].equals(old):
        raise RpsStorageError("Migration would change existing RPS history")
    with tempfile.TemporaryDirectory(prefix=".rps-migrate-", dir=root.parent) as temp:
        staging = Path(temp)
        partitions = {}
        for year, rows in merged.groupby(merged["date"].map(lambda value: value.year)):
            path = _partition_path(staging, int(year))
            _write_rps_parquet(path, rows)
            partitions[int(year)] = (path, len(rows))
        manifest = _manifest_for_partitions(
            universe=universe, partitions=partitions, previous=previous
        )
        manifest["migration"] = {
            "from_schema": previous["schema_version"],
            "source_manifest_sha256": rps_manifest_fingerprint(previous),
            "preserved_lookbacks": previous["lookbacks"],
        }
        write_json_atomically(staging / RPS_MANIFEST_NAME, manifest)
        validate_rps_dataset(staging, universe_path=universe_path)
        archive = root.parent / f".rps-v1-archive-{uuid.uuid4().hex}"
        replace_files_transactionally(
            root,
            staging,
            [
                *(str(asset["local_path"]) for asset in manifest["assets"].values()),
                RPS_MANIFEST_NAME,
            ],
            backup_root=archive,
            validate_after=lambda: validate_rps_dataset(
                root, universe_path=universe_path
            ),
        )
    return {
        "success": True,
        "status": "migrated",
        "schema_version": RPS_SCHEMA_VERSION,
        "rows_preserved": len(old),
        "added_lookbacks": list(missing),
        "archive": str(archive),
    }


def update_daily_rps(
    *,
    as_of_date: date | None = None,
    prices_root: Path = DEFAULT_PRICES_ROOT,
    root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> dict[str, Any]:
    """Calculate the default configured-horizon snapshot once and persist it."""

    if as_of_date is None:
        manifest = load_price_manifest(prices_root / "manifest.json")
        session = date.fromisoformat(str(manifest["latest_session"]))
    else:
        session = as_of_date
    snapshot = calculate_rps_snapshot(
        session,
        lookbacks=RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
    )
    return persist_rps_snapshot(snapshot, root=root, universe_path=universe_path)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from exc


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.rps_storage",
        description="Persist, query, validate, and backfill the RPS dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="calculate and upsert one session")
    update.add_argument("--as-of-date", type=_parse_date)
    backfill = subparsers.add_parser("backfill", help="vectorize complete RPS history")
    backfill.add_argument("--start", type=_parse_date, default=DEFAULT_BACKFILL_START)
    backfill.add_argument("--end", type=_parse_date)
    migration = subparsers.add_parser(
        "migrate", help="preserve v1 history and add missing horizons from local prices"
    )
    subparsers.add_parser("validate", help="validate all persisted RPS assets")
    for command in (update, backfill, migration):
        command.add_argument("--prices-root", type=Path, default=DEFAULT_PRICES_ROOT)
    for command in (update, backfill, migration, subparsers.choices["validate"]):
        command.add_argument("--rps-root", type=Path, default=DEFAULT_RPS_ROOT)
        command.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
        command.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for local RPS persistence."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "update":
            result = update_daily_rps(
                as_of_date=args.as_of_date,
                prices_root=args.prices_root,
                root=args.rps_root,
                universe_path=args.universe,
            )
        elif args.command == "backfill":
            result = backfill_rps_history(
                start_date=args.start,
                end_date=args.end,
                prices_root=args.prices_root,
                root=args.rps_root,
                universe_path=args.universe,
            )
        elif args.command == "migrate":
            result = migrate_rps_history(
                prices_root=args.prices_root,
                root=args.rps_root,
                universe_path=args.universe,
            )
        elif args.command == "validate":
            manifest = validate_rps_dataset(args.rps_root, universe_path=args.universe)
            result = {
                "schema_version": manifest["schema_version"],
                "latest_session": manifest["latest_session"],
                "total_row_count": manifest["total_row_count"],
                "success": True,
            }
        else:
            parser.error(f"unsupported command: {args.command}")
        if args.result_json is not None:
            write_json_atomically(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (RpsStorageError, OSError, ValueError) as exc:
        LOGGER.error("RPS storage operation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
