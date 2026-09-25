"""Observed daily MarketCap snapshots, independent of OHLCV and Universe ranks."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    determine_target_session,
    load_universe,
    universe_sha256,
)
from momentum_screener.storage_manifest import (
    ManifestError,
    build_asset_record,
    calculate_sha256,
    load_manifest,
    remove_owned_tree,
    replace_files_transactionally,
    resolve_local_asset_path,
    write_json_atomically,
)
from momentum_screener.universe import (
    UniverseBuildError,
    fetch_market_caps,
    normalize_ticker,
)

DEFAULT_MARKET_CAP_ROOT = Path("data/processed/market_cap")
DEFAULT_REFRESH_CALENDAR_DAYS = 7
MARKET_CAP_SCHEMA_VERSION = "market_cap_v1"
MARKET_CAP_MANIFEST_NAME = "manifest.json"
MISSING_TICKERS_NAME = "missing_tickers.csv"
MARKET_CAP_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("market_cap", pa.int64(), nullable=False),
    ]
)


class MarketCapStorageError(RuntimeError):
    """A snapshot or its committed local dataset is invalid."""


LOGGER = logging.getLogger(__name__)


def _partition_path(root: Path, year: int) -> Path:
    return root / "daily" / f"year={year}" / "market_cap.parquet"


def _normalize(rows: pd.DataFrame) -> pd.DataFrame:
    if not set(MARKET_CAP_SCHEMA.names).issubset(rows.columns):
        raise MarketCapStorageError("MarketCap rows require date, ticker, market_cap")
    frame = rows.loc[:, MARKET_CAP_SCHEMA.names].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    frame["ticker"] = frame["ticker"].map(normalize_ticker).astype("string")
    values = frame["market_cap"]
    # Do not coerce fractional, bool, null, or overflowing caps into plausible data.
    if not values.map(
        lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 2**63 - 1
    ).all():
        raise MarketCapStorageError("MarketCap must be a positive int64")
    frame["market_cap"] = values.astype("int64")
    if (
        frame[["date", "ticker"]].isna().any().any()
        or frame.duplicated(["date", "ticker"]).any()
    ):
        raise MarketCapStorageError("Invalid or duplicate MarketCap date/ticker keys")
    return frame.sort_values(["date", "ticker"], ignore_index=True)


def validate_market_cap_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise MarketCapStorageError("MarketCap manifest must be an object")
    manifest = dict(payload)
    if (
        manifest.get("schema_version") != MARKET_CAP_SCHEMA_VERSION
        or manifest.get("dataset_type") != "point_in_time_market_cap"
        or manifest.get("completed") is not True
    ):
        raise MarketCapStorageError("Unsupported or incomplete MarketCap manifest")
    digest = manifest.get("universe_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise MarketCapStorageError("Invalid MarketCap Universe hash")
    count = manifest.get("universe_ticker_count")
    if type(count) is not int or count <= 0:
        raise MarketCapStorageError("Invalid MarketCap Universe count")
    snapshots = manifest.get("snapshots")
    assets = manifest.get("assets")
    if (
        not isinstance(snapshots, Mapping)
        or not snapshots
        or not isinstance(assets, Mapping)
    ):
        raise MarketCapStorageError("MarketCap manifest requires snapshots and assets")
    totals: dict[str, int] = {}
    for session, info in snapshots.items():
        year = str(date.fromisoformat(session).year)
        if not isinstance(info, Mapping):
            raise MarketCapStorageError("Invalid MarketCap snapshot metadata")
        stored = info.get("stored_ticker_count")
        if (
            type(stored) is not int
            or not 0 <= stored <= count
            or info.get("requested_ticker_count") != count
            or info.get("missing_ticker_count") != count - stored
            or datetime.fromisoformat(info["observed_at"]).utcoffset() is None
        ):
            raise MarketCapStorageError(
                "Invalid MarketCap snapshot counts or observation time"
            )
        totals[year] = totals.get(year, 0) + stored
    if (
        manifest.get("earliest_session") != min(snapshots)
        or manifest.get("latest_session") != max(snapshots)
        or manifest.get("total_row_count") != sum(totals.values())
        or manifest.get("partition_row_counts") != totals
        or set(assets) != set(totals)
    ):
        raise MarketCapStorageError("MarketCap manifest date/count/partition mismatch")
    if datetime.fromisoformat(manifest["updated_at"]).utcoffset() is None:
        raise MarketCapStorageError("MarketCap updated_at requires timezone")
    for year, asset in assets.items():
        if not isinstance(asset, Mapping):
            raise MarketCapStorageError("Invalid MarketCap asset metadata")
        if (
            asset.get("asset_name") != f"market-cap-year-{year}.parquet"
            or asset.get("local_path") != f"daily/year={year}/market_cap.parquet"
            or type(asset.get("size_bytes")) is not int
            or asset["size_bytes"] <= 0
        ):
            raise MarketCapStorageError("Invalid MarketCap asset path or size")
        sha = asset.get("sha256", "")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise MarketCapStorageError("Invalid MarketCap asset hash")
    return manifest


def load_market_cap_manifest(root: Path = DEFAULT_MARKET_CAP_ROOT) -> dict[str, Any]:
    try:
        return validate_market_cap_manifest(
            json.loads((root / MARKET_CAP_MANIFEST_NAME).read_text())
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MarketCapStorageError(
            f"Unable to read MarketCap manifest: {exc}"
        ) from exc


def _read_partition(path: Path, year: int) -> pd.DataFrame:
    table = pq.read_table(path)
    if not table.schema.equals(MARKET_CAP_SCHEMA, check_metadata=False):
        raise MarketCapStorageError(f"Unexpected MarketCap parquet schema: {path}")
    rows = table.to_pandas()
    normalized = _normalize(rows)
    if not normalized["date"].map(lambda value: value.year == year).all():
        raise MarketCapStorageError("MarketCap rows belong to another year")
    rows["ticker"] = rows["ticker"].astype("string")
    if not rows.equals(normalized):
        raise MarketCapStorageError("MarketCap partition is not normalized/sorted")
    return normalized


def validate_market_cap_dataset(
    root: Path = DEFAULT_MARKET_CAP_ROOT, *, universe_path: Path = DEFAULT_UNIVERSE
) -> dict[str, Any]:
    manifest = load_market_cap_manifest(root)
    universe = load_universe(universe_path)
    if manifest["universe_sha256"] != universe_sha256(universe) or manifest[
        "universe_ticker_count"
    ] != len(universe):
        raise MarketCapStorageError("MarketCap Universe identity mismatch")
    observed: dict[str, int] = {}
    for year, asset in manifest["assets"].items():
        path = resolve_local_asset_path(root, asset["local_path"])
        if (
            path.stat().st_size != asset["size_bytes"]
            or calculate_sha256(path) != asset["sha256"]
        ):
            raise MarketCapStorageError("MarketCap partition size/hash mismatch")
        rows = _read_partition(path, int(year))
        if len(rows) != manifest["partition_row_counts"][year] or not set(
            rows["ticker"]
        ).issubset(universe):
            raise MarketCapStorageError("MarketCap partition count/ticker mismatch")
        observed.update(
            {day.isoformat(): len(group) for day, group in rows.groupby("date")}
        )
    expected = {
        day: info["stored_ticker_count"]
        for day, info in manifest["snapshots"].items()
        if info["stored_ticker_count"]
    }
    if observed != expected:
        raise MarketCapStorageError("MarketCap snapshot coverage differs from parquet")
    return manifest


def refresh_market_cap_snapshot(
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    root: Path = DEFAULT_MARKET_CAP_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    refresh_calendar_days: int = DEFAULT_REFRESH_CALENDAR_DAYS,
    now: datetime | None = None,
    fetch_func: Callable[[Sequence[str]], Mapping[str, int]] = fetch_market_caps,
) -> dict[str, Any]:
    """Refresh the target snapshot and recent missing provider observations.

    A snapshot is an observation made at observed_at, labelled by the successful
    price session. It is not an estimate of historical caps or exact closing cap.
    Previously stored values are never copied between sessions. Within the rolling
    calendar-day window, a fresh provider value can fill an observation that was
    previously missing; established observations are retained. The target session
    is always replaced by the current provider response. A stale/future manifest
    is rejected before contacting Yahoo.
    """

    if refresh_calendar_days < 0:
        raise ValueError("refresh_calendar_days cannot be negative")
    observed_at = now or datetime.now(UTC)
    price_manifest = load_manifest(prices_root / "manifest.json")
    session = date.fromisoformat(price_manifest["latest_session"])
    if session != determine_target_session(now=observed_at):
        raise MarketCapStorageError(
            "Price latest_session is stale or unsettled; update prices before MarketCap refresh"
        )
    universe = load_universe(universe_path)
    identity = universe_sha256(universe)
    if price_manifest["universe_sha256"] != identity or price_manifest[
        "universe_ticker_count"
    ] != len(universe):
        raise MarketCapStorageError("Price and MarketCap Universe identity mismatch")
    previous = None
    if (root / MARKET_CAP_MANIFEST_NAME).exists():
        previous = validate_market_cap_dataset(root, universe_path=universe_path)
        if previous["latest_session"] > session.isoformat():
            raise MarketCapStorageError(
                "Cannot replace a past MarketCap snapshot with current caps"
            )
    elif root.exists() and any(root.iterdir()):
        raise MarketCapStorageError("Non-empty MarketCap root has no valid manifest")
    caps = fetch_func(universe)
    valid = {
        ticker: caps[ticker]
        for ticker in universe
        if ticker in caps
        and type(caps[ticker]) is int
        and 0 < caps[ticker] <= 2**63 - 1
    }
    target_rows = _normalize(
        pd.DataFrame(
            [(session, ticker, value) for ticker, value in valid.items()],
            columns=MARKET_CAP_SCHEMA.names,
        )
    )
    refresh_start = session - timedelta(days=refresh_calendar_days)
    snapshots = dict(previous["snapshots"]) if previous else {}
    refresh_sessions = tuple(
        sorted(
            {
                session,
                *(
                    date.fromisoformat(value)
                    for value in snapshots
                    if refresh_start <= date.fromisoformat(value) <= session
                ),
            }
        )
    )
    years = tuple(sorted({value.year for value in refresh_sessions}))
    partition_rows = {
        year: (
            _read_partition(_partition_path(root, year), year)
            if _partition_path(root, year).exists()
            else target_rows.iloc[:0].copy()
        )
        for year in years
    }
    refreshed_rows: dict[date, pd.DataFrame] = {}
    missing_rows: list[tuple[date, str, str]] = []
    for refresh_session in refresh_sessions:
        existing = partition_rows[refresh_session.year]
        existing = existing.loc[existing["date"].eq(refresh_session)].copy()
        if refresh_session == session:
            refreshed = target_rows
        else:
            additions = target_rows.loc[
                ~target_rows["ticker"].isin(existing["ticker"])
            ].assign(date=refresh_session)
            refreshed = _normalize(pd.concat([existing, additions], ignore_index=True))
        refreshed_rows[refresh_session] = refreshed
        missing_for_session = sorted(set(universe).difference(refreshed["ticker"]))
        missing_rows.extend(
            (refresh_session, ticker, "missing_or_invalid_yahoo_market_cap")
            for ticker in missing_for_session
        )
        snapshots[refresh_session.isoformat()] = {
            "requested_ticker_count": len(universe),
            "stored_ticker_count": len(refreshed),
            "missing_ticker_count": len(missing_for_session),
            "observed_at": observed_at.isoformat(),
        }

    updated_partitions: dict[int, pd.DataFrame] = {}
    for year, rows in partition_rows.items():
        refreshed_dates = {value for value in refresh_sessions if value.year == year}
        replacement = [refreshed_rows[value] for value in sorted(refreshed_dates)]
        updated_partitions[year] = _normalize(
            pd.concat(
                [rows.loc[~rows["date"].isin(refreshed_dates)], *replacement],
                ignore_index=True,
            )
        )

    target_info = snapshots[session.isoformat()]
    requested_observation_count = len(refresh_sessions) * len(universe)
    stored_observation_count = sum(len(rows) for rows in refreshed_rows.values())
    missing_observation_count = requested_observation_count - stored_observation_count
    LOGGER.info(
        "MarketCap refresh %s..%s sessions=%d requested=%d stored=%d missing=%d; "
        "target tickers requested=%d stored=%d missing=%d",
        refresh_start.isoformat(),
        session.isoformat(),
        len(refresh_sessions),
        requested_observation_count,
        stored_observation_count,
        missing_observation_count,
        len(universe),
        target_info["stored_ticker_count"],
        target_info["missing_ticker_count"],
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".market-cap-refresh-", dir=root.parent
    ) as temp:
        staging = Path(temp)
        assets = dict(previous["assets"]) if previous else {}
        counts = dict(previous["partition_row_counts"]) if previous else {}
        paths: list[str] = []
        for year, rows in updated_partitions.items():
            staged = _partition_path(staging, year)
            staged.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(
                    rows, schema=MARKET_CAP_SCHEMA, preserve_index=False
                ),
                staged,
                compression="zstd",
            )
            relative = str(staged.relative_to(staging))
            paths.append(relative)
            assets[str(year)] = build_asset_record(
                staged,
                asset_name=f"market-cap-year-{year}.parquet",
                local_path=relative,
            )
            counts[str(year)] = len(rows)
        manifest = {
            "schema_version": MARKET_CAP_SCHEMA_VERSION,
            "dataset_type": "point_in_time_market_cap",
            "completed": True,
            "latest_session": max(snapshots),
            "earliest_session": min(snapshots),
            "total_row_count": sum(counts.values()),
            "partition_row_counts": counts,
            "assets": assets,
            "universe_sha256": identity,
            "universe_ticker_count": len(universe),
            "updated_at": observed_at.isoformat(),
            "snapshots": snapshots,
        }
        validate_market_cap_manifest(manifest)
        write_json_atomically(staging / MARKET_CAP_MANIFEST_NAME, manifest)
        pd.DataFrame(missing_rows, columns=("date", "ticker", "reason")).to_csv(
            staging / MISSING_TICKERS_NAME, index=False
        )
        backup = root.parent / f".market-cap-backup-{uuid.uuid4().hex}"
        replace_files_transactionally(
            root,
            staging,
            [*paths, MISSING_TICKERS_NAME, MARKET_CAP_MANIFEST_NAME],
            backup_root=backup,
            validate_after=lambda: validate_market_cap_dataset(
                root, universe_path=universe_path
            ),
        )
        remove_owned_tree(backup, parent=root.parent, prefix=".market-cap-backup-")
    return {
        "success": True,
        "snapshot_date": session.isoformat(),
        "refresh_start": refresh_start.isoformat(),
        "refresh_end": session.isoformat(),
        "refreshed_session_count": len(refresh_sessions),
        "requested_observation_count": requested_observation_count,
        "stored_observation_count": stored_observation_count,
        "missing_observation_count": missing_observation_count,
        **target_info,
        "missing_tickers_csv": str(root / MISSING_TICKERS_NAME),
    }


def read_market_cap(
    *,
    tickers: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    root: Path = DEFAULT_MARKET_CAP_ROOT,
) -> pd.DataFrame:
    """Read exact historical observations; missing dates/tickers are never filled."""
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    if not root.exists():
        return MARKET_CAP_SCHEMA.empty_table().to_pandas()
    manifest = load_market_cap_manifest(root)
    frames = []
    for year, asset in sorted(manifest["assets"].items()):
        if (start_date and int(year) < start_date.year) or (
            end_date and int(year) > end_date.year
        ):
            continue
        rows = _read_partition(
            resolve_local_asset_path(root, asset["local_path"]), int(year)
        )
        if start_date:
            rows = rows.loc[rows["date"].ge(start_date)]
        if end_date:
            rows = rows.loc[rows["date"].le(end_date)]
        if tickers is not None:
            rows = rows.loc[rows["ticker"].isin([normalize_ticker(t) for t in tickers])]
        frames.append(rows)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else MARKET_CAP_SCHEMA.empty_table().to_pandas()
    )


def get_market_cap(
    ticker: str, session: date, *, root: Path = DEFAULT_MARKET_CAP_ROOT
) -> int | None:
    rows = read_market_cap(
        tickers=[ticker], start_date=session, end_date=session, root=root
    )
    return None if rows.empty else int(rows.iloc[0]["market_cap"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh MarketCap for the latest settled price session and retry "
            "missing observations from the recent calendar-day window."
        )
    )
    parser.add_argument("command", choices=["refresh", "validate"])
    parser.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_MARKET_CAP_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--refresh-calendar-days",
        type=int,
        default=DEFAULT_REFRESH_CALENDAR_DAYS,
        help="retry missing observations this many calendar days before the target",
    )
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (
            refresh_market_cap_snapshot(
                prices_root=args.prices_root,
                root=args.root,
                universe_path=args.universe,
                refresh_calendar_days=args.refresh_calendar_days,
            )
            if args.command == "refresh"
            else validate_market_cap_dataset(args.root, universe_path=args.universe)
        )
        if args.result_json:
            write_json_atomically(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        MarketCapStorageError,
        UniverseBuildError,
        ManifestError,
        OSError,
        ValueError,
        pa.ArrowException,
    ) as exc:
        parser.exit(1, f"MarketCap operation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
