"""Shared manifest-backed local price queries for research consumers.

Projection and ticker/date predicates read existing canonical Parquet assets;
this module never downloads, adjusts or persists prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from stat import S_ISREG

import pandas as pd
import pyarrow.parquet as pq

from momentum_screener.prices import DEFAULT_OUTPUT_ROOT
from momentum_screener.storage_manifest import load_manifest, resolve_local_asset_path


@dataclass(frozen=True)
class LocalFile:
    """Hashable cache identity; reloading observes local file replacements."""

    path: str
    mtime_ns: int
    ctime_ns: int
    size: int

    @classmethod
    def inspect(cls, path: str | Path) -> LocalFile:
        resolved = Path(path).expanduser().resolve()
        info = resolved.stat()
        if not S_ISREG(info.st_mode):
            raise ValueError(f"Not a regular local file: {resolved}")
        return cls(str(resolved), info.st_mtime_ns, info.st_ctime_ns, info.st_size)


def price_files_in_range(
    start_date: date,
    end_date: date | None = None,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[LocalFile, ...]:
    """Fingerprint the manifest first, then only indexed years in the range.

    An omitted end selects through the committed latest session. Including
    the manifest invalidates caches when a new yearly partition is published.
    """
    path = prices_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Local marketData manifest not found: {path}")
    source = LocalFile.inspect(path)
    manifest = load_manifest(path)
    end = end_date or date.fromisoformat(manifest["latest_session"])
    files = tuple(
        LocalFile.inspect(resolve_local_asset_path(prices_root, asset["local_path"]))
        for year, asset in sorted(manifest["assets"].items())
        if year.isdigit() and start_date.year <= int(year) <= end.year
    )
    require_unchanged_files((source,))
    return (source, *files)


def require_unchanged_files(files: tuple[LocalFile, ...]) -> None:
    if any(LocalFile.inspect(source.path) != source for source in files):
        raise ValueError("Local price data changed while loading; retry the selection.")


def read_local_price_rows(
    tickers: Sequence[str],
    start_date: date,
    end_date: date | None,
    files: tuple[LocalFile, ...],
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read each selected Parquet once for the whole ticker batch.

    files contains Parquet identities only. Date/ticker keys are always retained;
    the caller selects price fields and decides how unavailable values behave.
    """
    selected = list(dict.fromkeys(["date", "ticker", *columns]))
    if not tickers or not files:
        return pd.DataFrame(columns=selected)
    filters = [("ticker", "in", list(tickers)), ("date", ">=", start_date)]
    if end_date is not None:
        filters.append(("date", "<=", end_date))
    require_unchanged_files(files)
    frames = [
        pq.read_table(source.path, columns=selected, filters=filters).to_pandas()
        for source in files
    ]
    require_unchanged_files(files)
    rows = pd.concat(frames, ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.date
    if rows["date"].isna().any() or rows.duplicated(["date", "ticker"]).any():
        raise ValueError("Local prices contain invalid or duplicate ticker dates.")
    return rows.sort_values(["ticker", "date"], ignore_index=True)
