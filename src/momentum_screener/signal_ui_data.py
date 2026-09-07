"""Local CSV normalization and bounded price reads for signal research.

No Signal Store queries, strategy computation, downloads, or persistence.
Keep this data adapter separate from Streamlit so another signal source can
later supply the same normalized frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from stat import S_ISREG

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from momentum_screener.prices import DEFAULT_OUTPUT_ROOT
from momentum_screener.storage_manifest import load_manifest, resolve_local_asset_path

SIGNAL_KEY = ["session", "strategy_id", "ticker"]


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


def read_signal_csv(source: LocalFile) -> tuple[pd.DataFrame, list[str]]:
    """Normalize one CSV; preserve diagnostics and report discarded rows."""

    if LocalFile.inspect(source.path) != source:
        raise ValueError("CSV changed while loading; click Load Signals again.")
    frame = pd.read_csv(
        source.path,
        encoding="utf-8-sig",
        dtype={
            "session": "string",
            "date": "string",
            "ticker": "string",
            "strategy_id": "string",
            "strategy_version": "string",
        },
        keep_default_na=False,  # A ticker such as NA is a symbol, not a null.
    )
    date_column = "session" if "session" in frame else "date"
    if date_column not in frame or "ticker" not in frame:
        raise ValueError("CSV requires 'session' (or 'date') and 'ticker' columns.")
    if frame.empty:
        raise ValueError("CSV contains no signal rows.")

    dates = pd.to_datetime(frame[date_column], format="mixed", errors="coerce")
    try:
        # Preserve the calendar date if all timestamps carry the same timezone.
        frame["session"] = dates.dt.tz_localize(None).dt.normalize()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Dates must use consistent, parseable calendar dates."
        ) from exc
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    invalid_dates = frame["session"].isna()
    invalid_tickers = frame["ticker"].isna() | frame["ticker"].eq("")
    warnings = []
    if invalid_dates.any():
        warnings.append(
            f"Ignored {int(invalid_dates.sum())} row(s) with invalid dates."
        )
    if invalid_tickers.any():
        warnings.append(
            f"Ignored {int(invalid_tickers.sum())} row(s) with empty tickers."
        )
    frame = frame.loc[~(invalid_dates | invalid_tickers)].copy()
    if frame.empty:
        raise ValueError(
            "CSV contains no valid signal rows (invalid dates or tickers)."
        )

    fallback = Path(source.path).stem
    if "strategy_id" not in frame:
        frame["strategy_id"] = fallback
    else:
        strategy = frame["strategy_id"].str.strip()
        missing = strategy.isna() | strategy.eq("")
        if missing.any():
            warnings.append(
                f"Used filename '{fallback}' for {int(missing.sum())} missing strategy id(s)."
            )
        frame["strategy_id"] = strategy.mask(missing, fallback)
    if "strategy_version" not in frame:
        frame["strategy_version"] = pd.NA
    # Normalize blank diagnostics to missing, including unioned CSV columns.
    frame = frame.replace("", pd.NA)
    frame["source_file"] = source.path
    if LocalFile.inspect(source.path) != source:
        raise ValueError("CSV changed while loading; click Load Signals again.")
    return frame.reset_index(drop=True), warnings


def combine_signals(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Union diagnostics; keep the first logical row and warn about conflicts."""

    if not frames:
        return pd.DataFrame(
            columns=[*SIGNAL_KEY, "strategy_version", "source_file"]
        ), []
    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicated = combined.duplicated(SIGNAL_KEY, keep=False)
    warnings = []
    if duplicated.any():
        # Provenance and the original date alias do not constitute diagnostics.
        compared = [c for c in combined if c not in {"source_file", "date"}]
        distinct = combined.loc[duplicated, compared].drop_duplicates()
        conflicting = distinct.loc[distinct.duplicated(SIGNAL_KEY, keep=False)]
        count = len(conflicting[SIGNAL_KEY].drop_duplicates())
        if count:
            warnings.append(
                f"{count} duplicate signal key(s) have conflicting diagnostics or metadata. "
                "Kept the first row in CSV path/row order; no fields were merged."
            )
        removed = int(combined.duplicated(SIGNAL_KEY).sum())
        warnings.append(
            f"Removed {removed} duplicate signal row(s); kept the first row."
        )
    return combined.drop_duplicates(SIGNAL_KEY, keep="first").reset_index(
        drop=True
    ), warnings


def maximum_price_window(signal_date: date) -> tuple[date, date]:
    """Inclusive calendar bounds, including leap-day/month-end clipping."""

    signal = pd.Timestamp(signal_date)
    return (
        (signal - pd.DateOffset(years=2)).date(),
        (signal + pd.DateOffset(years=1)).date(),
    )


@dataclass(frozen=True)
class PriceWindow:
    start: date
    end: date
    viewport_start: date
    viewport_end: date
    viewport_fallback: bool = False


def clip_price_window(
    signal_date: date, available_start: date, available_end: date
) -> PriceWindow | None:
    """Intersect maximum/default windows with actual data; None means no overlap.

    If the default viewport has no span of available data, show the available
    maximum window instead of returning an inverted or zero-width viewport.
    """

    if available_start > available_end:
        raise ValueError("available_start cannot be after available_end")
    earliest, latest = maximum_price_window(signal_date)
    start, end = max(earliest, available_start), min(latest, available_end)
    if start > end:
        return None
    signal = pd.Timestamp(signal_date)
    view_start = max(start, (signal - pd.DateOffset(months=3)).date())
    view_end = min(end, (signal + pd.DateOffset(months=1)).date())
    fallback = view_start >= view_end
    if fallback:
        view_start, view_end = start, end
    return PriceWindow(start, end, view_start, view_end, fallback)


def local_price_files(
    signal_date: date, prices_root: Path = DEFAULT_OUTPUT_ROOT
) -> tuple[LocalFile, ...]:
    """Resolve only relevant local assets through the existing manifest helpers."""

    manifest_path = prices_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Local marketData manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    start, end = maximum_price_window(signal_date)
    return tuple(
        LocalFile.inspect(resolve_local_asset_path(prices_root, asset["local_path"]))
        for year, asset in sorted(manifest["assets"].items())
        if year.isdigit() and start.year <= int(year) <= end.year
    )


def read_local_prices(
    ticker: str, signal_date: date, files: tuple[LocalFile, ...]
) -> pd.DataFrame:
    """Read persisted adj_close once, with ticker/date predicate pushdown.

    The project's adjusted OHLC helper preserves this same adj_close field.
    Do not use raw close or apply the adjustment factor a second time.
    """

    start, end = maximum_price_window(signal_date)
    frames = []
    for source in files:
        if LocalFile.inspect(source.path) != source:
            raise ValueError(
                "Local price data changed while loading; retry the selection."
            )
        frame = pq.read_table(
            source.path,
            columns=["date", "adj_close"],
            filters=[
                ("ticker", "=", ticker),
                ("date", ">=", start),
                ("date", "<=", end),
            ],
        ).to_pandas()
        if LocalFile.inspect(source.path) != source:
            raise ValueError(
                "Local price data changed while loading; retry the selection."
            )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "adjusted_close"])
    prices = pd.concat(frames, ignore_index=True).rename(
        columns={"adj_close": "adjusted_close"}
    )
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    values = pd.to_numeric(prices["adjusted_close"], errors="raise").astype("float64")
    if prices["date"].isna().any() or not (np.isfinite(values) & values.gt(0)).all():
        raise ValueError("Local prices contain invalid dates or adjusted close values.")
    if prices["date"].duplicated().any():
        raise ValueError("Local prices contain conflicting/duplicate ticker dates.")
    prices["adjusted_close"] = values
    return prices.sort_values("date", ignore_index=True)
