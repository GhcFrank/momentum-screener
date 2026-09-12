"""Local CSV normalization and bounded price reads for signal research.

No Signal Store queries, strategy computation, downloads, or persistence.
Keep this data adapter separate from Streamlit so another signal source can
later supply the same normalized frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from stat import S_ISDIR, S_ISREG

import numpy as np
import pandas as pd

from momentum_screener.local_price_data import (
    LocalFile,
    price_files_in_range,
    read_local_price_rows,
)
from momentum_screener.market_cap_storage import (
    DEFAULT_MARKET_CAP_ROOT,
    MARKET_CAP_MANIFEST_NAME,
    MARKET_CAP_SCHEMA,
    load_market_cap_manifest,
    read_market_cap,
)
from momentum_screener.prices import DEFAULT_OUTPUT_ROOT
from momentum_screener.rps import RPS_LOOKBACKS
from momentum_screener.rps_storage import (
    DEFAULT_RPS_ROOT,
    RPS_MANIFEST_NAME,
    RpsStorageError,
    load_rps_manifest,
    read_rps_snapshot,
)
from momentum_screener.storage_manifest import resolve_local_asset_path
from momentum_screener.strategy_data import calculate_turnover_columns
from momentum_screener.universe import normalize_ticker

SIGNAL_KEY = ["session", "strategy_id", "ticker"]


def discover_signal_csvs(paths: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Discover local CSV files without reading contents or recursing into folders.

    Preserve input order; sort each directory's immediate entries by filename.
    Return resolved paths only once, plus warnings for unusable inputs. A bad
    path or directory entry does not block discovery from the remaining paths.
    """

    discovered: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    def add_csv(path: Path) -> bool:
        try:
            if not S_ISREG(path.stat().st_mode):
                return False
            resolved = path.resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"Unable to inspect CSV: {path}: {exc}")
            return False
        if resolved not in seen:
            seen.add(resolved)
            discovered.append(resolved)
        return True

    for raw_path in paths:
        text = raw_path.strip()
        if not text:
            continue
        try:
            path = Path(text).expanduser()
            mode = path.stat().st_mode
            if S_ISDIR(mode):
                # Only immediate children; directories (including *.csv folders)
                # are never traversed, even when they contain more signal files.
                entries = sorted(path.iterdir())
                count = sum(
                    add_csv(entry)
                    for entry in entries
                    if entry.suffix.lower() == ".csv"
                )
                if not count:
                    warnings.append(f"No CSV files found in: {path}")
            elif S_ISREG(mode):
                if path.suffix.lower() == ".csv":
                    add_csv(path)
                else:
                    warnings.append(f"Unsupported file type: {path}")
            else:
                warnings.append(f"Not a regular file or directory: {path}")
        except FileNotFoundError:
            warnings.append(f"Path does not exist: {text}")
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"Unable to inspect path: {text}: {exc}")
    return discovered, warnings


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


def filter_tickers_by_strategies(
    signals: pd.DataFrame, session: date, selected_strategies: Sequence[str]
) -> list[str]:
    """Return sorted unique tickers present in EVERY selected strategy that day.

    An empty selection or a strategy with no rows that day yields no matches.
    Duplicate rows within one strategy cannot substitute for another strategy.
    """

    selected = set(selected_strategies)
    if not selected or signals.empty:
        return []
    rows = signals.loc[
        signals["session"].eq(pd.Timestamp(session))
        & signals["strategy_id"].isin(selected)
    ]
    counts = rows.groupby("ticker")["strategy_id"].nunique()
    return sorted(counts.index[counts.eq(len(selected))].tolist())


def local_rps_files(
    session: date, root: Path = DEFAULT_RPS_ROOT
) -> tuple[LocalFile, ...]:
    """Fingerprint the manifest and session's asset without duplicating its layout."""

    files = [LocalFile.inspect(root / RPS_MANIFEST_NAME)]
    manifest = load_rps_manifest(root)
    asset = manifest["assets"].get(str(session.year))
    if asset is not None:
        files.append(
            LocalFile.inspect(resolve_local_asset_path(root, asset["local_path"]))
        )
    return tuple(files)


def load_rps_for_session(
    session: date, files: tuple[LocalFile, ...], root: Path = DEFAULT_RPS_ROOT
) -> pd.DataFrame:
    """Read one complete local snapshot for caching, never calculate missing RPS.

    Include local_rps_files() in the cache key so edits and atomic replacements
    invalidate it. Storage errors propagate to the UI, which keeps tickers and
    displays missing RPS; an absent session already returns an empty snapshot.
    """

    if any(LocalFile.inspect(source.path) != source for source in files):
        raise RpsStorageError("Local RPS changed while loading; retry the selection.")
    snapshot = read_rps_snapshot(session, root=root)
    if any(LocalFile.inspect(source.path) != source for source in files):
        raise RpsStorageError("Local RPS changed while loading; retry the selection.")
    return snapshot


def build_ticker_rps_table(
    tickers: Sequence[str], snapshot: pd.DataFrame
) -> pd.DataFrame:
    """Join one session's RPS onto unique tickers, retaining every missing value.

    Read only snapshot values, never strategy CSV diagnostics. INVALID_RPS (-1),
    missing fields/rows, and nonfinite or out-of-range scores become NaN.
    """

    columns = [f"rps{lookback}" for lookback in RPS_LOOKBACKS]
    source = snapshot.reindex(columns=["ticker", *columns]).set_index("ticker")
    if source.index.has_duplicates:
        raise ValueError("RPS snapshot contains duplicate tickers.")
    table = source.reindex(pd.Index(sorted(set(tickers)), name="ticker"))
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").astype("float64")
        table[column] = values.where(values.between(0, 100))
    return table.reset_index().rename(
        columns={"ticker": "Ticker", **{column: column.upper() for column in columns}}
    )


def local_market_cap_files(
    session: date, root: Path = DEFAULT_MARKET_CAP_ROOT
) -> tuple[LocalFile, ...]:
    """Fingerprint the MarketCap manifest and the requested year's asset.

    A completely absent local MarketCap store is normal for older research
    environments and returns an empty cache identity. Once a Release is pulled,
    the new manifest and partition identities produce a different cache key.
    """

    if not root.exists():
        return ()
    manifest_path = root / MARKET_CAP_MANIFEST_NAME
    source = LocalFile.inspect(manifest_path)
    manifest = load_market_cap_manifest(root)
    files = [source]
    asset = manifest["assets"].get(str(session.year))
    if asset is not None:
        files.append(
            LocalFile.inspect(resolve_local_asset_path(root, asset["local_path"]))
        )
    if any(LocalFile.inspect(item.path) != item for item in files):
        raise ValueError("Local MarketCap changed while loading; retry the selection.")
    return tuple(files)


def load_turnover_for_session(
    session: date,
    tickers: Sequence[str],
    price_files: tuple[LocalFile, ...],
    market_cap_files: tuple[LocalFile, ...],
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
) -> pd.DataFrame:
    """Calculate exact-session UI turnover while retaining every requested ticker.

    Turnover is the project's MarketCap proxy ``raw close * volume / market_cap``.
    It is dynamically enriched for display and never read from or written to a
    signal CSV. Missing MarketCap stores, sessions and ticker observations remain
    float NaN; malformed stores and changed cache inputs still raise to the UI.
    """

    normalized: list[str] = []
    for ticker in tickers:
        value = normalize_ticker(ticker)
        if value is None:
            raise ValueError(f"Invalid ticker for turnover lookup: {ticker!r}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        return pd.DataFrame(
            {
                "ticker": pd.Series(dtype="string"),
                "turnover": pd.Series(dtype="float64"),
            }
        )

    sources = (*price_files, *market_cap_files)
    if any(LocalFile.inspect(item.path) != item for item in sources):
        raise ValueError("Local turnover inputs changed; retry the selection.")
    price_partitions = tuple(
        item for item in price_files if Path(item.path).suffix == ".parquet"
    )
    prices = read_local_price_rows(
        normalized,
        session,
        session,
        price_partitions,
        columns=("close", "volume"),
    )
    caps = (
        read_market_cap(
            tickers=normalized,
            start_date=session,
            end_date=session,
            root=market_cap_root,
        )
        if market_cap_files
        else MARKET_CAP_SCHEMA.empty_table().to_pandas()
    )
    keys = pd.DataFrame(
        {
            "date": session,
            "ticker": pd.Series(normalized, dtype="string"),
        }
    )
    prepared = keys.merge(
        prices.loc[:, ["date", "ticker", "close", "volume"]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    ).merge(
        caps.loc[:, ["date", "ticker", "market_cap"]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    result = calculate_turnover_columns(prepared).loc[:, ["ticker", "turnover"]]
    if any(LocalFile.inspect(item.path) != item for item in sources):
        raise ValueError("Local turnover inputs changed; retry the selection.")
    return result


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

    start, end = maximum_price_window(signal_date)
    return price_files_in_range(start, end, prices_root=prices_root)[1:]


def read_local_prices(
    ticker: str, signal_date: date, files: tuple[LocalFile, ...]
) -> pd.DataFrame:
    """Read persisted adj_close once, with ticker/date predicate pushdown.

    The project's adjusted OHLC helper preserves this same adj_close field.
    Do not use raw close or apply the adjustment factor a second time.
    """

    start, end = maximum_price_window(signal_date)
    prices = (
        read_local_price_rows((ticker,), start, end, files, columns=("adj_close",))
        .drop(columns="ticker")
        .rename(columns={"adj_close": "adjusted_close"})
    )
    if prices.empty:
        return pd.DataFrame(columns=["date", "adjusted_close"])
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    values = pd.to_numeric(prices["adjusted_close"], errors="raise").astype("float64")
    if prices["date"].isna().any() or not (np.isfinite(values) & values.gt(0)).all():
        raise ValueError("Local prices contain invalid dates or adjusted close values.")
    if prices["date"].duplicated().any():
        raise ValueError("Local prices contain conflicting/duplicate ticker dates.")
    prices["adjusted_close"] = values
    return prices.sort_values("date", ignore_index=True)
