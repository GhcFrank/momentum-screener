"""Shared point-in-time price and RPS access for strategy queries.

This layer reads existing stores and never persists data. RPS fallback always
ranks the complete Universe, even when only one ticker's result is requested.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import exchange_calendars as xcals  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa

from momentum_screener.market_cap_storage import (
    DEFAULT_MARKET_CAP_ROOT,
    MarketCapStorageError,
    read_market_cap,
    validate_market_cap_dataset,
)
from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    PriceUpdateError,
    load_universe,
    read_affected_partitions,
)
from momentum_screener.rps import (
    RPS_CALENDAR_NAME,
    RPS_LOOKBACKS,
    InvalidRpsSessionError,
    RpsTickerNotFoundError,
    _normalize_lookbacks,
    calculate_rps_snapshots,
    resolve_rps_session_dates,
)
from momentum_screener.rps_storage import (
    DEFAULT_RPS_ROOT,
    RPS_MANIFEST_NAME,
    RpsStorageError,
    read_rps_history,
)
from momentum_screener.storage_manifest import load_manifest


class StrategyDataError(RuntimeError):
    """A strategy date range cannot be queried safely."""


def load_strategy_market_cap(
    sessions: Iterable[date],
    *,
    root: Path = DEFAULT_MARKET_CAP_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> pd.DataFrame:
    """Require observations for every target session; retain missing ticker rows.

    No warmup snapshots are required. Only exact date/ticker observations are
    returned, with per-session coverage diagnostics for the current Universe.
    """

    requested = tuple(sorted(set(sessions)))
    if not requested:
        raise StrategyDataError("MarketCap query requires target sessions")
    try:
        manifest = validate_market_cap_dataset(root, universe_path=universe_path)
        missing = [
            day.isoformat()
            for day in requested
            if manifest["snapshots"]
            .get(day.isoformat(), {})
            .get("stored_ticker_count", 0)
            == 0
        ]
        if missing:
            raise StrategyDataError(
                f"MarketCap snapshots unavailable for {len(missing)} target session(s): "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        rows = read_market_cap(
            start_date=requested[0], end_date=requested[-1], root=root
        )
    except (MarketCapStorageError, OSError, ValueError, pa.ArrowException) as exc:
        raise StrategyDataError(
            f"MarketCap data unavailable for target sessions {requested[0]}..{requested[-1]}: {exc}"
        ) from exc
    rows = rows.loc[rows["date"].isin(requested)].copy()
    rows.attrs["session_counts"] = {
        day.isoformat(): {
            "market_cap_available_count": manifest["snapshots"][day.isoformat()][
                "stored_ticker_count"
            ],
            "market_cap_missing_ticker_count": manifest["snapshots"][day.isoformat()][
                "missing_ticker_count"
            ],
        }
        for day in requested
    }
    return rows


def normalize_date_values(values: pd.Series) -> pd.Series:
    """Normalize dates without filling unavailable rows."""

    normalized = pd.to_datetime(values, errors="coerce")
    if bool(normalized.isna().any()):
        raise ValueError("Strategy input contains invalid dates")
    if normalized.dt.tz is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.dt.date


def coerce_session_date(
    value: date | str, *, lookbacks: Iterable[int] = RPS_LOOKBACKS
) -> date:
    """Use the existing RPS date/calendar validation without date fallback."""

    return resolve_rps_session_dates(value, lookbacks=lookbacks).as_of_date


def resolve_history_bounds(
    start_date: date | str | None,
    end_date: date | str | None,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
) -> tuple[date, date]:
    """Resolve omitted bounds from the validated price manifest."""

    manifest = (
        load_manifest(prices_root / "manifest.json")
        if start_date is None or end_date is None
        else {}
    )
    start = coerce_session_date(
        start_date if start_date is not None else str(manifest["actual_min_date"]),
        lookbacks=lookbacks,
    )
    end = coerce_session_date(
        end_date if end_date is not None else str(manifest["latest_session"]),
        lookbacks=lookbacks,
    )
    if start > end:
        raise StrategyDataError("start_date cannot be after end_date")
    return start, end


def sessions_in_range(start_date: date, end_date: date) -> tuple[date, ...]:
    """Return inclusive XNYS sessions for a non-empty date range."""

    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date), pd.Timestamp(end_date)
    )
    result = tuple(value.date() for value in sessions)
    if not result:
        raise InvalidRpsSessionError("Strategy range has no XNYS sessions")
    return result


def resolve_strategy_sessions(end_date: date, count: int) -> tuple[date, ...]:
    """Return up to count sessions ending on an exact XNYS session."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("session count must be a positive integer")
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    session = calendar.date_to_session(pd.Timestamp(end_date), direction="none")
    end_index = int(calendar.sessions.get_loc(session))
    start_index = max(0, end_index - count + 1)
    return tuple(
        value.date() for value in calendar.sessions[start_index : end_index + 1]
    )


def load_strategy_price_history(
    *,
    end_date: date,
    start_date: date | None = None,
    required_sessions: int = 320,
    tickers: tuple[str, ...] | None = None,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    universe: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load existing validated price partitions with a session warmup.

    Retain the full first year, as Monthly Reversal historically does: sparse
    ticker histories can use those extra rows. Return only requested tickers
    and dates through end_date, plus the loaded distinct-session count. The
    existing partition reader validates against the full Universe before
    projecting tickers; passing candidates as its allowlist would be invalid.
    """

    load_start = resolve_strategy_sessions(start_date or end_date, required_sessions)[0]
    available_years = sorted(
        int(path.parent.name.removeprefix("year="))
        for path in (prices_root / "daily").glob("year=*/prices.parquet")
        if path.parent.name.removeprefix("year=").isdigit()
    )
    if not available_years or end_date.year not in available_years:
        raise PriceUpdateError(
            f"No complete price partition is available through {end_date.isoformat()}"
        )
    complete_universe = (
        universe if universe is not None else load_universe(universe_path)
    )
    first_year = max(load_start.year, available_years[0])
    rows = read_affected_partitions(
        prices_root,
        tuple(range(first_year, end_date.year + 1)),
        tickers=complete_universe,
    )
    mask = rows["date"].le(end_date)
    if tickers is not None:
        mask &= rows["ticker"].isin(tickers)
    rows = rows.loc[mask].copy()
    return rows, int(rows["date"].nunique())


def merge_prices_and_rps(
    prices: pd.DataFrame,
    rps_rows: pd.DataFrame,
    *,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
) -> pd.DataFrame:
    """Join exact date/ticker keys without forward filling or duplicate rows."""

    columns = [f"rps{value}" for value in _normalize_lookbacks(lookbacks)]
    left = prices.drop(columns=columns, errors="ignore").copy()
    return left.merge(
        rps_rows.loc[:, ["date", "ticker", *columns]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )


def _normalize_rps_rows(
    snapshots: pd.DataFrame,
    *,
    sessions: tuple[date, ...],
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    source = snapshots.reset_index(drop=True).copy()
    if "as_of_date" in source:
        source = source.rename(columns={"as_of_date": "date"})
    missing = sorted({"date", "ticker"}.difference(source.columns))
    if missing:
        raise ValueError(f"RPS snapshots are missing columns: {missing}")
    source["date"] = normalize_date_values(source["date"])
    source["ticker"] = source["ticker"].astype(str)
    source = source.loc[source["date"].isin(sessions) & source["ticker"].isin(tickers)]
    if bool(source.duplicated(["date", "ticker"]).any()):
        raise ValueError("RPS snapshots contain duplicate date/ticker rows")
    return source.set_index(["date", "ticker"])


def load_or_calculate_rps(
    sessions: Iterable[date],
    *,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    price_rows: pd.DataFrame | None = None,
    universe: tuple[str, ...] | None = None,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    result_tickers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Reuse supplied/stored RPS and batch-calculate only missing sessions.

    Injected rows override stored values; INVALID_RPS (-1) is an available
    calculation result, not a missing snapshot. Explicit NaN inputs also
    remain unavailable, preserving strategy injection behavior. Missing horizons are filled
    from the unchanged generic RPS engine. Returns/base dates are retained so
    an orchestrator can persist the same full snapshot once. A complete
    injected frame avoids both storage reads and recalculation.

    If price_rows is supplied, it must contain the entire Universe and every
    required RPS base session, never only a strategy's candidate prices.
    """

    horizons = _normalize_lookbacks(lookbacks)
    session_tuple = tuple(sorted(set(sessions)))
    if not session_tuple:
        raise InvalidRpsSessionError("sessions must contain at least one session")
    complete_universe = (
        universe if universe is not None else load_universe(universe_path)
    )
    tickers = result_tickers if result_tickers is not None else complete_universe
    unknown = sorted(set(tickers).difference(complete_universe))
    if unknown:
        raise RpsTickerNotFoundError(
            f"Requested result tickers are outside the Universe: {unknown}"
        )
    columns = [f"rps{value}" for value in horizons]
    expected = pd.MultiIndex.from_product(
        [session_tuple, tickers], names=["date", "ticker"]
    )
    stored = pd.DataFrame(index=expected, columns=columns, dtype="float64")
    covered = pd.DataFrame(False, index=expected, columns=columns)

    def fill_missing(rows: pd.DataFrame) -> None:
        normalized = _normalize_rps_rows(rows, sessions=session_tuple, tickers=tickers)
        row_present = expected.isin(normalized.index)
        for column in normalized:
            incoming = normalized[column].reindex(expected)
            if column not in stored:
                stored[column] = incoming
                covered[column] = row_present
            else:
                replace = row_present & ~covered[column]
                stored[column] = stored[column].where(~replace, incoming)
                covered[column] |= row_present

    if rps_snapshots is not None:
        fill_missing(rps_snapshots)
    missing = ~covered.loc[:, columns].all(axis="columns")
    if bool(missing.any()) and rps_root is not None:
        if (rps_root / RPS_MANIFEST_NAME).is_file():
            fill_missing(
                read_rps_history(
                    tickers=tickers,
                    start_date=session_tuple[0],
                    end_date=session_tuple[-1],
                    root=rps_root,
                )
            )
        elif rps_root.exists() and any(rps_root.iterdir()):
            raise RpsStorageError(
                f"Non-empty RPS root has no valid manifest: {rps_root}"
            )

    missing = ~covered.loc[:, columns].all(axis="columns")
    missing_sessions = tuple(
        sorted(set(stored.index.get_level_values("date")[missing]))
    )
    if missing_sessions:
        fill_missing(
            calculate_rps_snapshots(
                missing_sessions,
                lookbacks=horizons,
                prices_root=prices_root,
                universe_path=universe_path,
                price_rows=price_rows,
                universe=complete_universe,
                result_tickers=result_tickers,
            ),
        )
    result = stored.reset_index().sort_values(
        ["date", "ticker"], kind="mergesort", ignore_index=True
    )
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    return result
