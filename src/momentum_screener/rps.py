"""Rebuild cross-sectional RPS metrics from the daily-price dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    load_universe,
    read_affected_partitions,
)
from momentum_screener.universe import normalize_ticker

RPS_LOOKBACKS: Final[tuple[int, ...]] = (50, 120, 250)
RPS_CALENDAR_NAME: Final[str] = "XNYS"
RPS_PRICE_FIELD: Final[str] = "adj_close"
INVALID_RPS: Final[float] = -1.0


class RpsError(RuntimeError):
    """Base error for an RPS query that cannot be evaluated safely."""


class InvalidRpsSessionError(RpsError):
    """Raised when the requested date is not an XNYS trading session."""


class InsufficientRpsHistoryError(RpsError):
    """Raised when the shared market calendar cannot provide every lookback."""


class InvalidRpsLookbackError(RpsError):
    """Raised when configured RPS lookbacks are empty, invalid, or duplicated."""


class RpsTickerNotFoundError(RpsError):
    """Raised when a single-stock query is outside the requested Universe."""


@dataclass(frozen=True, slots=True)
class RpsSessionDates:
    """One shared set of market-session dates for an RPS cross section."""

    as_of_date: date
    base_dates: Mapping[int, date]

    @property
    def rps120_base_date(self) -> date:
        """Return the requested 120-session base date for compatibility."""

        return self.base_dates[120]

    @property
    def rps250_base_date(self) -> date:
        """Return the requested 250-session base date for compatibility."""

        return self.base_dates[250]


def _normalize_lookbacks(lookbacks: Iterable[int]) -> tuple[int, ...]:
    """Validate configured horizons and return them in deterministic order."""

    try:
        values = tuple(lookbacks)
    except TypeError as exc:
        raise InvalidRpsLookbackError(
            "RPS lookbacks must be an iterable of positive integers"
        ) from exc
    if not values:
        raise InvalidRpsLookbackError("RPS lookbacks must contain at least one horizon")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidRpsLookbackError(
                "RPS lookbacks must contain only integers, excluding bool"
            )
        if value <= 0:
            raise InvalidRpsLookbackError(
                "RPS lookbacks must contain only positive integers"
            )
    if len(set(values)) != len(values):
        raise InvalidRpsLookbackError(
            "RPS lookbacks must not contain duplicate horizons"
        )
    return tuple(sorted(values))


def _coerce_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise InvalidRpsSessionError(
            "as_of_date must be a date or an ISO YYYY-MM-DD string"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise InvalidRpsSessionError(
                f"as_of_date must use ISO YYYY-MM-DD format: {value!r}"
            ) from exc
    raise InvalidRpsSessionError(
        "as_of_date must be a date or an ISO YYYY-MM-DD string"
    )


def resolve_rps_session_dates(
    as_of_date: date | str,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
) -> RpsSessionDates:
    """Resolve requested trading-session base dates from the XNYS index."""

    normalized_lookbacks = _normalize_lookbacks(lookbacks)
    requested = _coerce_date(as_of_date)
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    try:
        session = calendar.date_to_session(pd.Timestamp(requested), direction="none")
    except ValueError as exc:
        raise InvalidRpsSessionError(
            f"as_of_date is not a valid {RPS_CALENDAR_NAME} trading session: "
            f"{requested.isoformat()}"
        ) from exc
    current_index = int(calendar.sessions.get_loc(session))
    longest_lookback = max(normalized_lookbacks)
    if current_index < longest_lookback:
        raise InsufficientRpsHistoryError(
            f"Insufficient {RPS_CALENDAR_NAME} session history before "
            f"{requested.isoformat()}: need {longest_lookback} prior sessions"
        )
    base_dates = {
        lookback: calendar.sessions[current_index - lookback].date()
        for lookback in normalized_lookbacks
    }
    return RpsSessionDates(
        as_of_date=session.date(), base_dates=MappingProxyType(base_dates)
    )


def calculate_cross_sectional_rps(
    *,
    current_prices: pd.Series,
    base_prices: pd.Series,
    universe: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate returns and normalized average ranks for one lookback."""

    index = pd.Index(universe, name="ticker")
    current = pd.to_numeric(current_prices.reindex(index), errors="coerce").astype(
        "float64"
    )
    base = pd.to_numeric(base_prices.reindex(index), errors="coerce").astype("float64")
    valid = (
        current.notna()
        & base.notna()
        & np.isfinite(current)
        & np.isfinite(base)
        & current.gt(0)
        & base.gt(0)
    )
    returns = current.div(base).sub(1.0)
    valid &= np.isfinite(returns)
    returns = returns.where(valid)

    rps = pd.Series(INVALID_RPS, index=index, dtype="float64")
    valid_returns = returns.loc[valid]
    valid_count = len(valid_returns)
    if valid_count == 1:
        rps.loc[valid_returns.index] = 100.0
    elif valid_count > 1:
        average_ranks = valid_returns.rank(method="average", ascending=True)
        rps.loc[valid_returns.index] = (
            100.0 * (average_ranks - 1.0) / (valid_count - 1.0)
        )
    return pd.DataFrame({"return": returns, "rps": rps}, index=index)


def _prices_on_date(prices: pd.DataFrame, session_date: date) -> pd.Series:
    rows = prices.loc[prices["date"].eq(session_date), ["ticker", RPS_PRICE_FIELD]]
    if rows.empty:
        return pd.Series(dtype="float64", name=RPS_PRICE_FIELD)
    return rows.set_index("ticker")[RPS_PRICE_FIELD]


def _calculate_snapshot_from_prices(
    *,
    session_dates: RpsSessionDates,
    prices: pd.DataFrame,
    universe: tuple[str, ...],
) -> pd.DataFrame:
    """Build one snapshot from an already loaded, validated price frame."""

    normalized_lookbacks = tuple(session_dates.base_dates)
    current_prices = _prices_on_date(prices, session_dates.as_of_date)
    metrics_by_lookback = {
        lookback: calculate_cross_sectional_rps(
            current_prices=current_prices,
            base_prices=_prices_on_date(prices, base_date),
            universe=universe,
        )
        for lookback, base_date in session_dates.base_dates.items()
    }

    index = pd.Index(universe, name="ticker")
    snapshot = pd.DataFrame(index=index)
    snapshot["ticker"] = index
    snapshot["as_of_date"] = session_dates.as_of_date
    for lookback in normalized_lookbacks:
        snapshot[f"rps{lookback}"] = metrics_by_lookback[lookback]["rps"]
    for lookback in normalized_lookbacks:
        snapshot[f"return_{lookback}"] = metrics_by_lookback[lookback]["return"]
    for lookback in normalized_lookbacks:
        snapshot[f"rps{lookback}_base_date"] = session_dates.base_dates[lookback]
    return snapshot


def calculate_rps_snapshot(
    as_of_date: date | str,
    *,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> pd.DataFrame:
    """Calculate cross-sectional RPS for configured session lookbacks."""

    session_dates = resolve_rps_session_dates(as_of_date, lookbacks=lookbacks)
    universe = load_universe(universe_path)
    years = tuple(
        sorted(
            {
                session_dates.as_of_date.year,
                *(base_date.year for base_date in session_dates.base_dates.values()),
            }
        )
    )
    prices = read_affected_partitions(prices_root, years, tickers=universe)
    return _calculate_snapshot_from_prices(
        session_dates=session_dates,
        prices=prices,
        universe=universe,
    )


def calculate_rps_snapshots(
    as_of_dates: Iterable[date | str],
    *,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    price_rows: pd.DataFrame | None = None,
    universe: tuple[str, ...] | None = None,
    result_tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate several RPS cross sections with one shared price read.

    ``price_rows`` and ``universe`` allow an orchestrator that already loaded
    validated v2 partitions to reuse those rows.  Rankings always use the
    complete Universe; ``result_tickers`` only projects the returned rows.
    """

    normalized_lookbacks = _normalize_lookbacks(lookbacks)
    requested_dates = tuple(as_of_dates)
    if not requested_dates:
        raise InvalidRpsSessionError("as_of_dates must contain at least one session")
    resolved_dates = tuple(
        resolve_rps_session_dates(value, lookbacks=normalized_lookbacks)
        for value in requested_dates
    )
    canonical_dates = tuple(item.as_of_date for item in resolved_dates)
    if len(set(canonical_dates)) != len(canonical_dates):
        raise InvalidRpsSessionError("as_of_dates must not contain duplicates")
    if canonical_dates != tuple(sorted(canonical_dates)):
        order = np.argsort(np.asarray(canonical_dates, dtype="datetime64[D]"))
        resolved_dates = tuple(resolved_dates[int(index)] for index in order)

    complete_universe = (
        universe if universe is not None else load_universe(universe_path)
    )
    if not complete_universe:
        raise RpsError("RPS Universe must contain at least one ticker")
    if price_rows is None:
        years = tuple(
            sorted(
                {item.as_of_date.year for item in resolved_dates}
                | {
                    base_date.year
                    for item in resolved_dates
                    for base_date in item.base_dates.values()
                }
            )
        )
        prices = read_affected_partitions(
            prices_root,
            years,
            tickers=complete_universe,
        )
    else:
        required = {"date", "ticker", RPS_PRICE_FIELD}
        missing = sorted(required.difference(price_rows.columns))
        if missing:
            raise RpsError(f"Preloaded RPS price rows are missing columns: {missing}")
        prices = price_rows

    projection: tuple[str, ...] | None = None
    if result_tickers is not None:
        projection = tuple(result_tickers)
        unknown = sorted(set(projection).difference(complete_universe))
        if unknown:
            raise RpsTickerNotFoundError(
                f"Requested result tickers are outside the Universe: {unknown}"
            )

    frames: list[pd.DataFrame] = []
    for session_dates in resolved_dates:
        snapshot = _calculate_snapshot_from_prices(
            session_dates=session_dates,
            prices=prices,
            universe=complete_universe,
        )
        if projection is not None:
            snapshot = snapshot.loc[list(projection)]
        frames.append(snapshot.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def get_stock_rps(
    ticker: str,
    as_of_date: date | str,
    *,
    lookbacks: Iterable[int] = RPS_LOOKBACKS,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
) -> pd.Series:
    """Return one row from the complete-Universe RPS snapshot."""

    normalized = normalize_ticker(ticker)
    if normalized is None:
        raise RpsTickerNotFoundError(f"Invalid ticker for RPS query: {ticker!r}")
    snapshot = calculate_rps_snapshot(
        as_of_date,
        lookbacks=lookbacks,
        prices_root=prices_root,
        universe_path=universe_path,
    )
    if normalized not in snapshot.index:
        raise RpsTickerNotFoundError(
            f"Ticker is not in the requested Universe: {normalized}"
        )
    return snapshot.loc[normalized].copy()
