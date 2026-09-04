"""Point-in-time implementation of the Monthly Reversal 6.2 strategy."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Final

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    PriceUpdateError,
    load_universe,
    read_affected_partitions,
)
from momentum_screener.rps import (
    INVALID_RPS,
    RPS_CALENDAR_NAME,
    InvalidRpsSessionError,
    calculate_rps_snapshots,
    resolve_rps_session_dates,
)
from momentum_screener.storage_manifest import load_manifest
from momentum_screener.technical_features import (
    add_adjusted_ohlc,
    highest_value,
    lowest_value,
    moving_average,
    rolling_count,
    safe_ratio,
)
from momentum_screener.universe import normalize_ticker

MONTHLY_REVERSAL_RPS_LOOKBACKS: Final[tuple[int, int]] = (50, 120)
MONTHLY_REVERSAL_SIGNAL_WINDOW: Final[int] = 15
MONTHLY_REVERSAL_REQUIRED_PRICE_ROWS: Final[int] = 250
MONTHLY_REVERSAL_REQUIRED_SIGNAL_ROWS: Final[int] = 264
# Load extra XNYS sessions so occasional missing ticker rows do not turn a
# sufficient history into an artificial partial window.  Indicators themselves
# still use their exact formula windows.
MONTHLY_REVERSAL_LOAD_SESSIONS: Final[int] = 320

_PRICE_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
)
FYX_BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "fyx11",
    "fyx12",
    "fyx1",
    "fyx130",
    "fyx131",
    "fyx13",
    "fyx21",
    "fyx22",
    "fyx23",
    "fyx2",
    "nh80",
    "fyx31",
    "fyx32",
    "fyx3",
    "fyx4",
    "nn200",
    "lnn200",
    "fyx51",
    "fyx52",
    "fyx5",
    "fyx601",
    "fyx602",
    "fyx603",
    "fyx61",
    "fyx62",
    "fyx63",
    "fyx6",
    "fyx71",
    "fyx72",
    "fyx73",
    "fyx7",
    "yxfz",
    "signal",
)
SCREEN_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "rps50",
    "rps120",
    "fyx1",
    "fyx2",
    "fyx3",
    "fyx4",
    "fyx5",
    "fyx6",
    "fyx7",
    "yxfz",
    "signal",
    "status",
)


class MonthlyReversalError(RuntimeError):
    """Base error for a Monthly Reversal query that cannot run safely."""


class MonthlyReversalTickerNotFoundError(MonthlyReversalError):
    """Raised when a requested ticker is not in the selected Universe."""


def calculate_yxfz(frame: pd.DataFrame) -> pd.Series:
    """Return ``FYX1 AND ... AND FYX7``, treating unavailable values as false."""

    columns = [f"fyx{index}" for index in range(1, 8)]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"YXFZ input is missing columns: {missing}")
    return frame.loc[:, columns].fillna(False).astype("bool").all(axis="columns")


def calculate_monthly_reversal_signal(
    yxfz: pd.Series,
    *,
    history_sufficient: pd.Series | None = None,
) -> pd.Series:
    """Return true when current YXFZ is the first true in the last 15 rows."""

    current = yxfz.fillna(False).astype("bool")
    prior_yxfz = (
        current.astype("int64")
        .shift(1)
        .rolling(
            window=MONTHLY_REVERSAL_SIGNAL_WINDOW - 1,
            min_periods=MONTHLY_REVERSAL_SIGNAL_WINDOW - 1,
        )
        .sum()
    )
    signal = current & prior_yxfz.eq(0)
    if history_sufficient is not None:
        signal &= history_sufficient.fillna(False).astype("bool")
    return signal.fillna(False).astype("bool")


def calculate_monthly_reversal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the complete FYX/YXFZ/signal history for one ticker.

    Rows must contain persisted v2 OHLC, plus point-in-time ``rps50`` and
    ``rps120`` where available.  Every rolling calculation is backward-looking,
    includes the current row, and requires a complete ticker-row window.
    """

    missing = sorted(set(_PRICE_INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly Reversal input is missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["date"] = _normalize_date_values(result["date"])
    tickers = result["ticker"].dropna().astype(str).unique()
    if len(tickers) != 1:
        raise ValueError("Monthly Reversal features require exactly one ticker")
    result = result.sort_values("date", kind="mergesort", ignore_index=True)
    if bool(result["date"].duplicated().any()):
        raise ValueError("Monthly Reversal input contains duplicate ticker dates")
    for column in ("rps50", "rps120"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )

    result = add_adjusted_ohlc(result)
    close = result["adj_close"]
    high = result["adj_high"]
    low = result["adj_low"]

    for window in (20, 120, 200, 250):
        result[f"ma{window}"] = moving_average(close, window)
    for window in (5, 10, 30, 50, 80, 120):
        result[f"hhv_h_{window}"] = highest_value(high, window)
    for window in (50, 70):
        result[f"hhv_c_{window}"] = highest_value(close, window)
    for window in (10, 20, 30, 50, 120, 200):
        result[f"llv_l_{window}"] = lowest_value(low, window)

    rps50 = result["rps50"]
    rps120 = result["rps120"]
    result["rps50_available"] = np.isfinite(rps50) & rps50.ge(0)
    result["rps120_available"] = np.isfinite(rps120) & rps120.ge(0)

    # FYX1 and the related high-RPS/new-high branch.
    result["fyx11"] = rps50.gt(87)
    result["fyx12"] = rps120.gt(90)
    result["fyx1"] = result["fyx11"] | result["fyx12"]
    result["fyx130"] = rps50.ge(90) | rps120.ge(90)
    result["fyx131"] = close.ge(result["hhv_c_70"])
    result["fyx13"] = result["fyx130"] & result["fyx131"]

    # FYX2: rising lows / compact structure.
    result["fyx21"] = result["llv_l_50"].gt(result["llv_l_200"]) & result["fyx13"]
    result["fyx22"] = result["llv_l_30"].gt(result["llv_l_120"]) & result["fyx13"]
    result["fyx23"] = result["llv_l_20"].gt(result["llv_l_50"]) & result["llv_l_10"].gt(
        result["llv_l_20"]
    )
    result["fyx2"] = result["fyx21"] | result["fyx22"] | result["fyx23"]

    # FYX3: first create each day's own NH80, then count its last ten rows.
    nh80_available = high.notna() & result["hhv_h_80"].notna()
    result["nh80"] = high.ge(result["hhv_h_80"])
    result["fyx31"] = rolling_count(result["nh80"], 10).gt(0) & rolling_count(
        nh80_available, 10
    ).eq(10)
    result["fyx32"] = (
        close.ge(result["hhv_c_50"]) | high.ge(result["hhv_h_50"])
    ) & result["fyx130"]
    result["fyx3"] = result["fyx31"] | result["fyx32"]

    # FYX4: trend position, with a guarded positive MA200 denominator.
    ma120_ma200 = safe_ratio(result["ma120"], result["ma200"])
    result["fyx4"] = (
        close.gt(result["ma20"]) & close.gt(result["ma200"]) & ma120_ma200.gt(0.9)
    )

    # FYX5: each historical comparison uses that row's own MA200.
    nn200_available = close.notna() & result["ma200"].notna()
    result["nn200"] = close.gt(result["ma200"])
    result["aa200"] = rolling_count(result["nn200"], 45).where(
        rolling_count(nn200_available, 45).eq(45)
    )
    result["fyx51"] = result["aa200"].gt(2) & result["aa200"].lt(45)
    lnn200_available = low.notna() & result["ma200"].notna()
    result["lnn200"] = low.lt(result["ma200"])
    result["laa200"] = rolling_count(result["lnn200"], 45).where(
        rolling_count(lnn200_available, 45).eq(45)
    )
    result["fyx52"] = result["laa200"].gt(0) & result["aa200"].gt(2)
    result["fyx5"] = result["fyx51"] | result["fyx52"]

    # FYX6: long trend and platform width.
    ma120_ref15 = result["ma120"].shift(15)
    ma200_ref15 = result["ma200"].shift(15)
    result["fyx601"] = result["ma120"].ge(ma120_ref15) | result["ma200"].ge(ma200_ref15)
    result["fyx602"] = result["ma120"].ge(ma120_ref15) & result["ma200"].ge(ma200_ref15)
    result["fyx603"] = result["ma120"].gt(result["ma200"]) & result["ma200"].gt(
        result["ma250"]
    )
    platform_ratio = safe_ratio(result["hhv_h_30"], result["llv_l_120"])
    result["fyx61"] = platform_ratio.lt(1.50) & result["fyx601"]
    result["fyx62"] = platform_ratio.lt(1.60) & result["fyx602"]
    result["fyx63"] = platform_ratio.lt(1.75) & result["fyx603"] & result["fyx13"]
    result["fyx6"] = result["fyx61"] | result["fyx62"] | result["fyx63"]

    # FYX7: proximity to prior highs, with positive guarded denominators.
    near_120_ratio = safe_ratio(result["hhv_h_5"], result["hhv_h_120"])
    close_10_ratio = safe_ratio(close, result["hhv_h_10"])
    result["fyx71"] = near_120_ratio.gt(0.85)
    result["fyx72"] = near_120_ratio.gt(0.80) & result["fyx13"]
    result["fyx73"] = close_10_ratio.gt(0.90)
    result["fyx7"] = (result["fyx71"] | result["fyx72"]) & result["fyx73"]

    result["yxfz"] = calculate_yxfz(result)

    result["history_session_count"] = np.arange(1, len(result) + 1, dtype="int64")
    result["history_sufficient"] = (
        result["history_session_count"].ge(MONTHLY_REVERSAL_REQUIRED_PRICE_ROWS)
        & result["ma250"].notna()
        & result["llv_l_200"].notna()
        & result["hhv_h_120"].notna()
    )
    result["signal_history_sufficient"] = (
        result["history_sufficient"]
        .astype("int64")
        .rolling(
            window=MONTHLY_REVERSAL_SIGNAL_WINDOW,
            min_periods=MONTHLY_REVERSAL_SIGNAL_WINDOW,
        )
        .sum()
        .eq(MONTHLY_REVERSAL_SIGNAL_WINDOW)
    )
    result["signal"] = calculate_monthly_reversal_signal(
        result["yxfz"],
        history_sufficient=result["signal_history_sufficient"],
    )

    result["price_available"] = result["adjusted_ohlc_valid"]
    no_rps = ~(result["rps50_available"] | result["rps120_available"])
    result["status"] = np.select(
        [
            ~result["adjusted_ohlc_valid"],
            ~result["history_sufficient"],
            no_rps,
        ],
        ["invalid_adjusted_ohlc", "insufficient_history", "rps_unavailable"],
        default="evaluated",
    )
    for column in FYX_BOOLEAN_COLUMNS:
        result[column] = result[column].fillna(False).astype("bool")
    return result


def calculate_monthly_reversal_history(
    ticker: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return complete explainable strategy rows for one ticker and date range.

    The returned range is warmed with older price rows and 14 older RPS
    snapshots, so its first row can be evaluated without look-ahead or a
    truncated BARSSINCEN-style window.
    """

    universe = load_universe(universe_path)
    normalized_ticker = normalize_ticker(ticker)
    if normalized_ticker is None or normalized_ticker not in universe:
        raise MonthlyReversalTickerNotFoundError(
            f"Ticker is not in the requested Universe: {ticker!r}"
        )
    requested_start, requested_end = _resolve_history_bounds(
        start_date,
        end_date,
        prices_root=prices_root,
    )
    requested_sessions = _sessions_in_range(requested_start, requested_end)
    evaluation_sessions = _sessions_ending_at(
        requested_end,
        len(requested_sessions) + MONTHLY_REVERSAL_SIGNAL_WINDOW - 1,
    )
    price_rows, loaded_session_count = _load_price_rows(
        start_date=evaluation_sessions[0],
        end_date=requested_end,
        prices_root=prices_root,
        universe=universe,
    )
    rps_rows = _get_rps_rows(
        evaluation_sessions,
        prices_root=prices_root,
        universe_path=universe_path,
        price_rows=price_rows,
        universe=universe,
        rps_snapshots=rps_snapshots,
        result_tickers=(normalized_ticker,),
    )
    ticker_prices = price_rows.loc[
        price_rows["ticker"].eq(normalized_ticker)
        & price_rows["date"].le(requested_end)
    ].copy()
    feature_input = _merge_prices_and_rps(ticker_prices, rps_rows)
    features = calculate_monthly_reversal_features(feature_input)
    result = features.loc[
        features["date"].ge(requested_start) & features["date"].le(requested_end)
    ].reset_index(drop=True)
    result.attrs.update(
        {
            "requested_start": requested_start,
            "requested_end": requested_end,
            "loaded_price_session_count": loaded_session_count,
            "rps_snapshot_count": len(evaluation_sessions),
        }
    )
    return result


def evaluate_monthly_reversal(
    ticker: str,
    as_of_date: date | str,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_snapshots: pd.DataFrame | None = None,
) -> pd.Series:
    """Return one stock's full Monthly Reversal diagnosis on one session."""

    requested = _coerce_session_date(as_of_date)
    history = calculate_monthly_reversal_history(
        ticker,
        start_date=requested,
        end_date=requested,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_snapshots=rps_snapshots,
    )
    matching = history.loc[history["date"].eq(requested)]
    if not matching.empty:
        return matching.iloc[0].copy()
    normalized = normalize_ticker(ticker)
    return _unavailable_explanation(normalized or str(ticker), requested)


def screen_monthly_reversal(
    as_of_date: date | str,
    *,
    signal_only: bool = True,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Screen the complete Universe, using current FYX1 as a safe prefilter."""

    requested = _coerce_session_date(as_of_date)
    universe = load_universe(universe_path)
    evaluation_sessions = _sessions_ending_at(
        requested,
        MONTHLY_REVERSAL_SIGNAL_WINDOW,
    )
    price_rows, loaded_session_count = _load_price_rows(
        start_date=evaluation_sessions[0],
        end_date=requested,
        prices_root=prices_root,
        universe=universe,
    )
    rps_rows = _get_rps_rows(
        evaluation_sessions,
        prices_root=prices_root,
        universe_path=universe_path,
        price_rows=price_rows,
        universe=universe,
        rps_snapshots=rps_snapshots,
    )
    current_rps = rps_rows.loc[rps_rows["date"].eq(requested)].set_index("ticker")
    fyx1_mask = current_rps["rps50"].gt(87) | current_rps["rps120"].gt(90)
    candidates = tuple(current_rps.index[fyx1_mask])

    current_rows: list[pd.Series] = []
    if candidates:
        candidate_prices = price_rows.loc[price_rows["ticker"].isin(candidates)]
        for ticker, ticker_prices in candidate_prices.groupby("ticker", sort=False):
            ticker_rps = rps_rows.loc[rps_rows["ticker"].eq(ticker)]
            feature_input = _merge_prices_and_rps(ticker_prices, ticker_rps)
            features = calculate_monthly_reversal_features(feature_input)
            current = features.loc[features["date"].eq(requested)]
            if not current.empty:
                current_rows.append(current.iloc[0])

    if current_rows:
        all_current = pd.DataFrame(current_rows)
        filter_column = "signal" if signal_only else "yxfz"
        selected = all_current.loc[all_current[filter_column]].copy()
        selected = selected.sort_values("ticker", kind="mergesort")
        result = selected.loc[:, SCREEN_COLUMNS].reset_index(drop=True)
        yxfz_count = int(all_current["yxfz"].sum())
        signal_count = int(all_current["signal"].sum())
    else:
        result = pd.DataFrame(columns=SCREEN_COLUMNS)
        yxfz_count = 0
        signal_count = 0

    result.attrs.update(
        {
            "as_of_date": requested,
            "universe_count": len(universe),
            "fyx1_candidate_count": len(candidates),
            "yxfz_count": yxfz_count,
            "signal_count": signal_count,
            "loaded_price_session_count": loaded_session_count,
            "rps_snapshot_count": len(evaluation_sessions),
            "fyx1_prefilter_used": True,
        }
    )
    return result


def _merge_prices_and_rps(
    prices: pd.DataFrame,
    rps_rows: pd.DataFrame,
) -> pd.DataFrame:
    left = prices.drop(columns=["rps50", "rps120"], errors="ignore").copy()
    right = rps_rows.loc[:, ["date", "ticker", "rps50", "rps120"]]
    return left.merge(
        right,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )


def _get_rps_rows(
    sessions: Iterable[date],
    *,
    prices_root: Path,
    universe_path: Path,
    price_rows: pd.DataFrame,
    universe: tuple[str, ...],
    rps_snapshots: pd.DataFrame | None,
    result_tickers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    session_tuple = tuple(sessions)
    if rps_snapshots is None:
        snapshots = calculate_rps_snapshots(
            session_tuple,
            lookbacks=MONTHLY_REVERSAL_RPS_LOOKBACKS,
            prices_root=prices_root,
            universe_path=universe_path,
            price_rows=price_rows,
            universe=universe,
            result_tickers=result_tickers,
        )
    else:
        snapshots = rps_snapshots.copy()
    return _normalize_rps_rows(
        snapshots,
        requested_sessions=session_tuple,
        result_tickers=result_tickers,
    )


def _normalize_rps_rows(
    snapshots: pd.DataFrame,
    *,
    requested_sessions: tuple[date, ...],
    result_tickers: tuple[str, ...] | None,
) -> pd.DataFrame:
    date_column = "as_of_date" if "as_of_date" in snapshots else "date"
    required = {date_column, "ticker", "rps50", "rps120"}
    missing = sorted(required.difference(snapshots.columns))
    if missing:
        raise ValueError(f"RPS snapshots are missing columns: {missing}")
    result = snapshots.loc[:, [date_column, "ticker", "rps50", "rps120"]].copy()
    result = result.rename(columns={date_column: "date"})
    result["date"] = _normalize_date_values(result["date"])
    result["ticker"] = result["ticker"].astype(str)
    for column in ("rps50", "rps120"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    result = result.loc[result["date"].isin(requested_sessions)]
    if result_tickers is not None:
        result = result.loc[result["ticker"].isin(result_tickers)]
    if bool(result.duplicated(["date", "ticker"]).any()):
        raise ValueError("RPS snapshots contain duplicate date/ticker rows")
    return result.sort_values(["date", "ticker"], kind="mergesort").reset_index(
        drop=True
    )


def _load_price_rows(
    *,
    start_date: date,
    end_date: date,
    prices_root: Path,
    universe: tuple[str, ...],
) -> tuple[pd.DataFrame, int]:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    start_session = calendar.date_to_session(pd.Timestamp(start_date), direction="none")
    start_index = int(calendar.sessions.get_loc(start_session))
    load_index = max(0, start_index - (MONTHLY_REVERSAL_LOAD_SESSIONS - 1))
    load_start = calendar.sessions[load_index].date()

    available_years = sorted(
        int(path.parent.name.removeprefix("year="))
        for path in (prices_root / "daily").glob("year=*/prices.parquet")
        if path.parent.name.removeprefix("year=").isdigit()
    )
    if not available_years or end_date.year not in available_years:
        raise PriceUpdateError(
            f"No complete price partition is available through {end_date.isoformat()}"
        )
    first_year = max(load_start.year, available_years[0])
    years = tuple(range(first_year, end_date.year + 1))
    rows = read_affected_partitions(prices_root, years, tickers=universe)
    rows = rows.loc[rows["date"].le(end_date)].copy()
    return rows, int(rows["date"].nunique())


def _resolve_history_bounds(
    start_date: date | str | None,
    end_date: date | str | None,
    *,
    prices_root: Path,
) -> tuple[date, date]:
    if start_date is None or end_date is None:
        manifest = load_manifest(prices_root / "manifest.json")
    else:
        manifest = None
    start_value = (
        start_date
        if start_date is not None
        else str(manifest["actual_min_date"] if manifest is not None else "")
    )
    end_value = (
        end_date
        if end_date is not None
        else str(manifest["latest_session"] if manifest is not None else "")
    )
    start = _coerce_session_date(start_value)
    end = _coerce_session_date(end_value)
    if start > end:
        raise MonthlyReversalError("start_date cannot be after end_date")
    return start, end


def _coerce_session_date(value: date | str) -> date:
    # Reuse the RPS module's exact XNYS validation and date coercion policy.
    return resolve_rps_session_dates(
        value,
        lookbacks=MONTHLY_REVERSAL_RPS_LOOKBACKS,
    ).as_of_date


def _sessions_in_range(start_date: date, end_date: date) -> tuple[date, ...]:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
    result = tuple(value.date() for value in sessions)
    if not result:
        raise InvalidRpsSessionError("Monthly Reversal range has no XNYS sessions")
    return result


def _sessions_ending_at(end_date: date, count: int) -> tuple[date, ...]:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    session = calendar.date_to_session(pd.Timestamp(end_date), direction="none")
    end_index = int(calendar.sessions.get_loc(session))
    start_index = max(0, end_index - count + 1)
    return tuple(
        value.date() for value in calendar.sessions[start_index : end_index + 1]
    )


def _normalize_date_values(values: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(values, errors="coerce")
    if bool(normalized.isna().any()):
        raise ValueError("Monthly Reversal input contains invalid dates")
    if normalized.dt.tz is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.dt.date


def _unavailable_explanation(ticker: str, requested: date) -> pd.Series:
    values: dict[str, object] = {
        "date": requested,
        "ticker": ticker,
        "rps50": INVALID_RPS,
        "rps120": INVALID_RPS,
        "price_available": False,
        "rps50_available": False,
        "rps120_available": False,
        "history_session_count": 0,
        "history_sufficient": False,
        "signal_history_sufficient": False,
        "status": "price_unavailable",
    }
    values.update({column: False for column in FYX_BOOLEAN_COLUMNS})
    return pd.Series(values)
