"""顺向火车2: Strong Momentum + Healthy Pullback + Trend Re-acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    load_universe,
)
from momentum_screener.rps import INVALID_RPS
from momentum_screener.rps_storage import DEFAULT_RPS_ROOT
from momentum_screener.strategy_data import (
    StrategyDataError,
    coerce_session_date,
    load_or_calculate_rps,
    load_strategy_price_history,
    merge_prices_and_rps,
    normalize_date_values,
    resolve_history_bounds,
    sessions_in_range,
)
from momentum_screener.technical_features import (
    add_adjusted_ohlc,
    bars_since_highest,
    highest_value,
    lowest_since_anchor,
    moving_average,
    rolling_count,
    rolling_every,
    safe_ratio,
    value_at_offset,
)
from momentum_screener.universe import normalize_ticker

STRATEGY_ID: Final[str] = "trend_reacceleration"
STRATEGY_VERSION: Final[str] = "1.0"
STRATEGY_NAME: Final[str] = "顺向火车2"
STRATEGY_DESCRIPTION: Final[str] = (
    "Strong Momentum + Healthy Pullback + Trend Re-acceleration"
)
TREND_REACCELERATION_RPS_LOOKBACKS: Final[tuple[int, int]] = (120, 250)
TREND_REACCELERATION_LOAD_SESSIONS: Final[int] = 320


@dataclass(frozen=True, slots=True)
class TrendReaccelerationConfig:
    """Formula parameters; diagnostic names retain their default window labels."""

    rps_sum_threshold: float = 185.0
    ma_long_count_window: int = 30
    ma_long_min_days: int = 25
    short_count_window: int = 4
    short_min_days: int = 3
    recent_high_lookback: int = 20
    max_drawdown: float = 0.25
    year_high_lookback: int = 250
    year_high_ratio_min: float = 0.80
    ma_trend_confirm_days: int = 5

    def __post_init__(self) -> None:
        for name in (
            "ma_long_count_window",
            "ma_long_min_days",
            "short_count_window",
            "short_min_days",
            "recent_high_lookback",
            "year_high_lookback",
            "ma_trend_confirm_days",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ma_long_min_days > self.ma_long_count_window:
            raise ValueError("ma_long_min_days cannot exceed ma_long_count_window")
        if self.short_min_days > self.short_count_window:
            raise ValueError("short_min_days cannot exceed short_count_window")
        for name, maximum in (
            ("rps_sum_threshold", 200),
            ("max_drawdown", 1),
            ("year_high_ratio_min", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} must be finite and within 0..{maximum}")

    @property
    def required_price_rows(self) -> int:
        """MA250 then COUNT(30) needs 250 + 30 - 1 = 279 ticker rows."""

        return max(
            250 + self.ma_long_count_window - 1,
            20 + self.short_count_window - 1,
            20 + self.ma_trend_confirm_days,
            self.year_high_lookback,
            self.recent_high_lookback,
        )


DEFAULT_CONFIG = TrendReaccelerationConfig()

BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "momentum_ok",
    "close_above_ma20",
    "above_ma250_30",
    "above_ma200_30",
    "above_ma10_4",
    "above_ma20_4",
    "trend_ok",
    "drawdown_ok",
    "no_deep_drawdown_since_high",
    "near_250d_high",
    "pullback_ok",
    "ma20_non_decreasing_5",
    "ma10_above_ma20_5",
    "ma10_rising",
    "ma20_rising",
    "ma10_above_ma20",
    "reacceleration_ok",
    "setup",
    "signal",
    "rps120_available",
    "rps250_available",
    "price_available",
    "history_sufficient",
)
SCREEN_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "adj_close",
    "adj_high",
    "adj_low",
    "rps120",
    "rps250",
    "rps_sum",
    "ma10",
    "ma20",
    "ma200",
    "ma250",
    "above_ma250_count",
    "above_ma200_count",
    "above_ma10_count",
    "above_ma20_count",
    "days_since_20d_high",
    "recent_high",
    "days_since_low",
    "recent_low",
    "drawdown",
    "annual_high_ratio",
    "history_session_count",
    *BOOLEAN_COLUMNS,
    "status",
)


class TrendReaccelerationError(StrategyDataError):
    """A Trend Re-acceleration query cannot run safely."""


class TrendReaccelerationTickerNotFoundError(TrendReaccelerationError):
    """The requested ticker is not in the selected Universe."""


def _empty_result() -> pd.DataFrame:
    result = pd.DataFrame(columns=SCREEN_COLUMNS)
    for column in BOOLEAN_COLUMNS:
        result[column] = result[column].astype("bool")
    return result


def calculate_trend_reacceleration_features(
    frame: pd.DataFrame, *, config: TrendReaccelerationConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Calculate full diagnostics from one ticker's persisted raw + adj OHLC.

    All OHLC formulae use add_adjusted_ohlc exactly once. Windows use trailing
    ticker rows without filling missing values. Incomplete formula histories
    cannot generate setup/signal. Signal equals setup on EVERY qualifying day;
    no prior-signal suppression is performed.
    """

    required = {"date", "ticker", "open", "high", "low", "close", "adj_close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Trend Re-acceleration input is missing columns: {missing}")
    if frame.empty:
        return _empty_result()
    result = frame.copy()
    result["date"] = normalize_date_values(result["date"])
    if result["ticker"].isna().any() or result["ticker"].astype(str).nunique() != 1:
        raise ValueError("Trend Re-acceleration features require exactly one ticker")
    result = result.sort_values("date", kind="mergesort", ignore_index=True)
    if bool(result["date"].duplicated().any()):
        raise ValueError("Trend Re-acceleration input contains duplicate ticker dates")
    for lookback in TREND_REACCELERATION_RPS_LOOKBACKS:
        column = f"rps{lookback}"
        if column not in result:
            result[column] = np.nan
        values = pd.to_numeric(result[column], errors="coerce").astype("float64")
        result[column] = values
        result[f"{column}_available"] = np.isfinite(values) & values.between(0, 100)
    result["rps_sum"] = result["rps120"] + result["rps250"]
    result["momentum_ok"] = (
        result["rps_sum"].gt(config.rps_sum_threshold)
        & result["rps120_available"]
        & result["rps250_available"]
    )

    result = add_adjusted_ohlc(result)
    close, high, low = result["adj_close"], result["adj_high"], result["adj_low"]
    for window in (10, 20, 200, 250):
        result[f"ma{window}"] = moving_average(close, window)
    for ma_window, count_window, minimum, label in (
        (250, config.ma_long_count_window, config.ma_long_min_days, "above_ma250_30"),
        (200, config.ma_long_count_window, config.ma_long_min_days, "above_ma200_30"),
        (10, config.short_count_window, config.short_min_days, "above_ma10_4"),
        (20, config.short_count_window, config.short_min_days, "above_ma20_4"),
    ):
        ma = result[f"ma{ma_window}"]
        count = rolling_count(close.gt(ma), count_window).where(
            rolling_every(close.notna() & ma.notna(), count_window)
        )
        result[f"above_ma{ma_window}_count"] = count
        result[label] = count.ge(minimum)
    result["close_above_ma20"] = close.gt(result["ma20"])
    result["trend_ok"] = (
        result["close_above_ma20"]
        & result["above_ma250_30"]
        & result["above_ma200_30"]
        & (result["above_ma10_4"] | result["above_ma20_4"])
    )

    anchor = bars_since_highest(high, config.recent_high_lookback)
    result["days_since_20d_high"] = anchor
    result["recent_high"] = value_at_offset(high, anchor)
    low_path = lowest_since_anchor(low, anchor)
    result["recent_low"] = low_path["lowest_value"]
    result["days_since_low"] = low_path["bars_since_low"]
    result["drawdown"] = safe_ratio(
        result["recent_high"] - result["recent_low"], result["recent_high"]
    ).mask(anchor.eq(0), 0.0)
    result["drawdown_ok"] = result["drawdown"].le(config.max_drawdown)
    # recent_low already measures the entire path's maximum drawdown.
    result["no_deep_drawdown_since_high"] = result["drawdown_ok"]
    result["annual_high_ratio"] = safe_ratio(
        close, highest_value(close, config.year_high_lookback)
    )
    result["near_250d_high"] = result["annual_high_ratio"].gt(
        config.year_high_ratio_min
    )
    result["pullback_ok"] = (
        result["drawdown_ok"]
        & result["no_deep_drawdown_since_high"]
        & result["near_250d_high"]
    )

    ma10, ma20 = result["ma10"], result["ma20"]
    result["ma20_non_decreasing_5"] = rolling_every(
        ma20.ge(ma20.shift(1)), config.ma_trend_confirm_days
    )
    result["ma10_above_ma20_5"] = rolling_every(
        ma10.ge(ma20), config.ma_trend_confirm_days
    )
    result["ma10_rising"] = ma10.gt(ma10.shift(1))
    result["ma20_rising"] = ma20.gt(ma20.shift(1))
    result["ma10_above_ma20"] = ma10.ge(ma20)
    result["reacceleration_ok"] = (
        result["ma20_non_decreasing_5"]
        & result["ma10_above_ma20_5"]
        & result["ma10_rising"]
        & result["ma20_rising"]
        & result["ma10_above_ma20"]
    )
    result["history_session_count"] = np.arange(1, len(result) + 1, dtype="int64")
    result["history_sufficient"] = rolling_every(
        result["adjusted_ohlc_valid"], config.required_price_rows
    )
    result["price_available"] = result["adjusted_ohlc_valid"]
    result["setup"] = (
        result["momentum_ok"]
        & result["trend_ok"]
        & result["pullback_ok"]
        & result["reacceleration_ok"]
        & result["history_sufficient"]
    )
    result["signal"] = result["setup"]
    result["status"] = np.select(
        [
            ~result["price_available"],
            ~result["history_sufficient"],
            ~(result["rps120_available"] & result["rps250_available"]),
        ],
        ["invalid_adjusted_ohlc", "insufficient_history", "rps_unavailable"],
        default="evaluated",
    )
    for column in BOOLEAN_COLUMNS:
        result[column] = result[column].fillna(False).astype("bool")
    return result


def calculate_trend_reacceleration_history(
    ticker: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    config: TrendReaccelerationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return a warmed, explainable ticker history over an inclusive date range."""

    universe = load_universe(universe_path)
    normalized = normalize_ticker(ticker)
    if normalized is None or normalized not in universe:
        raise TrendReaccelerationTickerNotFoundError(
            f"Ticker is not in the requested Universe: {ticker!r}"
        )
    try:
        start, end = resolve_history_bounds(
            start_date,
            end_date,
            prices_root=prices_root,
            lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS,
        )
    except StrategyDataError as exc:
        raise TrendReaccelerationError(str(exc)) from exc
    sessions = sessions_in_range(start, end)
    rps_rows = load_or_calculate_rps(
        sessions,
        lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
        result_tickers=(normalized,),
    )
    prices, loaded_count = load_strategy_price_history(
        start_date=start,
        end_date=end,
        required_sessions=max(
            TREND_REACCELERATION_LOAD_SESSIONS, config.required_price_rows
        ),
        tickers=(normalized,),
        prices_root=prices_root,
        universe=universe,
    )
    features = calculate_trend_reacceleration_features(
        merge_prices_and_rps(
            prices, rps_rows, lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS
        ),
        config=config,
    )
    result = features.loc[
        features["date"].ge(start) & features["date"].le(end)
    ].reset_index(drop=True)
    result.attrs.update(
        {
            "requested_start": start,
            "requested_end": end,
            "loaded_price_session_count": loaded_count,
            "rps_snapshot_count": len(sessions),
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
        }
    )
    return result


def evaluate_trend_reacceleration(
    ticker: str,
    as_of_date: date | str,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    config: TrendReaccelerationConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Evaluate one exact session, including unavailable/insufficient states."""

    requested = coerce_session_date(
        as_of_date, lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS
    )
    history = calculate_trend_reacceleration_history(
        ticker,
        requested,
        requested,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
        config=config,
    )
    if not history.empty:
        return history.iloc[0].copy()
    values: dict[str, object] = dict.fromkeys(SCREEN_COLUMNS, np.nan)
    values.update({column: False for column in BOOLEAN_COLUMNS})
    values.update(
        {
            "date": requested,
            "ticker": normalize_ticker(ticker) or ticker,
            "rps120": INVALID_RPS,
            "rps250": INVALID_RPS,
            "history_session_count": 0,
            "status": "price_unavailable",
        }
    )
    return pd.Series(values)


def screen_trend_reacceleration(
    as_of_date: date | str,
    *,
    signal_only: bool = True,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    config: TrendReaccelerationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Screen the Universe with an exact momentum prefilter and full diagnostics.

    signal_only=False selects setup, which equals signal for this strategy.
    Insufficient histories and missing current price rows are skipped.
    """

    requested = coerce_session_date(
        as_of_date, lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS
    )
    universe = load_universe(universe_path)
    rps_rows = load_or_calculate_rps(
        (requested,),
        lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
    )
    momentum = (rps_rows["rps120"] + rps_rows["rps250"]).gt(config.rps_sum_threshold)
    momentum &= rps_rows["rps120"].between(0, 100) & rps_rows["rps250"].between(0, 100)
    candidates = tuple(rps_rows.loc[momentum, "ticker"])
    current_rows: list[pd.Series] = []
    loaded_count = 0
    if candidates:
        prices, loaded_count = load_strategy_price_history(
            end_date=requested,
            tickers=candidates,
            prices_root=prices_root,
            universe=universe,
            required_sessions=max(
                TREND_REACCELERATION_LOAD_SESSIONS, config.required_price_rows
            ),
        )
        merged = merge_prices_and_rps(
            prices, rps_rows, lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS
        )
        for _, ticker_prices in merged.groupby("ticker", sort=False):
            features = calculate_trend_reacceleration_features(
                ticker_prices, config=config
            )
            current = features.loc[features["date"].eq(requested)]
            if not current.empty:
                current_rows.append(current.iloc[0])
    result = _empty_result()
    insufficient_count = 0
    if current_rows:
        current = pd.DataFrame(current_rows)
        insufficient_count = int(current["status"].eq("insufficient_history").sum())
        selected = current.loc[current["signal" if signal_only else "setup"]]
        result = selected.loc[:, SCREEN_COLUMNS].sort_values(
            "ticker", kind="mergesort", ignore_index=True
        )
    result.attrs.update(
        {
            "as_of_date": requested,
            "universe_count": len(universe),
            "momentum_candidate_count": len(candidates),
            "signal_count": len(result),
            "insufficient_history_count": insufficient_count,
            "loaded_price_session_count": loaded_count,
            "rps_snapshot_count": 1,
            "momentum_prefilter_used": True,
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
        }
    )
    return result
