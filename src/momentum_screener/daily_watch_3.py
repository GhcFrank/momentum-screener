"""Daily Watch 3: strong RPS near long-term adjusted price highs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.market_cap_storage import DEFAULT_MARKET_CAP_ROOT
from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    load_universe,
)
from momentum_screener.rps_storage import DEFAULT_RPS_ROOT
from momentum_screener.strategy_data import (
    StrategyDataError,
    calculate_turnover_columns,
    coerce_session_date,
    load_or_calculate_rps,
    load_strategy_market_cap,
    load_strategy_price_history,
    merge_prices_and_rps,
    normalize_date_values,
)
from momentum_screener.technical_features import (
    add_adjusted_ohlc,
    highest_value,
    rolling_count,
    rolling_every,
    safe_ratio,
)
from momentum_screener.universe import normalize_ticker

CORE_STRATEGY_ID: Final[str] = "daily_watch_3_core"
CORE_STRATEGY_VERSION: Final[str] = "1.0"
CORE_STRATEGY_NAME: Final[str] = "每日观察选股3 Core"

STRATEGY_ID: Final[str] = "daily_watch_3"
STRATEGY_VERSION: Final[str] = "1.0"
STRATEGY_NAME: Final[str] = "每日观察选股3"
STRATEGY_DESCRIPTION: Final[str] = (
    "Strong RPS + Long-term High Price Position + MarketCap Turnover Proxy < 20%"
)
RPS_LOOKBACKS: Final[tuple[int, int, int]] = (50, 120, 250)
LOAD_SESSIONS: Final[int] = 270


@dataclass(frozen=True, slots=True)
class DailyWatch3Config:
    recent_high_days: int = 5
    long_high_lookback: int = 250
    breakout_rps_threshold: float = 95.99
    breakout_rps120_rps50_threshold: float = 94.99
    near_high_ratio_1: float = 0.85
    near_high_rps_threshold_1: float = 96.99
    near_high_ratio_2: float = 0.70
    near_high_rps_threshold_2: float = 97.99
    max_turnover_proxy: float = 0.20

    def __post_init__(self) -> None:
        for name in ("recent_high_days", "long_high_lookback"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "breakout_rps_threshold",
            "breakout_rps120_rps50_threshold",
            "near_high_rps_threshold_1",
            "near_high_rps_threshold_2",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{name} must be finite and within 0..100")
        for name in ("near_high_ratio_1", "near_high_ratio_2", "max_turnover_proxy"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within 0..1")

    @property
    def required_price_rows(self) -> int:
        """Rows needed for five independently complete 250-row rolling highs."""

        return self.long_high_lookback + self.recent_high_days - 1


DEFAULT_CONFIG = DailyWatch3Config()
BOOLEAN_COLUMNS = (
    "rps_available",
    "price_available",
    "history_sufficient",
    "is_250d_close_high",
    "recent_250d_close_high",
    "rps_condition_1a",
    "rps_condition_1b",
    "high_breakout_ok",
    "near_high_85_ok",
    "near_high_70_ok",
    "core_signal",
    "market_cap_available",
    "turnover_market_cap_proxy_available",
    "normal_turnover",
    "setup",
    "signal",
)
SCREEN_COLUMNS = (
    "date",
    "ticker",
    "raw_close",
    "adj_close",
    "adj_high",
    "volume",
    "rps50",
    "rps120",
    "rps250",
    "high_close_250",
    "high_250_intraday",
    "price_position_250_high",
    "is_250d_close_high",
    "recent_250d_close_high",
    "rps_condition_1a",
    "rps_condition_1b",
    "high_breakout_ok",
    "near_high_85_ok",
    "near_high_70_ok",
    "core_signal",
    "market_cap",
    "market_cap_available",
    "dollar_volume",
    "turnover_market_cap_proxy",
    "turnover_market_cap_proxy_available",
    "normal_turnover",
    "rps_available",
    "price_available",
    "history_session_count",
    "history_sufficient",
    "setup",
    "signal",
    "status",
)


class DailyWatch3Error(StrategyDataError):
    """Daily Watch 3 cannot be evaluated with the requested data."""


def _empty_result() -> pd.DataFrame:
    rows = pd.DataFrame(columns=SCREEN_COLUMNS)
    for column in BOOLEAN_COLUMNS:
        rows[column] = rows[column].astype("bool")
    return rows


def daily_watch_3_rps_mask(
    rows: pd.DataFrame, *, config: DailyWatch3Config = DEFAULT_CONFIG
) -> pd.Series:
    """Return the exact union of RPS branches used by all three modules."""

    required = {"rps50", "rps120", "rps250"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Daily Watch 3 RPS input is missing columns: {missing}")
    rps50 = pd.to_numeric(rows["rps50"], errors="coerce")
    rps120 = pd.to_numeric(rows["rps120"], errors="coerce")
    rps250 = pd.to_numeric(rows["rps250"], errors="coerce")
    available = rps50.between(0, 100) & rps120.between(0, 100) & rps250.between(0, 100)
    return available & (
        rps120.gt(config.breakout_rps_threshold)
        | rps250.gt(config.breakout_rps_threshold)
        | (
            rps120.gt(config.breakout_rps120_rps50_threshold)
            & rps50.gt(config.breakout_rps120_rps50_threshold)
        )
    )


def calculate_daily_watch_3_features(
    frame: pd.DataFrame, *, config: DailyWatch3Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Calculate the shared Core and formal signal diagnostics once per ticker.

    Technical conditions use adjusted OHLC. ``turnover_market_cap_proxy`` is
    explicitly a MarketCap approximation, calculated from raw Close and raw
    Volume. It is not float-share turnover.
    """

    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Daily Watch 3 input is missing columns: {missing}")
    if frame.empty:
        return _empty_result()

    result = frame.copy()
    result["date"] = normalize_date_values(result["date"])
    if result["ticker"].isna().any() or result["ticker"].astype(str).nunique() != 1:
        raise ValueError("Daily Watch 3 features require exactly one ticker")
    result = result.sort_values("date", kind="mergesort", ignore_index=True)
    if result["date"].duplicated().any():
        raise ValueError("Daily Watch 3 input contains duplicate ticker dates")

    for column in ("rps50", "rps120", "rps250", "market_cap"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    result["rps_available"] = (
        result["rps50"].between(0, 100)
        & result["rps120"].between(0, 100)
        & result["rps250"].between(0, 100)
    )

    result["raw_close"] = pd.to_numeric(result["close"], errors="coerce").astype(
        "float64"
    )
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce").astype(
        "float64"
    )
    result = add_adjusted_ohlc(result)
    close = result["adj_close"]
    high = result["adj_high"]

    result["high_close_250"] = highest_value(close, config.long_high_lookback)
    result["is_250d_close_high"] = result["high_close_250"].notna() & close.eq(
        result["high_close_250"]
    )
    complete_recent_highs = rolling_every(
        result["high_close_250"].notna(), config.recent_high_days
    )
    result["recent_250d_close_high"] = (
        rolling_count(result["is_250d_close_high"], config.recent_high_days).ge(1)
        & complete_recent_highs
    )

    result["rps_condition_1a"] = result["rps_available"] & (
        result["rps120"].gt(config.breakout_rps_threshold)
        | result["rps250"].gt(config.breakout_rps_threshold)
    )
    result["rps_condition_1b"] = (
        result["rps_available"]
        & result["rps120"].gt(config.breakout_rps120_rps50_threshold)
        & result["rps50"].gt(config.breakout_rps120_rps50_threshold)
    )
    result["high_breakout_ok"] = result["recent_250d_close_high"] & (
        result["rps_condition_1a"] | result["rps_condition_1b"]
    )

    result["high_250_intraday"] = highest_value(high, config.long_high_lookback)
    result["price_position_250_high"] = safe_ratio(close, result["high_250_intraday"])
    result["near_high_85_ok"] = (
        result["rps_available"]
        & result["price_position_250_high"].ge(config.near_high_ratio_1)
        & (
            result["rps120"].gt(config.near_high_rps_threshold_1)
            | result["rps250"].gt(config.near_high_rps_threshold_1)
        )
    )
    result["near_high_70_ok"] = (
        result["rps_available"]
        & result["price_position_250_high"].ge(config.near_high_ratio_2)
        & (
            result["rps120"].gt(config.near_high_rps_threshold_2)
            | result["rps250"].gt(config.near_high_rps_threshold_2)
        )
    )

    result["price_available"] = result["adjusted_ohlc_valid"]
    result["history_session_count"] = np.arange(1, len(result) + 1)
    result["history_sufficient"] = rolling_every(
        result["adjusted_ohlc_valid"], config.required_price_rows
    )
    result["core_signal"] = (
        result["history_sufficient"]
        & result["rps_available"]
        & (
            result["high_breakout_ok"]
            | result["near_high_85_ok"]
            | result["near_high_70_ok"]
        )
    )

    turnover = calculate_turnover_columns(result)
    result["market_cap_available"] = np.isfinite(result["market_cap"]) & result[
        "market_cap"
    ].gt(0)
    result["dollar_volume"] = turnover["dollar_volume"]
    result["turnover_market_cap_proxy"] = turnover["turnover"]
    result["turnover_market_cap_proxy_available"] = turnover["turnover_available"]
    result["normal_turnover"] = result["turnover_market_cap_proxy_available"] & result[
        "turnover_market_cap_proxy"
    ].lt(config.max_turnover_proxy)
    result["signal"] = result["core_signal"] & result["normal_turnover"]
    result["setup"] = result["signal"]
    result["status"] = np.select(
        [
            ~result["price_available"],
            ~result["history_sufficient"],
            ~result["rps_available"],
            ~result["market_cap_available"],
            ~result["turnover_market_cap_proxy_available"],
        ],
        [
            "price_unavailable",
            "insufficient_history",
            "rps_unavailable",
            "market_cap_unavailable",
            "turnover_market_cap_proxy_unavailable",
        ],
        default="ok",
    )
    for column in BOOLEAN_COLUMNS:
        result[column] = result[column].fillna(False).astype("bool")
    return result


def calculate_daily_watch_3_core_features(
    frame: pd.DataFrame, *, config: DailyWatch3Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Select the turnover-free research signal from the shared diagnostics."""

    result = calculate_daily_watch_3_features(frame, config=config)
    if result.empty:
        return result
    result["signal"] = result["core_signal"]
    result["setup"] = result["core_signal"]
    result.loc[
        result["status"].isin(
            ["market_cap_unavailable", "turnover_market_cap_proxy_unavailable"]
        ),
        "status",
    ] = "ok"
    return result


def _injected_market_cap_rows(
    rows: pd.DataFrame, *, session: date, universe: tuple[str, ...]
) -> pd.DataFrame:
    required = {"date", "ticker", "market_cap"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"MarketCap rows are missing columns: {missing}")
    caps = rows.loc[:, ["date", "ticker", "market_cap"]].copy()
    caps["date"] = normalize_date_values(caps["date"])
    if caps.empty or not bool(caps["date"].eq(session).all()):
        raise ValueError("MarketCap rows must match the requested session exactly")
    caps["ticker"] = caps["ticker"].map(normalize_ticker).astype("string")
    if caps["ticker"].isna().any():
        raise ValueError("MarketCap rows contain an invalid ticker")
    if bool(caps.duplicated(["date", "ticker"]).any()):
        raise ValueError("MarketCap rows contain duplicate date/ticker keys")
    unknown = sorted(set(caps["ticker"]).difference(universe))
    if unknown:
        raise ValueError(
            f"MarketCap rows contain tickers outside the Universe: {unknown}"
        )
    values = pd.to_numeric(caps["market_cap"], errors="coerce")
    if not bool((np.isfinite(values) & values.gt(0)).all()):
        raise ValueError("MarketCap rows contain invalid market_cap values")
    caps["market_cap"] = values.astype("float64")
    session_key = session.isoformat()
    source_counts = rows.attrs.get("session_counts", {})
    counts = source_counts.get(session_key) if isinstance(source_counts, dict) else None
    if not isinstance(counts, dict):
        counts = {
            "market_cap_available_count": len(caps),
            "market_cap_missing_ticker_count": max(len(universe) - len(caps), 0),
        }
    caps.attrs["session_counts"] = {session_key: counts}
    return caps


def screen_daily_watch_3(
    as_of_date: date | str,
    *,
    signal_only: bool = True,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    market_cap_rows: pd.DataFrame | None = None,
    config: DailyWatch3Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Screen the formal signal with shared RPS and exact-session MarketCap."""

    requested = coerce_session_date(as_of_date, lookbacks=RPS_LOOKBACKS)
    universe = load_universe(universe_path)
    caps = (
        load_strategy_market_cap(
            (requested,), root=market_cap_root, universe_path=universe_path
        )
        if market_cap_rows is None
        else _injected_market_cap_rows(
            market_cap_rows, session=requested, universe=universe
        )
    )
    rps_rows = load_or_calculate_rps(
        (requested,),
        lookbacks=RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
    )
    candidates = tuple(rps_rows.loc[daily_watch_3_rps_mask(rps_rows), "ticker"])
    current_rows: list[pd.DataFrame] = []
    loaded_count = 0
    if candidates:
        prices, loaded_count = load_strategy_price_history(
            end_date=requested,
            tickers=candidates,
            prices_root=prices_root,
            universe=universe,
            required_sessions=max(LOAD_SESSIONS, config.required_price_rows),
        )
        prepared = merge_prices_and_rps(prices, rps_rows, lookbacks=RPS_LOOKBACKS)
        prepared = prepared.merge(
            caps, on=["date", "ticker"], how="left", validate="one_to_one"
        )
        for _, ticker_rows in prepared.groupby("ticker", sort=False):
            features = calculate_daily_watch_3_features(ticker_rows, config=config)
            current_rows.append(features.loc[features["date"].eq(requested)])

    result = _empty_result()
    insufficient_count = 0
    if current_rows:
        current = pd.concat(current_rows, ignore_index=True)
        insufficient_count = int(current["status"].eq("insufficient_history").sum())
        result = current.loc[
            current["signal" if signal_only else "setup"], SCREEN_COLUMNS
        ].sort_values("ticker", ignore_index=True)
    result.attrs.update(
        as_of_date=requested,
        universe_count=len(universe),
        rps_candidate_count=len(candidates),
        signal_count=len(result),
        insufficient_history_count=insufficient_count,
        loaded_price_session_count=loaded_count,
        rps_snapshot_count=1,
        rps_prefilter_used=True,
        turnover_definition="raw_close * volume / point_in_time_market_cap",
        turnover_is_market_cap_proxy=True,
        turnover_is_float_share_turnover=False,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        **caps.attrs["session_counts"][requested.isoformat()],
    )
    return result
