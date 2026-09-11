"""蓝色钻石: extreme momentum, controlled pullback and proximity to MA20."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from momentum_screener.market_cap_storage import DEFAULT_MARKET_CAP_ROOT
from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    load_universe,
)
from momentum_screener.rps import INVALID_RPS
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

STRATEGY_ID: Final[str] = "blue_diamond"
STRATEGY_VERSION: Final[str] = "1.0"
STRATEGY_NAME: Final[str] = "蓝色钻石"
STRATEGY_DESCRIPTION: Final[str] = (
    "Extreme Momentum + Strong Trend Structure + Controlled Pullback + Pullback to MA20"
)
BLUE_DIAMOND_RPS_LOOKBACKS: Final[tuple[int, int]] = (20, 50)
BLUE_DIAMOND_LOAD_SESSIONS: Final[int] = 280
_STRUCTURE_WINDOW = 20


@dataclass(frozen=True, slots=True)
class BlueDiamondConfig:
    rps20_threshold_strong: float = 98.0
    rps50_threshold_strong: float = 98.0
    rps50_threshold_combined: float = 97.0
    rps_sum_threshold: float = 190.0
    rps50_super_strong: float = 99.0
    pullback_max: float = 0.25
    price_position_250_min: float = 0.80
    ma20_distance_max: float = 1.005
    max_close_below_ma20_days: int = 2
    max_close_below_ma10_days: int = 8
    max_low_below_ma20_days: int = 4
    turnover_max: float = 0.10
    pullback_lookback: int = 20
    price_high_lookback: int = 250
    ma10: int = 10
    ma20: int = 20
    ma50: int = 50
    ma120: int = 120
    ma200: int = 200
    ma250: int = 250

    def __post_init__(self) -> None:
        for name in (
            "pullback_lookback",
            "price_high_lookback",
            "ma10",
            "ma20",
            "ma50",
            "ma120",
            "ma200",
            "ma250",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_close_below_ma20_days",
            "max_close_below_ma10_days",
            "max_low_below_ma20_days",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _STRUCTURE_WINDOW:
                raise ValueError(f"{name} must be within 0..20")
        for name, maximum in (
            ("rps20_threshold_strong", 100),
            ("rps50_threshold_strong", 100),
            ("rps50_threshold_combined", 100),
            ("rps_sum_threshold", 200),
            ("rps50_super_strong", 100),
            ("pullback_max", 1),
            ("price_position_250_min", 1),
            ("turnover_max", 1),
            ("ma20_distance_max", np.inf),
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
        """250 by default: MAs/HHV, own-day MA counts, and dynamic pullback path.

        A high up to N-1 bars ago needs up to N-1 post-anchor pullback values;
        the earliest of those in turn needs N highs (at most 2*N-2 rows).
        """
        return max(
            self.ma50,
            self.ma120,
            self.ma200,
            self.ma250,
            self.price_high_lookback,
            max(self.ma10, self.ma20) + _STRUCTURE_WINDOW - 1,
            self.pullback_lookback,
            2 * self.pullback_lookback - 2,
        )


DEFAULT_CONFIG = BlueDiamondConfig()
BOOLEAN_COLUMNS = (
    "controlled_pullback",
    "no_deep_pullback_since_high",
    "near_250_high",
    "extreme_rps",
    "near_ma20",
    "strong_ma_structure",
    "long_term_trend",
    "normal_turnover",
    "rps20_available",
    "rps50_available",
    "market_cap_available",
    "price_available",
    "history_sufficient",
    "setup",
    "signal",
)
SCREEN_COLUMNS = (
    "date",
    "ticker",
    "raw_close",
    "adj_close",
    "adj_high",
    "adj_low",
    "volume",
    "market_cap",
    "dollar_volume",
    "turnover",
    "rps20",
    "rps50",
    "rps_sum",
    "ma10",
    "ma20",
    "ma50",
    "ma120",
    "ma200",
    "ma250",
    "days_since_high_20",
    "days_since_low",
    "recent_high",
    "recent_low",
    "pullback",
    "high_250_close",
    "price_position_250",
    "ma20_distance",
    "close_below_ma20_count",
    "close_below_ma10_count",
    "low_below_ma20_count",
    "history_session_count",
    *BOOLEAN_COLUMNS,
    "status",
)


class BlueDiamondError(StrategyDataError):
    """Blue Diamond cannot be evaluated with the requested data."""


class BlueDiamondTickerNotFoundError(BlueDiamondError):
    """The ticker is outside the current Universe."""


def _empty_result() -> pd.DataFrame:
    rows = pd.DataFrame(columns=SCREEN_COLUMNS)
    for column in BOOLEAN_COLUMNS:
        rows[column] = rows[column].astype("bool")
    return rows


def extreme_rps_mask(
    rows: pd.DataFrame, *, config: BlueDiamondConfig = DEFAULT_CONFIG
) -> pd.Series:
    """The same exact cheap prefilter used by the full formula and both runners."""
    rps20 = pd.to_numeric(rows["rps20"], errors="coerce")
    rps50 = pd.to_numeric(rows["rps50"], errors="coerce")
    available = rps20.between(0, 100) & rps50.between(0, 100)
    return available & (
        rps50.ge(config.rps50_threshold_strong)
        | rps20.ge(config.rps20_threshold_strong)
        | (
            rps50.ge(config.rps50_threshold_combined)
            & (rps20 + rps50).ge(config.rps_sum_threshold)
        )
    )


def _no_deep_pullback_since_high(
    pullback: pd.Series, anchors: pd.Series, maximum: float
) -> pd.Series:
    """COUNT(pullback > maximum, current anchor offset) == 0, after the high.

    Each path row uses its own historical pullback and anchor. Missing path
    values fail closed; a high today has an empty count window and passes.
    """
    values = pullback.to_numpy(dtype="float64")
    result = np.zeros(len(values), dtype="bool")
    for position, offset in enumerate(anchors.to_numpy(dtype="float64")):
        if (
            not np.isfinite(offset)
            or offset < 0
            or offset > position
            or offset != int(offset)
        ):
            continue
        path = values[position - int(offset) + 1 : position + 1]
        result[position] = np.isfinite(path).all() and not (path > maximum).any()
    return pd.Series(result, index=pullback.index)


def calculate_blue_diamond_features(
    frame: pd.DataFrame, *, config: BlueDiamondConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Full trailing diagnostics for one ticker; adjusted OHLC exactly once.

    MarketCap and RPS can be missing on warmup rows. Turnover uses raw close.
    Every qualifying day signals, without cooldown or cross-day suppression.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Blue Diamond input is missing columns: {missing}")
    if frame.empty:
        return _empty_result()
    result = frame.copy()
    result["date"] = normalize_date_values(result["date"])
    if result["ticker"].isna().any() or result["ticker"].astype(str).nunique() != 1:
        raise ValueError("Blue Diamond features require exactly one ticker")
    result = result.sort_values("date", kind="mergesort", ignore_index=True)
    if result["date"].duplicated().any():
        raise ValueError("Blue Diamond input contains duplicate ticker dates")
    for column in ("rps20", "rps50", "market_cap"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    for column in ("rps20", "rps50"):
        result[f"{column}_available"] = result[column].between(0, 100)
    result["rps_sum"] = result["rps20"] + result["rps50"]
    result["extreme_rps"] = extreme_rps_mask(result, config=config)

    result["raw_close"] = pd.to_numeric(result["close"], errors="coerce").astype(
        "float64"
    )
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce").astype(
        "float64"
    )
    result = add_adjusted_ohlc(result)
    close, high, low = result["adj_close"], result["adj_high"], result["adj_low"]
    for label in (10, 20, 50, 120, 200, 250):
        result[f"ma{label}"] = moving_average(close, getattr(config, f"ma{label}"))

    anchor = bars_since_highest(high, config.pullback_lookback)
    result["days_since_high_20"] = anchor
    result["days_since_low"] = lowest_since_anchor(low, anchor)["bars_since_low"]
    result["recent_high"] = value_at_offset(high, anchor)
    result["recent_low"] = value_at_offset(low, result["days_since_low"])
    result["pullback"] = safe_ratio(
        result["recent_high"] - result["recent_low"], result["recent_high"]
    )
    result["no_deep_pullback_since_high"] = _no_deep_pullback_since_high(
        result["pullback"], anchor, config.pullback_max
    )
    result["controlled_pullback"] = (
        result["pullback"].le(config.pullback_max)
        & result["no_deep_pullback_since_high"]
    )
    result["high_250_close"] = highest_value(close, config.price_high_lookback)
    result["price_position_250"] = safe_ratio(close, result["high_250_close"])
    result["near_250_high"] = result["price_position_250"].gt(
        config.price_position_250_min
    )
    result["ma20_distance"] = safe_ratio(close, result["ma20"])
    result["near_ma20"] = result["ma20_distance"].lt(config.ma20_distance_max)

    for label, values, ma in (
        ("close_below_ma20_count", close, result["ma20"]),
        ("close_below_ma10_count", close, result["ma10"]),
        ("low_below_ma20_count", low, result["ma20"]),
    ):
        result[label] = rolling_count(values.lt(ma), _STRUCTURE_WINDOW).where(
            rolling_every(values.notna() & ma.notna(), _STRUCTURE_WINDOW)
        )
    result["strong_ma_structure"] = (
        result["close_below_ma20_count"].le(config.max_close_below_ma20_days)
        & result["close_below_ma10_count"].le(config.max_close_below_ma10_days)
        & (
            result["low_below_ma20_count"].le(config.max_low_below_ma20_days)
            | result["rps50"].ge(config.rps50_super_strong)
        )
    )
    result["long_term_trend"] = (
        result["ma50"].gt(result["ma120"])
        & result["ma50"].gt(result["ma200"])
        & result["ma50"].gt(result["ma250"])
    )
    turnover = calculate_turnover_columns(result)
    result["market_cap_available"] = turnover["turnover_available"]
    result["dollar_volume"] = turnover["dollar_volume"]
    result["turnover"] = turnover["turnover"]
    result["normal_turnover"] = result["market_cap_available"] & result["turnover"].lt(
        config.turnover_max
    )
    result["price_available"] = result["adjusted_ohlc_valid"]
    result["history_session_count"] = np.arange(1, len(result) + 1)
    result["history_sufficient"] = rolling_every(
        result["adjusted_ohlc_valid"], config.required_price_rows
    )
    result["signal"] = (
        result["controlled_pullback"]
        & result["near_250_high"]
        & result["extreme_rps"]
        & result["near_ma20"]
        & result["strong_ma_structure"]
        & result["long_term_trend"]
        & result["normal_turnover"]
        & result["history_sufficient"]
    )
    result["setup"] = result["signal"]
    result["status"] = np.select(
        [
            ~result["price_available"],
            ~result["history_sufficient"],
            ~(result["rps20_available"] & result["rps50_available"]),
            ~result["market_cap_available"],
        ],
        [
            "price_unavailable",
            "insufficient_history",
            "rps_unavailable",
            "market_cap_unavailable",
        ],
        default="ok",
    )
    for column in BOOLEAN_COLUMNS:
        result[column] = result[column].fillna(False).astype("bool")
    return result


def calculate_blue_diamond_history(
    ticker: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    config: BlueDiamondConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Read exact target observations and warmed price history for one ticker."""
    universe = load_universe(universe_path)
    normalized = normalize_ticker(ticker)
    if normalized is None or normalized not in universe:
        raise BlueDiamondTickerNotFoundError(
            f"Ticker is not in the requested Universe: {ticker!r}"
        )
    start, end = resolve_history_bounds(
        start_date,
        end_date,
        prices_root=prices_root,
        lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS,
    )
    sessions = sessions_in_range(start, end)
    caps = load_strategy_market_cap(
        sessions, root=market_cap_root, universe_path=universe_path
    )
    rps_rows = load_or_calculate_rps(
        sessions,
        lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS,
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
        tickers=(normalized,),
        prices_root=prices_root,
        universe=universe,
        required_sessions=max(BLUE_DIAMOND_LOAD_SESSIONS, config.required_price_rows),
    )
    prepared = merge_prices_and_rps(
        prices, rps_rows, lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS
    )
    features = calculate_blue_diamond_features(
        prepared.merge(caps, on=["date", "ticker"], how="left", validate="one_to_one"),
        config=config,
    )
    result = features.loc[features["date"].between(start, end)].reset_index(drop=True)
    result.attrs.update(
        requested_start=start,
        requested_end=end,
        loaded_price_session_count=loaded_count,
        rps_snapshot_count=len(sessions),
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        market_cap_session_counts=caps.attrs["session_counts"],
    )
    return result


def evaluate_blue_diamond(
    ticker: str,
    as_of_date: date | str,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    config: BlueDiamondConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Return exact-session diagnostics, or an explicit missing-price result."""
    requested = coerce_session_date(as_of_date, lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS)
    history = calculate_blue_diamond_history(
        ticker,
        requested,
        requested,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=rps_root,
        market_cap_root=market_cap_root,
        rps_snapshots=rps_snapshots,
        config=config,
    )
    if not history.empty:
        return history.iloc[0].copy()
    values = dict.fromkeys(SCREEN_COLUMNS, np.nan)
    values.update({column: False for column in BOOLEAN_COLUMNS})
    values.update(
        date=requested,
        ticker=normalize_ticker(ticker),
        rps20=INVALID_RPS,
        rps50=INVALID_RPS,
        history_session_count=0,
        status="price_unavailable",
    )
    return pd.Series(values)


def screen_blue_diamond(
    as_of_date: date | str,
    *,
    signal_only: bool = True,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
    rps_snapshots: pd.DataFrame | None = None,
    market_cap_rows: pd.DataFrame | None = None,
    config: BlueDiamondConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Prefilter exact RPS, then calculate price histories only for candidates.

    signal_only=False selects setup, identical to signal for this strategy.
    Missing ticker caps fail closed; an absent whole-session snapshot raises.
    """
    requested = coerce_session_date(as_of_date, lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS)
    if market_cap_rows is None:
        caps = load_strategy_market_cap(
            (requested,), root=market_cap_root, universe_path=universe_path
        )
        universe = load_universe(universe_path)
    else:
        universe = load_universe(universe_path)
        required_caps = {"date", "ticker", "market_cap"}
        missing_caps = sorted(required_caps.difference(market_cap_rows.columns))
        if missing_caps:
            raise ValueError(f"MarketCap rows are missing columns: {missing_caps}")
        caps = market_cap_rows.loc[:, ["date", "ticker", "market_cap"]].copy()
        caps["date"] = normalize_date_values(caps["date"])
        if caps.empty or not bool(caps["date"].eq(requested).all()):
            raise ValueError("MarketCap rows must match the requested session exactly")
        if caps["ticker"].isna().any():
            raise ValueError("MarketCap rows contain an invalid ticker")
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
        cap_values = pd.to_numeric(caps["market_cap"], errors="coerce")
        if not bool((np.isfinite(cap_values) & cap_values.gt(0)).all()):
            raise ValueError("MarketCap rows contain invalid market_cap values")
        caps["market_cap"] = cap_values.astype("float64")
        session_key = requested.isoformat()
        session_counts = market_cap_rows.attrs.get("session_counts", {})
        counts = (
            session_counts.get(session_key)
            if isinstance(session_counts, dict)
            else None
        )
        if not isinstance(counts, dict):
            counts = {
                "market_cap_available_count": len(caps),
                "market_cap_missing_ticker_count": max(len(universe) - len(caps), 0),
            }
        caps.attrs["session_counts"] = {session_key: counts}
    rps_rows = load_or_calculate_rps(
        (requested,),
        lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
    )
    candidates = tuple(
        rps_rows.loc[extreme_rps_mask(rps_rows, config=config), "ticker"]
    )
    current_rows = []
    loaded_count = 0
    if candidates:
        prices, loaded_count = load_strategy_price_history(
            end_date=requested,
            tickers=candidates,
            prices_root=prices_root,
            universe=universe,
            required_sessions=max(
                BLUE_DIAMOND_LOAD_SESSIONS, config.required_price_rows
            ),
        )
        prepared = merge_prices_and_rps(
            prices, rps_rows, lookbacks=BLUE_DIAMOND_RPS_LOOKBACKS
        )
        prepared = prepared.merge(
            caps, on=["date", "ticker"], how="left", validate="one_to_one"
        )
        for _, ticker_rows in prepared.groupby("ticker", sort=False):
            features = calculate_blue_diamond_features(ticker_rows, config=config)
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
        momentum_candidate_count=len(candidates),
        signal_count=len(result),
        insufficient_history_count=insufficient_count,
        loaded_price_session_count=loaded_count,
        rps_snapshot_count=1,
        momentum_prefilter_used=True,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        **caps.attrs["session_counts"][requested.isoformat()],
    )
    return result
