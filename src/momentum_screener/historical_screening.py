"""Batch retrospective signal replay, separate from notification and backtesting."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.monthly_reversal import (
    MONTHLY_REVERSAL_LOAD_SESSIONS,
    MONTHLY_REVERSAL_REQUIRED_SIGNAL_ROWS,
    MONTHLY_REVERSAL_RPS_LOOKBACKS,
    MONTHLY_REVERSAL_SIGNAL_WINDOW,
    calculate_monthly_reversal_features,
)
from momentum_screener.prices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNIVERSE,
    PriceBackfillError,
    load_universe,
    universe_sha256,
)
from momentum_screener.rps import RpsError
from momentum_screener.rps_storage import DEFAULT_RPS_ROOT, RpsStorageError
from momentum_screener.signal_store import (
    DATA_MODE,
    DEFAULT_SIGNAL_ROOT,
    SignalStoreError,
    replace_signal_range,
)
from momentum_screener.storage_manifest import ManifestError, load_manifest
from momentum_screener.strategy_data import (
    StrategyDataError,
    load_or_calculate_rps,
    load_strategy_price_history,
    merge_prices_and_rps,
    resolve_strategy_sessions,
    sessions_in_range,
)
from momentum_screener.trend_reacceleration import (
    DEFAULT_CONFIG,
    TREND_REACCELERATION_LOAD_SESSIONS,
    TREND_REACCELERATION_RPS_LOOKBACKS,
    TrendReaccelerationConfig,
    calculate_trend_reacceleration_features,
)
from momentum_screener.trend_reacceleration import (
    STRATEGY_ID as TREND_STRATEGY_ID,
)
from momentum_screener.trend_reacceleration import (
    STRATEGY_VERSION as TREND_STRATEGY_VERSION,
)

LOGGER = logging.getLogger(__name__)


class HistoricalScreeningError(RuntimeError):
    """The requested historical interval cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class HistoricalStrategy:
    """Small registry entry describing an existing strategy's batch needs."""

    strategy_id: str
    version: str
    lookbacks: tuple[int, ...]
    required_price_rows: int
    load_sessions: int
    prior_rps_rows: int
    calculate_features: Callable[[pd.DataFrame], pd.DataFrame]
    config: Mapping[str, Any]


SUPPORTED_STRATEGIES: Mapping[str, HistoricalStrategy] = MappingProxyType(
    {
        "monthly_reversal": HistoricalStrategy(
            strategy_id="monthly_reversal",
            version="6.2",
            lookbacks=MONTHLY_REVERSAL_RPS_LOOKBACKS,
            required_price_rows=MONTHLY_REVERSAL_REQUIRED_SIGNAL_ROWS,
            load_sessions=MONTHLY_REVERSAL_LOAD_SESSIONS,
            prior_rps_rows=MONTHLY_REVERSAL_SIGNAL_WINDOW - 1,
            calculate_features=calculate_monthly_reversal_features,
            config=MappingProxyType({}),
        ),
        TREND_STRATEGY_ID: HistoricalStrategy(
            strategy_id=TREND_STRATEGY_ID,
            version=TREND_STRATEGY_VERSION,
            lookbacks=TREND_REACCELERATION_RPS_LOOKBACKS,
            required_price_rows=DEFAULT_CONFIG.required_price_rows,
            load_sessions=TREND_REACCELERATION_LOAD_SESSIONS,
            prior_rps_rows=0,
            calculate_features=calculate_trend_reacceleration_features,
            config=MappingProxyType(asdict(DEFAULT_CONFIG)),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class HistoricalScreeningResult:
    """Summary of a completed run; detailed results are queried from the store."""

    requested_start: date
    requested_end: date | None
    actual_start: date
    actual_end: date
    session_count: int
    universe_count: int
    loaded_price_session_count: int
    rps_snapshot_count: int
    strategy_summaries: Mapping[str, Mapping[str, object]]
    output_store: Path
    generated_at: datetime
    force: bool
    data_mode: str = DATA_MODE

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat()
            if self.requested_end
            else "latest",
            "actual_start": self.actual_start.isoformat(),
            "actual_end": self.actual_end.isoformat(),
            "session_count": self.session_count,
            "universe_count": self.universe_count,
            "loaded_price_session_count": self.loaded_price_session_count,
            "rps_snapshot_count": self.rps_snapshot_count,
            "strategies": dict(self.strategy_summaries),
            "output_store": str(self.output_store),
            "generated_at": self.generated_at.isoformat(),
            "force": self.force,
            "data_mode": self.data_mode,
            "cache_policy": "recompute_replace",
        }


def _date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise HistoricalScreeningError(
            "Historical dates must be dates or YYYY-MM-DD strings"
        )
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalScreeningError("Historical dates must use YYYY-MM-DD") from exc


def _select_strategies(
    strategies: Sequence[str] | str | None,
    config: TrendReaccelerationConfig,
) -> tuple[HistoricalStrategy, ...]:
    ids = (
        tuple(SUPPORTED_STRATEGIES)
        if strategies is None
        else ((strategies,) if isinstance(strategies, str) else tuple(strategies))
    )
    if not ids:
        raise HistoricalScreeningError("Select at least one strategy")
    unknown = sorted(set(ids).difference(SUPPORTED_STRATEGIES))
    if unknown:
        raise HistoricalScreeningError(f"Unsupported strategies: {unknown}")
    registry = dict(SUPPORTED_STRATEGIES)
    registry[TREND_STRATEGY_ID] = replace(
        registry[TREND_STRATEGY_ID],
        required_price_rows=config.required_price_rows,
        calculate_features=partial(
            calculate_trend_reacceleration_features, config=config
        ),
        config=asdict(config),
    )
    return tuple(registry[value] for value in dict.fromkeys(ids))


def _resolve_sessions(
    requested_start: date,
    requested_end: date | None,
    manifest: Mapping[str, Any],
) -> tuple[date, ...]:
    end = min(
        requested_end or _date(manifest["latest_session"]),
        _date(manifest["latest_session"]),
    )
    start = max(requested_start, _date(manifest["actual_min_date"]))
    if requested_end is not None and requested_start > requested_end:
        raise HistoricalScreeningError("start_date cannot be after end_date")
    if start > end:
        raise HistoricalScreeningError(
            "Requested range has no available trading sessions"
        )
    try:
        return sessions_in_range(start, end)
    except (ValueError, RpsError) as exc:
        raise HistoricalScreeningError(
            "Requested range has no available XNYS sessions"
        ) from exc


def run_historical_screening(
    start_date: date | str,
    end_date: date | str | None = None,
    *,
    strategies: Sequence[str] | str | None = None,
    output_store: Path | None = None,
    force: bool = False,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    rps_root: Path | None = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    rps_snapshots: pd.DataFrame | None = None,
    trend_config: TrendReaccelerationConfig = DEFAULT_CONFIG,
) -> HistoricalScreeningResult:
    """Replay any natural-date interval in one batch and safely replace matches.

    Phase 1 ALWAYS recomputes the requested sessions; force=True explicitly
    requests the same policy. None output_store selects data/signals. None or
    'latest' end_date uses the validated price dataset's latest session.
    Inputs are retrospective_latest_data, not a historical Universe backtest.
    """

    selected = _select_strategies(strategies, trend_config)
    requested_start = _date(start_date)
    requested_end = (
        None if end_date is None or end_date == "latest" else _date(end_date)
    )
    manifest = load_manifest(prices_root / "manifest.json")
    sessions = _resolve_sessions(requested_start, requested_end, manifest)
    universe = load_universe(universe_path)
    universe_hash = universe_sha256(universe)
    if manifest["universe_sha256"] != universe_hash or manifest[
        "universe_ticker_count"
    ] != len(universe):
        raise HistoricalScreeningError(
            "Price manifest does not match the requested Universe"
        )

    lookbacks = tuple(sorted({value for item in selected for value in item.lookbacks}))
    prior_rows = max(item.prior_rps_rows for item in selected)
    rps_start = resolve_strategy_sessions(sessions[0], prior_rows + 1)[0]
    warmup = max(max(item.load_sessions, item.required_price_rows) for item in selected)
    warmup = max(warmup, max(lookbacks) + 1)
    LOGGER.info(
        "Historical screening requested=%s..%s actual=%s..%s sessions=%d strategies=%s",
        requested_start,
        requested_end or "latest",
        sessions[0],
        sessions[-1],
        len(sessions),
        ",".join(item.strategy_id for item in selected),
    )
    prices, loaded_count = load_strategy_price_history(
        start_date=rps_start,
        end_date=sessions[-1],
        required_sessions=warmup,
        prices_root=prices_root,
        universe=universe,
    )
    missing_sessions = set(sessions).difference(prices["date"])
    if missing_sessions:
        raise HistoricalScreeningError(
            f"Target sessions have no price data: {sorted(missing_sessions)}"
        )

    # Preserve the actual preceding ticker rows for Monthly Reversal even if
    # a ticker missed market sessions. Its rolling suppression uses row bars.
    if prior_rows:
        active = prices.loc[prices["date"].isin(sessions), "ticker"].unique()
        prior = prices.loc[
            prices["ticker"].isin(active) & prices["date"].lt(sessions[0])
        ]
        prior = prior.groupby("ticker", sort=False).tail(prior_rows)
        if not prior.empty:
            rps_start = min(rps_start, min(prior["date"]))
    rps_sessions = sessions_in_range(rps_start, sessions[-1])
    rps_rows = load_or_calculate_rps(
        rps_sessions,
        lookbacks=lookbacks,
        prices_root=prices_root,
        universe_path=universe_path,
        price_rows=prices,
        universe=universe,
        rps_root=rps_root,
        rps_snapshots=rps_snapshots,
    )
    prepared = merge_prices_and_rps(prices, rps_rows, lookbacks=lookbacks)
    LOGGER.info(
        "Prepared prices once: sessions=%d rows=%d; shared RPS sessions=%d",
        loaded_count,
        len(prices),
        len(rps_sessions),
    )
    matches: dict[str, list[pd.DataFrame]] = {item.strategy_id: [] for item in selected}
    templates: dict[str, pd.DataFrame] = {}
    for index, (_, ticker_rows) in enumerate(
        prepared.groupby("ticker", sort=False), start=1
    ):
        for item in selected:
            features = item.calculate_features(ticker_rows)
            if item.strategy_id not in templates:
                templates[item.strategy_id] = features.iloc[:0].copy()
            signals = features.loc[
                features["date"].isin(sessions) & features["signal"]
            ].copy()
            if not signals.empty:
                matches[item.strategy_id].append(signals)
        if index % 250 == 0:
            LOGGER.info(
                "Calculated full feature histories for %d/%d tickers",
                index,
                len(universe),
            )

    generated_at = datetime.now(UTC)
    signals_by_strategy: dict[str, pd.DataFrame] = {}
    coverage_frames: list[pd.DataFrame] = []
    summaries: dict[str, Mapping[str, object]] = {}
    for item in selected:
        frames = matches[item.strategy_id]
        rows = (
            pd.concat(frames, ignore_index=True)
            if frames
            else templates[item.strategy_id].copy()
        )
        rows = rows.rename(columns={"date": "session"})
        config_json = json.dumps(
            dict(item.config), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        metadata: dict[str, object] = {
            "strategy_id": item.strategy_id,
            "strategy_version": item.version,
            "generated_at": generated_at,
            "config_json": config_json,
            "config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
            "universe_hash": universe_hash,
            "input_fingerprint": None,
            "data_mode": DATA_MODE,
        }
        for column, value in metadata.items():
            rows[column] = value
        signals_by_strategy[item.strategy_id] = rows
        counts = rows.groupby("session").size()
        coverage = pd.DataFrame(
            {
                "session": sessions,
                "status": "complete",
                "signal_count": [int(counts.get(session, 0)) for session in sessions],
            }
        )
        for column, value in metadata.items():
            coverage[column] = value
        coverage_frames.append(coverage)
        summaries[item.strategy_id] = {
            "strategy_version": item.version,
            "sessions_evaluated": len(sessions),
            "signals": len(rows),
            "config_hash": metadata["config_hash"],
        }
    root = DEFAULT_SIGNAL_ROOT if output_store is None else Path(output_store)
    replace_signal_range(
        signals_by_strategy, pd.concat(coverage_frames, ignore_index=True), root=root
    )
    return HistoricalScreeningResult(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=sessions[0],
        actual_end=sessions[-1],
        session_count=len(sessions),
        universe_count=len(universe),
        loaded_price_session_count=loaded_count,
        rps_snapshot_count=len(rps_sessions),
        strategy_summaries=summaries,
        output_store=root,
        generated_at=generated_at,
        force=force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for bounded historical replay; print counts rather than ticker lists."""

    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.historical_screening",
        description="Batch historical screening into a derived Parquet signal store.",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument(
        "--end-date", default="latest", help="YYYY-MM-DD or latest (default)"
    )
    parser.add_argument(
        "--strategy", action="append", choices=tuple(SUPPORTED_STRATEGIES)
    )
    parser.add_argument("--output-store", type=Path, default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rps-root", type=Path, default=DEFAULT_RPS_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicit recompute; Phase 1 always recomputes the interval",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run_historical_screening(
            args.start_date,
            args.end_date,
            strategies=args.strategy,
            output_store=args.output_store,
            prices_root=args.prices_root,
            rps_root=args.rps_root,
            universe_path=args.universe,
            force=args.force,
        )
    except (
        HistoricalScreeningError,
        SignalStoreError,
        PriceBackfillError,
        RpsError,
        RpsStorageError,
        StrategyDataError,
        ManifestError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("Historical screening failed: %s", exc)
        return 1
    print("Historical screening complete")
    print(f"requested: {result.requested_start} -> {result.requested_end or 'latest'}")
    print(
        f"actual sessions: {result.actual_start} -> {result.actual_end} ({result.session_count})"
    )
    for strategy_id, summary in result.strategy_summaries.items():
        print(
            f"{strategy_id} {summary['strategy_version']}: sessions evaluated={summary['sessions_evaluated']}, signals={summary['signals']}"
        )
    print(f"store: {result.output_store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
