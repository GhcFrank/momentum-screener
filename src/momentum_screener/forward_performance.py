"""Signal-independent, signal-close-relative forward High/Low performance.

Uses canonical raw Close/High/Low, without adjustment or persistence. These
research outcomes use future prices and must never be fed into signal rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_screener.local_price_data import (
    LocalFile,
    price_files_in_range,
    read_local_price_rows,
    require_unchanged_files,
)
from momentum_screener.prices import DEFAULT_OUTPUT_ROOT
from momentum_screener.storage_manifest import load_manifest
from momentum_screener.strategy_data import normalize_date_values
from momentum_screener.universe import normalize_ticker

FORWARD_WINDOWS = (40, 120)
FORWARD_METRIC_COLUMNS = tuple(
    f"forward_{window}d_{metric}"
    for window in FORWARD_WINDOWS
    for metric in ("max_drawdown", "max_gain")
)


@dataclass(frozen=True)
class ForwardPerformance:
    forward_40d_max_drawdown: float | None = None
    forward_40d_max_gain: float | None = None
    forward_120d_max_drawdown: float | None = None
    forward_120d_max_gain: float | None = None


def _occurrences(signals: pd.DataFrame) -> pd.DataFrame:
    """Use the existing result keys, accepting signal_date as a public alias."""
    date_column = next(
        (name for name in ("signal_date", "session", "date") if name in signals),
        None,
    )
    if "ticker" not in signals or date_column is None:
        raise ValueError(
            "Forward performance requires ticker and signal_date/session/date"
        )
    rows = (
        signals.loc[:, ["ticker", date_column]]
        .rename(columns={date_column: "signal_date"})
        .copy()
    )
    rows["signal_date"] = normalize_date_values(rows["signal_date"])
    rows["ticker"] = rows["ticker"].map(normalize_ticker)
    if rows["ticker"].isna().any():
        raise ValueError("Forward performance requires valid tickers")
    return rows.drop_duplicates().sort_values(
        ["signal_date", "ticker"], ignore_index=True
    )


def calculate_forward_performance_for_signals(
    signals: pd.DataFrame,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    price_rows: pd.DataFrame | None = None,
    files: tuple[LocalFile, ...] | None = None,
) -> pd.DataFrame:
    """Return one numeric result per unique ticker/signal_date, regardless of strategy.

    Windows contain the first up to 40/120 available ticker trading rows strictly
    after the signal, never the signal bar. A partial window is valid. Missing or
    invalid exact-date Close, or no future bars, yields unavailable metrics.

    Each relevant local partition is read once for all tickers. Supplying already
    loaded canonical price_rows avoids I/O. Optional files is the complete cache
    identity from price_files_in_range: manifest first, then yearly Parquet files.
    """
    occurrences = _occurrences(signals)
    if occurrences.empty:
        return occurrences.reindex(
            columns=["ticker", "signal_date", *FORWARD_METRIC_COLUMNS]
        )
    if price_rows is None:
        sources = (
            files
            if files is not None
            else price_files_in_range(
                min(occurrences["signal_date"]), prices_root=prices_root
            )
        )
        if not sources:
            raise ValueError("Forward performance requires a local price manifest")
        require_unchanged_files(sources)
        manifest = load_manifest(Path(sources[0].path))
        price_rows = read_local_price_rows(
            tuple(occurrences["ticker"].unique()),
            min(occurrences["signal_date"]),
            date.fromisoformat(manifest["latest_session"]),
            sources[1:],
            columns=("close", "high", "low"),
        )
        require_unchanged_files(sources)

    required = ["date", "ticker", "close", "high", "low"]
    missing = sorted(set(required).difference(price_rows.columns))
    if missing:
        raise ValueError(f"Forward price rows are missing columns: {missing}")
    prices = price_rows.loc[:, required].copy()
    prices["date"] = normalize_date_values(prices["date"])
    prices["ticker"] = prices["ticker"].map(normalize_ticker)
    prices = prices.loc[prices["ticker"].isin(occurrences["ticker"])]
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Forward prices contain duplicate ticker dates")
    for column in ("close", "high", "low"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce").astype(
            "float64"
        )
    prices = prices.sort_values(["ticker", "date"], ignore_index=True)
    grouped = {ticker: frame for ticker, frame in prices.groupby("ticker", sort=False)}
    results = []
    for ticker, signal_date in occurrences.itertuples(index=False, name=None):
        metrics = asdict(ForwardPerformance())
        history = grouped.get(ticker)
        if history is not None:
            dates = history["date"].to_numpy()
            position = int(np.searchsorted(dates, signal_date))
            if position < len(history) and dates[position] == signal_date:
                base = history.iloc[position]["close"]
                if np.isfinite(base) and base > 0:
                    for window in FORWARD_WINDOWS:
                        future = history.iloc[position + 1 : position + 1 + window]
                        if future.empty:
                            continue
                        for field, metric, operation in (
                            ("low", "max_drawdown", "min"),
                            ("high", "max_gain", "max"),
                        ):
                            values = future[field]
                            # Canonical prices are positive and finite. Corrupt
                            # windows stay unavailable instead of silently dropping bars.
                            if not (np.isfinite(values) & values.gt(0)).all():
                                continue
                            with np.errstate(
                                over="ignore", divide="ignore", invalid="ignore"
                            ):
                                value = float(getattr(values, operation)() / base - 1)
                            if np.isfinite(value):
                                metrics[f"forward_{window}d_{metric}"] = value
        results.append({"ticker": ticker, "signal_date": signal_date, **metrics})
    result = pd.DataFrame(results)
    # DataFrame unavailable values are NaN; the single-occurrence API uses None.
    result[list(FORWARD_METRIC_COLUMNS)] = result[list(FORWARD_METRIC_COLUMNS)].astype(
        "float64"
    )
    return result


def calculate_signal_forward_performance(
    ticker: str,
    signal_date: date | str,
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    price_rows: pd.DataFrame | None = None,
) -> ForwardPerformance:
    """Convenience API returning four float-or-None values for one occurrence."""
    rows = calculate_forward_performance_for_signals(
        pd.DataFrame({"ticker": [ticker], "signal_date": [signal_date]}),
        prices_root=prices_root,
        price_rows=price_rows,
    )
    return ForwardPerformance(
        **{
            column: None
            if pd.isna(rows.iloc[0][column])
            else float(rows.iloc[0][column])
            for column in FORWARD_METRIC_COLUMNS
        }
    )
