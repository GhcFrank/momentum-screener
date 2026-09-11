"""Experimental first-setup-row variant of 顺向火车2."""

from __future__ import annotations

from typing import Final

import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.trend_reacceleration import (
    DEFAULT_CONFIG,
    TrendReaccelerationConfig,
    calculate_trend_reacceleration_features,
)

STRATEGY_ID: Final[str] = "trend_reacceleration_entry"
STRATEGY_VERSION: Final[str] = "1.0"
STRATEGY_NAME: Final[str] = "顺向火车2 · 首次触发（实验）"


def calculate_trend_reacceleration_entry_features(
    frame: pd.DataFrame, *, config: TrendReaccelerationConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Keep base diagnostics; signal only on entry into each setup episode.

    Evaluate the full warmed single-ticker history before any requested-date
    clipping, including the previous ticker row's RPS inputs.
    """
    result = calculate_trend_reacceleration_features(frame, config=config)
    if result.empty:
        return result
    previous_setup = result["setup"].shift(1, fill_value=False)
    result["signal"] = (result["setup"] & ~previous_setup).astype("bool")
    return result
