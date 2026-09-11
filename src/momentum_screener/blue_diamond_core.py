"""Research-only Blue Diamond variant without the turnover condition."""

from __future__ import annotations

from typing import Final

import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.blue_diamond import (
    DEFAULT_CONFIG,
    BlueDiamondConfig,
    calculate_blue_diamond_features,
)

STRATEGY_ID: Final[str] = "blue_diamond_core"
STRATEGY_VERSION: Final[str] = "1.0"
STRATEGY_NAME: Final[str] = "蓝色钻石 Core"


def calculate_blue_diamond_core_features(
    frame: pd.DataFrame, *, config: BlueDiamondConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Reuse all Blue Diamond diagnostics and select its turnover-free core."""
    result = calculate_blue_diamond_features(frame, config=config)
    if result.empty:
        return result
    result["signal"] = result["core_signal"]
    result["setup"] = result["core_signal"]
    # MarketCap and turnover remain optional diagnostics for this variant.
    result.loc[result["status"].eq("market_cap_unavailable"), "status"] = "ok"
    return result
