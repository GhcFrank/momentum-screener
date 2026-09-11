from dataclasses import replace

import pandas as pd

from momentum_screener import trend_reacceleration_entry as entry
from momentum_screener.trend_reacceleration import DEFAULT_CONFIG


def test_entry_keeps_only_first_setup_row_per_episode_and_preserves_diagnostics(
    monkeypatch,
):
    setup = [False, True, True, True, False, True, True]
    base = pd.DataFrame(
        {
            "setup": setup,
            "signal": setup,
            "momentum_ok": setup,
            "rps120": 95.0,
            "ma20": 100.0,
            "drawdown": 0.1,
        }
    )
    config = replace(DEFAULT_CONFIG, rps_sum_threshold=180)
    inputs = pd.DataFrame()

    def calculate(frame, *, config):
        assert frame is inputs
        assert config.rps_sum_threshold == 180
        return base.copy()

    monkeypatch.setattr(entry, "calculate_trend_reacceleration_features", calculate)
    result = entry.calculate_trend_reacceleration_entry_features(inputs, config=config)
    assert result["signal"].tolist() == [False, True, False, False, False, True, False]
    pd.testing.assert_frame_equal(
        result.drop(columns="signal"), base.drop(columns="signal")
    )
    assert base["signal"].tolist() == setup
