from dataclasses import asdict
from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from momentum_screener import local_price_data
from momentum_screener.forward_performance import (
    calculate_forward_performance_for_signals,
    calculate_signal_forward_performance,
)
from momentum_screener.local_price_data import price_files_in_range
from momentum_screener.storage_manifest import (
    PRICE_SCHEMA,
    SCHEMA_VERSION,
    build_asset_record,
    write_json_atomically,
)
from momentum_screener.strategy_data import resolve_strategy_sessions


def prices(future_count):
    return pd.DataFrame(
        {
            "date": resolve_strategy_sessions(date(2026, 9, 4), future_count + 1),
            "ticker": "AAA",
            "open": 100.0,
            "close": 100.0,
            "high": 120.0,
            "low": 90.0,
            "adj_close": 50.0,
            "volume": 100,
        }
    )


def test_full_windows_use_signal_raw_close_future_high_low_and_exclude_signal_bar():
    frame = prices(121)
    frame.loc[0, ["high", "low"]] = [10_000.0, 0.1]
    frame.loc[1, "low"] = 80.0
    frame.loc[40, "high"] = 130.0
    frame.loc[120, ["high", "low"]] = [150.0, 60.0]
    frame.loc[121, ["high", "low"]] = [20_000.0, 0.01]  # Beyond both windows.
    result = calculate_signal_forward_performance(
        "aaa", frame.iloc[0]["date"], price_rows=frame
    )
    assert asdict(result) == pytest.approx(
        {
            "forward_40d_max_drawdown": -0.2,
            "forward_40d_max_gain": 0.3,
            "forward_120d_max_drawdown": -0.4,
            "forward_120d_max_gain": 0.5,
        }
    )


def test_partial_windows_use_available_ticker_sessions_without_calendar_cutoff():
    for count in (5, 70):
        frame = prices(count)
        # Sparse ticker rows deliberately span more XNYS sessions than row count.
        frame["date"] = resolve_strategy_sessions(date(2026, 9, 4), count * 2 + 1)[::2]
        if count == 70:
            frame.loc[60, ["high", "low"]] = [160.0, 70.0]
        result = calculate_signal_forward_performance(
            "AAA", frame.iloc[0]["date"], price_rows=frame
        )
        assert result.forward_40d_max_drawdown == pytest.approx(-0.1)
        assert result.forward_40d_max_gain == pytest.approx(0.2)
        assert result.forward_120d_max_drawdown == pytest.approx(
            -0.1 if count == 5 else -0.3
        )
        assert result.forward_120d_max_gain == pytest.approx(0.2 if count == 5 else 0.6)


def test_no_future_or_missing_exact_signal_close_is_unavailable_without_date_fallback():
    frame = prices(5)
    signal_date = frame.iloc[0]["date"]
    cases = [(frame.iloc[-1]["date"], frame), (signal_date, frame.iloc[1:])]
    for value in (np.nan, 0.0):
        missing = frame.copy()
        missing.loc[0, "close"] = value
        cases.append((signal_date, missing))
    for session, rows in cases:
        result = calculate_signal_forward_performance("AAA", session, price_rows=rows)
        assert all(value is None for value in asdict(result).values())


def write_dataset(root, rows):
    assets, counts = {}, {}
    for year, group in rows.groupby(rows["date"].map(lambda value: value.year)):
        relative = f"daily/year={year}/prices.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(group, schema=PRICE_SCHEMA, preserve_index=False), path
        )
        assets[str(year)] = build_asset_record(
            path, asset_name=f"prices-year-{year}.parquet", local_path=relative
        )
        counts[str(year)] = len(group)
    coverage = root / "ticker_coverage.csv"
    coverage.write_text(
        "ticker,status,first_date,last_date,row_count,attempt_count,last_error\n"
    )
    assets["ticker_coverage"] = build_asset_record(
        coverage, asset_name="prices-ticker-coverage.csv", local_path=coverage.name
    )
    write_json_atomically(
        root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "source": "yahoo_finance_via_yfinance",
            "universe_sha256": "a" * 64,
            "universe_ticker_count": rows["ticker"].nunique(),
            "requested_start": "2025-01-01",
            "actual_min_date": str(min(rows["date"])),
            "actual_max_date": str(max(rows["date"])),
            "latest_session": str(max(rows["date"])),
            "last_successful_update_utc": "2026-09-04T22:00:00+00:00",
            "total_row_count": len(rows),
            "partition_row_counts": counts,
            "assets": assets,
        },
    )


def test_batch_reuses_yearly_reads_for_any_signal_and_detects_price_updates(
    tmp_path, monkeypatch
):
    frame = prices(5)
    frame["date"] = [
        date(2025, 12, 30),
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
    ]
    rows = pd.concat(
        [frame.assign(ticker=ticker) for ticker in ("AAA", "BBB", "UNSELECTED")],
        ignore_index=True,
    )
    write_dataset(tmp_path, rows)
    signals = pd.DataFrame(
        {
            "ticker": ["aaa", "AAA", "BBB", "AAA"],
            "session": [frame.iloc[0]["date"]] * 3 + [frame.iloc[1]["date"]],
            "strategy_id": [
                "anything",
                "future_strategy",
                "third_strategy",
                "anything",
            ],
        }
    )
    read = Mock(wraps=local_price_data.pq.read_table)
    monkeypatch.setattr(local_price_data.pq, "read_table", read)
    original_files = price_files_in_range(frame.iloc[0]["date"], prices_root=tmp_path)
    result = calculate_forward_performance_for_signals(
        signals, prices_root=tmp_path, files=original_files
    )
    assert len(result) == 3  # Same occurrence shared by two strategies calculates once.
    assert read.call_count == 2  # Two yearly files, independent of ticker count.
    assert all(
        call.kwargs["filters"][0] == ("ticker", "in", ["AAA", "BBB"])
        for call in read.call_args_list
    )
    assert result["forward_40d_max_gain"].tolist() == pytest.approx([0.2] * 3)
    assert result["forward_120d_max_drawdown"].tolist() == pytest.approx([-0.1] * 3)
    assert "strategy_id" not in result

    rows.loc[rows["date"].eq(max(rows["date"])), "high"] = 190.0
    write_dataset(tmp_path, rows)
    assert (
        price_files_in_range(frame.iloc[0]["date"], prices_root=tmp_path)
        != original_files
    )
    with pytest.raises(ValueError, match="changed while loading"):
        calculate_forward_performance_for_signals(signals, files=original_files)
    refreshed = calculate_forward_performance_for_signals(signals, prices_root=tmp_path)
    assert refreshed["forward_40d_max_gain"].tolist() == pytest.approx([0.9] * 3)
