from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.historical_screening as engine
import momentum_screener.strategy_data as data
from momentum_screener.monthly_reversal import (
    calculate_monthly_reversal_features,
    evaluate_monthly_reversal,
)
from momentum_screener.prices import universe_sha256
from momentum_screener.signal_store import (
    COMMON_COLUMNS,
    get_signal_calendar,
    get_signal_detail,
    get_signals_for_date,
    read_strategy_signals,
)
from momentum_screener.storage_manifest import (
    PRICE_SCHEMA,
    SCHEMA_VERSION,
    TICKER_COVERAGE_ASSET_NAME,
    build_asset_record,
    validate_manifest,
    write_json_atomically,
)
from momentum_screener.trend_reacceleration import (
    DEFAULT_CONFIG,
    calculate_trend_reacceleration_features,
    evaluate_trend_reacceleration,
)


@dataclass
class HistoricalDataset:
    prices_root: Path
    universe_path: Path
    output_store: Path
    prices: pd.DataFrame
    rps: pd.DataFrame
    universe: tuple[str, ...]

    def save(self) -> None:
        """Write real existing-format price partitions and a validated manifest."""
        counts: dict[str, int] = {}
        assets: dict[str, object] = {}
        for year, rows in self.prices.groupby(
            self.prices["date"].map(lambda value: value.year)
        ):
            relative = f"daily/year={year}/prices.parquet"
            path = self.prices_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(
                    rows.sort_values(["date", "ticker"]),
                    schema=PRICE_SCHEMA,
                    preserve_index=False,
                ),
                path,
                compression="zstd",
            )
            counts[str(year)] = len(rows)
            assets[str(year)] = build_asset_record(
                path, asset_name=f"prices-year-{year}.parquet", local_path=relative
            )
        coverage_path = self.prices_root / "ticker_coverage.csv"
        coverage_path.write_text(
            "ticker\n" + "\n".join(self.universe) + "\n", encoding="utf-8"
        )
        assets["ticker_coverage"] = build_asset_record(
            coverage_path,
            asset_name=TICKER_COVERAGE_ASSET_NAME,
            local_path="ticker_coverage.csv",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "source": "yahoo_finance_via_yfinance",
            "universe_sha256": universe_sha256(self.universe),
            "universe_ticker_count": len(self.universe),
            "requested_start": "2016-01-01",
            "actual_min_date": min(self.prices["date"]).isoformat(),
            "actual_max_date": max(self.prices["date"]).isoformat(),
            "latest_session": max(self.prices["date"]).isoformat(),
            "last_successful_update_utc": "2026-09-04T22:00:00+00:00",
            "total_row_count": len(self.prices),
            "partition_row_counts": counts,
            "assets": assets,
        }
        write_json_atomically(
            self.prices_root / "manifest.json", validate_manifest(manifest)
        )

    def run(
        self,
        start: date | str = "2026-09-02",
        end: date | str | None = "2026-09-04",
        **kwargs: Any,
    ) -> engine.HistoricalScreeningResult:
        options = {
            "prices_root": self.prices_root,
            "universe_path": self.universe_path,
            "rps_root": None,
            "rps_snapshots": self.rps,
            "output_store": self.output_store,
        }
        options.update(kwargs)
        return engine.run_historical_screening(start, end, **options)


@pytest.fixture
def dataset(tmp_path: Path) -> HistoricalDataset:
    sessions = data.resolve_strategy_sessions(date(2026, 9, 4), 350)
    tickers = ("TREND", "OLD", "FRESH", "ZERO", "SHORT", "GAP")
    prices: list[dict[str, object]] = []
    rps: list[dict[str, object]] = []
    for ticker in tickers:
        closes = np.full(len(sessions), 100.0)
        if ticker in {"TREND", "SHORT"}:
            closes = 50 + np.arange(len(sessions)) * 0.2
        elif ticker in {"OLD", "FRESH"}:
            closes[-20:] = np.linspace(101, 115, 20)
        elif ticker == "GAP":
            closes[-60:] = np.linspace(101, 115, 60)
        for index, (session, close) in enumerate(zip(sessions, closes, strict=True)):
            score50 = -1.0
            if (
                ticker == "OLD"
                and index >= len(sessions) - 4
                or ticker == "FRESH"
                and index == len(sessions) - 1
                or ticker == "GAP"
                and (index == len(sessions) - 21 or index >= len(sessions) - 3)
            ):
                score50 = 100.0
            rps.append(
                {
                    "date": session,
                    "ticker": ticker,
                    "rps50": score50,
                    "rps120": 95.0 if ticker in {"TREND", "SHORT"} else -1.0,
                    "rps250": 95.0 if ticker in {"TREND", "SHORT"} else -1.0,
                }
            )
            if (ticker == "SHORT" and index < len(sessions) - 278) or (
                ticker == "GAP" and len(sessions) - 20 <= index < len(sessions) - 4
            ):
                continue
            low = close - 1
            if (ticker in {"OLD", "FRESH"} and index == len(sessions) - 40) or (
                ticker == "GAP" and index == len(sessions) - 55
            ):
                low = 80.0
            prices.append(
                {
                    "date": session,
                    "ticker": ticker,
                    "open": close,
                    "high": close + 1,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": 100,
                }
            )
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("ticker\n" + "\n".join(tickers) + "\n", encoding="utf-8")
    result = HistoricalDataset(
        tmp_path / "prices",
        universe_path,
        tmp_path / "signals",
        pd.DataFrame(prices),
        pd.DataFrame(rps),
        tickers,
    )
    result.save()
    return result


def stable(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(columns=["generated_at"]).reset_index(drop=True)


def test_natural_dates_weekends_clamping_and_no_output_outside_range(
    dataset: HistoricalDataset,
) -> None:
    result = dataset.run("2026-08-29", "2026-09-06")
    assert result.actual_start == date(2026, 8, 31)
    assert result.actual_end == date(2026, 9, 4)
    assert result.session_count == 5
    assert result.loaded_price_session_count >= 320
    calendar = get_signal_calendar(
        "2026-01-01", "2026-12-31", root=dataset.output_store
    )
    assert len(calendar) == 10
    assert calendar["session"].min() == result.actual_start
    assert calendar["session"].max() == result.actual_end
    assert all(value.weekday() < 5 for value in calendar["session"])
    assert result.as_dict()["cache_policy"] == "recompute_replace"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-08-29", "2026-08-30"),
        ("2026-09-05", "2026-09-10"),
        ("2026-09-04", "2026-09-01"),
        ("2016-01-01", "2016-01-03"),
    ],
)
def test_empty_or_reversed_range_does_not_write_coverage(
    dataset: HistoricalDataset, start: str, end: str
) -> None:
    with pytest.raises(engine.HistoricalScreeningError):
        dataset.run(start, end)
    assert not dataset.output_store.exists()


def test_cross_year_arbitrary_range_and_latest_alias(
    dataset: HistoricalDataset,
) -> None:
    result = dataset.run("2025-12-27", "2026-01-06", strategies="trend_reacceleration")
    expected = data.sessions_in_range(date(2025, 12, 27), date(2026, 1, 6))
    assert result.session_count == len(expected)
    assert (dataset.output_store / "trend_reacceleration/2025.parquet").exists()
    assert (dataset.output_store / "trend_reacceleration/2026.parquet").exists()
    latest = dataset.run("2026-09-02", None)
    assert latest.actual_end == date(2026, 9, 4)
    assert latest.as_dict()["requested_end"] == "latest"


def test_batch_signals_and_every_diagnostic_match_existing_features(
    dataset: HistoricalDataset,
) -> None:
    result = dataset.run()
    prepared = data.merge_prices_and_rps(
        dataset.prices, dataset.rps, lookbacks=(50, 120, 250)
    )
    calculators = {
        "monthly_reversal": calculate_monthly_reversal_features,
        "trend_reacceleration": calculate_trend_reacceleration_features,
    }
    for strategy_id, calculate in calculators.items():
        expected_parts = []
        for _, prices in prepared.groupby("ticker", sort=False):
            features = calculate(prices)
            selected = features.loc[
                features["date"].between(result.actual_start, result.actual_end)
                & features["signal"]
            ]
            expected_parts.append(selected)
        expected = (
            pd.concat(expected_parts)
            .rename(columns={"date": "session"})
            .sort_values(["session", "ticker"], ignore_index=True)
        )
        actual = read_strategy_signals(strategy_id, root=dataset.output_store)
        pd.testing.assert_frame_equal(
            actual.loc[:, expected.columns], expected, check_dtype=False
        )
    assert (
        get_signals_for_date("2026-09-04", root=dataset.output_store)[
            "strategy_id"
        ].nunique()
        == 2
    )
    for strategy_id, ticker, evaluate in (
        ("monthly_reversal", "FRESH", evaluate_monthly_reversal),
        ("trend_reacceleration", "TREND", evaluate_trend_reacceleration),
    ):
        explanation = evaluate(
            ticker,
            "2026-09-04",
            prices_root=dataset.prices_root,
            universe_path=dataset.universe_path,
            rps_root=None,
            rps_snapshots=dataset.rps,
        )
        detail = get_signal_detail(
            "2026-09-04", strategy_id, ticker, root=dataset.output_store
        )
        assert detail is not None and bool(explanation["signal"])
        assert bool(detail["signal"]) == bool(explanation["signal"])


def test_monthly_prior_yxfz_before_requested_start_suppresses_first_day(
    dataset: HistoricalDataset,
) -> None:
    dataset.run()
    inputs = data.merge_prices_and_rps(
        dataset.prices, dataset.rps, lookbacks=(50, 120, 250)
    )
    for ticker in ("OLD", "GAP"):
        history = calculate_monthly_reversal_features(
            inputs.loc[inputs["ticker"].eq(ticker)]
        )
        current = history.loc[history["date"].eq(date(2026, 9, 2))].iloc[0]
        assert bool(current["yxfz"]), ticker
        assert not bool(current["signal"]), ticker
        assert (
            get_signal_detail(
                "2026-09-02", "monthly_reversal", ticker, root=dataset.output_store
            )
            is None
        )
    # GAP's prior true YXFZ is more than 14 MARKET sessions ago, but fewer
    # than 14 actual ticker rows ago. The batch preparation must include it.
    assert date(2026, 9, 2) - min(
        inputs.loc[inputs["ticker"].eq("GAP") & inputs["rps50"].eq(100), "date"]
    ) > timedelta(days=20)


def test_trend_repetition_and_insufficient_history_do_not_create_false_positives(
    dataset: HistoricalDataset,
) -> None:
    dataset.run()
    signals = read_strategy_signals("trend_reacceleration", root=dataset.output_store)
    assert signals["ticker"].tolist() == ["TREND"] * 3
    assert signals["session"].tolist() == [date(2026, 9, value) for value in (2, 3, 4)]
    assert signals["signal"].tolist() == [True] * 3
    calendar = get_signal_calendar(
        "2026-09-02",
        "2026-09-04",
        strategy_id="monthly_reversal",
        root=dataset.output_store,
    )
    assert calendar["signal_count"].tolist() == [0, 0, 1]
    assert calendar["status"].eq("complete").all()


def test_one_price_load_one_rps_preparation_and_one_feature_history_per_ticker(
    dataset: HistoricalDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_read = Mock(wraps=data.read_affected_partitions)
    rps_prepare = Mock(wraps=engine.load_or_calculate_rps)
    universe_read = Mock(wraps=engine.load_universe)
    monthly = Mock(wraps=calculate_monthly_reversal_features)
    trend = Mock(wraps=calculate_trend_reacceleration_features)
    registry = dict(engine.SUPPORTED_STRATEGIES)
    registry["monthly_reversal"] = replace(
        registry["monthly_reversal"], calculate_features=monthly
    )
    monkeypatch.setattr(engine, "SUPPORTED_STRATEGIES", registry)
    monkeypatch.setattr(engine, "calculate_trend_reacceleration_features", trend)
    monkeypatch.setattr(data, "read_affected_partitions", price_read)
    monkeypatch.setattr(engine, "load_or_calculate_rps", rps_prepare)
    monkeypatch.setattr(engine, "load_universe", universe_read)
    dataset.run("2026-08-29", "2026-09-04")
    assert (
        price_read.call_count == universe_read.call_count == rps_prepare.call_count == 1
    )
    assert rps_prepare.call_args.kwargs["lookbacks"] == (50, 120, 250)
    assert monthly.call_count == trend.call_count == len(dataset.universe)
    assert all(len(call.args[0]) >= 278 for call in monthly.call_args_list)


def test_missing_rps_fallback_reuses_preloaded_full_universe_prices(
    dataset: HistoricalDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    calculator = Mock(wraps=data.calculate_rps_snapshots)
    reader = Mock(wraps=data.read_affected_partitions)
    monkeypatch.setattr(data, "calculate_rps_snapshots", calculator)
    monkeypatch.setattr(data, "read_affected_partitions", reader)
    dataset.run(rps_snapshots=None)
    assert calculator.call_count == reader.call_count == 1
    assert calculator.call_args.kwargs["price_rows"] is not None
    assert set(calculator.call_args.kwargs["price_rows"]["ticker"]) == set(
        dataset.universe
    )
    assert calculator.call_args.kwargs["lookbacks"] == (50, 120, 250)


def test_recompute_replace_idempotency_and_changed_signal_removal(
    dataset: HistoricalDataset,
) -> None:
    dataset.run()
    first = read_strategy_signals("trend_reacceleration", root=dataset.output_store)
    first_calendar = get_signal_calendar(
        "2026-09-02", "2026-09-04", root=dataset.output_store
    )
    dataset.run(force=True)
    pd.testing.assert_frame_equal(
        stable(first),
        stable(
            read_strategy_signals("trend_reacceleration", root=dataset.output_store)
        ),
    )
    pd.testing.assert_frame_equal(
        stable(first_calendar),
        stable(
            get_signal_calendar("2026-09-02", "2026-09-04", root=dataset.output_store)
        ),
    )
    dataset.rps.loc[dataset.rps["ticker"].eq("TREND"), ["rps120", "rps250"]] = -1.0
    dataset.run(strategies=["trend_reacceleration"])
    assert read_strategy_signals(
        "trend_reacceleration", root=dataset.output_store
    ).empty
    assert (
        get_signal_detail(
            "2026-09-04", "monthly_reversal", "FRESH", root=dataset.output_store
        )
        is not None
    )
    assert (
        not get_signal_calendar("2026-09-02", "2026-09-04", root=dataset.output_store)
        .duplicated(["session", "strategy_id"])
        .any()
    )


def test_future_appends_do_not_change_past_signals_or_diagnostics(
    dataset: HistoricalDataset,
) -> None:
    dataset.run()
    before = {
        key: read_strategy_signals(key, root=dataset.output_store)
        for key in engine.SUPPORTED_STRATEGIES
    }
    future_dates = data.sessions_in_range(date(2026, 9, 5), date(2026, 9, 10))
    future = pd.concat(
        [
            dataset.prices.loc[dataset.prices["date"].eq(date(2026, 9, 4))].assign(
                date=session
            )
            for session in future_dates
        ],
        ignore_index=True,
    )
    for column in ("open", "high", "low", "close", "adj_close"):
        future[column] *= 100
    dataset.prices = pd.concat([dataset.prices, future], ignore_index=True)
    dataset.save()
    dataset.run()
    for key, original in before.items():
        pd.testing.assert_frame_equal(
            stable(original),
            stable(read_strategy_signals(key, root=dataset.output_store)),
        )

    # Include future bars in the SAME feature batch as well as in the source
    # dataset, so this checks rolling causality rather than only loader slicing.
    dataset.run(end="2026-09-10")
    for key, original in before.items():
        pd.testing.assert_frame_equal(
            stable(original),
            stable(
                read_strategy_signals(
                    key, "2026-09-02", "2026-09-04", root=dataset.output_store
                )
            ),
        )


def test_deleted_store_rebuilds_identical_results(dataset: HistoricalDataset) -> None:
    dataset.run()
    before = {
        key: read_strategy_signals(key, root=dataset.output_store)
        for key in engine.SUPPORTED_STRATEGIES
    }
    shutil.rmtree(dataset.output_store)
    dataset.run()
    for key, original in before.items():
        pd.testing.assert_frame_equal(
            stable(original),
            stable(read_strategy_signals(key, root=dataset.output_store)),
        )


def test_strategy_config_metadata_and_selection(dataset: HistoricalDataset) -> None:
    config = replace(DEFAULT_CONFIG, rps_sum_threshold=199.0)
    result = dataset.run(strategies="trend_reacceleration", trend_config=config)
    assert set(result.strategy_summaries) == {"trend_reacceleration"}
    calendar = get_signal_calendar(
        "2026-09-02", "2026-09-04", root=dataset.output_store
    )
    assert calendar["signal_count"].eq(0).all()
    assert json.loads(calendar.iloc[0]["config_json"])["rps_sum_threshold"] == 199.0
    assert calendar["input_fingerprint"].isna().all()
    assert calendar["universe_hash"].eq(universe_sha256(dataset.universe)).all()


@pytest.mark.parametrize("strategies", [[], ["unknown"]])
def test_invalid_selection_fails_before_writing(
    dataset: HistoricalDataset, strategies: list[str]
) -> None:
    with pytest.raises(engine.HistoricalScreeningError):
        dataset.run(strategies=strategies)
    assert not dataset.output_store.exists()


def test_missing_whole_market_session_does_not_become_complete_zero(
    dataset: HistoricalDataset,
) -> None:
    dataset.prices = dataset.prices.loc[dataset.prices["date"].ne(date(2026, 9, 3))]
    dataset.save()
    with pytest.raises(engine.HistoricalScreeningError, match="no price data"):
        dataset.run()
    assert not dataset.output_store.exists()


def test_formula_failure_does_not_publish_partial_strategy_coverage(
    dataset: HistoricalDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("formula failed")

    monkeypatch.setattr(engine, "calculate_trend_reacceleration_features", fail)
    with pytest.raises(RuntimeError, match="formula failed"):
        dataset.run()
    assert not dataset.output_store.exists()


def test_cli_runs_real_fixture_without_listing_stocks(
    dataset: HistoricalDataset,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_run = engine.run_historical_screening

    def run(*args: Any, **kwargs: Any) -> engine.HistoricalScreeningResult:
        kwargs["rps_snapshots"] = dataset.rps
        kwargs["rps_root"] = None
        return real_run(*args, **kwargs)

    monkeypatch.setattr(engine, "run_historical_screening", run)
    assert (
        engine.main(
            [
                "--start-date",
                "2026-09-02",
                "--end-date",
                "latest",
                "--prices-root",
                str(dataset.prices_root),
                "--universe",
                str(dataset.universe_path),
                "--output-store",
                str(dataset.output_store),
                "--strategy",
                "monthly_reversal",
                "--strategy",
                "trend_reacceleration",
                "--force",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Historical screening complete" in output
    assert "2026-09-02 -> 2026-09-04" in output
    assert "monthly_reversal 6.2" in output
    assert "trend_reacceleration 1.0" in output
    assert "FRESH" not in output
    assert "TREND" not in output


def test_cli_failure_returns_nonzero_and_historical_layer_has_no_notification_dependency(
    dataset: HistoricalDataset,
) -> None:
    assert (
        engine.main(
            ["--start-date", "bad-date", "--prices-root", str(dataset.prices_root)]
        )
        == 1
    )
    source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "import smtplib" not in source
    assert "from momentum_screener.daily_screening_notification" not in source
    assert "screen_monthly_reversal(" not in source
    assert "screen_trend_reacceleration(" not in source
    assert set(COMMON_COLUMNS) >= {
        "input_fingerprint",
        "config_hash",
        "universe_hash",
        "data_mode",
    }


def test_cli_reports_corrupt_rps_store_without_publishing_coverage(
    dataset: HistoricalDataset, caplog: pytest.LogCaptureFixture
) -> None:
    rps_root = dataset.output_store.parent / "broken-rps"
    rps_root.mkdir()
    (rps_root / "orphan.parquet").write_bytes(b"unindexed")
    assert (
        engine.main(
            [
                "--start-date",
                "2026-09-02",
                "--end-date",
                "2026-09-04",
                "--prices-root",
                str(dataset.prices_root),
                "--universe",
                str(dataset.universe_path),
                "--rps-root",
                str(rps_root),
                "--output-store",
                str(dataset.output_store),
            ]
        )
        == 1
    )
    assert "Non-empty RPS root has no valid manifest" in caplog.text
    assert not dataset.output_store.exists()


def test_blue_diamond_registry_market_cap_coverage_store_csv_and_live_queries(
    dataset: HistoricalDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    import momentum_screener.blue_diamond as blue
    from momentum_screener.market_cap_storage import refresh_market_cap_snapshot
    from momentum_screener.rps import RPS_LOOKBACKS, resolve_rps_session_dates
    from momentum_screener.rps_storage import persist_rps_snapshot, read_rps_snapshot
    from momentum_screener.signal_store import export_signal_csv
    from momentum_screener.signal_ui_data import (
        LocalFile,
        build_ticker_rps_table,
        filter_tickers_by_strategies,
        read_signal_csv,
    )

    session = date(2026, 9, 4)
    entry = engine.SUPPORTED_STRATEGIES["blue_diamond"]
    assert entry.version == "1.0" and entry.lookbacks == (20, 50)
    assert entry.required_price_rows == 250 and entry.prior_rps_rows == 0
    assert entry.requires_market_cap
    cap_root = tmp_path / "caps"
    rps_root = tmp_path / "persisted-rps"
    trend = dataset.prices.loc[dataset.prices["ticker"].eq("TREND")].copy()
    for offset in (-2, -1):
        closing = trend["adj_close"].iloc[offset - 19 : offset].mean() * 0.999
        index = trend.index[offset]
        trend.loc[index, ["open", "close", "adj_close"]] = closing
        trend.loc[index, "high"] = closing + 0.2
        trend.loc[index, "low"] = closing - 0.2
    # FRESH has the same qualifying prices/RPS, but no exact-date cap.
    dataset.prices = pd.concat(
        [
            dataset.prices.loc[~dataset.prices["ticker"].isin(["TREND", "FRESH"])],
            trend,
            trend.assign(ticker="FRESH"),
        ],
        ignore_index=True,
    )
    dataset.save()
    dataset.rps["rps20"] = -1.0
    dataset.rps.loc[
        dataset.rps["ticker"].isin(["TREND", "FRESH"]), ["rps20", "rps50"]
    ] = [98.0, 95.0]
    snapshot = dataset.rps.loc[dataset.rps["date"].eq(session)].copy()
    for lookback, base in resolve_rps_session_dates(session).base_dates.items():
        snapshot[f"return_{lookback}"] = 0.1
        snapshot[f"rps{lookback}_base_date"] = base
    persist_rps_snapshot(snapshot, root=rps_root, universe_path=dataset.universe_path)
    monkeypatch.setattr(
        data,
        "calculate_rps_snapshots",
        Mock(side_effect=AssertionError("Persisted RPS must be reused")),
    )
    options = {
        "strategies": "blue_diamond",
        "market_cap_root": cap_root,
        "rps_root": rps_root,
        "rps_snapshots": None,
    }

    # Existing strategies neither load nor require MarketCap.
    dataset.run(session, session, market_cap_root=cap_root)
    prior_coverage = (dataset.output_store / "coverage.parquet").read_bytes()
    with pytest.raises(data.StrategyDataError, match="MarketCap data unavailable"):
        dataset.run(session, session, **options)
    assert (dataset.output_store / "coverage.parquet").read_bytes() == prior_coverage

    refresh_market_cap_snapshot(
        prices_root=dataset.prices_root,
        root=cap_root,
        universe_path=dataset.universe_path,
        now=datetime(2026, 9, 4, 22, tzinfo=UTC),
        fetch_func=lambda _: {"TREND": 1_000_000},
    )
    with pytest.raises(data.StrategyDataError, match="2026-09-03"):
        dataset.run("2026-09-03", session, **options)
    assert (dataset.output_store / "coverage.parquet").read_bytes() == prior_coverage

    summary = dataset.run(session, session, **options).strategy_summaries[
        "blue_diamond"
    ]
    assert summary["signals"] == 1
    assert summary["market_cap_available_count"] == 1
    assert summary["market_cap_missing_ticker_count"] == len(dataset.universe) - 1
    stored = read_strategy_signals("blue_diamond", root=dataset.output_store)
    assert stored["ticker"].tolist() == ["TREND"]
    assert stored["status"].tolist() == ["ok"]
    common = {
        "prices_root": dataset.prices_root,
        "universe_path": dataset.universe_path,
        "rps_root": rps_root,
        "market_cap_root": cap_root,
    }
    explanation = blue.evaluate_blue_diamond("TREND", session, **common)
    pd.testing.assert_series_equal(
        stored.iloc[0].rename({"session": "date"})[list(blue.SCREEN_COLUMNS)],
        explanation[list(blue.SCREEN_COLUMNS)],
        check_names=False,
    )
    missing = blue.evaluate_blue_diamond("FRESH", session, **common)
    assert missing["status"] == "market_cap_unavailable" and not missing["signal"]
    price_reader = Mock(wraps=blue.load_strategy_price_history)
    monkeypatch.setattr(blue, "load_strategy_price_history", price_reader)
    screened = blue.screen_blue_diamond(session, **common)
    assert screened["ticker"].tolist() == ["TREND"]
    assert set(price_reader.call_args.kwargs["tickers"]) == {"TREND", "FRESH"}
    assert (
        screened.attrs["market_cap_missing_ticker_count"] == len(dataset.universe) - 1
    )
    with pytest.raises(data.StrategyDataError, match="2026-09-03"):
        blue.screen_blue_diamond(date(2026, 9, 3), **common)

    csv = tmp_path / "research" / "signals.csv"
    export_signal_csv(csv, session, session, root=dataset.output_store)
    signals, warnings = read_signal_csv(LocalFile.inspect(csv))
    assert not warnings and "blue_diamond" in set(signals["strategy_id"])
    blue_rows = signals.loc[signals["strategy_id"].eq("blue_diamond")]
    assert blue_rows["strategy_version"].tolist() == ["1.0"]
    assert blue_rows["turnover"].tolist() == pytest.approx(stored["turnover"].tolist())
    assert filter_tickers_by_strategies(signals, session, ["blue_diamond"]) == ["TREND"]
    assert (
        filter_tickers_by_strategies(
            signals, session, ["blue_diamond", "monthly_reversal"]
        )
        == []
    )
    table = build_ticker_rps_table(["TREND"], read_rps_snapshot(session, root=rps_root))
    assert list(table) == ["Ticker", *[f"RPS{n}" for n in RPS_LOOKBACKS]]
    assert table["RPS20"].tolist() == [98.0]

    # Re-running with no RPS candidates must replace prior signals with a real
    # complete/zero result, still using the same store/export path.
    snapshot[["rps20", "rps50"]] = -1.0
    persist_rps_snapshot(snapshot, root=rps_root, universe_path=dataset.universe_path)
    assert (
        engine.main(
            [
                "--start-date",
                str(session),
                "--end-date",
                str(session),
                "--strategy",
                "blue_diamond",
                "--prices-root",
                str(dataset.prices_root),
                "--rps-root",
                str(rps_root),
                "--market-cap-root",
                str(cap_root),
                "--universe",
                str(dataset.universe_path),
                "--output-store",
                str(dataset.output_store),
                "--export-csv",
                str(csv),
            ]
        )
        == 0
    )
    assert read_strategy_signals("blue_diamond", root=dataset.output_store).empty
    assert pd.read_csv(csv).empty
