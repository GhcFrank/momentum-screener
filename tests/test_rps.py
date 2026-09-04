from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from momentum_screener.rps import (
    INVALID_RPS,
    RPS_CALENDAR_NAME,
    RPS_PRICE_FIELD,
    InsufficientRpsHistoryError,
    InvalidRpsSessionError,
    RpsTickerNotFoundError,
    calculate_cross_sectional_rps,
    calculate_rps_snapshot,
    get_stock_rps,
    resolve_rps_session_dates,
)
from momentum_screener.storage_manifest import PRICE_SCHEMA

AS_OF_DATE = date(2026, 8, 31)


def _xnys_dates() -> tuple[date, date, date, date, date, date]:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    as_of_session = calendar.date_to_session(pd.Timestamp(AS_OF_DATE), direction="none")
    current_index = int(calendar.sessions.get_loc(as_of_session))
    return (
        as_of_session.date(),
        calendar.sessions[current_index - 120].date(),
        calendar.sessions[current_index - 250].date(),
        calendar.sessions[current_index - 121].date(),
        calendar.sessions[current_index - 251].date(),
        calendar.sessions[current_index - 1].date(),
    )


def _write_universe(path: Path, tickers: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"{ticker},{ticker} Inc.,{1000 - index},{index + 1}\n"
        for index, ticker in enumerate(tickers)
    )
    path.write_text(
        "ticker,company_name,market_cap,market_cap_rank\n" + rows,
        encoding="utf-8",
    )


def _write_price_partitions(
    prices_root: Path,
    records: list[tuple[date, str, float, float]],
) -> None:
    by_year: dict[int, list[tuple[date, str, float, float]]] = {}
    for record in records:
        by_year.setdefault(record[0].year, []).append(record)
    for year, year_records in by_year.items():
        ordered = sorted(year_records, key=lambda value: (value[0], value[1]))
        table = pa.Table.from_arrays(
            [
                pa.array([value[0] for value in ordered], type=pa.date32()),
                pa.array([value[1] for value in ordered], type=pa.string()),
                pa.array([value[2] for value in ordered], type=pa.float64()),
                pa.array([value[2] for value in ordered], type=pa.float64()),
                pa.array([value[2] for value in ordered], type=pa.float64()),
                pa.array([value[2] for value in ordered], type=pa.float64()),
                pa.array([value[3] for value in ordered], type=pa.float64()),
                pa.array([100] * len(ordered), type=pa.int64()),
            ],
            schema=PRICE_SCHEMA,
        )
        path = prices_root / "daily" / f"year={year}" / "prices.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")


@pytest.fixture
def rps_dataset(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    as_of, base120, base250, prior120, prior250, prior_current = _xnys_dates()
    tickers = ("AAA", "BBB", "CCC", "MISS120", "MISS250", "MISSCUR")
    universe_path = tmp_path / "universe.csv"
    prices_root = tmp_path / "prices"
    _write_universe(universe_path, tickers)

    adjusted: dict[tuple[date, str], float] = {
        (as_of, "AAA"): 12.0,
        (as_of, "BBB"): 20.0,
        (as_of, "CCC"): 40.0,
        (as_of, "MISS120"): 30.0,
        (as_of, "MISS250"): 30.0,
        (base120, "AAA"): 10.0,
        (base120, "BBB"): 10.0,
        (base120, "CCC"): 10.0,
        (base120, "MISS250"): 10.0,
        (base120, "MISSCUR"): 10.0,
        (base250, "AAA"): 10.0,
        (base250, "BBB"): 20.0,
        (base250, "CCC"): 10.0,
        (base250, "MISS120"): 10.0,
        (base250, "MISSCUR"): 10.0,
        (prior120, "MISS120"): 10.0,
        (prior250, "MISS250"): 10.0,
        (prior_current, "MISSCUR"): 10.0,
    }
    close_overrides = {
        (as_of, "AAA"): 1000.0,
        (as_of, "BBB"): 2.0,
        (as_of, "CCC"): 3.0,
    }
    records = [
        (row_date, ticker, close_overrides.get((row_date, ticker), price), price)
        for (row_date, ticker), price in adjusted.items()
    ]
    _write_price_partitions(prices_root, records)
    return universe_path, prices_root, tickers


def test_cross_sectional_rps_normalizes_lowest_and_highest() -> None:
    result = calculate_cross_sectional_rps(
        current_prices=pd.Series({"AAA": 11.0, "BBB": 15.0, "CCC": 20.0}),
        base_prices=pd.Series({"AAA": 10.0, "BBB": 10.0, "CCC": 10.0}),
        universe=("AAA", "BBB", "CCC"),
    )

    assert result.loc["AAA", "rps"] == pytest.approx(0.0)
    assert result.loc["BBB", "rps"] == pytest.approx(50.0)
    assert result.loc["CCC", "rps"] == pytest.approx(100.0)
    assert result.loc["AAA", "return"] < result.loc["BBB", "return"]
    assert result.loc["BBB", "return"] < result.loc["CCC", "return"]


def test_cross_sectional_rps_uses_average_ties_and_excludes_invalid() -> None:
    result = calculate_cross_sectional_rps(
        current_prices=pd.Series(
            {"AAA": 20.0, "BBB": 20.0, "CCC": 40.0, "BAD": float("nan")}
        ),
        base_prices=pd.Series({"AAA": 10.0, "BBB": 10.0, "CCC": 10.0}),
        universe=("AAA", "BBB", "CCC", "BAD"),
    )

    assert result.loc["AAA", "rps"] == pytest.approx(25.0)
    assert result.loc["BBB", "rps"] == pytest.approx(25.0)
    assert result.loc["CCC", "rps"] == pytest.approx(100.0)
    assert result.loc["BAD", "rps"] == INVALID_RPS
    assert pd.isna(result.loc["BAD", "return"])


def test_cross_sectional_rps_single_valid_stock_is_100() -> None:
    result = calculate_cross_sectional_rps(
        current_prices=pd.Series({"AAA": 20.0}),
        base_prices=pd.Series({"AAA": 10.0}),
        universe=("AAA", "MISSING"),
    )

    assert result.loc["AAA", "rps"] == 100.0
    assert result.loc["MISSING", "rps"] == INVALID_RPS


def test_snapshot_calculates_independent_rps120_and_rps250(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset

    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert snapshot.loc["AAA", "rps120"] == pytest.approx(0.0)
    assert snapshot.loc["BBB", "rps120"] == pytest.approx(100.0 / 3.0)
    assert snapshot.loc["MISS250", "rps120"] == pytest.approx(200.0 / 3.0)
    assert snapshot.loc["CCC", "rps120"] == pytest.approx(100.0)
    assert snapshot.loc["BBB", "rps250"] == pytest.approx(0.0)
    assert snapshot.loc["AAA", "rps250"] == pytest.approx(100.0 / 3.0)
    assert snapshot.loc["MISS120", "rps250"] == pytest.approx(200.0 / 3.0)
    assert snapshot.loc["CCC", "rps250"] == pytest.approx(100.0)


def test_snapshot_uses_exact_t_minus_120_and_t_minus_250_sessions(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset
    as_of, base120, base250, _, _, _ = _xnys_dates()

    snapshot = calculate_rps_snapshot(
        as_of.isoformat(),
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert snapshot["as_of_date"].unique().tolist() == [as_of]
    assert snapshot["rps120_base_date"].unique().tolist() == [base120]
    assert snapshot["rps250_base_date"].unique().tolist() == [base250]


def test_snapshot_uses_adjusted_close_as_standard_price_field(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset

    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert RPS_PRICE_FIELD == "adj_close"
    assert snapshot.loc["AAA", "return_120"] == pytest.approx(0.2)
    assert snapshot.loc["BBB", "rps120"] > snapshot.loc["AAA", "rps120"]


def test_missing_base_dates_do_not_fall_back_and_metrics_are_independent(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset

    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert snapshot.loc["MISS120", "rps120"] == INVALID_RPS
    assert snapshot.loc["MISS120", "rps250"] >= 0.0
    assert snapshot.loc["MISS250", "rps120"] >= 0.0
    assert snapshot.loc["MISS250", "rps250"] == INVALID_RPS


def test_missing_as_of_price_invalidates_both_metrics(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset

    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert snapshot.loc["MISSCUR", "rps120"] == INVALID_RPS
    assert snapshot.loc["MISSCUR", "rps250"] == INVALID_RPS


def test_snapshot_always_contains_complete_universe(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, tickers = rps_dataset

    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert len(snapshot) == len(tickers)
    assert snapshot.index.tolist() == list(tickers)
    assert snapshot["ticker"].tolist() == list(tickers)
    assert {"ticker", "as_of_date", "rps120", "rps250"}.issubset(snapshot.columns)


def test_get_stock_rps_is_exact_row_from_full_market_snapshot(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
) -> None:
    universe_path, prices_root, _ = rps_dataset
    snapshot = calculate_rps_snapshot(
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    stock = get_stock_rps(
        "aaa",
        AS_OF_DATE,
        prices_root=prices_root,
        universe_path=universe_path,
    )

    assert stock["rps120"] == snapshot.loc["AAA", "rps120"]
    assert stock["rps250"] == snapshot.loc["AAA", "rps250"]
    assert stock["rps120"] == 0.0
    assert stock["rps120"] != 100.0


def test_invalid_as_of_date_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidRpsSessionError, match="not a valid XNYS"):
        calculate_rps_snapshot(
            "2026-08-30",
            prices_root=tmp_path / "prices",
            universe_path=tmp_path / "universe.csv",
        )


def test_insufficient_market_session_history_raises_clear_error(
    tmp_path: Path,
) -> None:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    early_session = calendar.sessions[249].date()

    with pytest.raises(InsufficientRpsHistoryError, match="need 250 prior sessions"):
        calculate_rps_snapshot(
            early_session,
            prices_root=tmp_path / "prices",
            universe_path=tmp_path / "universe.csv",
        )


@pytest.mark.parametrize("ticker", ["ZZZZZ", "NOT_IN_UNIVERSE", "^INVALID"])
def test_get_stock_rps_rejects_unknown_or_invalid_ticker(
    rps_dataset: tuple[Path, Path, tuple[str, ...]],
    ticker: str,
) -> None:
    universe_path, prices_root, _ = rps_dataset

    with pytest.raises(RpsTickerNotFoundError, match="not in|Invalid"):
        get_stock_rps(
            ticker,
            AS_OF_DATE,
            prices_root=prices_root,
            universe_path=universe_path,
        )


def test_resolve_dates_matches_explicit_shared_session_indices() -> None:
    calendar = xcals.get_calendar(RPS_CALENDAR_NAME)
    session = calendar.date_to_session(pd.Timestamp(AS_OF_DATE), direction="none")
    current_index = int(calendar.sessions.get_loc(session))

    resolved = resolve_rps_session_dates(AS_OF_DATE)

    assert resolved.rps120_base_date == calendar.sessions[current_index - 120].date()
    assert resolved.rps250_base_date == calendar.sessions[current_index - 250].date()
