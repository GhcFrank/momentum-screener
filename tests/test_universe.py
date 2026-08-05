from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from momentum_screener.universe import (
    PAGE_SIZE,
    SORT_FIELD,
    UNIVERSE_COLUMNS,
    UniverseBuildError,
    build_query,
    build_universe,
    extract_quotes,
    fetch_screener_pages,
    is_probable_adr,
    load_manual_exclusions,
    normalize_ticker,
    parse_candidate,
    process_candidates,
    run_build,
    write_universe_atomically,
)


def make_quote(index: int, *, market_cap: int | None = None) -> dict[str, Any]:
    return {
        "symbol": f"T{index:04d}",
        "longName": f"Test Company {index}",
        "marketCap": market_cap if market_cap is not None else 10_000_000 - index,
    }


def paged_screen(records: Sequence[object], calls: list[dict[str, object]]):
    def screen(query: object, **kwargs: object) -> dict[str, object]:
        calls.append({"query": query, **kwargs})
        offset = cast(int, kwargs["offset"])
        size = cast(int, kwargs["size"])
        return {"quotes": records[offset : offset + size]}

    return screen


def test_query_and_full_2000_row_pagination() -> None:
    records = [make_quote(index) for index in range(2_000)]
    calls: list[dict[str, object]] = []

    fetch_result = fetch_screener_pages(
        build_query(),
        target_size=2_000,
        screen_func=paged_screen(records, calls),
    )
    rows, processed = build_universe(fetch_result.records, target_size=2_000)

    assert len(rows) == 2_000
    assert processed.unique_candidate_count == 2_000
    assert [call["offset"] for call in calls] == list(range(0, 2_000, 250))
    assert all(call["size"] == PAGE_SIZE for call in calls)
    assert all(call["sortField"] == SORT_FIELD for call in calls)
    assert all(call["sortAsc"] is False for call in calls)
    assert build_query().to_dict() == {
        "operator": "AND",
        "operands": [
            {"operator": "EQ", "operands": ["region", "us"]},
            {"operator": "GT", "operands": [SORT_FIELD, 0]},
        ],
    }
    assert [row.market_cap_rank for row in rows] == list(range(1, 2_001))


def test_fetch_continues_after_filtered_first_eight_pages() -> None:
    records = [
        {
            "symbol": f"ADR{index:04d}",
            "longName": f"Issuer {index} ADR",
            "marketCap": 20_000_000 - index,
        }
        for index in range(10)
    ]
    records.extend(make_quote(index) for index in range(2_000))
    calls: list[dict[str, object]] = []

    result = fetch_screener_pages(
        build_query(),
        target_size=2_000,
        screen_func=paged_screen(records, calls),
    )

    assert len(calls) == 9
    assert calls[-1]["offset"] == 2_000
    assert len(process_candidates(result.records).candidates) == 2_000


def test_stable_sort_deduplication_and_ticker_normalization() -> None:
    records = [
        {"symbol": " zzz ", "shortName": "Zed", "marketCap": 100},
        {"symbol": "aaa", "displayName": "Alpha", "marketCap": 100},
        {"symbol": "aaa", "longName": "Alpha Old", "marketCap": 90},
        {"symbol": "BRK.B", "longName": "Berkshire", "marketCap": 80},
    ]

    rows, processed = build_universe(records, target_size=3)

    assert [row.ticker for row in rows] == ["AAA", "ZZZ", "BRK-B"]
    assert [row.market_cap for row in rows] == [100, 100, 80]
    assert processed.unique_candidate_count == 3
    assert normalize_ticker(" BF.B ") == "BF-B"
    assert normalize_ticker("already-valid") == "ALREADY-VALID"


@pytest.mark.parametrize(
    "name",
    [
        "Example American Depositary Receipt",
        "Example Depositary Receipt Series A",
        "Example Depositary Shares",
        "Example ADR Holdings",
    ],
)
def test_adr_names_are_excluded(name: str) -> None:
    assert is_probable_adr(name)
    result = process_candidates(
        [{"symbol": "TEST", "longName": name, "marketCap": 100}]
    )
    assert result.adr_excluded_count == 1
    assert result.candidates == ()


@pytest.mark.parametrize(
    "name",
    ["Madrigal Pharmaceuticals, Inc.", "Adroit Industries", "ADS Corporation"],
)
def test_unrelated_company_names_are_not_excluded(name: str) -> None:
    assert not is_probable_adr(name)


def test_manual_exclusion_file_is_optional_normalized_and_effective(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.txt"
    assert load_manual_exclusions(missing_path) == frozenset()

    exclusion_path = tmp_path / "exclude_tickers.txt"
    exclusion_path.write_text("# reviewed\n\nbrk.b\n AAPL \n", encoding="utf-8")
    exclusions = load_manual_exclusions(exclusion_path)
    assert exclusions == frozenset({"BRK-B", "AAPL"})

    result = process_candidates(
        [
            {"symbol": "BRK.B", "longName": "Berkshire", "marketCap": 100},
            {"symbol": "MSFT", "longName": "Microsoft", "marketCap": 90},
        ],
        exclusions,
    )
    assert [candidate.ticker for candidate in result.candidates] == ["MSFT"]
    assert result.manual_excluded_count == 1


def test_invalid_manual_exclusion_has_file_and_line_context(tmp_path: Path) -> None:
    path = tmp_path / "exclude_tickers.txt"
    path.write_text("AAPL\nBAD/TICKER\n", encoding="utf-8")

    with pytest.raises(UniverseBuildError, match=r"line 2.*BAD/TICKER"):
        load_manual_exclusions(path)


@pytest.mark.parametrize(
    "record",
    [
        {"symbol": "", "longName": "Empty", "marketCap": 1},
        {"symbol": "^GSPC", "longName": "Index", "marketCap": 1},
        {"symbol": "EUR=USD", "longName": "FX", "marketCap": 1},
        {"symbol": "BAD/ONE", "longName": "Bad", "marketCap": 1},
        {"symbol": "AAPL", "longName": "Apple", "marketCap": None},
        {"symbol": "AAPL", "longName": "Apple", "marketCap": 0},
        {"symbol": "AAPL", "longName": "Apple", "marketCap": -1},
        {"symbol": "AAPL", "longName": "Apple", "marketCap": "not-a-number"},
        {"symbol": "AAPL", "longName": "Apple", "marketCap": 1.5},
        {"symbol": "AAPL", "marketCap": 1},
    ],
)
def test_invalid_tickers_names_and_market_caps_are_skipped(
    record: dict[str, object],
) -> None:
    assert parse_candidate(record) is None


def test_market_cap_fallbacks_are_supported_in_priority_order() -> None:
    preferred = parse_candidate(
        {
            "symbol": "AAA",
            "longName": "Alpha",
            "marketCap": 200,
            SORT_FIELD: 100,
        }
    )
    fallback = parse_candidate(
        {"symbol": "BBB", "shortName": "Beta", SORT_FIELD: "123.0"}
    )
    nested = parse_candidate(
        {
            "symbol": "CCC",
            "displayName": "Gamma",
            "lastclosemarketcap": {"lasttwelvemonths": 50},
        }
    )

    assert preferred is not None and preferred.market_cap == 200
    assert fallback is not None and fallback.market_cap == 123
    assert nested is not None and nested.market_cap == 50


@pytest.mark.parametrize(
    "response, message",
    [
        (None, "expected a mapping"),
        ({}, "missing 'quotes' list"),
        ({"quotes": {}}, "'quotes' must be a list"),
    ],
)
def test_malformed_yahoo_responses_raise_clear_errors(
    response: object, message: str
) -> None:
    with pytest.raises(UniverseBuildError, match=re.escape(message)):
        extract_quotes(response)


def test_network_failure_has_finite_retry_and_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def flaky_screen(query: object, **kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary timeout")
        return {"quotes": [make_quote(1)]}

    result = fetch_screener_pages(
        build_query(),
        target_size=1,
        max_attempts=3,
        retry_base_delay=0.25,
        screen_func=flaky_screen,
        sleep_func=delays.append,
    )

    assert len(result.records) == 1
    assert result.pages_requested == 1
    assert result.request_attempt_count == 3
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_retry_exhaustion_raises_with_offset_context() -> None:
    attempts = 0

    def broken_screen(query: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("offline")

    with pytest.raises(UniverseBuildError, match=r"offset 0 after 2 attempts"):
        fetch_screener_pages(
            build_query(),
            target_size=1,
            max_attempts=2,
            retry_base_delay=0,
            screen_func=broken_screen,
            sleep_func=lambda _: None,
        )
    assert attempts == 2


def test_insufficient_candidates_does_not_replace_existing_universe(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "universe.csv"
    output_path.write_text("existing universe\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    with pytest.raises(UniverseBuildError, match="Insufficient valid unique"):
        run_build(
            target_size=3,
            output_path=output_path,
            screen_func=paged_screen([make_quote(1), make_quote(2)], calls),
            retry_base_delay=0,
        )

    assert output_path.read_text(encoding="utf-8") == "existing universe\n"
    assert not (tmp_path / "archive").exists()


def test_atomic_write_archives_old_file_and_preserves_exact_columns(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "universe.csv"
    output_path.write_text("old data\n", encoding="utf-8")
    rows, _ = build_universe([make_quote(1), make_quote(2)], target_size=2)

    archive_path = write_universe_atomically(output_path, rows)

    assert archive_path is not None
    assert archive_path.read_text(encoding="utf-8") == "old data\n"
    assert re.fullmatch(r"universe_\d{4}-\d{2}-\d{2}_\d{6}\.csv", archive_path.name)
    with output_path.open(encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        output_rows = list(reader)
    assert tuple(reader.fieldnames or ()) == UNIVERSE_COLUMNS
    assert len(output_rows) == 2
    assert [int(row["market_cap_rank"]) for row in output_rows] == [1, 2]


def test_successful_build_writes_report_and_adr_review(tmp_path: Path) -> None:
    output_path = tmp_path / "universe.csv"
    (tmp_path / "exclude_tickers.txt").write_text("MANUAL\n", encoding="utf-8")
    records: list[object] = [
        {"symbol": "ADR1", "longName": "Issuer ADR", "marketCap": 1_000},
        {"symbol": "MANUAL", "longName": "Manual Co", "marketCap": 900},
        {"symbol": "BAD", "longName": "Bad Co", "marketCap": 0},
        {"symbol": "BBB", "longName": "Beta", "marketCap": 800},
        {"symbol": "AAA", "longName": "Alpha", "marketCap": 800},
        {"symbol": "CCC", "longName": "Gamma", "marketCap": 700},
    ]
    calls: list[dict[str, object]] = []

    report = run_build(
        target_size=3,
        output_path=output_path,
        screen_func=paged_screen(records, calls),
        retry_base_delay=0,
    )

    persisted_report = json.loads(
        (tmp_path / "universe_build_report.json").read_text(encoding="utf-8")
    )
    assert persisted_report == report
    assert report == {
        **report,
        "source": "yfinance_screener",
        "target_size": 3,
        "pages_requested": 1,
        "request_attempt_count": 1,
        "raw_candidate_count": 6,
        "unique_candidate_count": 5,
        "invalid_record_count": 1,
        "adr_excluded_count": 1,
        "manual_excluded_count": 1,
        "final_count": 3,
        "largest_market_cap": 800,
        "smallest_market_cap": 700,
    }
    with output_path.open(encoding="utf-8", newline="") as output_file:
        universe_rows = list(csv.DictReader(output_file))
    assert [row["ticker"] for row in universe_rows] == ["AAA", "BBB", "CCC"]

    review_path = tmp_path / "review" / "adr_excluded.csv"
    with review_path.open(encoding="utf-8", newline="") as review_file:
        review_reader = csv.DictReader(review_file)
        review_rows = list(review_reader)
    assert tuple(review_reader.fieldnames or ()) == (
        "ticker",
        "company_name",
        "market_cap",
        "exclusion_reason",
    )
    assert review_rows[0]["ticker"] == "ADR1"


def test_empty_adr_review_still_has_header(tmp_path: Path) -> None:
    output_path = tmp_path / "universe.csv"
    run_build(
        target_size=1,
        output_path=output_path,
        screen_func=lambda query, **kwargs: {"quotes": [make_quote(1)]},
    )

    review_path = tmp_path / "review" / "adr_excluded.csv"
    with review_path.open(encoding="utf-8", newline="") as review_file:
        reader = csv.reader(review_file)
        assert list(reader) == [
            ["ticker", "company_name", "market_cap", "exclusion_reason"]
        ]
