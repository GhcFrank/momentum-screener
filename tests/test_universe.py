from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import momentum_screener.universe as universe_module
from momentum_screener.universe import (
    ALLOWED_YAHOO_EXCHANGES,
    EXCLUDED_REVIEW_COLUMNS,
    FINAL_AUDIT_COLUMNS,
    FORBIDDEN_YAHOO_EXCHANGES,
    PAGE_SIZE,
    SORT_FIELD,
    UNIVERSE_COLUMNS,
    ExcludedSecurity,
    SymbolDirectory,
    SymbolDirectoryRecord,
    UniverseBuildError,
    adr_exclusion_reason,
    build_query,
    build_symbol_directory,
    build_universe,
    download_symbol_directory,
    evaluate_candidate,
    fetch_screener_pages,
    is_foreign_otc_suffix,
    is_probable_adr,
    load_manual_exclusions,
    main,
    non_common_security_reason,
    normalize_ticker,
    parse_nasdaq_listed,
    parse_other_listed,
    process_candidates,
    run_build,
    validate_universe_file,
    write_universe_atomically,
)

NASDAQ_HEADER = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares"
)
OTHER_HEADER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
    "Test Issue|NASDAQ Symbol"
)


def nasdaq_file(rows: Sequence[str]) -> str:
    return "\n".join(
        [
            NASDAQ_HEADER,
            *rows,
            "File Creation Time: 20260805|||||||",
            "",
        ]
    )


def other_file(rows: Sequence[str]) -> str:
    return "\n".join(
        [
            OTHER_HEADER,
            *rows,
            "File Creation Time: 20260805|||||||",
            "",
        ]
    )


def listing(
    ticker: str,
    *,
    security_name: str = "Example Corporation Common Stock",
    listing_exchange: str = "NASDAQ",
    etf: bool = False,
    test_issue: bool = False,
    category: str | None = "Q",
    financial_status: str | None = "N",
) -> SymbolDirectoryRecord:
    return SymbolDirectoryRecord(
        raw_symbol=ticker.replace("-", "."),
        ticker=ticker,
        security_name=security_name,
        listing_exchange=listing_exchange,
        etf=etf,
        test_issue=test_issue,
        market_category=category,
        financial_status=financial_status,
    )


def directory(*records: SymbolDirectoryRecord) -> SymbolDirectory:
    return SymbolDirectory(
        records={record.ticker: record for record in records},
        raw_record_count=len(records),
    )


def quote(
    ticker: str,
    *,
    market_cap: object = 1_000,
    exchange: str = "NMS",
    name: str | None = None,
    quote_type: object = "EQUITY",
) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": ticker,
        "marketCap": market_cap,
        "exchange": exchange,
        "quoteType": quote_type,
    }
    if name is not None:
        result["longName"] = name
    else:
        result["longName"] = f"{ticker} Incorporated"
    return result


def make_large_fixture(count: int) -> tuple[list[dict[str, object]], SymbolDirectory]:
    records: list[dict[str, object]] = []
    listings: list[SymbolDirectoryRecord] = []
    for index in range(count):
        ticker = f"T{index:04d}"
        records.append(quote(ticker, market_cap=10_000_000 - index))
        listings.append(listing(ticker))
    return records, directory(*listings)


def paged_screen(records: Sequence[object], calls: list[dict[str, object]]):
    def screen(query: object, **kwargs: object) -> dict[str, object]:
        calls.append({"query": query, **kwargs})
        offset = cast(int, kwargs["offset"])
        size = cast(int, kwargs["size"])
        return {"quotes": list(records[offset : offset + size])}

    return screen


def test_query_has_explicit_exchange_allowlist_and_no_otc_codes() -> None:
    payload = build_query().to_dict()
    encoded = json.dumps(payload)
    assert payload["operator"] == "AND"
    assert '"region", "us"' in encoded
    assert SORT_FIELD in encoded
    assert set(ALLOWED_YAHOO_EXCHANGES).issubset(
        set(re.findall(r'"([A-Z]{3})"', encoded))
    )
    assert not set(FORBIDDEN_YAHOO_EXCHANGES).intersection(encoded.split('"'))


def test_pagination_fetches_2000_validated_candidates_in_250_pages() -> None:
    records, symbols = make_large_fixture(2_000)
    calls: list[dict[str, object]] = []
    result = fetch_screener_pages(
        build_query(),
        symbol_directory=symbols,
        target_size=2_000,
        screen_func=paged_screen(records, calls),
    )
    assert len(result.records) == 2_000
    assert [call["offset"] for call in calls] == list(range(0, 2_000, 250))
    assert all(call["size"] == PAGE_SIZE for call in calls)
    assert all(call["sortField"] == SORT_FIELD for call in calls)
    assert all(call["sortAsc"] is False for call in calls)


def test_pagination_continues_after_filtered_candidates() -> None:
    valid_records, valid_directory = make_large_fixture(2_000)
    excluded_records = [
        quote(f"X{index:04d}", market_cap=20_000_000 - index) for index in range(300)
    ]
    excluded_listings = [listing(f"X{index:04d}", etf=True) for index in range(300)]
    symbols = directory(*excluded_listings, *valid_directory.records.values())
    records = [*excluded_records, *valid_records]
    calls: list[dict[str, object]] = []
    result = fetch_screener_pages(
        build_query(),
        symbol_directory=symbols,
        target_size=2_000,
        screen_func=paged_screen(records, calls),
    )
    assert len(process_candidates(result.records, symbols).candidates) == 2_000
    assert len(result.records) == 2_300
    assert calls[-1]["offset"] == 2_250


def test_nasdaq_directory_parsing_and_footer_ignored() -> None:
    text = nasdaq_file(
        [
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
            "BRK.B|Example Class B Common Stock|G|N|N|100|N|N",
        ]
    )
    records = parse_nasdaq_listed(text)
    assert len(records) == 2
    assert records[0] == listing("AAPL", security_name="Apple Inc. - Common Stock")
    assert records[1].ticker == "BRK-B"
    assert records[1].raw_symbol == "BRK.B"
    assert records[1].market_category == "G"


def test_other_directory_parsing_and_exchange_mapping() -> None:
    text = other_file(
        [
            "IBM|IBM Common Stock|N|IBM|N|100|N|IBM",
            "LEU|Centrus Energy Class A Common Stock|A|LEU|N|100|N|LEU",
            "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY",
            "MTEST|NYSE Texas Test Common Stock|M|MTEST|N|100|Y|MTEST",
        ]
    )
    records = parse_other_listed(text)
    assert [record.listing_exchange for record in records] == [
        "NYSE",
        "NYSE American",
        "NYSE Arca",
        "NYSE Texas",
    ]
    assert records[2].etf is True


def test_other_directory_normalizes_special_non_common_symbols() -> None:
    text = other_file(
        [
            "ABR$D|Arbor Realty Trust Series D Preferred Stock|N|ABRpD|N|100|N|ABR-D",
            "AAC.U|Ares Acquisition Units|N|AAC.U|N|100|N|AAC=",
        ]
    )
    records = parse_other_listed(text)
    assert records[0].ticker == "ABR-D"
    assert records[0].raw_symbol == "ABR-D"
    assert records[1].ticker == "AAC-U"
    assert records[1].raw_symbol == "AAC.U"


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        (parse_nasdaq_listed, "Symbol|Security Name|ETF\nA|A Common Stock|N\n"),
        (parse_other_listed, "ACT Symbol|Security Name|ETF\nA|A Common Stock|N\n"),
    ],
)
def test_directory_missing_required_columns_fails(parser: Any, text: str) -> None:
    with pytest.raises(UniverseBuildError, match="missing required columns"):
        parser(text)


def test_directory_invalid_boolean_flag_fails_closed() -> None:
    text = nasdaq_file(["AAPL|Apple Common Stock|Q|N|N|100||N"])
    with pytest.raises(UniverseBuildError, match="invalid ETF flag"):
        parse_nasdaq_listed(text)


def test_combined_directory_rejects_conflicting_symbols() -> None:
    nasdaq = nasdaq_file(["DUP|First Common Stock|Q|N|N|100|N|N"])
    other = other_file(["DUP|Second Common Stock|N|DUP|N|100|N|DUP"])
    with pytest.raises(UniverseBuildError, match="conflicting records"):
        build_symbol_directory(nasdaq, other)


def test_directory_download_is_bounded_persisted_and_never_degrades(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    payloads = {
        "nasdaqlisted.txt": nasdaq_file(
            ["AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N"]
        ).encode(),
        "otherlisted.txt": other_file(
            ["IBM|IBM Common Stock|N|IBM|N|100|N|IBM"]
        ).encode(),
    }

    def download(url: str, timeout: float) -> bytes:
        calls.append(url)
        return payloads[Path(url).name]

    symbols = download_symbol_directory(
        tmp_path / "raw", download_func=download, sleep_func=lambda _: None
    )
    assert set(symbols.records) == {"AAPL", "IBM"}
    assert len(calls) == 2
    assert (tmp_path / "raw/nasdaqlisted.txt").read_bytes() == payloads[
        "nasdaqlisted.txt"
    ]

    attempts = 0

    def broken(url: str, timeout: float) -> bytes:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("offline")

    with pytest.raises(UniverseBuildError, match="after 2 attempts"):
        download_symbol_directory(
            tmp_path / "failed",
            max_attempts=2,
            retry_base_delay=0,
            download_func=broken,
            sleep_func=lambda _: None,
        )
    assert attempts == 2
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize(
    ("record", "symbols", "reason"),
    [
        (quote("MISSING"), directory(), "not_in_symbol_directory"),
        (quote("OTC", exchange="PNK"), directory(), "exchange_not_allowed"),
        (
            quote("ARCA", exchange="PCX"),
            directory(listing("ARCA", listing_exchange="NYSE Arca", category=None)),
            "exchange_not_allowed",
        ),
        (quote("BAD/F"), directory(), "invalid_ticker"),
        (quote("AAPL", market_cap=0), directory(listing("AAPL")), "invalid_market_cap"),
    ],
)
def test_basic_fail_closed_candidate_reasons(
    record: dict[str, object], symbols: SymbolDirectory, reason: str
) -> None:
    result = evaluate_candidate(record, symbols)
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == reason


@pytest.mark.parametrize(
    ("ticker", "yahoo_exchange", "listing_exchange", "category"),
    [
        ("NASD", "NMS", "NASDAQ", "Q"),
        ("NYSE", "NYQ", "NYSE", None),
        ("AMEX", "ASE", "NYSE American", None),
    ],
)
def test_target_listing_exchanges_are_allowed(
    ticker: str,
    yahoo_exchange: str,
    listing_exchange: str,
    category: str | None,
) -> None:
    symbols = directory(
        listing(
            ticker,
            listing_exchange=listing_exchange,
            category=category,
            financial_status="N" if listing_exchange == "NASDAQ" else None,
        )
    )
    result = evaluate_candidate(quote(ticker, exchange=yahoo_exchange), symbols)
    assert not isinstance(result, ExcludedSecurity)


def test_nasdaq_market_category_must_match_yahoo_exchange() -> None:
    result = evaluate_candidate(
        quote("CAPM", exchange="NMS"),
        directory(listing("CAPM", category="S")),
    )
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "exchange_not_allowed"


@pytest.mark.parametrize(
    ("etf", "test_issue", "reason"),
    [(True, False, "etf"), (False, True, "test_issue")],
)
def test_etf_and_test_issue_flags_exclude(
    etf: bool, test_issue: bool, reason: str
) -> None:
    symbols = directory(listing("FLAG", etf=etf, test_issue=test_issue))
    result = evaluate_candidate(quote("FLAG"), symbols)
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == reason


@pytest.mark.parametrize(
    "security_name",
    [
        "Issuer American Depositary Shares",
        "Issuer American Depository Receipt",
        "Issuer Depositary Receipt Common Stock",
        "Issuer Depository Share",
        "Issuer ADR",
        "Issuer ADS",
        "Issuer GDR",
    ],
)
def test_adr_ads_gdr_official_names_are_excluded(security_name: str) -> None:
    assert is_probable_adr(security_name)
    result = evaluate_candidate(
        quote("ADRX"), directory(listing("ADRX", security_name=security_name))
    )
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "adr_or_ads"


@pytest.mark.parametrize(
    "security_name",
    [
        "Madrigal Pharmaceuticals Common Stock",
        "Adroit Industries Common Stock",
        "Roadshow Media Common Stock",
    ],
)
def test_adr_word_boundaries_do_not_harm_ordinary_names(security_name: str) -> None:
    assert adr_exclusion_reason(security_name) is None


@pytest.mark.parametrize("ticker", ["ASMLF", "TCEHY", "SSNLF", "BABAF"])
def test_five_character_f_or_y_suffix_is_defensively_excluded(ticker: str) -> None:
    assert is_foreign_otc_suffix(ticker)
    result = evaluate_candidate(quote(ticker), directory(listing(ticker)))
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "foreign_otc_suffix"


@pytest.mark.parametrize("ticker", ["F", "Y", "FIVE", "YARD", "BRK-B", "FOURF1"])
def test_normal_tickers_containing_f_or_y_are_not_suffix_excluded(ticker: str) -> None:
    assert not is_foreign_otc_suffix(ticker)


@pytest.mark.parametrize(
    "security_name",
    [
        "Issuer Preferred Stock",
        "Issuer PFD Series A",
        "Issuer Warrant",
        "Issuer Warrants",
        "Issuer Rights",
        "Issuer Unit",
        "Issuer Units",
        "Issuer ETF",
        "Issuer ETN",
        "Issuer Fund",
        "Issuer Notes",
        "Issuer Bond",
        "Issuer NextShares",
    ],
)
def test_explicit_non_common_security_names_are_excluded(
    security_name: str,
) -> None:
    assert non_common_security_reason(security_name) == "non_common_security"


@pytest.mark.parametrize(
    "security_name",
    [
        "Example REIT",
        "Example Trust Common Stock",
        "Example L.P. Common Stock",
        "Example Shares of Beneficial Interest",
    ],
)
def test_reit_trust_and_lp_are_not_blanket_excluded(security_name: str) -> None:
    assert non_common_security_reason(security_name) is None


def test_ambiguous_security_type_fails_closed() -> None:
    result = evaluate_candidate(
        quote("AMB"),
        directory(listing("AMB", security_name="Ambiguous Corporation Class A")),
    )
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "ambiguous_security_type"


@pytest.mark.parametrize("quote_type", ["ETF", "FUND", "INDEX", "CRYPTOCURRENCY"])
def test_non_equity_quote_type_is_excluded(quote_type: str) -> None:
    result = evaluate_candidate(
        quote("TYPE", quote_type=quote_type), directory(listing("TYPE"))
    )
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "non_common_security"


def test_missing_quote_type_can_still_pass_directory_common_stock_proof() -> None:
    result = evaluate_candidate(
        quote("EQTY", quote_type=None), directory(listing("EQTY"))
    )
    assert not isinstance(result, ExcludedSecurity)
    assert result.quote_type == "EQUITY"


def test_company_name_falls_back_to_official_security_name() -> None:
    record = quote("NAME")
    record.pop("longName")
    result = evaluate_candidate(
        record,
        directory(
            listing("NAME", security_name="Official Name Incorporated Common Stock")
        ),
    )
    assert not isinstance(result, ExcludedSecurity)
    assert result.company_name == "Official Name Incorporated Common Stock"


def test_manual_exclusions_are_optional_normalized_and_audited(tmp_path: Path) -> None:
    assert load_manual_exclusions(tmp_path / "missing.txt") == frozenset()
    path = tmp_path / "exclude_tickers.txt"
    path.write_text("# comment\n\nbrk.b\n AAPL \n", encoding="utf-8")
    exclusions = load_manual_exclusions(path)
    assert exclusions == frozenset({"BRK-B", "AAPL"})
    result = evaluate_candidate(quote("AAPL"), directory(listing("AAPL")), exclusions)
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "manual_exclusion"


def test_invalid_manual_exclusion_has_line_context(tmp_path: Path) -> None:
    path = tmp_path / "exclude_tickers.txt"
    path.write_text("AAPL\nBAD/TICKER\n", encoding="utf-8")
    with pytest.raises(UniverseBuildError, match=r"line 2.*BAD/TICKER"):
        load_manual_exclusions(path)


@pytest.mark.parametrize("market_cap", [None, 0, -1, 1.5, float("nan"), "bad"])
def test_invalid_market_caps_are_excluded(market_cap: object) -> None:
    result = evaluate_candidate(
        quote("CAP", market_cap=market_cap), directory(listing("CAP"))
    )
    assert isinstance(result, ExcludedSecurity)
    assert result.exclusion_reason == "invalid_market_cap"


def test_ticker_normalization_deduplication_and_stable_sort() -> None:
    symbols = directory(listing("BRK-B"), listing("AAA"), listing("ZZZ"))
    records = [
        quote(" zzz ", market_cap=100),
        quote("aaa", market_cap=100),
        quote("BRK.B", market_cap=80),
        quote("AAA", market_cap=90),
    ]
    processed = process_candidates(records, symbols)
    assert [item.ticker for item in processed.candidates] == ["AAA", "ZZZ", "BRK-B"]
    assert [item.market_cap for item in processed.candidates] == [100, 100, 80]
    assert sum(item.exclusion_reason == "duplicate" for item in processed.excluded) == 1
    assert normalize_ticker(" BF.B ") == "BF-B"
    assert normalize_ticker("already-valid") == "ALREADY-VALID"


def test_build_exact_2000_ranked_rows_and_audit() -> None:
    records, symbols = make_large_fixture(2_000)
    rows, audit, processed = build_universe(
        records, symbol_directory=symbols, target_size=2_000
    )
    assert len(rows) == len(audit) == 2_000
    assert len({row.ticker for row in rows}) == 2_000
    assert [row.market_cap_rank for row in rows] == list(range(1, 2_001))
    assert tuple(asdict_keys(rows[0])) == UNIVERSE_COLUMNS
    assert all(not is_foreign_otc_suffix(row.ticker) for row in rows)
    assert len(processed.candidates) == 2_000


def asdict_keys(value: object) -> tuple[str, ...]:
    return tuple(value.__dataclass_fields__)  # type: ignore[attr-defined]


def test_insufficient_candidates_fail_without_relaxing_rules() -> None:
    records, symbols = make_large_fixture(2)
    with pytest.raises(UniverseBuildError, match="needed 3, found 2"):
        build_universe(records, symbol_directory=symbols, target_size=3)


def test_yahoo_retry_is_finite_with_backoff() -> None:
    attempts = 0
    delays: list[float] = []
    symbols = directory(listing("AAA"))

    def flaky(query: object, **kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return {"quotes": [quote("AAA")]}

    result = fetch_screener_pages(
        build_query(),
        symbol_directory=symbols,
        target_size=1,
        max_attempts=3,
        retry_base_delay=0.25,
        screen_func=flaky,
        sleep_func=delays.append,
    )
    assert result.request_attempt_count == 3
    assert delays == [0.25, 0.5]


def write_simple_build_fixture(
    root: Path,
) -> tuple[Path, list[dict[str, object]], SymbolDirectory]:
    output = root / "data/universe/universe.csv"
    records = [quote("AAA", market_cap=300), quote("BBB", market_cap=200)]
    symbols = directory(listing("AAA"), listing("BBB"))
    return output, records, symbols


def test_failed_build_keeps_old_universe(tmp_path: Path) -> None:
    output, records, symbols = write_simple_build_fixture(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_text("old universe\n", encoding="utf-8")
    with pytest.raises(UniverseBuildError, match="needed 3, found 2"):
        run_build(
            target_size=3,
            output_path=output,
            max_candidates=3,
            symbol_directory=symbols,
            screen_func=paged_screen(records, []),
            retry_base_delay=0,
        )
    assert output.read_text(encoding="utf-8") == "old universe\n"
    assert not (output.parent / "archive").exists()


def test_successful_build_archives_and_writes_reports_without_touching_prices(
    tmp_path: Path,
) -> None:
    output, records, symbols = write_simple_build_fixture(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_text("old universe\n", encoding="utf-8")
    prices = tmp_path / "data/processed/prices/sentinel.bin"
    prices.parent.mkdir(parents=True)
    prices.write_bytes(b"do not touch")
    before = prices.stat()
    report = run_build(
        target_size=2,
        output_path=output,
        max_candidates=2,
        symbol_directory=symbols,
        screen_func=paged_screen(records, []),
        retry_base_delay=0,
        generated_at=datetime(2026, 8, 5, 12, 0, tzinfo=UTC),
    )
    archives = list((output.parent / "archive").glob("universe_*.csv"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == "old universe\n"
    assert prices.read_bytes() == b"do not touch"
    assert prices.stat().st_mtime_ns == before.st_mtime_ns
    assert report["validation_passed"] is True
    assert report["source"] == "yfinance_screener_plus_nasdaq_symbol_directory"
    assert report["final_count"] == 2
    assert report["exchange_counts"] == {"NASDAQ": 2}
    assert (
        tuple(
            csv.DictReader(output.open(encoding="utf-8", newline="")).fieldnames or ()
        )
        == UNIVERSE_COLUMNS
    )


def test_build_report_and_review_counts_are_real(tmp_path: Path) -> None:
    output = tmp_path / "data/universe/universe.csv"
    records = [
        quote("AAA", market_cap=500),
        quote("BBB", market_cap=400),
        quote("ETF1", market_cap=900),
        quote("ADR1", market_cap=800),
        quote("MISS", market_cap=700),
        quote("CAP0", market_cap=0),
    ]
    symbols = directory(
        listing("AAA"),
        listing("BBB"),
        listing("ETF1", etf=True),
        listing("ADR1", security_name="Issuer ADS"),
    )
    report = run_build(
        target_size=2,
        output_path=output,
        max_candidates=6,
        symbol_directory=symbols,
        screen_func=paged_screen(records, []),
        retry_base_delay=0,
    )
    assert report["raw_yfinance_candidate_count"] == 6
    assert report["validated_candidate_count"] == 2
    assert report["excluded_etf_count"] == 1
    assert report["excluded_adr_count"] == 1
    assert report["excluded_not_in_symbol_directory_count"] == 1
    assert report["invalid_market_cap_count"] == 1
    persisted = json.loads(
        (output.parent / "universe_build_report.json").read_text(encoding="utf-8")
    )
    assert persisted == report
    with (output.parent / "review/excluded_securities.csv").open(
        encoding="utf-8", newline=""
    ) as review_file:
        reader = csv.DictReader(review_file)
        excluded = list(reader)
    assert tuple(reader.fieldnames or ()) == EXCLUDED_REVIEW_COLUMNS
    assert {row["exclusion_reason"] for row in excluded} == {
        "adr_or_ads",
        "etf",
        "invalid_market_cap",
        "not_in_symbol_directory",
    }
    with (output.parent / "review/final_universe_audit.csv").open(
        encoding="utf-8", newline=""
    ) as audit_file:
        audit_reader = csv.DictReader(audit_file)
        audit = list(audit_reader)
    assert tuple(audit_reader.fieldnames or ()) == FINAL_AUDIT_COLUMNS
    assert [row["ticker"] for row in audit] == ["AAA", "BBB"]


def test_staged_validation_failure_does_not_replace_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, records, symbols = write_simple_build_fixture(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_text("old universe\n", encoding="utf-8")
    real_reader = universe_module._read_universe_csv

    def reject_staging(path: Path):
        if ".universe_build_staging" in str(path):
            raise UniverseBuildError("simulated staged validation failure")
        return real_reader(path)

    monkeypatch.setattr(universe_module, "_read_universe_csv", reject_staging)
    with pytest.raises(UniverseBuildError, match="simulated staged"):
        run_build(
            target_size=2,
            output_path=output,
            max_candidates=2,
            symbol_directory=symbols,
            screen_func=paged_screen(records, []),
        )
    assert output.read_text(encoding="utf-8") == "old universe\n"
    assert not (output.parent / "archive").exists()


def test_atomic_single_file_writer_archives_and_preserves_columns(
    tmp_path: Path,
) -> None:
    records, symbols = make_large_fixture(2)
    rows, _, _ = build_universe(records, symbol_directory=symbols, target_size=2)
    output = tmp_path / "universe.csv"
    output.write_text("old\n", encoding="utf-8")
    archive = write_universe_atomically(output, rows)
    assert archive is not None and archive.read_text(encoding="utf-8") == "old\n"
    assert re.fullmatch(r"universe_\d{4}-\d{2}-\d{2}_\d{6}\.csv", archive.name)


def test_validate_file_works_offline_and_rejects_suffix(tmp_path: Path) -> None:
    output, records, symbols = write_simple_build_fixture(tmp_path)
    run_build(
        target_size=2,
        output_path=output,
        max_candidates=2,
        symbol_directory=symbols,
        screen_func=paged_screen(records, []),
    )
    result = validate_universe_file(output, target_size=2, require_audit=True)
    assert result["rows"] == 2
    assert result["audit_validated"] is True
    text = output.read_text(encoding="utf-8").replace("AAA", "ASMLF", 1)
    output.write_text(text, encoding="utf-8")
    with pytest.raises(UniverseBuildError, match="Foreign OTC suffix"):
        validate_universe_file(output, target_size=2)


def test_run_build_downloads_both_directory_files_with_mocks(tmp_path: Path) -> None:
    output = tmp_path / "data/universe/universe.csv"
    payloads = {
        "nasdaqlisted.txt": nasdaq_file(
            [
                "AAA|Alpha Common Stock|Q|N|N|100|N|N",
                "BBB|Beta Common Stock|Q|N|N|100|N|N",
            ]
        ).encode(),
        "otherlisted.txt": other_file(
            ["IBM|IBM Common Stock|N|IBM|N|100|N|IBM"]
        ).encode(),
    }

    def download(url: str, timeout: float) -> bytes:
        return payloads[Path(url).name]

    report = run_build(
        target_size=2,
        output_path=output,
        max_candidates=2,
        directory_download_func=download,
        screen_func=paged_screen(
            [quote("AAA", market_cap=2), quote("BBB", market_cap=1)], []
        ),
        sleep_func=lambda _: None,
    )
    assert report["symbol_directory_record_count"] == 3
    assert (output.parent / "raw/nasdaqlisted.txt").is_file()
    assert (output.parent / "raw/otherlisted.txt").is_file()


def test_cli_build_validate_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(universe_module, "run_build", lambda **kwargs: {})
    assert main(["build", "--target-size", "2"]) == 0
    monkeypatch.setattr(
        universe_module, "validate_universe_file", lambda *args, **kwargs: {}
    )
    assert main(["validate", "--target-size", "2"]) == 0

    def fail(**kwargs: object) -> dict[str, object]:
        raise UniverseBuildError("expected")

    monkeypatch.setattr(universe_module, "run_build", fail)
    assert main(["build"]) == 1
    with pytest.raises(SystemExit):
        main(["build", "--target-size", "0"])
