"""Build a market-cap-ranked US primary-exchange common-stock universe."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.request import Request, urlopen

import yfinance as yf  # type: ignore[import-untyped]
from yfinance import EquityQuery  # type: ignore[import-untyped]

from momentum_screener.storage_manifest import (
    remove_owned_tree,
    replace_files_transactionally,
)

LOGGER = logging.getLogger(__name__)

PAGE_SIZE = 250
DEFAULT_TARGET_SIZE = 2_000
DEFAULT_MAX_CANDIDATES = 5_000
DEFAULT_OUTPUT = Path("data/universe/universe.csv")
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_DOWNLOAD_TIMEOUT = 30.0
SORT_FIELD = "lastclosemarketcap.lasttwelvemonths"
UNIVERSE_COLUMNS = ("ticker", "company_name", "market_cap", "market_cap_rank")
EXCLUDED_REVIEW_COLUMNS = (
    "ticker",
    "company_name",
    "security_name",
    "yahoo_exchange",
    "listing_exchange",
    "market_cap",
    "exclusion_reason",
)
FINAL_AUDIT_COLUMNS = (
    "ticker",
    "company_name",
    "security_name",
    "market_cap",
    "market_cap_rank",
    "yahoo_exchange",
    "listing_exchange",
    "etf",
    "test_issue",
    "quote_type",
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

ALLOWED_YAHOO_EXCHANGES = frozenset({"NMS", "NGM", "NCM", "NYQ", "ASE"})
FORBIDDEN_YAHOO_EXCHANGES = frozenset(
    {"PNK", "OEM", "OQB", "OQX", "PCX", "BTS", "CXI", "YHD"}
)
ALLOWED_LISTING_EXCHANGES = frozenset({"NASDAQ", "NYSE", "NYSE American"})

_NASDAQ_REQUIRED_COLUMNS = frozenset(
    {
        "Symbol",
        "Security Name",
        "Market Category",
        "ETF",
        "Test Issue",
        "Financial Status",
    }
)
_OTHER_REQUIRED_COLUMNS = frozenset(
    {
        "ACT Symbol",
        "Security Name",
        "Exchange",
        "CQS Symbol",
        "ETF",
        "Test Issue",
        "NASDAQ Symbol",
    }
)
_NASDAQ_MARKET_TO_YAHOO = {"Q": "NMS", "G": "NGM", "S": "NCM"}
_OTHER_EXCHANGE_NAMES = {
    "A": "NYSE American",
    "M": "NYSE Texas",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}
_LISTING_TO_YAHOO = {
    "NYSE": frozenset({"NYQ"}),
    "NYSE American": frozenset({"ASE"}),
}

_TICKER_PATTERN = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_ADR_PHRASE_PATTERN = re.compile(
    r"\b(?:"
    r"AMERICAN\s+DEPOSIT(?:ARY|ORY)|"
    r"DEPOSIT(?:ARY|ORY)\s+RECEIPTS?|"
    r"DEPOSIT(?:ARY|ORY)\s+SHARES?"
    r")\b",
    re.IGNORECASE,
)
_ADR_WORD_PATTERN = re.compile(r"\b(?:ADR|ADS|GDR)\b", re.IGNORECASE)
_NON_COMMON_PATTERN = re.compile(
    r"\b(?:ETF|ETN|FUND|WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREFERENCE|PFD|"
    r"DEBENTURES?|BONDS?|NOTES?|NEXTSHARES)\b",
    re.IGNORECASE,
)
_COMMON_SECURITY_PATTERN = re.compile(
    r"\b(?:COMMON\s+STOCK|COMMON\s+SHARES?|ORDINARY\s+SHARES?|CAPITAL\s+STOCK|"
    r"SHARES?\s+OF\s+BENEFICIAL\s+INTEREST|VOTING\s+SHARES?|"
    r"SUBORDINATE\s+VOTING\s+SHARES?|LIMITED\s+PARTNERSHIP\s+INTERESTS?|REIT)\b",
    re.IGNORECASE,
)


class UniverseBuildError(RuntimeError):
    """Raised when a Universe cannot be safely validated or published."""


@dataclass(frozen=True, slots=True)
class SymbolDirectoryRecord:
    """One normalized Nasdaq Trader listing record."""

    raw_symbol: str
    ticker: str
    security_name: str
    listing_exchange: str
    etf: bool
    test_issue: bool
    market_category: str | None = None
    financial_status: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolDirectory:
    """A complete, validated pair of Nasdaq Trader directory files."""

    records: Mapping[str, SymbolDirectoryRecord]
    raw_record_count: int


@dataclass(frozen=True, slots=True)
class Candidate:
    """A fully validated common-stock candidate."""

    ticker: str
    company_name: str
    security_name: str
    market_cap: int
    yahoo_exchange: str
    listing_exchange: str
    quote_type: str


@dataclass(frozen=True, slots=True)
class UniverseRow:
    """A row in the formal Universe CSV."""

    ticker: str
    company_name: str
    market_cap: int
    market_cap_rank: int


@dataclass(frozen=True, slots=True)
class FinalAuditRow:
    """A formal row plus source fields retained for human audit."""

    ticker: str
    company_name: str
    security_name: str
    market_cap: int
    market_cap_rank: int
    yahoo_exchange: str
    listing_exchange: str
    etf: str
    test_issue: str
    quote_type: str


@dataclass(frozen=True, slots=True)
class ExcludedSecurity:
    """One raw Yahoo record excluded with a stable reason."""

    ticker: str
    company_name: str
    security_name: str
    yahoo_exchange: str
    listing_exchange: str
    market_cap: int | None
    exclusion_reason: str


@dataclass(frozen=True, slots=True)
class ProcessedCandidates:
    """Pure qualification output for accumulated Yahoo records."""

    candidates: tuple[Candidate, ...]
    excluded: tuple[ExcludedSecurity, ...]
    raw_candidate_count: int
    unique_candidate_count: int


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Raw screener records and bounded request accounting."""

    records: tuple[object, ...]
    pages_requested: int
    request_attempt_count: int


ScreenFunction = Callable[..., object]
SleepFunction = Callable[[float], None]
DirectoryDownloadFunction = Callable[[str, float], bytes]


def build_query() -> EquityQuery:
    """Build the explicit US-region, primary-exchange, positive-cap query."""

    return EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery(
                "is-in",
                ["exchange", *sorted(ALLOWED_YAHOO_EXCHANGES)],
            ),
            EquityQuery("gt", [SORT_FIELD, 0]),
        ],
    )


def extract_quotes(response: object) -> list[object]:
    """Extract and strictly validate Yahoo's quote list."""

    if not isinstance(response, Mapping):
        raise UniverseBuildError(
            "Yahoo screener returned an invalid response: expected a mapping, "
            f"got {type(response).__name__}"
        )
    quotes = response.get("quotes")
    if not isinstance(quotes, list):
        raise UniverseBuildError(
            "Yahoo screener returned an invalid response: missing or invalid 'quotes' list"
        )
    return quotes


def normalize_ticker(value: object) -> str | None:
    """Normalize a listing symbol into Yahoo's dot-to-hyphen form."""

    if not isinstance(value, str):
        return None
    ticker = value.strip().upper().replace(".", "-")
    if not ticker or any(character in ticker for character in "^=/"):
        return None
    if not _TICKER_PATTERN.fullmatch(ticker):
        return None
    return ticker


def is_foreign_otc_suffix(ticker: str) -> bool:
    """Apply the required five-character F/Y defensive exclusion."""

    return len(ticker) == 5 and ticker.endswith(("F", "Y"))


def adr_exclusion_reason(security_name: str) -> str | None:
    """Return the stable ADR/ADS exclusion enum when a name matches."""

    if _ADR_PHRASE_PATTERN.search(security_name) or _ADR_WORD_PATTERN.search(
        security_name
    ):
        return "adr_or_ads"
    return None


def is_probable_adr(security_name: str) -> bool:
    """Return whether an official Security Name is ADR/ADS/GDR-like."""

    return adr_exclusion_reason(security_name) is not None


def non_common_security_reason(security_name: str) -> str | None:
    """Classify explicit non-common or ambiguous official security names."""

    if _NON_COMMON_PATTERN.search(security_name):
        return "non_common_security"
    if not _COMMON_SECURITY_PATTERN.search(security_name):
        return "ambiguous_security_type"
    return None


def _parse_flag(value: object, *, field: str, source: str, line: int) -> bool:
    normalized = str(value).strip().upper()
    if normalized == "Y":
        return True
    if normalized == "N":
        return False
    raise UniverseBuildError(
        f"{source} has invalid {field} flag at line {line}: {value!r}"
    )


def _directory_rows(
    text: str,
    *,
    source: str,
    required_columns: frozenset[str],
) -> tuple[tuple[dict[str, str], int], ...]:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise UniverseBuildError(f"{source} is empty or invalid")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="|")
    fieldnames = tuple((value or "").strip() for value in (reader.fieldnames or ()))
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise UniverseBuildError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )
    rows: list[tuple[dict[str, str], int]] = []
    for line_number, raw in enumerate(reader, start=2):
        normalized = {
            str(key).strip(): str(value or "").strip()
            for key, value in raw.items()
            if key is not None
        }
        first_value = next(iter(normalized.values()), "")
        if first_value.casefold().startswith("file creation time"):
            continue
        if not any(normalized.values()):
            continue
        rows.append((normalized, line_number))
    if not rows:
        raise UniverseBuildError(f"{source} contains no listing records")
    return tuple(rows)


def parse_nasdaq_listed(text: str) -> tuple[SymbolDirectoryRecord, ...]:
    """Parse and validate ``nasdaqlisted.txt``."""

    records: list[SymbolDirectoryRecord] = []
    for row, line_number in _directory_rows(
        text,
        source="nasdaqlisted.txt",
        required_columns=_NASDAQ_REQUIRED_COLUMNS,
    ):
        raw_symbol = row["Symbol"]
        ticker = normalize_ticker(raw_symbol)
        security_name = row["Security Name"]
        market_category = row["Market Category"].upper()
        financial_status = row["Financial Status"].upper()
        if (
            ticker is None
            or not security_name
            or not market_category
            or not financial_status
        ):
            raise UniverseBuildError(
                f"nasdaqlisted.txt has an invalid critical field at line {line_number}"
            )
        records.append(
            SymbolDirectoryRecord(
                raw_symbol=raw_symbol,
                ticker=ticker,
                security_name=security_name,
                listing_exchange="NASDAQ",
                etf=_parse_flag(
                    row["ETF"], field="ETF", source="nasdaqlisted.txt", line=line_number
                ),
                test_issue=_parse_flag(
                    row["Test Issue"],
                    field="Test Issue",
                    source="nasdaqlisted.txt",
                    line=line_number,
                ),
                market_category=market_category,
                financial_status=financial_status,
            )
        )
    return tuple(records)


def parse_other_listed(text: str) -> tuple[SymbolDirectoryRecord, ...]:
    """Parse and validate ``otherlisted.txt`` with actual exchange codes."""

    records: list[SymbolDirectoryRecord] = []
    for row, line_number in _directory_rows(
        text,
        source="otherlisted.txt",
        required_columns=_OTHER_REQUIRED_COLUMNS,
    ):
        # Prefer the directory's normalized symbol.  Some legitimate
        # non-common rows use ``=``, ``+``, or ``^`` there; their ACT symbol
        # has a dot suffix that can be normalized with the same dot-to-hyphen
        # rule used for ordinary share classes.  Keeping those rows lets the
        # security-name filters reject them instead of treating a valid
        # directory file as corrupt.
        raw_symbol = ""
        ticker: str | None = None
        for symbol_field in ("NASDAQ Symbol", "ACT Symbol", "CQS Symbol"):
            symbol_value = row[symbol_field]
            normalized = normalize_ticker(symbol_value)
            if normalized is not None:
                raw_symbol = symbol_value
                ticker = normalized
                break
        security_name = row["Security Name"]
        exchange_code = row["Exchange"].upper()
        listing_exchange = _OTHER_EXCHANGE_NAMES.get(exchange_code)
        if ticker is None or not security_name or listing_exchange is None:
            raise UniverseBuildError(
                f"otherlisted.txt has an invalid critical field at line {line_number}"
            )
        records.append(
            SymbolDirectoryRecord(
                raw_symbol=raw_symbol,
                ticker=ticker,
                security_name=security_name,
                listing_exchange=listing_exchange,
                etf=_parse_flag(
                    row["ETF"], field="ETF", source="otherlisted.txt", line=line_number
                ),
                test_issue=_parse_flag(
                    row["Test Issue"],
                    field="Test Issue",
                    source="otherlisted.txt",
                    line=line_number,
                ),
            )
        )
    return tuple(records)


def build_symbol_directory(nasdaq_text: str, other_text: str) -> SymbolDirectory:
    """Combine both validated directory files without ambiguous duplicate symbols."""

    source_records = (
        *parse_nasdaq_listed(nasdaq_text),
        *parse_other_listed(other_text),
    )
    indexed: dict[str, SymbolDirectoryRecord] = {}
    for record in source_records:
        existing = indexed.get(record.ticker)
        if existing is not None and existing != record:
            raise UniverseBuildError(
                f"Symbol Directory contains conflicting records for {record.ticker}"
            )
        indexed[record.ticker] = record
    return SymbolDirectory(records=indexed, raw_record_count=len(source_records))


def _default_directory_download(url: str, timeout: float) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "momentum-screener-universe-builder"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_with_retries(
    url: str,
    *,
    max_attempts: int,
    retry_base_delay: float,
    timeout: float,
    download_func: DirectoryDownloadFunction,
    sleep_func: SleepFunction,
) -> bytes:
    for attempt in range(1, max_attempts + 1):
        try:
            payload = download_func(url, timeout)
            if not isinstance(payload, bytes) or not payload:
                raise UniverseBuildError("download returned no bytes")
            return payload
        except Exception as exc:
            if attempt == max_attempts:
                raise UniverseBuildError(
                    f"Symbol Directory download failed for {Path(url).name} after "
                    f"{max_attempts} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            delay = retry_base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Symbol Directory download failed for %s (attempt %d/%d); "
                "retrying in %.1f seconds",
                Path(url).name,
                attempt,
                max_attempts,
                delay,
            )
            sleep_func(delay)
    raise AssertionError("bounded directory retry loop did not terminate")


def _decode_directory_file(payload: bytes, name: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise UniverseBuildError(f"Unable to decode {name} as UTF-8") from exc


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def download_symbol_directory(
    raw_directory: Path,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    download_func: DirectoryDownloadFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> SymbolDirectory:
    """Download, validate, and persist both Nasdaq Trader source files."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if retry_base_delay < 0 or timeout <= 0:
        raise ValueError("retry delay cannot be negative and timeout must be positive")
    call_download = download_func or _default_directory_download
    nasdaq_bytes = _download_with_retries(
        NASDAQ_LISTED_URL,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        timeout=timeout,
        download_func=call_download,
        sleep_func=sleep_func,
    )
    other_bytes = _download_with_retries(
        OTHER_LISTED_URL,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        timeout=timeout,
        download_func=call_download,
        sleep_func=sleep_func,
    )
    nasdaq_text = _decode_directory_file(nasdaq_bytes, "nasdaqlisted.txt")
    other_text = _decode_directory_file(other_bytes, "otherlisted.txt")
    directory = build_symbol_directory(nasdaq_text, other_text)
    _write_bytes_atomically(raw_directory / "nasdaqlisted.txt", nasdaq_bytes)
    _write_bytes_atomically(raw_directory / "otherlisted.txt", other_bytes)
    return directory


def load_manual_exclusions(path: Path) -> frozenset[str]:
    """Load normalized tickers from an optional comment-friendly file."""

    if not path.exists():
        return frozenset()
    if not path.is_file():
        raise UniverseBuildError(f"Manual exclusion path is not a file: {path}")
    exclusions: set[str] = set()
    with path.open(encoding="utf-8") as exclusion_file:
        for line_number, raw_line in enumerate(exclusion_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ticker = normalize_ticker(line)
            if ticker is None:
                raise UniverseBuildError(
                    f"Invalid ticker in manual exclusion file {path} at line "
                    f"{line_number}: {line!r}"
                )
            exclusions.add(ticker)
    return frozenset(exclusions)


def _parse_market_cap(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"[0-9]+(?:\.0+)?", stripped):
            return None
        try:
            decimal_value = Decimal(stripped)
        except InvalidOperation:
            return None
        if decimal_value != decimal_value.to_integral_value():
            return None
        parsed = int(decimal_value)
    else:
        return None
    return parsed if parsed > 0 else None


def _extract_market_cap(record: Mapping[str, object]) -> int | None:
    for field_name in ("marketCap", SORT_FIELD):
        if field_name in record:
            parsed = _parse_market_cap(record[field_name])
            if parsed is not None:
                return parsed
    nested = record.get("lastclosemarketcap")
    if isinstance(nested, Mapping):
        return _parse_market_cap(nested.get("lasttwelvemonths"))
    return None


def _company_name(record: Mapping[str, object]) -> str:
    for field_name in ("longName", "shortName", "displayName"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _excluded(
    *,
    ticker: str,
    company_name: str,
    directory_record: SymbolDirectoryRecord | None,
    yahoo_exchange: str,
    market_cap: int | None,
    reason: str,
) -> ExcludedSecurity:
    return ExcludedSecurity(
        ticker=ticker,
        company_name=company_name,
        security_name=directory_record.security_name if directory_record else "",
        yahoo_exchange=yahoo_exchange,
        listing_exchange=(
            directory_record.listing_exchange if directory_record else ""
        ),
        market_cap=market_cap,
        exclusion_reason=reason,
    )


def _expected_yahoo_exchanges(record: SymbolDirectoryRecord) -> frozenset[str]:
    if record.listing_exchange == "NASDAQ":
        expected = _NASDAQ_MARKET_TO_YAHOO.get(record.market_category or "")
        return frozenset({expected}) if expected else frozenset()
    return _LISTING_TO_YAHOO.get(record.listing_exchange, frozenset())


def evaluate_candidate(
    raw_record: object,
    symbol_directory: SymbolDirectory,
    manual_exclusions: frozenset[str] = frozenset(),
) -> Candidate | ExcludedSecurity:
    """Fail closed while qualifying one raw Yahoo screener record."""

    if not isinstance(raw_record, Mapping):
        return _excluded(
            ticker="",
            company_name="",
            directory_record=None,
            yahoo_exchange="",
            market_cap=None,
            reason="invalid_ticker",
        )
    raw_symbol = raw_record.get("symbol")
    ticker = normalize_ticker(raw_symbol)
    company_name = _company_name(raw_record)
    yahoo_exchange_value = raw_record.get("exchange")
    yahoo_exchange = (
        yahoo_exchange_value.strip().upper()
        if isinstance(yahoo_exchange_value, str)
        else ""
    )
    market_cap = _extract_market_cap(raw_record)
    if ticker is None:
        return _excluded(
            ticker=str(raw_symbol or "").strip().upper(),
            company_name=company_name,
            directory_record=None,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="invalid_ticker",
        )
    if market_cap is None:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=None,
            yahoo_exchange=yahoo_exchange,
            market_cap=None,
            reason="invalid_market_cap",
        )
    if yahoo_exchange not in ALLOWED_YAHOO_EXCHANGES:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=None,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="exchange_not_allowed",
        )
    if is_foreign_otc_suffix(ticker):
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=None,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="foreign_otc_suffix",
        )
    directory_record = symbol_directory.records.get(ticker)
    if directory_record is None:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=None,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="not_in_symbol_directory",
        )
    if (
        directory_record.listing_exchange not in ALLOWED_LISTING_EXCHANGES
        or yahoo_exchange not in _expected_yahoo_exchanges(directory_record)
    ):
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="exchange_not_allowed",
        )
    if directory_record.etf:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="etf",
        )
    if directory_record.test_issue:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="test_issue",
        )
    if (
        directory_record.listing_exchange == "NASDAQ"
        and directory_record.financial_status != "N"
    ):
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="ambiguous_security_type",
        )
    quote_type_value = raw_record.get("quoteType")
    quote_type = (
        quote_type_value.strip().upper() if isinstance(quote_type_value, str) else ""
    )
    if quote_type and quote_type != "EQUITY":
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="non_common_security",
        )
    if ticker in manual_exclusions:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="manual_exclusion",
        )
    if adr_exclusion_reason(directory_record.security_name):
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason="adr_or_ads",
        )
    security_reason = non_common_security_reason(directory_record.security_name)
    if security_reason is not None:
        return _excluded(
            ticker=ticker,
            company_name=company_name,
            directory_record=directory_record,
            yahoo_exchange=yahoo_exchange,
            market_cap=market_cap,
            reason=security_reason,
        )
    return Candidate(
        ticker=ticker,
        company_name=company_name or directory_record.security_name,
        security_name=directory_record.security_name,
        market_cap=market_cap,
        yahoo_exchange=yahoo_exchange,
        listing_exchange=directory_record.listing_exchange,
        quote_type=quote_type or "EQUITY",
    )


def process_candidates(
    records: Iterable[object],
    symbol_directory: SymbolDirectory,
    manual_exclusions: frozenset[str] = frozenset(),
) -> ProcessedCandidates:
    """Qualify, deduplicate, and stably sort accumulated Yahoo records."""

    records_list = list(records)
    valid_by_ticker: dict[str, Candidate] = {}
    excluded: list[ExcludedSecurity] = []
    observed_normalized_tickers: set[str] = set()
    for record in records_list:
        result = evaluate_candidate(record, symbol_directory, manual_exclusions)
        if isinstance(result, ExcludedSecurity):
            excluded.append(result)
            if result.ticker:
                observed_normalized_tickers.add(result.ticker)
            continue
        observed_normalized_tickers.add(result.ticker)
        existing = valid_by_ticker.get(result.ticker)
        if existing is None:
            valid_by_ticker[result.ticker] = result
            continue
        if result.market_cap > existing.market_cap:
            duplicate = existing
            valid_by_ticker[result.ticker] = result
        else:
            duplicate = result
        excluded.append(
            ExcludedSecurity(
                ticker=duplicate.ticker,
                company_name=duplicate.company_name,
                security_name=duplicate.security_name,
                yahoo_exchange=duplicate.yahoo_exchange,
                listing_exchange=duplicate.listing_exchange,
                market_cap=duplicate.market_cap,
                exclusion_reason="duplicate",
            )
        )
    candidates = sorted(
        valid_by_ticker.values(), key=lambda item: (-item.market_cap, item.ticker)
    )
    excluded.sort(
        key=lambda item: (
            item.exclusion_reason,
            item.ticker,
            -(item.market_cap or 0),
        )
    )
    return ProcessedCandidates(
        candidates=tuple(candidates),
        excluded=tuple(excluded),
        raw_candidate_count=len(records_list),
        unique_candidate_count=len(observed_normalized_tickers),
    )


def _record_fingerprint(record: object) -> str:
    try:
        return json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(record)


def _request_page_with_retries(
    query: EquityQuery,
    *,
    offset: int,
    page_size: int,
    max_attempts: int,
    retry_base_delay: float,
    screen_func: ScreenFunction,
    sleep_func: SleepFunction,
) -> tuple[object, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = screen_func(
                query,
                offset=offset,
                size=page_size,
                sortField=SORT_FIELD,
                sortAsc=False,
            )
            return response, attempt
        except Exception as exc:
            if attempt == max_attempts:
                raise UniverseBuildError(
                    f"Yahoo screener request failed at offset {offset} after "
                    f"{max_attempts} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            delay = retry_base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Yahoo screener request failed at offset %d (attempt %d/%d); "
                "retrying in %.1f seconds",
                offset,
                attempt,
                max_attempts,
                delay,
            )
            sleep_func(delay)
    raise AssertionError("bounded Yahoo retry loop did not terminate")


def fetch_screener_pages(
    query: EquityQuery,
    *,
    symbol_directory: SymbolDirectory,
    target_size: int = DEFAULT_TARGET_SIZE,
    manual_exclusions: frozenset[str] = frozenset(),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    page_size: int = PAGE_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    screen_func: ScreenFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> FetchResult:
    """Page until 2,000 fully validated candidates, exhaustion, or safety cap."""

    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    if not 1 <= page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {PAGE_SIZE}")
    if max_candidates < target_size:
        raise ValueError("max_candidates must be at least target_size")
    if max_attempts <= 0 or retry_base_delay < 0:
        raise ValueError("retry settings are invalid")
    call_screen = screen_func or yf.screen
    records: list[object] = []
    seen_fingerprints: set[str] = set()
    pages_requested = 0
    request_attempt_count = 0
    offset = 0
    while len(records) < max_candidates:
        response, attempts = _request_page_with_retries(
            query,
            offset=offset,
            page_size=page_size,
            max_attempts=max_attempts,
            retry_base_delay=retry_base_delay,
            screen_func=call_screen,
            sleep_func=sleep_func,
        )
        pages_requested += 1
        request_attempt_count += attempts
        quotes = extract_quotes(response)
        if not quotes:
            break
        new_records: list[object] = []
        for record in quotes:
            fingerprint = _record_fingerprint(record)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                new_records.append(record)
        if not new_records:
            LOGGER.warning("Yahoo returned no new records at offset %d", offset)
            break
        records.extend(new_records[: max_candidates - len(records)])
        processed = process_candidates(records, symbol_directory, manual_exclusions)
        if len(processed.candidates) >= target_size:
            break
        if len(quotes) < page_size:
            break
        offset += page_size
    return FetchResult(
        records=tuple(records),
        pages_requested=pages_requested,
        request_attempt_count=request_attempt_count,
    )


def build_universe(
    records: Iterable[object],
    *,
    symbol_directory: SymbolDirectory,
    target_size: int = DEFAULT_TARGET_SIZE,
    manual_exclusions: frozenset[str] = frozenset(),
) -> tuple[tuple[UniverseRow, ...], tuple[FinalAuditRow, ...], ProcessedCandidates]:
    """Build exactly ``target_size`` ranked rows plus a source-backed audit."""

    processed = process_candidates(records, symbol_directory, manual_exclusions)
    if len(processed.candidates) < target_size:
        reasons = Counter(item.exclusion_reason for item in processed.excluded)
        raise UniverseBuildError(
            f"Insufficient validated common stocks: needed {target_size}, found "
            f"{len(processed.candidates)} from {processed.raw_candidate_count} raw "
            f"candidates; exclusions={dict(sorted(reasons.items()))}"
        )
    selected = processed.candidates[:target_size]
    rows = tuple(
        UniverseRow(
            ticker=candidate.ticker,
            company_name=candidate.company_name,
            market_cap=candidate.market_cap,
            market_cap_rank=rank,
        )
        for rank, candidate in enumerate(selected, start=1)
    )
    audit = tuple(
        FinalAuditRow(
            ticker=candidate.ticker,
            company_name=candidate.company_name,
            security_name=candidate.security_name,
            market_cap=candidate.market_cap,
            market_cap_rank=rank,
            yahoo_exchange=candidate.yahoo_exchange,
            listing_exchange=candidate.listing_exchange,
            etf="N",
            test_issue="N",
            quote_type=candidate.quote_type,
        )
        for rank, candidate in enumerate(selected, start=1)
    )
    validate_universe(rows, target_size=target_size)
    validate_final_audit(
        rows,
        audit,
        manual_exclusions=manual_exclusions,
        target_size=target_size,
    )
    return rows, audit, processed


def validate_universe(
    rows: Sequence[UniverseRow],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> None:
    """Validate every formal CSV invariant, including F/Y suffix defenses."""

    if len(rows) != target_size:
        raise UniverseBuildError(
            f"Universe must contain exactly {target_size} rows; found {len(rows)}"
        )
    tickers: list[str] = []
    for expected_rank, row in enumerate(rows, start=1):
        if normalize_ticker(row.ticker) != row.ticker:
            raise UniverseBuildError(f"Invalid normalized ticker: {row.ticker!r}")
        if is_foreign_otc_suffix(row.ticker):
            raise UniverseBuildError(
                f"Foreign OTC suffix ticker is forbidden: {row.ticker}"
            )
        if not row.company_name.strip():
            raise UniverseBuildError(f"Missing company name for {row.ticker}")
        if (
            isinstance(row.market_cap, bool)
            or not isinstance(row.market_cap, int)
            or row.market_cap <= 0
        ):
            raise UniverseBuildError(f"Invalid market cap for {row.ticker}")
        if row.market_cap_rank != expected_rank:
            raise UniverseBuildError(
                f"Expected rank {expected_rank}, found {row.market_cap_rank}"
            )
        tickers.append(row.ticker)
    if len(set(tickers)) != len(tickers):
        raise UniverseBuildError("Universe contains duplicate tickers")
    if list(rows) != sorted(rows, key=lambda item: (-item.market_cap, item.ticker)):
        raise UniverseBuildError(
            "Universe is not sorted by descending market cap and ascending ticker"
        )


def validate_final_audit(
    rows: Sequence[UniverseRow],
    audit: Sequence[FinalAuditRow],
    *,
    manual_exclusions: frozenset[str] = frozenset(),
    target_size: int = DEFAULT_TARGET_SIZE,
) -> None:
    """Validate source exchange and security-type evidence for every final row."""

    if len(audit) != target_size or len(rows) != target_size:
        raise UniverseBuildError("Final audit row count differs from Universe")
    for row, evidence in zip(rows, audit, strict=True):
        if (
            evidence.ticker != row.ticker
            or evidence.company_name != row.company_name
            or evidence.market_cap != row.market_cap
            or evidence.market_cap_rank != row.market_cap_rank
        ):
            raise UniverseBuildError(f"Final audit mismatch for {row.ticker}")
        if evidence.yahoo_exchange not in ALLOWED_YAHOO_EXCHANGES:
            raise UniverseBuildError(f"Disallowed Yahoo exchange for {row.ticker}")
        if evidence.listing_exchange not in ALLOWED_LISTING_EXCHANGES:
            raise UniverseBuildError(f"Disallowed listing exchange for {row.ticker}")
        if evidence.etf != "N" or evidence.test_issue != "N":
            raise UniverseBuildError(f"ETF or Test Issue in final audit: {row.ticker}")
        if evidence.quote_type.upper() != "EQUITY":
            raise UniverseBuildError(f"Non-equity quoteType for {row.ticker}")
        if adr_exclusion_reason(evidence.security_name):
            raise UniverseBuildError(f"ADR/ADS in final audit: {row.ticker}")
        if non_common_security_reason(evidence.security_name) is not None:
            raise UniverseBuildError(
                f"Non-common security in final audit: {row.ticker}"
            )
        if row.ticker in manual_exclusions:
            raise UniverseBuildError(
                f"Manually excluded ticker in final audit: {row.ticker}"
            )


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_universe_csv(path: Path, rows: Sequence[UniverseRow]) -> None:
    _write_csv(
        path,
        fieldnames=UNIVERSE_COLUMNS,
        rows=(asdict(row) for row in rows),
    )


def _read_universe_csv(path: Path) -> tuple[UniverseRow, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != UNIVERSE_COLUMNS:
                raise UniverseBuildError(
                    f"Unexpected Universe CSV columns in {path}: {reader.fieldnames}"
                )
            return tuple(
                UniverseRow(
                    ticker=row["ticker"],
                    company_name=row["company_name"],
                    market_cap=int(row["market_cap"]),
                    market_cap_rank=int(row["market_cap_rank"]),
                )
                for row in reader
            )
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        if isinstance(exc, UniverseBuildError):
            raise
        raise UniverseBuildError(f"Unable to parse Universe CSV {path}: {exc}") from exc


def _read_final_audit(path: Path) -> tuple[FinalAuditRow, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != FINAL_AUDIT_COLUMNS:
                raise UniverseBuildError(
                    f"Unexpected final audit columns in {path}: {reader.fieldnames}"
                )
            return tuple(
                FinalAuditRow(
                    ticker=row["ticker"],
                    company_name=row["company_name"],
                    security_name=row["security_name"],
                    market_cap=int(row["market_cap"]),
                    market_cap_rank=int(row["market_cap_rank"]),
                    yahoo_exchange=row["yahoo_exchange"],
                    listing_exchange=row["listing_exchange"],
                    etf=row["etf"],
                    test_issue=row["test_issue"],
                    quote_type=row["quote_type"],
                )
                for row in reader
            )
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        if isinstance(exc, UniverseBuildError):
            raise
        raise UniverseBuildError(f"Unable to parse final audit {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _unique_archive_path(archive_dir: Path, generated_at: datetime) -> Path:
    timestamp = generated_at.astimezone(UTC).replace(microsecond=0)
    for seconds_to_add in range(86_400):
        candidate_timestamp = timestamp + timedelta(seconds=seconds_to_add)
        candidate = archive_dir / (
            f"universe_{candidate_timestamp.strftime('%Y-%m-%d_%H%M%S')}.csv"
        )
        if not candidate.exists():
            return candidate
    raise UniverseBuildError(f"Unable to allocate archive path in {archive_dir}")


def write_universe_atomically(
    output_path: Path,
    rows: Sequence[UniverseRow],
    *,
    generated_at: datetime | None = None,
) -> Path | None:
    """Retain the legacy single-file safe writer used by callers and tests."""

    validate_universe(rows, target_size=len(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at or datetime.now(UTC)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        _write_universe_csv(temp_path, rows)
        validate_universe(_read_universe_csv(temp_path), target_size=len(rows))
        archive_path: Path | None = None
        if output_path.exists():
            if not output_path.is_file():
                raise UniverseBuildError(
                    f"Universe output is not a file: {output_path}"
                )
            archive_dir = output_path.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = _unique_archive_path(archive_dir, timestamp)
            shutil.copy2(output_path, archive_path)
        os.replace(temp_path, output_path)
        temp_path = None
        return archive_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _build_report(
    *,
    generated_at: datetime,
    target_size: int,
    fetch_result: FetchResult,
    symbol_directory: SymbolDirectory,
    processed: ProcessedCandidates,
    rows: Sequence[UniverseRow],
    audit: Sequence[FinalAuditRow],
) -> dict[str, object]:
    reason_counts = Counter(item.exclusion_reason for item in processed.excluded)
    exchange_counts = Counter(item.listing_exchange for item in audit)
    tickers = [row.ticker for row in rows]
    return {
        "generated_at_utc": generated_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": "yfinance_screener_plus_nasdaq_symbol_directory",
        "target_size": target_size,
        "pages_requested": fetch_result.pages_requested,
        "request_attempt_count": fetch_result.request_attempt_count,
        "raw_yfinance_candidate_count": processed.raw_candidate_count,
        "symbol_directory_record_count": symbol_directory.raw_record_count,
        "validated_candidate_count": len(processed.candidates),
        "final_count": len(rows),
        "exchange_counts": dict(sorted(exchange_counts.items())),
        "excluded_exchange_count": reason_counts["exchange_not_allowed"],
        "excluded_not_in_symbol_directory_count": reason_counts[
            "not_in_symbol_directory"
        ],
        "excluded_adr_count": reason_counts["adr_or_ads"],
        "excluded_foreign_otc_suffix_count": reason_counts["foreign_otc_suffix"],
        "excluded_etf_count": reason_counts["etf"],
        "excluded_test_issue_count": reason_counts["test_issue"],
        "excluded_non_common_security_count": reason_counts["non_common_security"],
        "excluded_ambiguous_security_type_count": reason_counts[
            "ambiguous_security_type"
        ],
        "manual_excluded_count": reason_counts["manual_exclusion"],
        "invalid_market_cap_count": reason_counts["invalid_market_cap"],
        "invalid_ticker_count": reason_counts["invalid_ticker"],
        "duplicate_count": reason_counts["duplicate"],
        "largest_market_cap": rows[0].market_cap,
        "smallest_market_cap": rows[-1].market_cap,
        "five_char_ending_f_count": sum(
            len(ticker) == 5 and ticker.endswith("F") for ticker in tickers
        ),
        "five_char_ending_y_count": sum(
            len(ticker) == 5 and ticker.endswith("Y") for ticker in tickers
        ),
        "validation_passed": True,
    }


def _stage_build_outputs(
    staging_root: Path,
    *,
    output_name: str,
    rows: Sequence[UniverseRow],
    audit: Sequence[FinalAuditRow],
    excluded: Sequence[ExcludedSecurity],
    report: Mapping[str, object],
) -> None:
    _write_universe_csv(staging_root / output_name, rows)
    _write_csv(
        staging_root / "review/excluded_securities.csv",
        fieldnames=EXCLUDED_REVIEW_COLUMNS,
        rows=(asdict(item) for item in excluded),
    )
    _write_csv(
        staging_root / "review/final_universe_audit.csv",
        fieldnames=FINAL_AUDIT_COLUMNS,
        rows=(asdict(item) for item in audit),
    )
    _write_json(staging_root / "universe_build_report.json", report)


def validate_universe_file(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    audit_path: Path | None = None,
    exclusion_path: Path | None = None,
    require_audit: bool = False,
) -> dict[str, object]:
    """Validate an existing Universe without any network access."""

    rows = _read_universe_csv(output_path)
    validate_universe(rows, target_size=target_size)
    resolved_audit = (
        audit_path or output_path.parent / "review/final_universe_audit.csv"
    )
    manual_exclusions = load_manual_exclusions(
        exclusion_path or output_path.parent / "exclude_tickers.txt"
    )
    audit_validated = False
    exchange_counts: dict[str, int] = {}
    if resolved_audit.is_file():
        audit = _read_final_audit(resolved_audit)
        validate_final_audit(
            rows,
            audit,
            manual_exclusions=manual_exclusions,
            target_size=target_size,
        )
        audit_validated = True
        exchange_counts = dict(
            sorted(Counter(item.listing_exchange for item in audit).items())
        )
    elif require_audit:
        raise UniverseBuildError(f"Final audit file is missing: {resolved_audit}")
    report_path = output_path.parent / "universe_build_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UniverseBuildError(f"Build report is invalid: {exc}") from exc
        if (
            not isinstance(report, Mapping)
            or report.get("validation_passed") is not True
            or report.get("final_count") != target_size
        ):
            raise UniverseBuildError("Build report does not confirm this Universe")
    result = {
        "rows": len(rows),
        "unique_tickers": len({row.ticker for row in rows}),
        "five_char_ending_f": sum(
            len(row.ticker) == 5 and row.ticker.endswith("F") for row in rows
        ),
        "five_char_ending_y": sum(
            len(row.ticker) == 5 and row.ticker.endswith("Y") for row in rows
        ),
        "rank_min": rows[0].market_cap_rank,
        "rank_max": rows[-1].market_cap_rank,
        "audit_validated": audit_validated,
        "exchange_counts": exchange_counts,
    }
    LOGGER.info("Validated Universe at %s: %s", output_path, result)
    return result


def _publish_build_outputs(
    output_path: Path,
    staging_root: Path,
    *,
    generated_at: datetime,
    rows: Sequence[UniverseRow],
    audit: Sequence[FinalAuditRow],
    manual_exclusions: frozenset[str],
) -> Path | None:
    staged_output = staging_root / output_path.name
    staged_audit = staging_root / "review/final_universe_audit.csv"
    staged_rows = _read_universe_csv(staged_output)
    validate_universe(staged_rows, target_size=len(rows))
    staged_audit_rows = _read_final_audit(staged_audit)
    validate_final_audit(
        staged_rows,
        staged_audit_rows,
        manual_exclusions=manual_exclusions,
        target_size=len(rows),
    )
    report = json.loads(
        (staging_root / "universe_build_report.json").read_text(encoding="utf-8")
    )
    if report.get("validation_passed") is not True or report.get("final_count") != len(
        rows
    ):
        raise UniverseBuildError("Staged build report failed validation")

    archive_path: Path | None = None
    if output_path.exists():
        if not output_path.is_file():
            raise UniverseBuildError(f"Universe output is not a file: {output_path}")
        archive_dir = output_path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = _unique_archive_path(archive_dir, generated_at)
        shutil.copy2(output_path, archive_path)

    backup_parent = output_path.parent / ".universe_build_backup"
    build_id = staging_root.name
    backup_root = backup_parent / build_id
    replacement_paths = [
        "review/excluded_securities.csv",
        "review/final_universe_audit.csv",
        "universe_build_report.json",
        output_path.name,
    ]

    def validate_published() -> None:
        validate_universe_file(
            output_path,
            target_size=len(rows),
            audit_path=output_path.parent / "review/final_universe_audit.csv",
            require_audit=True,
        )

    replace_files_transactionally(
        output_path.parent,
        staging_root,
        replacement_paths,
        backup_root=backup_root,
        validate_after=validate_published,
    )
    if backup_root.exists():
        remove_owned_tree(backup_root, parent=backup_parent, prefix="build-")
    return archive_path


def run_build(
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    output_path: Path = DEFAULT_OUTPUT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    screen_func: ScreenFunction | None = None,
    directory_download_func: DirectoryDownloadFunction | None = None,
    symbol_directory: SymbolDirectory | None = None,
    sleep_func: SleepFunction = time.sleep,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Download both sources, build, validate, archive, and atomically publish."""

    if target_size > max_candidates:
        raise UniverseBuildError(
            f"target_size {target_size} exceeds safety limit {max_candidates}"
        )
    exclusion_path = output_path.parent / "exclude_tickers.txt"
    manual_exclusions = load_manual_exclusions(exclusion_path)
    directory = symbol_directory or download_symbol_directory(
        output_path.parent / "raw",
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        timeout=download_timeout,
        download_func=directory_download_func,
        sleep_func=sleep_func,
    )
    fetch_result = fetch_screener_pages(
        build_query(),
        symbol_directory=directory,
        target_size=target_size,
        manual_exclusions=manual_exclusions,
        max_candidates=max_candidates,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        screen_func=screen_func,
        sleep_func=sleep_func,
    )
    rows, audit, processed = build_universe(
        fetch_result.records,
        symbol_directory=directory,
        target_size=target_size,
        manual_exclusions=manual_exclusions,
    )
    timestamp = generated_at or datetime.now(UTC)
    report = _build_report(
        generated_at=timestamp,
        target_size=target_size,
        fetch_result=fetch_result,
        symbol_directory=directory,
        processed=processed,
        rows=rows,
        audit=audit,
    )
    staging_parent = output_path.parent / ".universe_build_staging"
    staging_root = staging_parent / f"build-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        _stage_build_outputs(
            staging_root,
            output_name=output_path.name,
            rows=rows,
            audit=audit,
            excluded=processed.excluded,
            report=report,
        )
        archive_path = _publish_build_outputs(
            output_path,
            staging_root,
            generated_at=timestamp,
            rows=rows,
            audit=audit,
            manual_exclusions=manual_exclusions,
        )
    finally:
        if staging_root.exists():
            remove_owned_tree(staging_root, parent=staging_parent, prefix="build-")
    LOGGER.info(
        "Built %d common stocks at %s from %d Yahoo candidates and %d directory records",
        len(rows),
        output_path,
        processed.raw_candidate_count,
        directory.raw_record_count,
    )
    if archive_path is not None:
        LOGGER.info("Archived prior Universe at %s", archive_path)
    return report


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.universe",
        description=(
            "Build a primary-US-exchange common-stock Universe from Yahoo and "
            "Nasdaq Trader."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build and persist the Universe")
    build_parser.add_argument(
        "--target-size", type=_positive_integer, default=DEFAULT_TARGET_SIZE
    )
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument(
        "--max-candidates", type=_positive_integer, default=DEFAULT_MAX_CANDIDATES
    )
    build_parser.add_argument(
        "--max-attempts", type=_positive_integer, default=DEFAULT_MAX_ATTEMPTS
    )
    build_parser.add_argument(
        "--retry-base-delay",
        type=_nonnegative_float,
        default=DEFAULT_RETRY_BASE_DELAY,
    )
    build_parser.add_argument(
        "--download-timeout", type=_positive_float, default=DEFAULT_DOWNLOAD_TIMEOUT
    )
    validate_parser = subparsers.add_parser(
        "validate", help="validate the current Universe without network access"
    )
    validate_parser.add_argument(
        "--target-size", type=_positive_integer, default=DEFAULT_TARGET_SIZE
    )
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser.add_argument("--require-audit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "build":
            run_build(
                target_size=args.target_size,
                output_path=args.output,
                max_candidates=args.max_candidates,
                max_attempts=args.max_attempts,
                retry_base_delay=args.retry_base_delay,
                download_timeout=args.download_timeout,
            )
        elif args.command == "validate":
            validate_universe_file(
                args.output,
                target_size=args.target_size,
                require_audit=args.require_audit,
            )
        else:
            parser.error(f"unsupported command: {args.command}")
    except (UniverseBuildError, OSError, ValueError) as exc:
        LOGGER.error("Universe operation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
