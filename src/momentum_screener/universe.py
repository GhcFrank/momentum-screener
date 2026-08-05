"""Build a static, market-cap-ranked US equity universe with yfinance."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yfinance as yf  # type: ignore[import-untyped]
from yfinance import EquityQuery  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)

PAGE_SIZE = 250
DEFAULT_TARGET_SIZE = 2_000
DEFAULT_MAX_CANDIDATES = 4_000
DEFAULT_OUTPUT = Path("data/universe/universe.csv")
SORT_FIELD = "lastclosemarketcap.lasttwelvemonths"
UNIVERSE_COLUMNS = ("ticker", "company_name", "market_cap", "market_cap_rank")
ADR_REVIEW_COLUMNS = ("ticker", "company_name", "market_cap", "exclusion_reason")

_TICKER_PATTERN = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_ADR_WORD_PATTERN = re.compile(r"\bADR\b", re.IGNORECASE)


class UniverseBuildError(RuntimeError):
    """Raised when a universe cannot be built without risking existing output."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A parsed, unranked Yahoo screener candidate."""

    ticker: str
    company_name: str
    market_cap: int


@dataclass(frozen=True, slots=True)
class UniverseRow:
    """A validated row in the final universe."""

    ticker: str
    company_name: str
    market_cap: int
    market_cap_rank: int


@dataclass(frozen=True, slots=True)
class ExcludedCandidate:
    """A candidate omitted by an auditable exclusion rule."""

    ticker: str
    company_name: str
    market_cap: int
    exclusion_reason: str


@dataclass(frozen=True, slots=True)
class ProcessedCandidates:
    """Pure processing results for a collection of raw Yahoo records."""

    candidates: tuple[Candidate, ...]
    adr_excluded: tuple[ExcludedCandidate, ...]
    raw_candidate_count: int
    unique_candidate_count: int
    invalid_record_count: int
    adr_excluded_count: int
    manual_excluded_count: int


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Raw screener records and request accounting."""

    records: tuple[object, ...]
    pages_requested: int
    request_attempt_count: int


ScreenFunction = Callable[..., object]
SleepFunction = Callable[[float], None]


def build_query() -> EquityQuery:
    """Return the Yahoo query defining positive-market-cap US equities."""

    return EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", [SORT_FIELD, 0]),
        ],
    )


def extract_quotes(response: object) -> list[object]:
    """Extract and validate the quote list returned by ``yf.screen``."""

    if not isinstance(response, Mapping):
        raise UniverseBuildError(
            "Yahoo screener returned an invalid response: expected a mapping, "
            f"got {type(response).__name__}"
        )
    if "quotes" not in response:
        raise UniverseBuildError(
            "Yahoo screener returned an invalid response: missing 'quotes' list"
        )
    quotes = response["quotes"]
    if not isinstance(quotes, list):
        raise UniverseBuildError(
            "Yahoo screener returned an invalid response: 'quotes' must be a list, "
            f"got {type(quotes).__name__}"
        )
    return quotes


def normalize_ticker(value: object) -> str | None:
    """Normalize a Yahoo ticker, returning ``None`` for invalid symbols."""

    if not isinstance(value, str):
        return None
    ticker = value.strip().upper().replace(".", "-")
    if not ticker or any(character in ticker for character in "^=/"):
        return None
    if not _TICKER_PATTERN.fullmatch(ticker):
        return None
    return ticker


def adr_exclusion_reason(company_name: str) -> str | None:
    """Return the matched ADR name rule, if any."""

    folded_name = company_name.casefold()
    phrase_rules = (
        ("american depositary", "name_contains_american_depositary"),
        ("depositary receipt", "name_contains_depositary_receipt"),
        ("depositary shares", "name_contains_depositary_shares"),
    )
    for phrase, reason in phrase_rules:
        if phrase in folded_name:
            return reason
    if _ADR_WORD_PATTERN.search(company_name):
        return "name_contains_adr_word"
    return None


def is_probable_adr(company_name: str) -> bool:
    """Return whether a company name matches the auditable ADR name rules."""

    return adr_exclusion_reason(company_name) is not None


def _parse_market_cap(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        market_cap = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        market_cap = int(value)
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
        market_cap = int(decimal_value)
    else:
        return None
    return market_cap if market_cap > 0 else None


def _extract_market_cap(record: Mapping[str, object]) -> int | None:
    for field_name in ("marketCap", SORT_FIELD):
        if field_name in record:
            parsed = _parse_market_cap(record[field_name])
            if parsed is not None:
                return parsed

    nested_market_cap = record.get("lastclosemarketcap")
    if isinstance(nested_market_cap, Mapping):
        return _parse_market_cap(nested_market_cap.get("lasttwelvemonths"))
    return None


def parse_candidate(record: object) -> Candidate | None:
    """Parse one Yahoo record without applying ADR or manual exclusions."""

    if not isinstance(record, Mapping):
        return None

    ticker = normalize_ticker(record.get("symbol"))
    if ticker is None:
        return None

    company_name: str | None = None
    for field_name in ("longName", "shortName", "displayName"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            company_name = value.strip()
            break
    if company_name is None:
        return None

    market_cap = _extract_market_cap(record)
    if market_cap is None:
        return None

    return Candidate(
        ticker=ticker,
        company_name=company_name,
        market_cap=market_cap,
    )


def load_manual_exclusions(path: Path) -> frozenset[str]:
    """Load normalized tickers from an optional comment-friendly text file."""

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
                    f"Invalid ticker in manual exclusion file {path} "
                    f"at line {line_number}: {line!r}"
                )
            exclusions.add(ticker)
    return frozenset(exclusions)


def process_candidates(
    records: Iterable[object],
    manual_exclusions: frozenset[str] = frozenset(),
) -> ProcessedCandidates:
    """Parse, deduplicate, exclude, and sort raw records without I/O."""

    records_list = list(records)
    invalid_record_count = 0
    unique_candidates: dict[str, Candidate] = {}

    for record in records_list:
        candidate = parse_candidate(record)
        if candidate is None:
            invalid_record_count += 1
            continue
        existing = unique_candidates.get(candidate.ticker)
        if existing is None or candidate.market_cap > existing.market_cap:
            unique_candidates[candidate.ticker] = candidate

    included: list[Candidate] = []
    adr_excluded: list[ExcludedCandidate] = []
    manual_excluded_count = 0

    for candidate in unique_candidates.values():
        reason = adr_exclusion_reason(candidate.company_name)
        if reason is not None:
            adr_excluded.append(
                ExcludedCandidate(
                    ticker=candidate.ticker,
                    company_name=candidate.company_name,
                    market_cap=candidate.market_cap,
                    exclusion_reason=reason,
                )
            )
        elif candidate.ticker in manual_exclusions:
            manual_excluded_count += 1
        else:
            included.append(candidate)

    included.sort(key=lambda candidate: (-candidate.market_cap, candidate.ticker))
    adr_excluded.sort(key=lambda candidate: (-candidate.market_cap, candidate.ticker))

    return ProcessedCandidates(
        candidates=tuple(included),
        adr_excluded=tuple(adr_excluded),
        raw_candidate_count=len(records_list),
        unique_candidate_count=len(unique_candidates),
        invalid_record_count=invalid_record_count,
        adr_excluded_count=len(adr_excluded),
        manual_excluded_count=manual_excluded_count,
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
                    "Yahoo screener request failed "
                    f"at offset {offset} after {max_attempts} attempts: {exc}"
                ) from exc
            delay = retry_base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Yahoo screener request failed at offset %d (attempt %d/%d): %s; "
                "retrying in %.1f seconds",
                offset,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep_func(delay)
    raise AssertionError("retry loop exited unexpectedly")


def fetch_screener_pages(
    query: EquityQuery,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    manual_exclusions: frozenset[str] = frozenset(),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    page_size: int = PAGE_SIZE,
    max_attempts: int = 3,
    retry_base_delay: float = 1.0,
    screen_func: ScreenFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> FetchResult:
    """Fetch pages until enough valid unique candidates or a safety stop."""

    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    if not 1 <= page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {PAGE_SIZE}")
    if max_candidates < target_size:
        raise ValueError("max_candidates must be at least target_size")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if retry_base_delay < 0:
        raise ValueError("retry_base_delay cannot be negative")

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
            LOGGER.warning(
                "Yahoo returned no new records at offset %d; stopping pagination",
                offset,
            )
            break

        remaining_capacity = max_candidates - len(records)
        records.extend(new_records[:remaining_capacity])

        processed = process_candidates(records, manual_exclusions)
        if len(processed.candidates) >= target_size:
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
    target_size: int = DEFAULT_TARGET_SIZE,
    manual_exclusions: frozenset[str] = frozenset(),
) -> tuple[tuple[UniverseRow, ...], ProcessedCandidates]:
    """Build and validate exactly ``target_size`` ranked universe rows."""

    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    processed = process_candidates(records, manual_exclusions)
    if len(processed.candidates) < target_size:
        raise UniverseBuildError(
            "Insufficient valid unique candidates: "
            f"needed {target_size}, found {len(processed.candidates)} "
            f"from {processed.raw_candidate_count} raw records "
            f"({processed.invalid_record_count} invalid, "
            f"{processed.adr_excluded_count} ADR excluded, "
            f"{processed.manual_excluded_count} manually excluded)"
        )

    rows = tuple(
        UniverseRow(
            ticker=candidate.ticker,
            company_name=candidate.company_name,
            market_cap=candidate.market_cap,
            market_cap_rank=rank,
        )
        for rank, candidate in enumerate(processed.candidates[:target_size], start=1)
    )
    validate_universe(rows, target_size=target_size)
    return rows, processed


def validate_universe(
    rows: Sequence[UniverseRow],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> None:
    """Validate all invariants required by the persisted universe."""

    if len(rows) != target_size:
        raise UniverseBuildError(
            f"Universe must contain exactly {target_size} rows; found {len(rows)}"
        )

    tickers: list[str] = []
    for expected_rank, row in enumerate(rows, start=1):
        if normalize_ticker(row.ticker) != row.ticker:
            raise UniverseBuildError(f"Invalid normalized ticker: {row.ticker!r}")
        if not isinstance(row.company_name, str) or not row.company_name.strip():
            raise UniverseBuildError(f"Missing company name for ticker {row.ticker}")
        if isinstance(row.market_cap, bool) or not isinstance(row.market_cap, int):
            raise UniverseBuildError(
                f"Market cap must be an integer for ticker {row.ticker}"
            )
        if row.market_cap <= 0:
            raise UniverseBuildError(
                f"Market cap must be positive for ticker {row.ticker}"
            )
        if row.market_cap_rank != expected_rank:
            raise UniverseBuildError(
                f"Expected rank {expected_rank}, found {row.market_cap_rank} "
                f"for ticker {row.ticker}"
            )
        tickers.append(row.ticker)

    if len(set(tickers)) != len(tickers):
        raise UniverseBuildError("Universe contains duplicate tickers")

    expected_order = sorted(rows, key=lambda row: (-row.market_cap, row.ticker))
    if list(rows) != expected_order:
        raise UniverseBuildError(
            "Universe is not sorted by descending market cap and ascending ticker"
        )


def _write_universe_csv(path: Path, rows: Sequence[UniverseRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
        output_file.flush()
        os.fsync(output_file.fileno())


def _read_universe_csv(path: Path) -> tuple[UniverseRow, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != UNIVERSE_COLUMNS:
                raise UniverseBuildError(
                    f"Unexpected universe CSV columns in {path}: {reader.fieldnames}"
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
    except (KeyError, TypeError, ValueError) as exc:
        raise UniverseBuildError(
            f"Unable to parse staged universe CSV {path}: {exc}"
        ) from exc


def _unique_archive_path(archive_dir: Path, generated_at: datetime) -> Path:
    timestamp = generated_at.astimezone(UTC).replace(microsecond=0)
    for seconds_to_add in range(86_400):
        candidate_timestamp = timestamp + timedelta(seconds=seconds_to_add)
        candidate = archive_dir / (
            f"universe_{candidate_timestamp.strftime('%Y-%m-%d_%H%M%S')}.csv"
        )
        if not candidate.exists():
            return candidate
    raise UniverseBuildError(
        f"Could not allocate a unique archive filename in {archive_dir}"
    )


def write_universe_atomically(
    output_path: Path,
    rows: Sequence[UniverseRow],
    *,
    generated_at: datetime | None = None,
) -> Path | None:
    """Validate, stage, archive any old file, and atomically replace output."""

    validate_universe(rows, target_size=len(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at or datetime.now(UTC)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        _write_universe_csv(temp_path, rows)
        staged_rows = _read_universe_csv(temp_path)
        validate_universe(staged_rows, target_size=len(rows))

        archive_path: Path | None = None
        if output_path.exists():
            if not output_path.is_file():
                raise UniverseBuildError(
                    f"Universe output path is not a file: {output_path}"
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


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            json.dump(payload, output_file, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        with temp_path.open(encoding="utf-8") as input_file:
            json.load(input_file)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_adr_review_atomically(
    path: Path, excluded: Sequence[ExcludedCandidate]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=ADR_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(asdict(candidate) for candidate in excluded)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _build_report(
    *,
    generated_at: datetime,
    target_size: int,
    fetch_result: FetchResult,
    processed: ProcessedCandidates,
    rows: Sequence[UniverseRow],
) -> dict[str, object]:
    return {
        "generated_at_utc": generated_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": "yfinance_screener",
        "target_size": target_size,
        "pages_requested": fetch_result.pages_requested,
        "request_attempt_count": fetch_result.request_attempt_count,
        "raw_candidate_count": processed.raw_candidate_count,
        "unique_candidate_count": processed.unique_candidate_count,
        "invalid_record_count": processed.invalid_record_count,
        "adr_excluded_count": processed.adr_excluded_count,
        "manual_excluded_count": processed.manual_excluded_count,
        "final_count": len(rows),
        "largest_market_cap": rows[0].market_cap,
        "smallest_market_cap": rows[-1].market_cap,
    }


def run_build(
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    output_path: Path = DEFAULT_OUTPUT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_attempts: int = 3,
    retry_base_delay: float = 1.0,
    screen_func: ScreenFunction | None = None,
    sleep_func: SleepFunction = time.sleep,
) -> dict[str, object]:
    """Run the complete universe build and return its persisted report."""

    if target_size > max_candidates:
        raise UniverseBuildError(
            f"target_size {target_size} exceeds candidate safety limit {max_candidates}"
        )

    exclusion_path = output_path.parent / "exclude_tickers.txt"
    manual_exclusions = load_manual_exclusions(exclusion_path)
    fetch_result = fetch_screener_pages(
        build_query(),
        target_size=target_size,
        manual_exclusions=manual_exclusions,
        max_candidates=max_candidates,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        screen_func=screen_func,
        sleep_func=sleep_func,
    )
    rows, processed = build_universe(
        fetch_result.records,
        target_size=target_size,
        manual_exclusions=manual_exclusions,
    )
    generated_at = datetime.now(UTC)
    report = _build_report(
        generated_at=generated_at,
        target_size=target_size,
        fetch_result=fetch_result,
        processed=processed,
        rows=rows,
    )

    archive_path = write_universe_atomically(
        output_path,
        rows,
        generated_at=generated_at,
    )
    report_path = output_path.parent / "universe_build_report.json"
    review_path = output_path.parent / "review" / "adr_excluded.csv"
    _write_json_atomically(report_path, report)
    _write_adr_review_atomically(review_path, processed.adr_excluded)

    LOGGER.info(
        "Built %d-ticker universe at %s from %d raw candidates across %d pages",
        len(rows),
        output_path,
        processed.raw_candidate_count,
        fetch_result.pages_requested,
    )
    if archive_path is not None:
        LOGGER.info("Archived previous universe at %s", archive_path)
    LOGGER.info(
        "Wrote build report to %s and ADR review to %s", report_path, review_path
    )
    return report


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.universe",
        description="Build a static US equity universe from the Yahoo Finance screener.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build and persist the universe")
    build_parser.add_argument(
        "--target-size",
        type=_positive_integer,
        default=DEFAULT_TARGET_SIZE,
        help=f"number of ranked tickers to persist (default: {DEFAULT_TARGET_SIZE})",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"universe CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m momentum_screener.universe``."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "build":
        try:
            run_build(target_size=args.target_size, output_path=args.output)
        except (UniverseBuildError, OSError) as exc:
            LOGGER.error("Universe build failed: %s", exc)
            return 1
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
