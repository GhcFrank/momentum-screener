"""Maintain presentation-only Yahoo company classifications for the Universe."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock

import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

from momentum_screener.prices import DEFAULT_UNIVERSE, load_universe
from momentum_screener.universe import normalize_ticker

LOGGER = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = Path("data/universe/ticker_metadata.csv")
METADATA_COLUMNS = ("ticker", "sector", "industry", "updated_at")
DEFAULT_MAX_WORKERS = 4
MAX_WORKERS = 16
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.5
MetadataProvider = Callable[[str], Mapping[str, object]]


class CompanyMetadataError(ValueError):
    """Raised when company metadata cannot be read or safely refreshed."""


@dataclass(frozen=True, slots=True)
class CompanyMetadataRefreshResult:
    universe_count: int
    requested_count: int
    request_failure_count: int
    sector_available_count: int
    industry_available_count: int
    missing_sector_count: int
    missing_industry_count: int
    output_path: Path


def _empty_company_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="string") for column in METADATA_COLUMNS},
        columns=METADATA_COLUMNS,
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def format_metadata_value(value: object) -> str:
    """Convert one optional classification to its presentation fallback."""

    return _optional_text(value) or "N/A"


def load_company_metadata(
    path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Load normalized metadata; a missing file is an empty, usable lookup."""

    if not path.exists():
        return _empty_company_metadata()
    if not path.is_file():
        raise CompanyMetadataError(f"Company metadata path is not a file: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CompanyMetadataError(
            f"Unable to read company metadata CSV {path}: {exc}"
        ) from exc
    if "ticker" not in frame:
        raise CompanyMetadataError(
            f"Company metadata CSV is missing required 'ticker' column: {path}"
        )
    for column in METADATA_COLUMNS[1:]:
        if column not in frame:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")

    normalized: list[str] = []
    for row_number, value in enumerate(frame["ticker"], start=2):
        ticker = normalize_ticker(value)
        if ticker is None:
            raise CompanyMetadataError(
                f"Invalid ticker at {path}:{row_number}: {value!r}"
            )
        normalized.append(ticker)
    frame = frame.loc[:, METADATA_COLUMNS].copy()
    frame["ticker"] = pd.Series(normalized, dtype="string")
    if frame["ticker"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["ticker"].duplicated(False), "ticker"].unique()
        )
        raise CompanyMetadataError(
            f"Company metadata contains duplicate tickers: {', '.join(duplicates[:5])}"
        )
    for column in METADATA_COLUMNS[1:]:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.eq(""), pd.NA)
    return frame.sort_values("ticker", kind="mergesort", ignore_index=True)


def enrich_company_metadata(
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Left-join classifications without changing row order or other columns."""

    if ticker_column not in rows:
        raise CompanyMetadataError(f"Rows are missing ticker column: {ticker_column}")
    attrs = rows.attrs.copy()
    result = rows.drop(columns=["sector", "industry"], errors="ignore").copy()
    result["__metadata_order"] = range(len(result))
    result["__metadata_ticker"] = (
        result[ticker_column].map(normalize_ticker).astype("string")
    )
    lookup = metadata.reindex(columns=METADATA_COLUMNS).copy()
    if not lookup.empty and lookup["ticker"].duplicated().any():
        raise CompanyMetadataError("Company metadata contains duplicate tickers")
    lookup = lookup.rename(columns={"ticker": "__metadata_ticker"})
    lookup["__metadata_ticker"] = lookup["__metadata_ticker"].astype("string")
    result = result.merge(
        lookup,
        on="__metadata_ticker",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.sort_values("__metadata_order", kind="mergesort").drop(
        columns=["__metadata_order", "__metadata_ticker"]
    )
    result.index = rows.index
    result.attrs = attrs
    return result


def get_company_metadata(
    tickers: Sequence[str],
    metadata: pd.DataFrame | None = None,
    *,
    path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Return an order-preserving metadata lookup for the requested tickers."""

    source = load_company_metadata(path) if metadata is None else metadata
    keys = pd.DataFrame({"ticker": pd.Series(tickers, dtype="string")})
    return enrich_company_metadata(keys, source)


def fetch_yahoo_company_metadata(ticker: str) -> Mapping[str, object]:
    """Fetch Yahoo's own classification for one ticker."""

    info = yf.Ticker(ticker).get_info()
    if not isinstance(info, Mapping):
        raise CompanyMetadataError(
            f"Yahoo returned {type(info).__name__}, expected a mapping"
        )
    return {"sector": info.get("sector"), "industry": info.get("industry")}


def _write_company_metadata_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            frame.to_csv(output, index=False, na_rep="")
            output.flush()
            os.fsync(output.fileno())
        load_company_metadata(temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refresh_company_metadata(
    *,
    force: bool = False,
    universe_path: Path = DEFAULT_UNIVERSE,
    output_path: Path = DEFAULT_METADATA_PATH,
    provider: MetadataProvider = fetch_yahoo_company_metadata,
    max_workers: int = DEFAULT_MAX_WORKERS,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    refreshed_on: date | None = None,
) -> CompanyMetadataRefreshResult:
    """Refresh missing or all current-Universe metadata with bounded concurrency."""

    if not 1 <= max_workers <= MAX_WORKERS:
        raise CompanyMetadataError(f"max_workers must be between 1 and {MAX_WORKERS}")
    if request_interval_seconds < 0:
        raise CompanyMetadataError("request_interval_seconds cannot be negative")
    tickers = load_universe(universe_path)
    existing = load_company_metadata(output_path)
    current = get_company_metadata(tickers, existing).loc[:, METADATA_COLUMNS].copy()
    missing = current[["sector", "industry"]].isna().any(axis=1)
    targets = list(tickers) if force else current.loc[missing, "ticker"].tolist()
    LOGGER.info(
        "Refreshing company metadata universe=%d requested=%d force=%s",
        len(tickers),
        len(targets),
        force,
    )

    responses: dict[str, Mapping[str, object]] = {}
    failures: dict[str, Exception] = {}
    if targets:
        request_lock = Lock()
        next_request_at = [time.monotonic()]

        def request(ticker: str) -> Mapping[str, object]:
            # Space starts globally while still allowing a few requests in flight.
            with request_lock:
                delay = next_request_at[0] - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                next_request_at[0] = time.monotonic() + request_interval_seconds
            return provider(ticker)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(request, ticker): ticker for ticker in targets}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    response = future.result()
                    if not isinstance(response, Mapping):
                        raise CompanyMetadataError(
                            f"provider returned {type(response).__name__}, expected a mapping"
                        )
                    if not any(
                        _optional_text(response.get(column)) is not None
                        for column in ("sector", "industry")
                    ):
                        raise CompanyMetadataError(
                            "provider returned neither sector nor industry"
                        )
                    responses[ticker] = response
                except Exception as exc:  # noqa: BLE001
                    # A provider failure for one ticker must not cancel the batch.
                    failures[ticker] = exc
                    if len(failures) <= 5:
                        LOGGER.warning(
                            "Company metadata request failed for %s: %s", ticker, exc
                        )
                    else:
                        LOGGER.debug(
                            "Company metadata request failed for %s: %s", ticker, exc
                        )
    if len(failures) > 5:
        LOGGER.warning(
            "%d additional company metadata requests failed; see DEBUG logs for details",
            len(failures) - 5,
        )

    refreshed_date = (refreshed_on or datetime.now(UTC).date()).isoformat()
    current = current.set_index("ticker", drop=False)
    for ticker, response in responses.items():
        refreshed_any = False
        for column in ("sector", "industry"):
            value = _optional_text(response.get(column))
            if value is not None and (force or pd.isna(current.at[ticker, column])):
                current.at[ticker, column] = value
                refreshed_any = True
        if refreshed_any:
            current.at[ticker, "updated_at"] = refreshed_date
    current = current.reset_index(drop=True).loc[:, METADATA_COLUMNS]

    # Avoid creating a plausible-looking but entirely empty dataset when Yahoo is
    # unavailable on the first refresh. Existing last-known-good files remain safe.
    if targets and not responses and existing.empty:
        raise CompanyMetadataError(
            "No company metadata requests succeeded; output file was not created"
        )
    _write_company_metadata_atomically(current, output_path)

    sector_available = int(current["sector"].notna().sum())
    industry_available = int(current["industry"].notna().sum())
    result = CompanyMetadataRefreshResult(
        universe_count=len(tickers),
        requested_count=len(targets),
        request_failure_count=len(failures),
        sector_available_count=sector_available,
        industry_available_count=industry_available,
        missing_sector_count=len(tickers) - sector_available,
        missing_industry_count=len(tickers) - industry_available,
        output_path=output_path,
    )
    LOGGER.info("Company metadata refresh complete: %s", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.company_metadata",
        description="Maintain presentation-only Yahoo company metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--force", action="store_true")
    refresh_parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    refresh_parser.add_argument("--output", type=Path, default=DEFAULT_METADATA_PATH)
    refresh_parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    refresh_parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
        help="minimum seconds between Yahoo request starts",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        result = refresh_company_metadata(
            force=args.force,
            universe_path=args.universe,
            output_path=args.output,
            max_workers=args.max_workers,
            request_interval_seconds=args.request_interval,
        )
    except Exception:
        LOGGER.exception("Company metadata refresh failed")
        return 1
    print("Company metadata refresh complete:")
    print(f"universe={result.universe_count}")
    print(f"sector_available={result.sector_available_count}")
    print(f"industry_available={result.industry_available_count}")
    print(f"missing_sector={result.missing_sector_count}")
    print(f"missing_industry={result.missing_industry_count}")
    print(f"request_failures={result.request_failure_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
