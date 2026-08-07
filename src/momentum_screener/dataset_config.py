"""Stable identity configuration for the authoritative daily-price dataset."""

from datetime import date

DATASET_SCHEMA_VERSION = "daily_prices_v1"
DEFAULT_BACKFILL_START = date(2016, 1, 1)
EXPECTED_UNIVERSE_SIZE = 2000
DEFAULT_RELEASE_TAG = "marketData"
DATASET_IDENTITY_FIELDS = (
    "schema_version",
    "universe_sha256",
    "requested_start",
    "universe_ticker_count",
)
