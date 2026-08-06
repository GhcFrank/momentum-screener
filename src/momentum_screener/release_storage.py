"""Synchronize the daily-price dataset through a fixed GitHub Release."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from momentum_screener.dataset_config import (
    DATASET_IDENTITY_FIELDS,
    DATASET_SCHEMA_VERSION,
    DEFAULT_BACKFILL_START,
    DEFAULT_RELEASE_TAG,
    EXPECTED_UNIVERSE_SIZE,
)
from momentum_screener.storage_manifest import (
    DOWNLOAD_FAILURES_ASSET_NAME,
    MANIFEST_ASSET_NAME,
    ManifestError,
    build_asset_record,
    calculate_sha256,
    load_manifest,
    remove_owned_tree,
    replace_files_transactionally,
    resolve_local_asset_path,
    validate_asset_size_and_hash,
    validate_managed_asset,
    validate_manifest,
    validate_safe_relative_path,
    write_json_atomically,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data/processed/prices")
DEFAULT_UNIVERSE = Path("data/universe/universe.csv")
DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_API_RETRIES = 3
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MIGRATION_PLAN_NAME = "release_migration_plan.json"
OBSOLETE_ASSETS_REPORT_NAME = "remote_obsolete_assets.json"
REMOTE_IDENTITY_MISMATCH_MESSAGE = (
    "Remote Release dataset identity does not match the current local dataset. "
    "The market-data Release must be bootstrapped or replaced with the new 2016+ "
    "dataset."
)


class ReleaseStorageError(RuntimeError):
    """Raised when Release synchronization or publication cannot finish safely."""


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """The fields that prevent one logical price dataset from mixing with another."""

    schema_version: str
    universe_sha256: str
    requested_start: str
    universe_ticker_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "universe_sha256": self.universe_sha256,
            "requested_start": self.requested_start,
            "universe_ticker_count": self.universe_ticker_count,
        }


def dataset_identity_from_manifest(
    manifest: Mapping[str, object], *, context: str
) -> DatasetIdentity:
    """Parse identity fields without accepting missing or weakly typed values."""

    schema_version = manifest.get("schema_version")
    universe_hash = manifest.get("universe_sha256")
    requested_start = manifest.get("requested_start")
    ticker_count = manifest.get("universe_ticker_count")
    if not isinstance(schema_version, str):
        raise ReleaseStorageError(f"{context} schema_version is invalid")
    if not isinstance(universe_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", universe_hash
    ):
        raise ReleaseStorageError(f"{context} universe_sha256 is invalid")
    if not isinstance(requested_start, str):
        raise ReleaseStorageError(f"{context} requested_start is invalid")
    try:
        date.fromisoformat(requested_start)
    except ValueError as exc:
        raise ReleaseStorageError(
            f"{context} requested_start is invalid: {requested_start!r}"
        ) from exc
    if isinstance(ticker_count, bool) or not isinstance(ticker_count, int):
        raise ReleaseStorageError(f"{context} universe_ticker_count is invalid")
    return DatasetIdentity(
        schema_version=schema_version,
        universe_sha256=universe_hash,
        requested_start=requested_start,
        universe_ticker_count=ticker_count,
    )


def dataset_identity_differences(
    expected: DatasetIdentity, actual: DatasetIdentity
) -> dict[str, dict[str, str | int]]:
    """Return exact identity mismatches suitable for bounded reports."""

    expected_values = expected.as_dict()
    actual_values = actual.as_dict()
    return {
        field: {
            "expected": expected_values[field],
            "actual": actual_values[field],
        }
        for field in DATASET_IDENTITY_FIELDS
        if expected_values[field] != actual_values[field]
    }


def require_remote_dataset_identity(
    payload: Mapping[str, object], expected: DatasetIdentity
) -> dict[str, Any]:
    """Require the remote manifest to represent the exact expected dataset."""

    try:
        remote = dataset_identity_from_manifest(payload, context="Remote manifest")
    except ReleaseStorageError as exc:
        raise ReleaseStorageError(
            f"{REMOTE_IDENTITY_MISMATCH_MESSAGE} Detail: {exc}"
        ) from exc
    differences = dataset_identity_differences(expected, remote)
    if payload.get("completed") is not True:
        differences["completed"] = {
            "expected": "true",
            "actual": str(payload.get("completed")),
        }
    if differences:
        raise ReleaseStorageError(
            f"{REMOTE_IDENTITY_MISMATCH_MESSAGE} Differences: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
    return validate_remote_manifest(payload)


def load_local_dataset_identity(
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> tuple[DatasetIdentity, tuple[str, ...], dict[str, Any]]:
    """Load the runtime Universe and require its local manifest to match config."""

    expected, tickers = runtime_dataset_identity(
        universe_path=universe_path,
        expected_universe_size=expected_universe_size,
    )
    manifest = load_manifest(prices_root / "manifest.json")
    local_identity = dataset_identity_from_manifest(manifest, context="Local manifest")
    differences = dataset_identity_differences(expected, local_identity)
    if differences:
        raise ReleaseStorageError(
            "Local dataset identity does not match the current Universe/config: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
    return expected, tickers, manifest


def runtime_dataset_identity(
    *,
    universe_path: Path = DEFAULT_UNIVERSE,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> tuple[DatasetIdentity, tuple[str, ...]]:
    """Build the expected identity from code configuration and current Universe."""

    from momentum_screener.prices import load_universe, universe_sha256

    tickers = load_universe(universe_path)
    if len(tickers) != expected_universe_size:
        raise ReleaseStorageError(
            f"Local Universe has {len(tickers)} tickers; expected "
            f"{expected_universe_size}"
        )
    expected = DatasetIdentity(
        schema_version=DATASET_SCHEMA_VERSION,
        universe_sha256=universe_sha256(tickers),
        requested_start=DEFAULT_BACKFILL_START.isoformat(),
        universe_ticker_count=len(tickers),
    )
    return expected, tickers


class ResponseLike(Protocol):
    """The small urllib response surface used by this module."""

    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None: ...


OpenFunction = Callable[[Request, float], ResponseLike]
SleepFunction = Callable[[float], None]


def _default_open(request: Request, timeout: float) -> ResponseLike:
    return urlopen(request, timeout=timeout)  # type: ignore[return-value]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_repository(value: str) -> str:
    candidate = value.strip().removesuffix(".git")
    if (
        not _REPOSITORY_PATTERN.fullmatch(candidate)
        or candidate.startswith(".")
        or "/." in candidate
    ):
        raise ReleaseStorageError(
            f"GitHub repository must use OWNER/REPO syntax, found {value!r}"
        )
    return candidate


def _repository_from_remote_url(value: str) -> str | None:
    remote = value.strip()
    if remote.startswith("git@github.com:"):
        return _validate_repository(remote.removeprefix("git@github.com:"))
    parsed = urlparse(remote)
    if parsed.hostname != "github.com":
        return None
    candidate = parsed.path.strip("/")
    if not candidate:
        return None
    return _validate_repository(candidate)


def _repository_from_git_config(path: Path) -> str | None:
    if not path.is_file():
        return None
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, configparser.Error) as exc:
        raise ReleaseStorageError(
            f"Unable to parse read-only Git config {path}: {exc}"
        ) from exc
    section_names = [
        'remote "origin"',
        *(section for section in parser.sections() if section.startswith('remote "')),
    ]
    seen: set[str] = set()
    for section in section_names:
        if section in seen or not parser.has_option(section, "url"):
            continue
        seen.add(section)
        repository = _repository_from_remote_url(parser.get(section, "url"))
        if repository is not None:
            return repository
    return None


def resolve_repository(
    cli_repository: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    git_config_path: Path = Path(".git/config"),
) -> str:
    """Resolve OWNER/REPO without invoking Git or changing repository metadata."""

    environment = os.environ if environ is None else environ
    for value in (
        cli_repository,
        environment.get("GITHUB_REPOSITORY"),
        environment.get("MOMENTUM_SCREENER_REPOSITORY"),
    ):
        if value:
            return _validate_repository(value)
    from_config = _repository_from_git_config(git_config_path)
    if from_config is not None:
        return from_config
    raise ReleaseStorageError(
        "GitHub repository is unavailable; pass --repository OWNER/REPO or set "
        "GITHUB_REPOSITORY or MOMENTUM_SCREENER_REPOSITORY"
    )


def resolve_github_token(environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve a GitHub token without logging or persisting it."""

    environment = os.environ if environ is None else environ
    return environment.get("GITHUB_TOKEN") or environment.get("GH_TOKEN") or None


@dataclass(slots=True)
class GitHubClient:
    """Finite-retry GitHub API client with no import-time network activity."""

    token: str | None = None
    api_base: str = DEFAULT_API_BASE
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_API_RETRIES
    open_func: OpenFunction = _default_open
    sleep_func: SleepFunction = time.sleep

    def _headers(
        self, *, accept: str = "application/vnd.github+json"
    ) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "momentum-screener-release-storage",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _open_with_retries(
        self, request_factory: Callable[[], Request]
    ) -> ResponseLike:
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        for attempt in range(self.max_retries + 1):
            try:
                return self.open_func(request_factory(), self.timeout)
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_retries:
                    message = f"GitHub API request failed with HTTP {exc.code}"
                    if exc.code in {401, 403, 404} and not self.token:
                        message += "; a token may be required for a private repository"
                    raise ReleaseStorageError(message) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise ReleaseStorageError(
                        f"GitHub API request failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
            self.sleep_func(min(2**attempt, 8))
        raise AssertionError("bounded GitHub retry loop did not terminate")

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: bytes | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Request one JSON object from GitHub."""

        def make_request() -> Request:
            headers = self._headers()
            if content_type:
                headers["Content-Type"] = content_type
            return Request(url, data=payload, headers=headers, method=method)

        with self._open_with_retries(make_request) as response:
            raw = response.read()
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseStorageError("GitHub API returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ReleaseStorageError("GitHub API returned a non-object JSON response")
        return decoded

    def download_to(self, asset: Mapping[str, Any], destination: Path) -> int:
        """Stream a Release asset to a new staging path."""

        url = asset.get("url") or asset.get("browser_download_url")
        if not isinstance(url, str):
            raise ReleaseStorageError(
                f"Release asset {asset.get('name')!r} has no download URL"
            )

        def make_request() -> Request:
            return Request(
                url,
                headers=self._headers(accept="application/octet-stream"),
                method="GET",
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        try:
            with (
                self._open_with_retries(make_request) as response,
                destination.open("xb") as output_file,
            ):
                while chunk := response.read(1024 * 1024):
                    output_file.write(chunk)
                    bytes_written += len(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        expected_size = asset.get("size")
        if isinstance(expected_size, int) and expected_size != bytes_written:
            destination.unlink(missing_ok=True)
            raise ReleaseStorageError(
                f"Downloaded Release asset {asset.get('name')!r} has size "
                f"{bytes_written}, expected {expected_size}"
            )
        return bytes_written

    def request_empty(self, method: str, url: str) -> None:
        """Run an API operation whose response body is not used."""

        def make_request() -> Request:
            return Request(url, headers=self._headers(), method=method)

        with self._open_with_retries(make_request) as response:
            response.read()

    def upload_file(
        self,
        upload_url: str,
        *,
        asset_name: str,
        path: Path,
    ) -> dict[str, Any]:
        """Upload one local file to a Release upload endpoint."""

        data = path.read_bytes()
        base_url = upload_url.split("{", maxsplit=1)[0]
        url = f"{base_url}?{urlencode({'name': asset_name})}"
        content_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"

        def make_request() -> Request:
            headers = self._headers()
            headers["Content-Type"] = content_type
            return Request(url, data=data, headers=headers, method="POST")

        with self._open_with_retries(make_request) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseStorageError(
                f"Upload response for {asset_name!r} was invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ReleaseStorageError(
                f"Upload response for {asset_name!r} was not an object"
            )
        return payload


def get_release_metadata(
    client: GitHubClient,
    repository: str,
    release_tag: str = DEFAULT_RELEASE_TAG,
) -> dict[str, Any]:
    """Fetch a fixed-tag Release or raise a concise contextual error."""

    url = (
        f"{client.api_base}/repos/{quote(repository, safe='/')}/releases/tags/"
        f"{quote(release_tag, safe='')}"
    )
    try:
        payload = client.request_json("GET", url)
    except ReleaseStorageError as exc:
        raise ReleaseStorageError(
            f"Unable to load GitHub Release tag {release_tag!r} for {repository}: {exc}"
        ) from exc
    if not isinstance(payload.get("assets"), list):
        raise ReleaseStorageError("GitHub Release metadata has no asset list")
    if not isinstance(payload.get("upload_url"), str):
        raise ReleaseStorageError("GitHub Release metadata has no upload_url")
    return payload


def _release_asset_index(release: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in release.get("assets", []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise ReleaseStorageError("Release contains malformed asset metadata")
        asset = dict(raw)
        name = str(asset["name"])
        size = asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseStorageError(f"Release asset {name!r} has an invalid size")
        if not isinstance(asset.get("url") or asset.get("browser_download_url"), str):
            raise ReleaseStorageError(
                f"Release asset {name!r} has no valid download URL"
            )
        if name in result:
            raise ReleaseStorageError(f"Release contains duplicate asset name {name!r}")
        result[name] = asset
    return result


def download_release_asset(
    client: GitHubClient,
    asset: Mapping[str, Any],
    destination: Path,
) -> int:
    """Download one named Release asset through the injected client."""

    return client.download_to(asset, destination)


def validate_remote_manifest(payload: object) -> dict[str, Any]:
    """Validate a remote manifest as a completed trusted dataset index."""

    try:
        return validate_manifest(payload, require_completed=True, require_assets=True)
    except ManifestError as exc:
        raise ReleaseStorageError(f"Remote manifest is invalid: {exc}") from exc


def _download_and_validate_manifest(
    client: GitHubClient,
    release_assets: Mapping[str, Mapping[str, Any]],
    destination: Path,
    *,
    expected_identity: DatasetIdentity | None = None,
) -> dict[str, Any]:
    metadata = release_assets.get(MANIFEST_ASSET_NAME)
    if metadata is None:
        raise ReleaseStorageError(
            f"Release asset {MANIFEST_ASSET_NAME!r} does not exist"
        )
    download_release_asset(client, metadata, destination)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseStorageError(
            "Remote prices manifest contains invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReleaseStorageError("Remote prices manifest must be a JSON object")
    manifest = (
        require_remote_dataset_identity(payload, expected_identity)
        if expected_identity is not None
        else validate_remote_manifest(payload)
    )
    for key, asset in manifest["assets"].items():
        remote = release_assets.get(str(asset["asset_name"]))
        if remote is None:
            raise ReleaseStorageError(
                f"Manifest asset {key!r} is absent from the Release: "
                f"{asset['asset_name']}"
            )
        remote_size = remote.get("size")
        if isinstance(remote_size, int) and remote_size != asset["size_bytes"]:
            raise ReleaseStorageError(
                f"Release metadata size differs from manifest for {asset['asset_name']}"
            )
    return manifest


def _download_manifest_payload(
    client: GitHubClient,
    release_assets: Mapping[str, Mapping[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    """Download only the remote manifest and parse it without touching local data."""

    metadata = release_assets.get(MANIFEST_ASSET_NAME)
    if metadata is None:
        raise ReleaseStorageError(
            f"Release asset {MANIFEST_ASSET_NAME!r} does not exist"
        )
    download_release_asset(client, metadata, destination)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseStorageError(
            "Remote prices manifest contains invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ReleaseStorageError("Remote prices manifest must be a JSON object")
    return payload


def build_pull_plan(
    manifest: Mapping[str, Any],
    output_root: Path,
    *,
    force: bool = False,
    years: Collection[int] | None = None,
    include_auxiliary: Collection[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return manifest keys needing download and keys already byte-identical."""

    selected_years = set(years) if years is not None else None
    auxiliary = set(include_auxiliary or ())
    download: list[str] = []
    unchanged: list[str] = []
    for key, asset in manifest["assets"].items():
        if selected_years is not None:
            if key.isdigit() and int(key) not in selected_years:
                continue
            if not key.isdigit() and key not in auxiliary:
                continue
        if force:
            download.append(str(key))
            continue
        local = resolve_local_asset_path(output_root, asset["local_path"])
        matches = (
            local.is_file()
            and local.stat().st_size == asset["size_bytes"]
            and calculate_sha256(local) == asset["sha256"]
        )
        if not matches:
            download.append(str(key))
        else:
            unchanged.append(str(key))
    return download, unchanged


def _validate_downloaded_assets(
    manifest: Mapping[str, Any],
    staging_root: Path,
    keys: Sequence[str],
) -> None:
    for key in keys:
        asset = manifest["assets"][key]
        path = resolve_local_asset_path(staging_root, asset["local_path"])
        validate_managed_asset(path, key=key, asset=asset)


def validate_coverage_ticker_set(path: Path, expected_tickers: Collection[str]) -> None:
    """Require coverage to contain every expected ticker exactly once."""

    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseStorageError(
            f"Unable to read ticker coverage {path}: {exc}"
        ) from exc
    tickers = [str(row.get("ticker", "")) for row in rows]
    expected = set(expected_tickers)
    if (
        len(tickers) != len(expected)
        or len(set(tickers)) != len(tickers)
        or set(tickers) != expected
    ):
        raise ReleaseStorageError(
            "Remote ticker coverage does not exactly match the current Universe"
        )


def _local_previous_session(output_root: Path) -> str | None:
    path = output_root / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("latest_session") or payload.get("actual_max_date")
    return value if isinstance(value, str) else None


def _validate_post_pull(
    output_root: Path,
    manifest: Mapping[str, Any],
    verified_keys: Sequence[str],
    expected_tickers: Collection[str] | None = None,
) -> None:
    disk_manifest = load_manifest(output_root / "manifest.json")
    if disk_manifest != manifest:
        raise ReleaseStorageError("Installed manifest differs from remote manifest")
    for key in verified_keys:
        asset = manifest["assets"][key]
        path = resolve_local_asset_path(output_root, asset["local_path"])
        validate_managed_asset(path, key=key, asset=asset)
    if expected_tickers is not None and "ticker_coverage" in verified_keys:
        validate_coverage_ticker_set(
            output_root / "ticker_coverage.csv", expected_tickers
        )


def pull_release_dataset(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RELEASE_TAG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    years: Collection[int] | None = None,
    include_auxiliary: Collection[str] | None = None,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
    expected_identity: DatasetIdentity | None = None,
    expected_tickers: Collection[str] | None = None,
) -> dict[str, Any]:
    """Transactionally synchronize manifest-selected Release assets."""

    resolved_repository = resolve_repository(repository, environ=environ)
    github = client or GitHubClient(token=resolve_github_token(environ))
    started = _utc_now_iso()
    local_previous = _local_previous_session(output_root)
    release = get_release_metadata(github, resolved_repository, release_tag)
    release_assets = _release_asset_index(release)

    if dry_run:
        with tempfile.TemporaryDirectory(
            prefix="momentum-release-dry-run-"
        ) as temp_dir:
            manifest = _download_and_validate_manifest(
                github,
                release_assets,
                Path(temp_dir) / "manifest.json",
                expected_identity=expected_identity,
            )
        download_keys, unchanged_keys = build_pull_plan(
            manifest,
            output_root,
            force=force,
            years=years,
            include_auxiliary=include_auxiliary,
        )
        return {
            "repository": resolved_repository,
            "release_tag": release_tag,
            "remote_latest_session": manifest.get("latest_session"),
            "local_previous_session": local_previous,
            "would_download_assets": download_keys,
            "unchanged_assets": unchanged_keys,
            "dry_run": True,
        }

    sync_id = uuid.uuid4().hex
    staging_parent = output_root / ".sync_staging"
    backup_parent = output_root / ".sync_backup"
    staging_root = staging_parent / f"sync-{sync_id}"
    backup_root = backup_parent / f"sync-{sync_id}"
    staging_root.mkdir(parents=True, exist_ok=False)
    downloaded_bytes = 0
    try:
        manifest_path = staging_root / "manifest.json"
        manifest = _download_and_validate_manifest(
            github,
            release_assets,
            manifest_path,
            expected_identity=expected_identity,
        )
        download_keys, unchanged_keys = build_pull_plan(
            manifest,
            output_root,
            force=force,
            years=years,
            include_auxiliary=include_auxiliary,
        )
        for key in download_keys:
            asset = manifest["assets"][key]
            metadata = release_assets[str(asset["asset_name"])]
            destination = resolve_local_asset_path(staging_root, asset["local_path"])
            downloaded_bytes += download_release_asset(github, metadata, destination)
        _validate_downloaded_assets(manifest, staging_root, download_keys)
        for key in unchanged_keys:
            asset = manifest["assets"][key]
            local = resolve_local_asset_path(output_root, asset["local_path"])
            validate_managed_asset(local, key=key, asset=asset)
        if expected_tickers is not None and "ticker_coverage" in {
            *download_keys,
            *unchanged_keys,
        }:
            coverage_asset = manifest["assets"]["ticker_coverage"]
            coverage_path = (
                resolve_local_asset_path(staging_root, coverage_asset["local_path"])
                if "ticker_coverage" in download_keys
                else resolve_local_asset_path(output_root, coverage_asset["local_path"])
            )
            validate_coverage_ticker_set(coverage_path, expected_tickers)

        finished = _utc_now_iso()
        report = {
            "sync_id": sync_id,
            "repository": resolved_repository,
            "release_tag": release_tag,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "remote_latest_session": manifest.get("latest_session"),
            "local_previous_session": local_previous,
            "downloaded_assets": [
                manifest["assets"][key]["asset_name"] for key in download_keys
            ],
            "unchanged_assets": [
                manifest["assets"][key]["asset_name"] for key in unchanged_keys
            ],
            "downloaded_bytes": downloaded_bytes,
            "success": True,
        }
        write_json_atomically(staging_root / "sync_report.json", report)
        replacement_paths = [
            *(str(manifest["assets"][key]["local_path"]) for key in download_keys),
            "sync_report.json",
            "manifest.json",
        ]
        verified_keys = [*download_keys, *unchanged_keys]
        replace_files_transactionally(
            output_root,
            staging_root,
            replacement_paths,
            backup_root=backup_root,
            validate_after=lambda: _validate_post_pull(
                output_root,
                manifest,
                verified_keys,
                expected_tickers,
            ),
        )
        LOGGER.info(
            "Remote latest session: %s; local previous session: %s; "
            "downloaded assets: %d; unchanged assets: %d; verified assets: %d; "
            "local dataset updated: yes",
            manifest.get("latest_session"),
            local_previous,
            len(download_keys),
            len(unchanged_keys),
            len(verified_keys),
        )
        return report
    finally:
        if staging_root.exists():
            remove_owned_tree(staging_root, parent=staging_parent, prefix="sync-")
        if backup_root.exists():
            remove_owned_tree(backup_root, parent=backup_parent, prefix="sync-")


def _parse_iso_manifest_date(manifest: Mapping[str, Any], key: str) -> date:
    value = manifest.get(key)
    if not isinstance(value, str):
        raise ReleaseStorageError(f"Remote manifest is missing {key}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseStorageError(
            f"Remote manifest {key} is invalid: {value!r}"
        ) from exc


def pull_update_inputs(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RELEASE_TAG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    refresh_calendar_days: int = 550,
    settlement_delay_minutes: int = 90,
    timezone: str = "America/New_York",
    target_date: date | None = None,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
    expected_identity: DatasetIdentity | None = None,
    expected_tickers: Collection[str] | None = None,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> dict[str, Any]:
    """Pull only the partitions and coverage required by the next daily update."""

    from momentum_screener.prices import (  # local import avoids import-time coupling
        affected_partition_years,
        calculate_refresh_start,
        determine_target_session,
    )

    if expected_identity is None:
        expected_identity, loaded_tickers, _ = load_local_dataset_identity(
            prices_root=output_root,
            universe_path=universe_path,
            expected_universe_size=expected_universe_size,
        )
        expected_tickers = loaded_tickers
    if expected_tickers is None:
        raise ReleaseStorageError(
            "Expected Universe tickers are required for update input validation"
        )
    resolved_repository = resolve_repository(repository, environ=environ)
    github = client or GitHubClient(token=resolve_github_token(environ))
    release = get_release_metadata(github, resolved_repository, release_tag)
    release_assets = _release_asset_index(release)
    with tempfile.TemporaryDirectory(prefix="momentum-update-inputs-") as temp_dir:
        manifest = _download_and_validate_manifest(
            github,
            release_assets,
            Path(temp_dir) / "manifest.json",
            expected_identity=expected_identity,
        )
    dataset_start = _parse_iso_manifest_date(manifest, "requested_start")
    target = determine_target_session(
        target_date=target_date,
        settlement_delay_minutes=settlement_delay_minutes,
        timezone=timezone,
    )
    refresh_start = calculate_refresh_start(
        dataset_start, target, refresh_calendar_days
    )
    years = affected_partition_years(refresh_start, target)
    return pull_release_dataset(
        repository=resolved_repository,
        release_tag=release_tag,
        output_root=output_root,
        force=False,
        years=years,
        include_auxiliary={"ticker_coverage"},
        dry_run=dry_run,
        client=github,
        environ=environ,
        expected_identity=expected_identity,
        expected_tickers=expected_tickers,
    )


def prepare_bootstrap_manifest(
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Index an existing completed Backfill for the first manual Release upload."""

    from momentum_screener.prices import index_release_assets

    path = prices_root / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseStorageError(
            f"Unable to read bootstrap manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReleaseStorageError("Bootstrap manifest must be a JSON object")
    try:
        indexed = index_release_assets(prices_root, payload)
    except (ManifestError, OSError, ValueError) as exc:
        raise ReleaseStorageError(f"Unable to index bootstrap assets: {exc}") from exc
    if not dry_run:
        write_json_atomically(path, indexed)
    LOGGER.info(
        "%s bootstrap manifest with %d managed assets",
        "Validated" if dry_run else "Wrote",
        len(indexed["assets"]),
    )
    return indexed


def _ordered_bootstrap_assets(
    prices_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, object]]:
    assets = manifest["assets"]
    ordered_keys = [
        *sorted(
            (key for key in assets if str(key).isdigit()), key=lambda key: int(key)
        ),
        *(
            key
            for key in (
                "ticker_coverage",
                "download_failures",
                "update_missing_tickers",
                "update_report",
            )
            if key in assets
        ),
    ]
    result: list[dict[str, object]] = []
    for order, key in enumerate(ordered_keys, start=1):
        asset = dict(assets[key])
        path = resolve_local_asset_path(prices_root, asset["local_path"])
        validate_managed_asset(path, key=str(key), asset=asset)
        asset["publish_order"] = order
        result.append(asset)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    result.append(
        {
            "local_path": "manifest.json",
            "asset_name": MANIFEST_ASSET_NAME,
            "size_bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "publish_order": len(result) + 1,
        }
    )
    return result


def validate_local_bootstrap_dataset(
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> tuple[DatasetIdentity, dict[str, Any], dict[str, Any]]:
    """Fully validate local data and build the candidate authoritative manifest."""

    from momentum_screener.prices import index_release_assets, validate_backfill_dataset

    expected, _, local_manifest = load_local_dataset_identity(
        prices_root=prices_root,
        universe_path=universe_path,
        expected_universe_size=expected_universe_size,
    )
    acceptance = validate_backfill_dataset(prices_root, universe_path=universe_path)
    if acceptance["failed_ticker_count"] != 0:
        raise ReleaseStorageError("Local bootstrap dataset contains failed tickers")
    years = sorted(int(year) for year in acceptance["partition_row_counts"])
    if not years or years[0] != DEFAULT_BACKFILL_START.year:
        raise ReleaseStorageError(
            f"Local bootstrap partitions must begin in {DEFAULT_BACKFILL_START.year}"
        )
    if any(year < DEFAULT_BACKFILL_START.year for year in years):
        raise ReleaseStorageError(
            "Local bootstrap dataset contains a pre-2016 partition"
        )
    try:
        candidate = index_release_assets(prices_root, local_manifest)
    except (ManifestError, OSError, ValueError) as exc:
        raise ReleaseStorageError(f"Unable to build bootstrap manifest: {exc}") from exc
    if "download_failures" not in candidate["assets"]:
        raise ReleaseStorageError(f"Bootstrap requires {DOWNLOAD_FAILURES_ASSET_NAME}")
    return expected, candidate, acceptance


def validate_local_dataset_acceptance(
    *,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> dict[str, Any]:
    """Fully validate the current authoritative local dataset."""

    _, _, acceptance = validate_local_bootstrap_dataset(
        prices_root=prices_root,
        universe_path=universe_path,
        expected_universe_size=expected_universe_size,
    )
    return acceptance


def _migration_plan(
    *,
    release_tag: str,
    local_manifest: Mapping[str, Any],
    bootstrap_assets: Sequence[Mapping[str, object]],
    remote_release: Mapping[str, Any] | None = None,
    remote_manifest_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    local_identity = dataset_identity_from_manifest(
        local_manifest, context="Local bootstrap manifest"
    )
    partition_counts = local_manifest["partition_row_counts"]
    managed_names = {str(item["asset_name"]) for item in bootstrap_assets}
    if remote_release is None:
        remote_dataset: dict[str, object] = {
            "release_tag": release_tag,
            "manifest_found": "not_checked",
            "schema_version": "not_checked",
            "universe_sha256": "not_checked",
            "universe_ticker_count": "not_checked",
            "requested_start": "not_checked",
            "latest_session": "not_checked",
        }
        identity_matches: bool | str = "not_checked"
        obsolete: list[str] = []
        missing: list[str] = []
    else:
        release_assets = _release_asset_index(remote_release)
        obsolete = sorted(set(release_assets) - managed_names)
        missing = sorted(managed_names - set(release_assets))
        remote_dataset = {
            "release_tag": release_tag,
            "manifest_found": remote_manifest_payload is not None,
            "schema_version": None,
            "universe_sha256": None,
            "universe_ticker_count": None,
            "requested_start": None,
            "latest_session": None,
        }
        identity_matches = False
        if remote_manifest_payload is not None:
            for key in (
                "schema_version",
                "universe_sha256",
                "universe_ticker_count",
                "requested_start",
                "latest_session",
            ):
                remote_dataset[key] = remote_manifest_payload.get(key)
            try:
                remote_identity = dataset_identity_from_manifest(
                    remote_manifest_payload, context="Remote manifest"
                )
                identity_matches = (
                    not dataset_identity_differences(local_identity, remote_identity)
                    and remote_manifest_payload.get("completed") is True
                )
            except ReleaseStorageError:
                identity_matches = False
    return {
        "generated_at_utc": _utc_now_iso(),
        "local_dataset": {
            **local_identity.as_dict(),
            "actual_min_date": local_manifest.get("actual_min_date"),
            "actual_max_date": local_manifest.get("actual_max_date"),
            "latest_session": local_manifest.get("latest_session"),
            "total_row_count": local_manifest.get("total_row_count"),
            "partition_years": sorted(int(year) for year in partition_counts),
        },
        "remote_dataset": remote_dataset,
        "identity_matches": identity_matches,
        "bootstrap_required": identity_matches is not True,
        "assets_to_upload": [dict(item) for item in bootstrap_assets],
        "missing_remote_assets": missing,
        "obsolete_remote_assets": obsolete,
        "daily_workflow_ready": identity_matches is True and not missing,
        "publication_strategy": "replace formal assets; upload manifest last",
        "partial_publication_risk": (
            "A failed pre-manifest upload can leave replaced assets beside the old "
            "manifest; rerun bootstrap before daily workflow."
        ),
    }


def check_release_dataset(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RELEASE_TAG,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
    expected_identity: DatasetIdentity | None = None,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> dict[str, Any]:
    """Read only the remote manifest and report whether daily workflow is safe."""

    if expected_identity is None:
        expected_identity, _, local_manifest = load_local_dataset_identity(
            prices_root=prices_root,
            universe_path=universe_path,
            expected_universe_size=expected_universe_size,
        )
    else:
        local_manifest = load_manifest(prices_root / "manifest.json")
    resolved_repository = resolve_repository(repository, environ=environ)
    github = client or GitHubClient(token=resolve_github_token(environ))
    release = get_release_metadata(github, resolved_repository, release_tag)
    release_assets = _release_asset_index(release)
    payload: dict[str, Any] | None = None
    error: str | None = None
    remote_manifest: dict[str, Any] | None = None
    inspected_manifest: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="momentum-release-check-") as temp_dir:
        try:
            payload = _download_manifest_payload(
                github, release_assets, Path(temp_dir) / "manifest.json"
            )
            remote_manifest = require_remote_dataset_identity(
                payload, expected_identity
            )
            inspected_manifest = remote_manifest
        except (ReleaseStorageError, ManifestError) as exc:
            error = str(exc)
            if payload is not None:
                try:
                    inspected_manifest = validate_remote_manifest(payload)
                except ReleaseStorageError:
                    inspected_manifest = None
    managed_names = (
        {
            MANIFEST_ASSET_NAME,
            *(
                str(asset["asset_name"])
                for asset in inspected_manifest["assets"].values()
            ),
        }
        if inspected_manifest is not None
        else set()
    )
    missing = sorted(managed_names - set(release_assets))
    obsolete = sorted(set(release_assets) - managed_names) if managed_names else []
    workflow_ready = remote_manifest is not None and not missing
    result = {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "local_schema_version": expected_identity.schema_version,
        "remote_schema_version": payload.get("schema_version") if payload else None,
        "local_universe_sha256": expected_identity.universe_sha256,
        "remote_universe_sha256": payload.get("universe_sha256") if payload else None,
        "local_universe_ticker_count": expected_identity.universe_ticker_count,
        "remote_universe_ticker_count": (
            payload.get("universe_ticker_count") if payload else None
        ),
        "local_requested_start": expected_identity.requested_start,
        "remote_requested_start": payload.get("requested_start") if payload else None,
        "local_latest_session": local_manifest.get("latest_session"),
        "remote_latest_session": payload.get("latest_session") if payload else None,
        "remote_asset_count": len(release_assets),
        "managed_asset_count": len(managed_names),
        "missing_managed_assets": missing,
        "obsolete_remote_assets": obsolete,
        "obsolete_asset_count": len(obsolete),
        "dataset_identity_match": remote_manifest is not None,
        "workflow_ready": workflow_ready,
        "error": error,
    }
    LOGGER.info(
        "Dataset check: local_count=%d local_hash=%s remote_hash=%s local_start=%s "
        "remote_start=%s local_latest=%s remote_latest=%s assets=%d "
        "obsolete=%d ready=%s",
        result["local_universe_ticker_count"],
        result["local_universe_sha256"],
        result["remote_universe_sha256"],
        result["local_requested_start"],
        result["remote_requested_start"],
        result["local_latest_session"],
        result["remote_latest_session"],
        result["remote_asset_count"],
        result["obsolete_asset_count"],
        "yes" if workflow_ready else "no",
    )
    return result


def bootstrap_dataset(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RELEASE_TAG,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    confirm_replace_dataset: bool = False,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> dict[str, Any]:
    """Plan or explicitly replace the Release baseline, committing manifest last."""

    expected, candidate_manifest, _ = validate_local_bootstrap_dataset(
        prices_root=prices_root,
        universe_path=universe_path,
        expected_universe_size=expected_universe_size,
    )
    bootstrap_assets = _ordered_bootstrap_assets(prices_root, candidate_manifest)
    plan_path = prices_root / MIGRATION_PLAN_NAME
    if not confirm_replace_dataset:
        plan = _migration_plan(
            release_tag=release_tag,
            local_manifest=candidate_manifest,
            bootstrap_assets=bootstrap_assets,
        )
        write_json_atomically(plan_path, plan)
        return plan
    if dry_run:
        raise ReleaseStorageError(
            "--dry-run cannot be combined with --confirm-replace-dataset"
        )

    resolved_repository = resolve_repository(repository, environ=environ)
    github = client or GitHubClient(token=resolve_github_token(environ))
    release = get_release_metadata(github, resolved_repository, release_tag)
    release_assets = _release_asset_index(release)
    remote_payload: dict[str, Any] | None = None
    if MANIFEST_ASSET_NAME in release_assets:
        with tempfile.TemporaryDirectory(
            prefix="momentum-bootstrap-remote-"
        ) as temp_dir:
            try:
                remote_payload = _download_manifest_payload(
                    github, release_assets, Path(temp_dir) / "manifest.json"
                )
            except ReleaseStorageError:
                remote_payload = None
    plan = _migration_plan(
        release_tag=release_tag,
        local_manifest=candidate_manifest,
        bootstrap_assets=bootstrap_assets,
        remote_release=release,
        remote_manifest_payload=remote_payload,
    )
    plan["repository"] = resolved_repository
    write_json_atomically(plan_path, plan)

    manifest_item = bootstrap_assets[-1]
    if manifest_item["asset_name"] != MANIFEST_ASSET_NAME:
        raise ReleaseStorageError("Bootstrap manifest must be published last")
    uploaded: list[str] = []
    for item in bootstrap_assets[:-1]:
        path = resolve_local_asset_path(prices_root, item["local_path"])
        validate_asset_size_and_hash(path, item)
        upload_release_asset(
            github,
            repository=resolved_repository,
            release=release,
            asset_name=str(item["asset_name"]),
            path=path,
        )
        uploaded.append(str(item["asset_name"]))
        release = get_release_metadata(github, resolved_repository, release_tag)
        uploaded_asset = _release_asset_index(release).get(str(item["asset_name"]))
        if uploaded_asset is None or uploaded_asset.get("size") != item["size_bytes"]:
            raise ReleaseStorageError(
                f"Bootstrap asset verification failed: {item['asset_name']}"
            )

    with tempfile.TemporaryDirectory(prefix="momentum-bootstrap-manifest-") as temp_dir:
        candidate_path = Path(temp_dir) / "manifest.json"
        write_json_atomically(candidate_path, candidate_manifest)
        validate_asset_size_and_hash(candidate_path, manifest_item)
        upload_release_asset(
            github,
            repository=resolved_repository,
            release=release,
            asset_name=MANIFEST_ASSET_NAME,
            path=candidate_path,
        )
        uploaded.append(MANIFEST_ASSET_NAME)
    release = get_release_metadata(github, resolved_repository, release_tag)
    final_assets = _release_asset_index(release)
    with tempfile.TemporaryDirectory(prefix="momentum-bootstrap-verify-") as temp_dir:
        verified_path = Path(temp_dir) / "manifest.json"
        verified_payload = _download_manifest_payload(
            github, final_assets, verified_path
        )
        verified_manifest = require_remote_dataset_identity(verified_payload, expected)
    if verified_manifest != candidate_manifest:
        raise ReleaseStorageError(
            "Remote manifest differs from the validated bootstrap manifest"
        )
    for asset in candidate_manifest["assets"].values():
        remote_asset = final_assets.get(str(asset["asset_name"]))
        if remote_asset is None or remote_asset.get("size") != asset["size_bytes"]:
            raise ReleaseStorageError(
                f"Remote bootstrap asset is missing or wrong-sized: {asset['asset_name']}"
            )

    write_json_atomically(prices_root / "manifest.json", candidate_manifest)
    obsolete_report = {
        "generated_at_utc": _utc_now_iso(),
        "repository": resolved_repository,
        "release_tag": release_tag,
        "obsolete_remote_assets": plan["obsolete_remote_assets"],
        "deleted": False,
    }
    write_json_atomically(prices_root / OBSOLETE_ASSETS_REPORT_NAME, obsolete_report)
    return {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "uploaded_assets": uploaded,
        "manifest_uploaded_last": uploaded[-1] == MANIFEST_ASSET_NAME,
        "obsolete_remote_assets": plan["obsolete_remote_assets"],
        "success": True,
    }


def _load_publish_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseStorageError(f"Unable to read publish plan {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ReleaseStorageError("Publish plan must be an object with an assets list")
    assets = payload["assets"]
    if not assets:
        raise ReleaseStorageError("Publish plan contains no assets")
    orders: list[int] = []
    names: set[str] = set()
    paths: set[str] = set()
    for item in assets:
        if not isinstance(item, Mapping):
            raise ReleaseStorageError("Publish plan asset must be an object")
        local_path = str(validate_safe_relative_path(item.get("local_path")))
        asset_name = item.get("asset_name")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        order = item.get("publish_order")
        if (
            not isinstance(asset_name, str)
            or Path(asset_name).name != asset_name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or isinstance(order, bool)
            or not isinstance(order, int)
            or order <= 0
        ):
            raise ReleaseStorageError(f"Publish plan asset is invalid: {asset_name!r}")
        if asset_name in names or local_path in paths:
            raise ReleaseStorageError(
                "Publish plan has duplicate assets or local paths"
            )
        names.add(asset_name)
        paths.add(local_path)
        orders.append(order)
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise ReleaseStorageError("Publish plan order must be unique and ascending")
    if assets[-1].get("asset_name") != MANIFEST_ASSET_NAME:
        raise ReleaseStorageError("Publish plan must upload the manifest last")
    return payload


def _validate_publish_plan_assets(
    prices_root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Fully validate a publication plan against its authoritative local files."""

    local_identity = dataset_identity_from_manifest(
        manifest, context="Local publish manifest"
    )
    raw_plan_identity = plan.get("dataset_identity")
    if not isinstance(raw_plan_identity, Mapping):
        raise ReleaseStorageError("Publish plan has no dataset identity")
    plan_identity = dataset_identity_from_manifest(
        raw_plan_identity, context="Publish plan"
    )
    differences = dataset_identity_differences(local_identity, plan_identity)
    if differences:
        raise ReleaseStorageError(
            "Publish plan dataset identity differs from local manifest: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
    manifest_assets_by_name = {
        str(asset["asset_name"]): (str(key), asset)
        for key, asset in manifest["assets"].items()
    }
    for item in plan["assets"]:
        path = resolve_local_asset_path(prices_root, item["local_path"])
        validate_asset_size_and_hash(path, item)
        if item["asset_name"] == MANIFEST_ASSET_NAME:
            continue
        managed = manifest_assets_by_name.get(str(item["asset_name"]))
        if managed is None:
            raise ReleaseStorageError(
                f"Publish plan asset is not managed by the manifest: {item['asset_name']}"
            )
        key, manifest_asset = managed
        if any(
            item[field] != manifest_asset[field]
            for field in ("asset_name", "local_path", "size_bytes", "sha256")
        ):
            raise ReleaseStorageError(
                f"Publish plan differs from manifest for {item['asset_name']}"
            )
        validate_managed_asset(path, key=key, asset=manifest_asset)
    if manifest.get("latest_session") != plan.get("target_session"):
        raise ReleaseStorageError(
            "Publish plan target_session differs from the local manifest"
        )


def _load_local_update_report(
    prices_root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or "update_report" not in assets:
        raise ReleaseStorageError("Local manifest does not manage an update report")
    report_asset = assets["update_report"]
    if not isinstance(report_asset, Mapping):
        raise ReleaseStorageError("Local update report asset mapping is invalid")
    report_path = resolve_local_asset_path(prices_root, report_asset["local_path"])
    validate_managed_asset(report_path, key="update_report", asset=report_asset)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseStorageError(
            f"Unable to read local update report {report_path}: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("local_update_success") is not True:
        raise ReleaseStorageError(
            "Local update report does not record local_update_success=true"
        )
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ReleaseStorageError("Local update report has no run_id")
    if manifest.get("last_update_run_id") != run_id:
        raise ReleaseStorageError(
            "Local update report run_id differs from the local manifest"
        )
    target_session = report.get("target_session")
    if target_session != manifest.get("latest_session"):
        raise ReleaseStorageError(
            "Local update report target_session differs from the local manifest"
        )
    raw_years = report.get("changed_partition_years")
    if (
        not isinstance(raw_years, list)
        or not raw_years
        or any(
            isinstance(year, bool) or not isinstance(year, int) for year in raw_years
        )
    ):
        raise ReleaseStorageError(
            "Local update report changed_partition_years is invalid"
        )
    years = sorted(set(raw_years))
    if years != raw_years:
        raise ReleaseStorageError(
            "Local update report changed_partition_years must be unique and sorted"
        )
    requested_start = date.fromisoformat(str(manifest["requested_start"]))
    if any(year < requested_start.year or str(year) not in assets for year in years):
        raise ReleaseStorageError(
            "Local update report references an unmanaged partition year"
        )
    expected_paths = [
        *(f"daily/year={year}/prices.parquet" for year in years),
        "ticker_coverage.csv",
        "update_missing_tickers.csv",
        "update_report.json",
        "manifest.json",
    ]
    if report.get("changed_local_assets") != expected_paths:
        raise ReleaseStorageError(
            "Local update report changed_local_assets is inconsistent"
        )
    return report, years


def build_publish_plan(
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repository: str,
    release_tag: str = DEFAULT_RELEASE_TAG,
) -> dict[str, Any]:
    """Build and fully validate a Release plan from one committed local update."""

    resolved_repository = _validate_repository(repository)
    if not release_tag:
        raise ReleaseStorageError("Release tag cannot be empty")
    manifest = load_manifest(prices_root / "manifest.json")
    report, years = _load_local_update_report(prices_root, manifest)
    manifest_assets = manifest["assets"]
    ordered_keys = [
        *(str(year) for year in years),
        "ticker_coverage",
        "update_missing_tickers",
        "update_report",
    ]
    assets: list[dict[str, object]] = []
    for order, key in enumerate(ordered_keys, start=1):
        if key not in manifest_assets:
            raise ReleaseStorageError(
                f"Local manifest does not manage publish asset {key!r}"
            )
        asset = dict(manifest_assets[key])
        asset["publish_order"] = order
        assets.append(asset)
    manifest_asset: dict[str, object] = dict(
        build_asset_record(
            prices_root / "manifest.json",
            asset_name=MANIFEST_ASSET_NAME,
            local_path="manifest.json",
        )
    )
    manifest_asset["publish_order"] = len(assets) + 1
    assets.append(manifest_asset)
    plan: dict[str, Any] = {
        "run_id": report["run_id"],
        "target_session": report["target_session"],
        "repository": resolved_repository,
        "release_tag": release_tag,
        "dataset_identity": dataset_identity_from_manifest(
            manifest, context="Local publish manifest"
        ).as_dict(),
        "changed_partition_years": years,
        "assets": assets,
    }
    plan_path = prices_root / "release_publish_plan.json"
    write_json_atomically(plan_path, plan)
    stored_plan = _load_publish_plan(plan_path)
    if stored_plan != plan:
        raise ReleaseStorageError("Stored publish plan differs from the built plan")
    _validate_publish_plan_assets(prices_root, stored_plan, manifest)
    return stored_plan


def upload_release_asset(
    client: GitHubClient,
    *,
    repository: str,
    release: Mapping[str, Any],
    asset_name: str,
    path: Path,
) -> dict[str, Any]:
    """Replace one fixed-name Release asset and validate the upload response."""

    existing = _release_asset_index(release).get(asset_name)
    if existing is not None:
        asset_id = existing.get("id")
        if not isinstance(asset_id, int):
            raise ReleaseStorageError(
                f"Existing Release asset {asset_name!r} has no numeric id"
            )
        url = (
            f"{client.api_base}/repos/{quote(repository, safe='/')}/releases/assets/"
            f"{asset_id}"
        )
        client.request_empty("DELETE", url)
    upload_url = release.get("upload_url")
    if not isinstance(upload_url, str):
        raise ReleaseStorageError("Release has no upload URL")
    uploaded = client.upload_file(upload_url, asset_name=asset_name, path=path)
    if (
        uploaded.get("name") != asset_name
        or uploaded.get("size") != path.stat().st_size
    ):
        raise ReleaseStorageError(
            f"GitHub upload response did not validate for asset {asset_name!r}"
        )
    return uploaded


def publish_update(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RELEASE_TAG,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
    expected_identity: DatasetIdentity | None = None,
    expected_universe_size: int = EXPECTED_UNIVERSE_SIZE,
) -> dict[str, Any]:
    """Publish validated update assets in order, committing with manifest last."""

    if expected_identity is None:
        expected_identity, _, _ = load_local_dataset_identity(
            prices_root=prices_root,
            universe_path=universe_path,
            expected_universe_size=expected_universe_size,
        )
    else:
        local_manifest = load_manifest(prices_root / "manifest.json")
        local_identity = dataset_identity_from_manifest(
            local_manifest, context="Local manifest"
        )
        differences = dataset_identity_differences(expected_identity, local_identity)
        if differences:
            raise ReleaseStorageError(
                "Local publish manifest identity mismatch: "
                f"{json.dumps(differences, sort_keys=True)}"
            )
    resolved_repository = resolve_repository(repository, environ=environ)
    plan = build_publish_plan(
        prices_root,
        repository=resolved_repository,
        release_tag=release_tag,
    )
    if dry_run:
        return {
            "repository": resolved_repository,
            "release_tag": release_tag,
            "target_session": plan.get("target_session"),
            "assets": [item["asset_name"] for item in plan["assets"]],
            "dry_run": True,
            "release_publish_success": False,
        }

    if client is None:
        token = resolve_github_token(environ)
        if token is None:
            raise ReleaseStorageError(
                "GitHub token is required to publish Release assets; set "
                "GITHUB_TOKEN or GH_TOKEN"
            )
        github = GitHubClient(token=token)
    else:
        github = client
    release = get_release_metadata(github, resolved_repository, release_tag)
    release_assets = _release_asset_index(release)
    with tempfile.TemporaryDirectory(prefix="momentum-publish-identity-") as temp_dir:
        _download_and_validate_manifest(
            github,
            release_assets,
            Path(temp_dir) / "manifest.json",
            expected_identity=expected_identity,
        )
    uploaded_names: list[str] = []
    for item in plan["assets"]:
        asset_name = str(item["asset_name"])
        path = resolve_local_asset_path(prices_root, item["local_path"])
        validate_asset_size_and_hash(path, item)
        upload_release_asset(
            github,
            repository=resolved_repository,
            release=release,
            asset_name=asset_name,
            path=path,
        )
        uploaded_names.append(asset_name)
        release = get_release_metadata(github, resolved_repository, release_tag)
        uploaded = _release_asset_index(release).get(asset_name)
        if uploaded is None or uploaded.get("size") != path.stat().st_size:
            raise ReleaseStorageError(
                f"Uploaded asset is missing or has wrong size: {asset_name}"
            )

    final_assets = _release_asset_index(release)
    manifest_metadata = final_assets.get(MANIFEST_ASSET_NAME)
    if manifest_metadata is None:
        raise ReleaseStorageError("Published manifest is absent from Release metadata")
    with tempfile.TemporaryDirectory(prefix="momentum-publish-verify-") as temp_dir:
        remote_path = Path(temp_dir) / "manifest.json"
        download_release_asset(github, manifest_metadata, remote_path)
        local_manifest_path = prices_root / "manifest.json"
        if calculate_sha256(remote_path) != calculate_sha256(local_manifest_path):
            raise ReleaseStorageError("Remote manifest hash differs after publication")
        remote_manifest = require_remote_dataset_identity(
            json.loads(remote_path.read_text(encoding="utf-8")),
            expected_identity,
        )
    if remote_manifest.get("latest_session") != plan.get("target_session"):
        raise ReleaseStorageError("Remote manifest target session validation failed")
    if remote_manifest.get("completed") is not True:
        raise ReleaseStorageError("Remote manifest is not completed after publication")
    for item in plan["assets"][:-1]:
        if str(item["asset_name"]) not in final_assets:
            raise ReleaseStorageError(
                f"Published asset absent after manifest commit: {item['asset_name']}"
            )
    result = {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "target_session": plan.get("target_session"),
        "uploaded_assets": uploaded_names,
        "manifest_uploaded_last": True,
        "release_publish_success": True,
        "success": True,
    }
    LOGGER.info(
        "Published %d assets through session %s; manifest uploaded last",
        len(uploaded_names),
        plan.get("target_session"),
    )
    return result


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from exc


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _year(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a four-digit year") from exc
    if len(value) != 4 or not 1900 <= parsed <= 9999:
        raise argparse.ArgumentTypeError("must be a four-digit year")
    return parsed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.release_storage",
        description="Synchronize validated daily prices through a GitHub Release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pull = subparsers.add_parser("pull", help="synchronize the Release dataset")
    pull.add_argument("--repository")
    pull.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    pull.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    pull.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    pull.add_argument("--force", action="store_true")
    pull.add_argument("--year", action="append", type=_year, dest="years")
    pull.add_argument("--dry-run", action="store_true")

    inputs = subparsers.add_parser(
        "pull-update-inputs", help="restore only the next update's inputs"
    )
    inputs.add_argument("--repository")
    inputs.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    inputs.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    inputs.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    inputs.add_argument(
        "--refresh-calendar-days", type=_nonnegative_integer, default=550
    )
    inputs.add_argument(
        "--settlement-delay-minutes", type=_nonnegative_integer, default=90
    )
    inputs.add_argument("--timezone", default="America/New_York")
    inputs.add_argument("--target-date", type=_parse_date)
    inputs.add_argument("--dry-run", action="store_true")

    publish = subparsers.add_parser(
        "publish-update", help="publish a validated local update plan"
    )
    publish.add_argument("--repository")
    publish.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    publish.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    publish.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--result-json", type=Path)
    check = subparsers.add_parser(
        "check", help="read only the remote manifest and verify dataset identity"
    )
    check.add_argument("--repository")
    check.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    check.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    check.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    check.add_argument("--result-json", type=Path)
    dataset_bootstrap = subparsers.add_parser(
        "bootstrap",
        help="plan or explicitly replace the authoritative Release dataset",
    )
    dataset_bootstrap.add_argument("--repository")
    dataset_bootstrap.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    dataset_bootstrap.add_argument(
        "--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    dataset_bootstrap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    dataset_bootstrap.add_argument("--dry-run", action="store_true")
    dataset_bootstrap.add_argument("--confirm-replace-dataset", action="store_true")
    bootstrap = subparsers.add_parser(
        "prepare-bootstrap",
        help="index an existing Backfill for the first manual Release upload",
    )
    bootstrap.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    bootstrap.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Release pull and publication."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "pull":
            expected_identity, expected_tickers = runtime_dataset_identity(
                universe_path=args.universe
            )
            pull_release_dataset(
                repository=args.repository,
                release_tag=args.release_tag,
                output_root=args.output_root,
                force=args.force,
                years=args.years,
                dry_run=args.dry_run,
                expected_identity=expected_identity,
                expected_tickers=expected_tickers,
            )
        elif args.command == "pull-update-inputs":
            pull_update_inputs(
                repository=args.repository,
                release_tag=args.release_tag,
                output_root=args.output_root,
                universe_path=args.universe,
                refresh_calendar_days=args.refresh_calendar_days,
                settlement_delay_minutes=args.settlement_delay_minutes,
                timezone=args.timezone,
                target_date=args.target_date,
                dry_run=args.dry_run,
            )
        elif args.command == "publish-update":
            result = publish_update(
                repository=args.repository,
                release_tag=args.release_tag,
                prices_root=args.prices_root,
                universe_path=args.universe,
                dry_run=args.dry_run,
            )
            if args.result_json is not None:
                write_json_atomically(args.result_json, result)
        elif args.command == "check":
            result = check_release_dataset(
                repository=args.repository,
                release_tag=args.release_tag,
                prices_root=args.prices_root,
                universe_path=args.universe,
            )
            if args.result_json is not None:
                write_json_atomically(args.result_json, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            if result["workflow_ready"] is not True:
                return 1
        elif args.command == "bootstrap":
            bootstrap_dataset(
                repository=args.repository,
                release_tag=args.release_tag,
                prices_root=args.prices_root,
                universe_path=args.universe,
                confirm_replace_dataset=args.confirm_replace_dataset,
                dry_run=args.dry_run,
            )
        elif args.command == "prepare-bootstrap":
            prepare_bootstrap_manifest(
                prices_root=args.prices_root,
                dry_run=args.dry_run,
            )
        else:
            parser.error(f"unsupported command: {args.command}")
    except (ReleaseStorageError, ManifestError, OSError, ValueError) as exc:
        LOGGER.error("Release storage operation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
