from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import momentum_screener.release_storage as release_module
import momentum_screener.storage_manifest as manifest_module
from momentum_screener.dataset_config import DEFAULT_RELEASE_TAG
from momentum_screener.prices import universe_sha256
from momentum_screener.release_storage import (
    DatasetIdentity,
    GitHubClient,
    ReleaseStorageError,
    bootstrap_dataset,
    build_publish_plan,
    build_pull_plan,
    check_release_dataset,
    main,
    prepare_bootstrap_manifest,
    publish_update,
    pull_release_dataset,
    pull_update_inputs,
    require_remote_dataset_identity,
    resolve_github_token,
    resolve_repository,
    validate_remote_manifest,
)
from momentum_screener.storage_manifest import (
    COVERAGE_COLUMNS,
    FAILURE_COLUMNS,
    PRICE_SCHEMA,
    UPDATE_MISSING_COLUMNS,
    ManifestError,
    build_asset_record,
    calculate_sha256,
    validate_managed_asset,
    write_json_atomically,
)

TEST_IDENTITY = DatasetIdentity(
    schema_version="daily_prices_v1",
    universe_sha256="a" * 64,
    requested_start="2016-01-01",
    universe_ticker_count=1,
)


def test_all_release_operations_share_market_data_default_tag() -> None:
    assert DEFAULT_RELEASE_TAG == "marketData"
    for operation in (
        pull_release_dataset,
        pull_update_inputs,
        publish_update,
        check_release_dataset,
        bootstrap_dataset,
    ):
        assert (
            inspect.signature(operation).parameters["release_tag"].default
            == "marketData"
        )


def test_release_storage_has_no_legacy_tag_fallback_or_case_normalization() -> None:
    source = Path(release_module.__file__).read_text(encoding="utf-8")
    config = Path("src/momentum_screener/dataset_config.py").read_text(encoding="utf-8")
    legacy_tag = "market" + "-data"
    assert legacy_tag not in source
    assert legacy_tag not in config
    assert "release_tag.lower" not in source
    assert "release_tag.casefold" not in source


def write_price_asset(path: Path, year: int = 2026) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array([date(year, 1, 2)], type=pa.date32()),
            pa.array(["AAA"], type=pa.string()),
            pa.array([10.0], type=pa.float64()),
            pa.array([9.0], type=pa.float64()),
            pa.array([100], type=pa.int64()),
        ],
        schema=PRICE_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_remote_dataset(
    root: Path,
    *,
    parquet_mode: str = "valid",
    coverage_columns: tuple[str, ...] = COVERAGE_COLUMNS,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    source = root / "remote"
    source.mkdir(parents=True)
    price = source / "prices-year-2026.parquet"
    if parquet_mode == "corrupt":
        price.write_bytes(b"not parquet")
    elif parquet_mode == "wrong_schema":
        pq.write_table(pa.table({"bad": [1]}), price)
    elif parquet_mode == "wrong_year":
        write_price_asset(price, 2025)
    else:
        write_price_asset(price)
    coverage = source / "prices-ticker-coverage.csv"
    write_csv(
        coverage,
        coverage_columns,
        [
            {
                key: value
                for key, value in {
                    "ticker": "AAA",
                    "status": "success",
                    "first_date": "2026-01-02",
                    "last_date": "2026-01-02",
                    "row_count": 1,
                    "attempt_count": 1,
                    "last_error": "",
                }.items()
                if key in coverage_columns
            }
        ],
    )
    missing = source / "prices-update-missing-tickers.csv"
    write_csv(missing, UPDATE_MISSING_COLUMNS, [])
    report = source / "prices-update-report.json"
    write_json_atomically(
        report,
        {
            "run_id": "run-1",
            "target_session": "2026-01-02",
            "affected_years": [2026],
            "changed_partition_years": [2026],
            "changed_local_assets": [
                "daily/year=2026/prices.parquet",
                "ticker_coverage.csv",
                "update_missing_tickers.csv",
                "update_report.json",
                "manifest.json",
            ],
            "local_update_success": True,
            "success": True,
        },
    )
    assets = {
        "2026": build_asset_record(
            price,
            asset_name="prices-year-2026.parquet",
            local_path="daily/year=2026/prices.parquet",
        ),
        "ticker_coverage": build_asset_record(
            coverage,
            asset_name="prices-ticker-coverage.csv",
            local_path="ticker_coverage.csv",
        ),
        "update_missing_tickers": build_asset_record(
            missing,
            asset_name="prices-update-missing-tickers.csv",
            local_path="update_missing_tickers.csv",
        ),
        "update_report": build_asset_record(
            report,
            asset_name="prices-update-report.json",
            local_path="update_report.json",
        ),
    }
    manifest = {
        "schema_version": "daily_prices_v1",
        "source": "yahoo_finance_via_yfinance",
        "requested_start": "2016-01-01",
        "latest_session": "2026-01-02",
        "actual_min_date": "2026-01-02",
        "actual_max_date": "2026-01-02",
        "last_successful_update_utc": "2026-01-03T00:00:00Z",
        "last_update_run_id": "run-1",
        "last_update_target_session": "2026-01-02",
        "universe_sha256": "a" * 64,
        "universe_ticker_count": 1,
        "successful_ticker_count": 1,
        "no_data_ticker_count": 0,
        "failed_ticker_count": 0,
        "total_row_count": 1,
        "partition_row_counts": {"2026": 1},
        "assets": assets,
        "completed": True,
    }
    manifest_path = source / "prices-manifest.json"
    write_json_atomically(manifest_path, manifest)
    content = {path.name: path.read_bytes() for path in source.iterdir()}
    release_assets = [
        {
            "id": index,
            "name": name,
            "size": len(data),
            "url": f"https://api.example/assets/{index}",
        }
        for index, (name, data) in enumerate(content.items(), start=1)
    ]
    release = {
        "id": 1,
        "tag_name": "marketData",
        "upload_url": "https://uploads.example/releases/1/assets{?name,label}",
        "assets": release_assets,
    }
    return release, content, manifest


def make_local_2016_dataset(
    root: Path,
) -> tuple[Path, Path, DatasetIdentity]:
    universe = root / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text(
        "ticker,company_name,market_cap,market_cap_rank\nAAA,AAA Inc.,100,1\n",
        encoding="utf-8",
    )
    prices_root = root / "prices"
    price = prices_root / "daily/year=2016/prices.parquet"
    write_price_asset(price, 2016)
    coverage = prices_root / "ticker_coverage.csv"
    write_csv(
        coverage,
        COVERAGE_COLUMNS,
        [
            {
                "ticker": "AAA",
                "status": "success",
                "first_date": "2016-01-02",
                "last_date": "2016-01-02",
                "row_count": 1,
                "attempt_count": 1,
                "last_error": "",
            }
        ],
    )
    write_csv(prices_root / "download_failures.csv", FAILURE_COLUMNS, [])
    identity = DatasetIdentity(
        schema_version="daily_prices_v1",
        universe_sha256=universe_sha256(("AAA",)),
        requested_start="2016-01-01",
        universe_ticker_count=1,
    )
    manifest = {
        **identity.as_dict(),
        "source": "yahoo_finance_via_yfinance",
        "yfinance_version": "test",
        "generated_at_utc": "2016-01-03T00:00:00Z",
        "last_successful_update_utc": "2016-01-03T00:00:00Z",
        "requested_end_exclusive": "2016-01-03",
        "actual_min_date": "2016-01-02",
        "actual_max_date": "2016-01-02",
        "latest_session": "2016-01-02",
        "successful_ticker_count": 1,
        "no_data_ticker_count": 0,
        "failed_ticker_count": 0,
        "total_row_count": 1,
        "partition_row_counts": {"2016": 1},
        "assets": {
            "2016": build_asset_record(
                price,
                asset_name="prices-year-2016.parquet",
                local_path="daily/year=2016/prices.parquet",
            ),
            "ticker_coverage": build_asset_record(
                coverage,
                asset_name="prices-ticker-coverage.csv",
                local_path="ticker_coverage.csv",
            ),
        },
        "completed": True,
    }
    write_json_atomically(prices_root / "manifest.json", manifest)
    return universe, prices_root, identity


def make_release_for_manifest(
    prices_root: Path,
    manifest: Mapping[str, Any],
    *,
    extra_assets: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    content = {
        str(asset["asset_name"]): (prices_root / str(asset["local_path"])).read_bytes()
        for asset in manifest["assets"].values()
    }
    content["prices-manifest.json"] = (
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    ).encode()
    content.update(extra_assets or {})
    assets = [
        {
            "id": index,
            "name": name,
            "size": len(data),
            "url": f"https://api.example/assets/{index}",
        }
        for index, (name, data) in enumerate(content.items(), start=1)
    ]
    return (
        {
            "id": 1,
            "tag_name": "marketData",
            "upload_url": "https://uploads.example/releases/1/assets{?name,label}",
            "assets": assets,
        },
        content,
    )


class FakeClient:
    api_base = "https://api.example"

    def __init__(self, release: dict[str, Any], content: Mapping[str, bytes]) -> None:
        self.release = release
        self.content = dict(content)
        self.destinations: list[Path] = []
        self.uploaded: list[str] = []
        self.fail_upload: str | None = None

    def request_json(self, method: str, url: str, **kwargs: object) -> dict[str, Any]:
        assert method == "GET"
        return self.release

    def download_to(self, asset: Mapping[str, Any], destination: Path) -> int:
        name = str(asset["name"])
        data = self.content[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        self.destinations.append(destination)
        return len(data)

    def request_empty(self, method: str, url: str) -> None:
        assert method == "DELETE"
        asset_id = int(url.rsplit("/", maxsplit=1)[1])
        self.release["assets"] = [
            item for item in self.release["assets"] if item["id"] != asset_id
        ]

    def upload_file(
        self, upload_url: str, *, asset_name: str, path: Path
    ) -> dict[str, Any]:
        if asset_name == self.fail_upload:
            raise ReleaseStorageError(f"simulated upload failure for {asset_name}")
        data = path.read_bytes()
        self.uploaded.append(asset_name)
        self.content[asset_name] = data
        asset = {
            "id": 100 + len(self.uploaded),
            "name": asset_name,
            "size": len(data),
            "url": f"https://api.example/uploaded/{asset_name}",
        }
        self.release["assets"].append(asset)
        return asset


@pytest.mark.parametrize(
    ("cli", "environment", "expected"),
    [
        ("cli-owner/cli-repo", {"GITHUB_REPOSITORY": "env/repo"}, "cli-owner/cli-repo"),
        (None, {"GITHUB_REPOSITORY": "env/repo"}, "env/repo"),
        (None, {"MOMENTUM_SCREENER_REPOSITORY": "fallback/repo"}, "fallback/repo"),
    ],
)
def test_resolve_repository_priority(
    tmp_path: Path,
    cli: str | None,
    environment: dict[str, str],
    expected: str,
) -> None:
    assert (
        resolve_repository(
            cli, environ=environment, git_config_path=tmp_path / "missing"
        )
        == expected
    )


@pytest.mark.parametrize(
    ("remote_url", "expected"),
    [
        ("https://github.com/example/project.git", "example/project"),
        ("git@github.com:example/project.git", "example/project"),
        ("ssh://git@github.com/example/project.git", "example/project"),
    ],
)
def test_resolve_repository_read_only_git_config(
    tmp_path: Path, remote_url: str, expected: str
) -> None:
    config = tmp_path / ".git" / "config"
    config.parent.mkdir()
    config.write_text(f'[remote "origin"]\n\turl = {remote_url}\n', encoding="utf-8")
    assert resolve_repository(environ={}, git_config_path=config) == expected


def test_resolve_repository_missing_fails_without_running_commands(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReleaseStorageError, match="--repository"):
        resolve_repository(environ={}, git_config_path=tmp_path / "missing")
    source = Path(release_module.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "os.system" not in source


def test_token_priority_and_absence() -> None:
    assert (
        resolve_github_token({"GITHUB_TOKEN": "first", "GH_TOKEN": "second"}) == "first"
    )
    assert resolve_github_token({"GH_TOKEN": "second"}) == "second"
    assert resolve_github_token({}) is None


def test_release_lookup_failure_has_tag_and_repository_context(
    tmp_path: Path,
) -> None:
    class MissingReleaseClient(FakeClient):
        def request_json(
            self, method: str, url: str, **kwargs: object
        ) -> dict[str, Any]:
            raise ReleaseStorageError("HTTP 404")

    release, content, _ = make_remote_dataset(tmp_path)
    client = MissingReleaseClient(release, content)
    with pytest.raises(ReleaseStorageError, match="marketData.*owner/repo"):
        pull_release_dataset(
            repository="owner/repo",
            output_root=tmp_path / "output",
            client=cast(Any, client),
        )


@pytest.mark.parametrize("mode", ["missing", "invalid_json"])
def test_pull_rejects_missing_or_corrupt_manifest_asset(
    tmp_path: Path, mode: str
) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    if mode == "missing":
        release["assets"] = [
            asset
            for asset in release["assets"]
            if asset["name"] != "prices-manifest.json"
        ]
    else:
        content["prices-manifest.json"] = b"{broken"
        for asset in release["assets"]:
            if asset["name"] == "prices-manifest.json":
                asset["size"] = len(content["prices-manifest.json"])
    client = FakeClient(release, content)
    with pytest.raises(ReleaseStorageError, match="does not exist|invalid JSON"):
        pull_release_dataset(
            repository="owner/repo",
            output_root=tmp_path / "output",
            client=cast(Any, client),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="future"), "schema_version"),
        (lambda value: value.update(completed=False), "completed"),
        (
            lambda value: value["assets"]["2026"].update(
                local_path="../escape.parquet"
            ),
            "unsafe",
        ),
        (
            lambda value: value["assets"]["2026"].update(
                local_path="/absolute.parquet"
            ),
            "absolute",
        ),
        (lambda value: value["assets"]["2026"].update(sha256="bad"), "sha256"),
        (lambda value: value["assets"]["2026"].update(size_bytes=-1), "size_bytes"),
    ],
)
def test_validate_remote_manifest_rejects_unsafe_or_unsupported_values(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    _, _, manifest = make_remote_dataset(tmp_path)
    mutation(manifest)
    with pytest.raises(ReleaseStorageError, match=message):
        validate_remote_manifest(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(universe_sha256="b" * 64),
        lambda value: value.update(requested_start="2010-01-01"),
        lambda value: value.update(schema_version="future"),
        lambda value: value.update(universe_ticker_count=2000),
        lambda value: value.update(completed=False),
    ],
)
def test_remote_dataset_identity_mismatch_requires_bootstrap(
    tmp_path: Path, mutation: Any
) -> None:
    _, _, manifest = make_remote_dataset(tmp_path)
    mutation(manifest)

    with pytest.raises(ReleaseStorageError, match="must be bootstrapped"):
        require_remote_dataset_identity(manifest, TEST_IDENTITY)


def test_pull_update_identity_mismatch_stops_before_year_assets(
    tmp_path: Path,
) -> None:
    universe, prices_root, identity = make_local_2016_dataset(tmp_path / "local")
    candidate = prepare_bootstrap_manifest(prices_root=prices_root, dry_run=True)
    remote = dict(candidate)
    remote["universe_sha256"] = "b" * 64
    release, content = make_release_for_manifest(prices_root, remote)
    client = FakeClient(release, content)
    before = (prices_root / "manifest.json").read_bytes()

    with pytest.raises(ReleaseStorageError, match="must be bootstrapped"):
        pull_update_inputs(
            repository="owner/repo",
            output_root=prices_root,
            universe_path=universe,
            target_date=date(2016, 1, 4),
            client=cast(Any, client),
            expected_identity=identity,
            expected_tickers=("AAA",),
        )

    assert len(client.destinations) == 1
    assert client.destinations[0].name == "manifest.json"
    assert (prices_root / "manifest.json").read_bytes() == before


def test_pull_update_validates_coverage_before_replacement(tmp_path: Path) -> None:
    universe, prices_root, identity = make_local_2016_dataset(tmp_path / "local")
    candidate = prepare_bootstrap_manifest(prices_root=prices_root, dry_run=True)
    bad_coverage = tmp_path / "bad-coverage.csv"
    write_csv(
        bad_coverage,
        COVERAGE_COLUMNS,
        [
            {
                "ticker": "WRONG",
                "status": "success",
                "first_date": "2016-01-02",
                "last_date": "2016-01-02",
                "row_count": 1,
                "attempt_count": 1,
                "last_error": "",
            }
        ],
    )
    candidate = dict(candidate)
    candidate["assets"] = dict(candidate["assets"])
    candidate["assets"]["ticker_coverage"] = build_asset_record(
        bad_coverage,
        asset_name="prices-ticker-coverage.csv",
        local_path="ticker_coverage.csv",
    )
    release, content = make_release_for_manifest(prices_root, candidate)
    content["prices-ticker-coverage.csv"] = bad_coverage.read_bytes()
    for asset in release["assets"]:
        if asset["name"] == "prices-ticker-coverage.csv":
            asset["size"] = bad_coverage.stat().st_size
    content["prices-manifest.json"] = (
        json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    ).encode()
    for asset in release["assets"]:
        if asset["name"] == "prices-manifest.json":
            asset["size"] = len(content["prices-manifest.json"])
    client = FakeClient(release, content)
    old_coverage = (prices_root / "ticker_coverage.csv").read_bytes()

    with pytest.raises(ReleaseStorageError, match="coverage"):
        pull_update_inputs(
            repository="owner/repo",
            output_root=prices_root,
            universe_path=universe,
            target_date=date(2016, 1, 4),
            client=cast(Any, client),
            expected_identity=identity,
            expected_tickers=("AAA",),
        )

    assert (prices_root / "ticker_coverage.csv").read_bytes() == old_coverage


def test_build_pull_plan_compares_actual_hash_size_missing_force_and_year(
    tmp_path: Path,
) -> None:
    _, content, manifest = make_remote_dataset(tmp_path)
    output = tmp_path / "output"
    price = output / "daily/year=2026/prices.parquet"
    price.parent.mkdir(parents=True)
    price.write_bytes(content["prices-year-2026.parquet"])
    download, unchanged = build_pull_plan(
        manifest, output, years={2026}, include_auxiliary=set()
    )
    assert download == []
    assert unchanged == ["2026"]
    price.write_bytes(b"same size but wrong".ljust(price.stat().st_size, b"x"))
    download, _ = build_pull_plan(manifest, output, years={2026})
    assert download == ["2026"]
    price.write_bytes(content["prices-year-2026.parquet"][:-1])
    download, _ = build_pull_plan(manifest, output, years={2026})
    assert download == ["2026"]
    price.unlink()
    download, _ = build_pull_plan(manifest, output, years={2026})
    assert download == ["2026"]
    price.parent.mkdir(parents=True, exist_ok=True)
    price.write_bytes(content["prices-year-2026.parquet"])
    download, _ = build_pull_plan(manifest, output, force=True, years={2026})
    assert download == ["2026"]
    download, unchanged = build_pull_plan(manifest, output, years={2025})
    assert download == []
    assert unchanged == []


def test_pull_downloads_to_staging_validates_and_restores_layout(
    tmp_path: Path,
) -> None:
    release, content, manifest = make_remote_dataset(tmp_path)
    client = FakeClient(release, content)
    output = tmp_path / "prices"
    unmanaged = output / "keep.txt"
    unmanaged.parent.mkdir(parents=True)
    unmanaged.write_text("keep", encoding="utf-8")

    report = pull_release_dataset(
        repository="owner/repo", output_root=output, client=cast(Any, client)
    )

    assert report["success"] is True
    assert len(report["downloaded_assets"]) == 4
    assert (output / "daily/year=2026/prices.parquet").is_file()
    assert (output / "ticker_coverage.csv").is_file()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert unmanaged.read_text(encoding="utf-8") == "keep"
    assert all(".sync_staging" in str(path) for path in client.destinations[1:])
    assert not any((output / ".sync_staging").glob("sync-*"))
    assert not any((output / ".sync_backup").glob("sync-*"))


def test_pull_skips_unchanged_and_force_redownloads(tmp_path: Path) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    output = tmp_path / "prices"
    first_client = FakeClient(release, content)
    pull_release_dataset(
        repository="owner/repo", output_root=output, client=cast(Any, first_client)
    )
    second_client = FakeClient(release, content)
    unchanged = pull_release_dataset(
        repository="owner/repo", output_root=output, client=cast(Any, second_client)
    )
    assert unchanged["downloaded_assets"] == []
    assert len(unchanged["unchanged_assets"]) == 4
    forced_client = FakeClient(release, content)
    forced = pull_release_dataset(
        repository="owner/repo",
        output_root=output,
        force=True,
        client=cast(Any, forced_client),
    )
    assert len(forced["downloaded_assets"]) == 4


def test_pull_year_downloads_only_requested_year(tmp_path: Path) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    client = FakeClient(release, content)
    output = tmp_path / "prices"
    report = pull_release_dataset(
        repository="owner/repo",
        output_root=output,
        years={2026},
        client=cast(Any, client),
    )
    assert report["downloaded_assets"] == ["prices-year-2026.parquet"]
    assert (output / "daily/year=2026/prices.parquet").is_file()
    assert not (output / "ticker_coverage.csv").exists()


def test_pull_dry_run_does_not_write_output(tmp_path: Path) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    client = FakeClient(release, content)
    output = tmp_path / "prices"
    result = pull_release_dataset(
        repository="owner/repo",
        output_root=output,
        dry_run=True,
        client=cast(Any, client),
    )
    assert result["dry_run"] is True
    assert not output.exists()


@pytest.mark.parametrize("mode", ["size", "hash"])
def test_pull_download_size_or_hash_mismatch_never_replaces(
    tmp_path: Path, mode: str
) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    output = tmp_path / "prices"
    output.mkdir()
    old = b'{"old": true}\n'
    (output / "manifest.json").write_bytes(old)
    client = FakeClient(release, content)
    original_download = client.download_to

    def altered_download(asset: Mapping[str, Any], destination: Path) -> int:
        written = original_download(asset, destination)
        if asset["name"] == "prices-year-2026.parquet":
            data = destination.read_bytes()
            if mode == "size":
                destination.write_bytes(data[:-1])
                return written - 1
            destination.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        return written

    client.download_to = altered_download  # type: ignore[method-assign]
    with pytest.raises(ManifestError, match="size mismatch|SHA-256 mismatch"):
        pull_release_dataset(
            repository="owner/repo", output_root=output, client=cast(Any, client)
        )
    assert (output / "manifest.json").read_bytes() == old


@pytest.mark.parametrize(
    ("parquet_mode", "coverage_columns", "message"),
    [
        ("corrupt", COVERAGE_COLUMNS, "Unable to read Parquet"),
        ("wrong_schema", COVERAGE_COLUMNS, "schema mismatch"),
        ("wrong_year", COVERAGE_COLUMNS, "contains years"),
        ("valid", ("ticker", "status"), "CSV header mismatch"),
    ],
)
def test_pull_invalid_managed_asset_never_replaces_local_manifest(
    tmp_path: Path,
    parquet_mode: str,
    coverage_columns: tuple[str, ...],
    message: str,
) -> None:
    release, content, _ = make_remote_dataset(
        tmp_path, parquet_mode=parquet_mode, coverage_columns=coverage_columns
    )
    output = tmp_path / "prices"
    output.mkdir()
    old_manifest = b'{"old": true}\n'
    (output / "manifest.json").write_bytes(old_manifest)
    client = FakeClient(release, content)
    with pytest.raises((ManifestError, ReleaseStorageError), match=message):
        pull_release_dataset(
            repository="owner/repo", output_root=output, client=cast(Any, client)
        )
    assert (output / "manifest.json").read_bytes() == old_manifest


def test_pull_replacement_failure_rolls_back_every_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    output = tmp_path / "prices"
    output.mkdir()
    old_manifest = b'{"old": true}\n'
    old_coverage = b"old coverage\n"
    (output / "manifest.json").write_bytes(old_manifest)
    (output / "ticker_coverage.csv").write_bytes(old_coverage)
    real_replace = manifest_module.os.replace
    failed = False

    def fail_once(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if (
            not failed
            and ".sync_staging" in str(source)
            and str(destination).endswith("ticker_coverage.csv")
        ):
            failed = True
            raise OSError("simulated replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(manifest_module.os, "replace", fail_once)
    client = FakeClient(release, content)
    with pytest.raises(OSError, match="simulated"):
        pull_release_dataset(
            repository="owner/repo", output_root=output, client=cast(Any, client)
        )
    assert (output / "manifest.json").read_bytes() == old_manifest
    assert (output / "ticker_coverage.csv").read_bytes() == old_coverage
    assert not (output / "daily/year=2026/prices.parquet").exists()


def test_pull_passes_manifest_as_last_transactional_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    observed: list[str] = []
    real_replace = release_module.replace_files_transactionally

    def record_order(
        root: Path,
        staging_root: Path,
        relative_paths: list[str],
        **kwargs: Any,
    ) -> None:
        observed.extend(relative_paths)
        real_replace(root, staging_root, relative_paths, **kwargs)

    monkeypatch.setattr(release_module, "replace_files_transactionally", record_order)
    pull_release_dataset(
        repository="owner/repo",
        output_root=tmp_path / "prices",
        client=cast(Any, FakeClient(release, content)),
    )
    assert observed[-1] == "manifest.json"


def test_pull_report_has_required_fields_and_no_token(tmp_path: Path) -> None:
    release, content, _ = make_remote_dataset(tmp_path)
    client = FakeClient(release, content)
    output = tmp_path / "prices"
    report = pull_release_dataset(
        repository="owner/repo",
        output_root=output,
        client=cast(Any, client),
        environ={"GITHUB_TOKEN": "super-secret"},
    )
    disk = json.loads((output / "sync_report.json").read_text(encoding="utf-8"))
    assert disk == report
    assert {
        "sync_id",
        "repository",
        "release_tag",
        "started_at_utc",
        "finished_at_utc",
        "remote_latest_session",
        "local_previous_session",
        "downloaded_assets",
        "unchanged_assets",
        "downloaded_bytes",
        "success",
    }.issubset(report)
    assert "super-secret" not in json.dumps(report)


def test_check_is_read_only_ready_and_reports_obsolete_assets(tmp_path: Path) -> None:
    universe, prices_root, identity = make_local_2016_dataset(tmp_path / "local")
    candidate = prepare_bootstrap_manifest(prices_root=prices_root, dry_run=True)
    release, content = make_release_for_manifest(
        prices_root,
        candidate,
        extra_assets={"prices-year-2010.parquet": b"obsolete"},
    )
    client = FakeClient(release, content)
    before = {
        path.relative_to(prices_root): path.read_bytes()
        for path in prices_root.rglob("*")
        if path.is_file()
    }

    result = check_release_dataset(
        repository="owner/repo",
        prices_root=prices_root,
        universe_path=universe,
        client=cast(Any, client),
        expected_identity=identity,
    )

    after = {
        path.relative_to(prices_root): path.read_bytes()
        for path in prices_root.rglob("*")
        if path.is_file()
    }
    assert result["workflow_ready"] is True
    assert result["dataset_identity_match"] is True
    assert result["release_tag"] == "marketData"
    assert result["local_schema_version"] == "daily_prices_v1"
    assert result["remote_schema_version"] == "daily_prices_v1"
    assert result["local_universe_ticker_count"] == identity.universe_ticker_count
    assert result["remote_universe_ticker_count"] == identity.universe_ticker_count
    assert result["obsolete_remote_assets"] == ["prices-year-2010.parquet"]
    assert before == after


def test_check_identity_mismatch_or_missing_asset_is_not_ready(
    tmp_path: Path,
) -> None:
    universe, prices_root, identity = make_local_2016_dataset(tmp_path / "local")
    candidate = prepare_bootstrap_manifest(prices_root=prices_root, dry_run=True)
    mismatching = dict(candidate)
    mismatching["requested_start"] = "2010-01-01"
    release, content = make_release_for_manifest(prices_root, mismatching)
    mismatch = check_release_dataset(
        repository="owner/repo",
        prices_root=prices_root,
        universe_path=universe,
        client=cast(Any, FakeClient(release, content)),
        expected_identity=identity,
    )
    assert mismatch["workflow_ready"] is False
    assert "bootstrapped" in str(mismatch["error"])

    release, content = make_release_for_manifest(prices_root, candidate)
    release["assets"] = [
        asset
        for asset in release["assets"]
        if asset["name"] != "prices-ticker-coverage.csv"
    ]
    missing = check_release_dataset(
        repository="owner/repo",
        prices_root=prices_root,
        universe_path=universe,
        client=cast(Any, FakeClient(release, content)),
        expected_identity=identity,
    )
    assert missing["workflow_ready"] is False
    assert missing["missing_managed_assets"] == ["prices-ticker-coverage.csv"]


def materialize_publish_fixture(tmp_path: Path) -> Path:
    _, content, manifest = make_remote_dataset(tmp_path)
    root = tmp_path / "publish"
    mappings = {
        "prices-year-2026.parquet": "daily/year=2026/prices.parquet",
        "prices-ticker-coverage.csv": "ticker_coverage.csv",
        "prices-update-missing-tickers.csv": "update_missing_tickers.csv",
        "prices-update-report.json": "update_report.json",
    }
    for name, local_path in mappings.items():
        destination = root / local_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content[name])
    write_json_atomically(root / "manifest.json", manifest)
    assert not (root / "release_publish_plan.json").exists()
    return root


def test_prepare_bootstrap_indexes_existing_data_without_network(
    tmp_path: Path,
) -> None:
    _, content, manifest = make_remote_dataset(tmp_path)
    root = tmp_path / "bootstrap"
    price = root / "daily/year=2026/prices.parquet"
    price.parent.mkdir(parents=True)
    price.write_bytes(content["prices-year-2026.parquet"])
    (root / "ticker_coverage.csv").write_bytes(content["prices-ticker-coverage.csv"])
    legacy = dict(manifest)
    legacy.pop("assets")
    legacy.pop("latest_session")
    legacy.pop("last_successful_update_utc")
    write_json_atomically(root / "manifest.json", legacy)
    before = (root / "manifest.json").read_bytes()

    preview = prepare_bootstrap_manifest(prices_root=root, dry_run=True)
    assert len(preview["assets"]) == 2
    assert (root / "manifest.json").read_bytes() == before
    indexed = prepare_bootstrap_manifest(prices_root=root)
    assert indexed["latest_session"] == "2026-01-02"
    assert indexed["assets"]["2026"]["sha256"] == calculate_sha256(price)


def test_bootstrap_default_only_writes_local_migration_plan(tmp_path: Path) -> None:
    universe, prices_root, _ = make_local_2016_dataset(tmp_path / "local")
    before_manifest = (prices_root / "manifest.json").read_bytes()

    plan = bootstrap_dataset(
        prices_root=prices_root,
        universe_path=universe,
        expected_universe_size=1,
    )

    assert plan["identity_matches"] == "not_checked"
    assert plan["bootstrap_required"] is True
    assert plan["daily_workflow_ready"] is False
    assert plan["remote_dataset"]["release_tag"] == "marketData"
    names = [item["asset_name"] for item in plan["assets_to_upload"]]
    assert names == [
        "prices-year-2016.parquet",
        "prices-ticker-coverage.csv",
        "prices-download-failures.csv",
        "prices-manifest.json",
    ]
    assert names[-1] == "prices-manifest.json"
    assert (prices_root / "release_migration_plan.json").is_file()
    assert (prices_root / "manifest.json").read_bytes() == before_manifest
    assert not (prices_root / "remote_obsolete_assets.json").exists()
    custom_plan = bootstrap_dataset(
        prices_root=prices_root,
        universe_path=universe,
        release_tag="CuStOmTag",
        expected_universe_size=1,
    )
    assert custom_plan["remote_dataset"]["release_tag"] == "CuStOmTag"


def test_bootstrap_confirm_uploads_manifest_last_and_keeps_obsolete_assets(
    tmp_path: Path,
) -> None:
    universe, prices_root, _ = make_local_2016_dataset(tmp_path / "local")
    release, content, _ = make_remote_dataset(tmp_path / "old-remote")
    client = FakeClient(release, content)

    result = bootstrap_dataset(
        repository="owner/repo",
        prices_root=prices_root,
        universe_path=universe,
        confirm_replace_dataset=True,
        client=cast(Any, client),
        expected_universe_size=1,
    )

    assert result["success"] is True
    assert client.uploaded[-1] == "prices-manifest.json"
    assert result["manifest_uploaded_last"] is True
    assert "prices-year-2026.parquet" in result["obsolete_remote_assets"]
    assert any(
        asset["name"] == "prices-year-2026.parquet"
        for asset in client.release["assets"]
    )
    local_manifest = json.loads(
        (prices_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert "download_failures" in local_manifest["assets"]
    obsolete = json.loads(
        (prices_root / "remote_obsolete_assets.json").read_text(encoding="utf-8")
    )
    assert obsolete["deleted"] is False
    assert "prices-year-2026.parquet" in obsolete["obsolete_remote_assets"]


def test_bootstrap_failure_before_manifest_never_uploads_manifest(
    tmp_path: Path,
) -> None:
    universe, prices_root, _ = make_local_2016_dataset(tmp_path / "local")
    release, content, _ = make_remote_dataset(tmp_path / "old-remote")
    client = FakeClient(release, content)
    client.fail_upload = "prices-download-failures.csv"

    with pytest.raises(ReleaseStorageError, match="simulated upload failure"):
        bootstrap_dataset(
            repository="owner/repo",
            prices_root=prices_root,
            universe_path=universe,
            confirm_replace_dataset=True,
            client=cast(Any, client),
            expected_universe_size=1,
        )

    assert "prices-manifest.json" not in client.uploaded


def test_bootstrap_rejects_pre_2016_partition_before_network(tmp_path: Path) -> None:
    universe, prices_root, _ = make_local_2016_dataset(tmp_path / "local")
    old_path = prices_root / "daily/year=2015/prices.parquet"
    write_price_asset(old_path, 2015)
    manifest_path = prices_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partition_row_counts"]["2015"] = 1
    manifest["total_row_count"] = 2
    manifest["assets"]["2015"] = build_asset_record(
        old_path,
        asset_name="prices-year-2015.parquet",
        local_path="daily/year=2015/prices.parquet",
    )
    write_json_atomically(manifest_path, manifest)

    with pytest.raises(ManifestError, match="before requested_start"):
        bootstrap_dataset(
            prices_root=prices_root,
            universe_path=universe,
            expected_universe_size=1,
        )


def test_build_publish_plan_validates_hash_and_manifest_last(tmp_path: Path) -> None:
    root = materialize_publish_fixture(tmp_path)
    plan = build_publish_plan(root, repository="owner/repo")
    assert plan["assets"][-1]["asset_name"] == "prices-manifest.json"
    assert plan["repository"] == "owner/repo"
    assert plan["release_tag"] == "marketData"
    assert (root / "release_publish_plan.json").is_file()
    custom_plan = build_publish_plan(
        root,
        repository="owner/repo",
        release_tag="CuStOmTag",
    )
    assert custom_plan["release_tag"] == "CuStOmTag"
    (root / "update_report.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ManifestError, match="size mismatch|SHA-256 mismatch"):
        build_publish_plan(root, repository="owner/repo")


def test_publish_uploads_parquet_first_and_manifest_last(tmp_path: Path) -> None:
    root = materialize_publish_fixture(tmp_path)
    release, content, _ = make_remote_dataset(tmp_path / "other")
    client = FakeClient(release, content)
    result = publish_update(
        repository="owner/repo",
        prices_root=root,
        client=cast(Any, client),
        expected_identity=TEST_IDENTITY,
    )
    assert result["success"] is True
    assert result["release_publish_success"] is True
    assert client.uploaded[0] == "prices-year-2026.parquet"
    assert client.uploaded[-1] == "prices-manifest.json"
    assert result["manifest_uploaded_last"] is True


def test_publish_failure_before_manifest_never_uploads_manifest(tmp_path: Path) -> None:
    root = materialize_publish_fixture(tmp_path)
    release, content, _ = make_remote_dataset(tmp_path / "other")
    client = FakeClient(release, content)
    client.fail_upload = "prices-update-report.json"
    with pytest.raises(ReleaseStorageError, match="simulated upload failure"):
        publish_update(
            repository="owner/repo",
            prices_root=root,
            client=cast(Any, client),
            expected_identity=TEST_IDENTITY,
        )
    assert "prices-manifest.json" not in client.uploaded


def test_publish_identity_mismatch_never_uploads_any_asset(tmp_path: Path) -> None:
    root = materialize_publish_fixture(tmp_path)
    release, content, manifest = make_remote_dataset(tmp_path / "other")
    manifest["universe_sha256"] = "b" * 64
    content["prices-manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    for asset in release["assets"]:
        if asset["name"] == "prices-manifest.json":
            asset["size"] = len(content["prices-manifest.json"])
    client = FakeClient(release, content)

    with pytest.raises(ReleaseStorageError, match="must be bootstrapped"):
        publish_update(
            repository="owner/repo",
            prices_root=root,
            client=cast(Any, client),
            expected_identity=TEST_IDENTITY,
        )

    assert client.uploaded == []


def test_publish_dry_run_does_not_construct_network_client(tmp_path: Path) -> None:
    root = materialize_publish_fixture(tmp_path)
    result = publish_update(
        repository="owner/repo",
        prices_root=root,
        dry_run=True,
        environ={},
        expected_identity=TEST_IDENTITY,
    )
    assert result["dry_run"] is True
    assert result["release_publish_success"] is False
    assert result["assets"][-1] == "prices-manifest.json"


def test_publish_requires_repository_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = materialize_publish_fixture(tmp_path)

    def missing_repository(*args: object, **kwargs: object) -> str:
        raise ReleaseStorageError("GitHub repository is unavailable")

    monkeypatch.setattr(release_module, "resolve_repository", missing_repository)
    with pytest.raises(ReleaseStorageError, match="repository is unavailable"):
        publish_update(
            prices_root=root,
            environ={},
            expected_identity=TEST_IDENTITY,
        )

    monkeypatch.setattr(
        release_module,
        "resolve_repository",
        lambda *args, **kwargs: "owner/repo",
    )
    with pytest.raises(ReleaseStorageError, match="GitHub token is required"):
        publish_update(
            prices_root=root,
            environ={},
            expected_identity=TEST_IDENTITY,
        )


def test_github_api_retry_is_finite_and_token_not_in_error() -> None:
    calls = 0

    def unavailable(request: Request, timeout: float) -> Any:
        nonlocal calls
        calls += 1
        raise URLError("offline")

    client = GitHubClient(
        token="super-secret",
        max_retries=2,
        open_func=unavailable,
        sleep_func=lambda _: None,
    )
    with pytest.raises(ReleaseStorageError) as raised:
        client.request_json("GET", "https://api.example/test")
    assert calls == 3
    assert "super-secret" not in str(raised.value)


def test_release_cli_defaults_and_case_preserving_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, list[str]] = {
        command: []
        for command in (
            "pull",
            "pull-update-inputs",
            "publish-update",
            "check",
            "bootstrap",
        )
    }
    monkeypatch.setattr(
        release_module,
        "runtime_dataset_identity",
        lambda **kwargs: (TEST_IDENTITY, ("AAA",)),
    )
    monkeypatch.setattr(
        release_module,
        "pull_release_dataset",
        lambda **kwargs: observed["pull"].append(str(kwargs["release_tag"])) or {},
    )
    monkeypatch.setattr(
        release_module,
        "pull_update_inputs",
        lambda **kwargs: (
            observed["pull-update-inputs"].append(str(kwargs["release_tag"])) or {}
        ),
    )
    monkeypatch.setattr(
        release_module,
        "publish_update",
        lambda **kwargs: (
            observed["publish-update"].append(str(kwargs["release_tag"])) or {}
        ),
    )
    monkeypatch.setattr(
        release_module,
        "check_release_dataset",
        lambda **kwargs: (
            observed["check"].append(str(kwargs["release_tag"]))
            or {"workflow_ready": True}
        ),
    )
    monkeypatch.setattr(
        release_module,
        "bootstrap_dataset",
        lambda **kwargs: observed["bootstrap"].append(str(kwargs["release_tag"])) or {},
    )

    assert main(["pull", "--repository", "owner/repo", "--dry-run"]) == 0
    assert main(["pull-update-inputs", "--repository", "owner/repo", "--dry-run"]) == 0
    assert main(["publish-update", "--repository", "owner/repo", "--dry-run"]) == 0
    assert main(["check", "--repository", "owner/repo"]) == 0
    assert main(["bootstrap", "--dry-run"]) == 0
    assert all(values == ["marketData"] for values in observed.values())

    assert (
        main(
            [
                "check",
                "--repository",
                "owner/repo",
                "--release-tag",
                "CuStOmTag",
            ]
        )
        == 0
    )
    assert observed["check"][-1] == "CuStOmTag"


@pytest.mark.parametrize(
    "command",
    ("check", "bootstrap", "pull", "pull-update-inputs", "publish-update"),
)
def test_release_cli_help_displays_market_data_default(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main([command, "--help"])

    assert raised.value.code == 0
    assert "GitHub Release tag (default: marketData)" in capsys.readouterr().out


def test_cli_success_failure_and_help_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_module, "pull_release_dataset", lambda **kwargs: {})
    assert main(["pull", "--repository", "owner/repo", "--dry-run"]) == 0

    def fail(**kwargs: object) -> dict[str, Any]:
        raise ReleaseStorageError("expected")

    monkeypatch.setattr(release_module, "pull_release_dataset", fail)
    assert main(["pull", "--repository", "owner/repo"]) == 1
    with pytest.raises(SystemExit) as raised:
        main(["pull", "--help"])
    assert raised.value.code == 0


def test_check_and_bootstrap_cli_exit_codes_and_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_module,
        "check_release_dataset",
        lambda **kwargs: {"workflow_ready": True},
    )
    assert main(["check", "--repository", "owner/repo"]) == 0
    monkeypatch.setattr(
        release_module,
        "check_release_dataset",
        lambda **kwargs: {"workflow_ready": False, "error": "identity mismatch"},
    )
    assert main(["check", "--repository", "owner/repo"]) == 1

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        release_module,
        "bootstrap_dataset",
        lambda **kwargs: calls.append(dict(kwargs)) or {},
    )
    assert main(["bootstrap", "--dry-run"]) == 0
    assert calls[-1]["confirm_replace_dataset"] is False
    assert calls[-1]["dry_run"] is True

    for command in ("check", "bootstrap", "pull-update-inputs", "publish-update"):
        with pytest.raises(SystemExit) as raised:
            main([command, "--help"])
        assert raised.value.code == 0


def test_validate_managed_asset_rejects_unsorted_parquet(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    table = pa.Table.from_arrays(
        [
            pa.array([date(2026, 1, 3), date(2026, 1, 2)], type=pa.date32()),
            pa.array(["AAA", "AAA"]),
            pa.array([10.0, 9.0]),
            pa.array([10.0, 9.0]),
            pa.array([100, 90], type=pa.int64()),
        ],
        schema=PRICE_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")
    asset = build_asset_record(
        path,
        asset_name="prices-year-2026.parquet",
        local_path="daily/year=2026/prices.parquet",
    )
    with pytest.raises(ManifestError, match="not sorted"):
        validate_managed_asset(path, key="2026", asset=asset)
