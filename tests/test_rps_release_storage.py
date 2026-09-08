from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.rps_release_storage as release_module
from momentum_screener.release_storage import GitHubClient, ReleaseStorageError
from momentum_screener.rps_release_storage import (
    RPS_RELEASE_MANIFEST_ASSET_NAME,
    RpsReleaseStorageError,
    check_rps_release,
    publish_rps_release,
    pull_rps_release,
)
from momentum_screener.rps_storage import load_rps_manifest, persist_rps_snapshot


def _write_universe(path: Path, tickers: tuple[str, ...]) -> None:
    rows = "".join(
        f"{ticker},{ticker} Inc.,{1000 - index},{index + 1}\n"
        for index, ticker in enumerate(tickers)
    )
    path.write_text(
        "ticker,company_name,market_cap,market_cap_rank\n" + rows,
        encoding="utf-8",
    )


def _build_local_dataset(tmp_path: Path) -> tuple[Path, Path]:
    tickers = ("AAA", "BBB")
    universe_path = tmp_path / "universe.csv"
    root = tmp_path / "rps"
    _write_universe(universe_path, tickers)
    snapshot = pd.DataFrame(
        {
            "ticker": tickers,
            "as_of_date": date(2026, 8, 31),
            "rps20": [0.0, 100.0],
            "rps50": [0.0, 100.0],
            "rps120": [100.0, 0.0],
            "rps250": [0.0, 100.0],
            "return_20": [0.01, 0.02],
            "return_50": [0.1, 0.2],
            "return_120": [0.3, 0.4],
            "return_250": [np.nan, 0.5],
            "rps20_base_date": date(2026, 8, 3),
            "rps50_base_date": date(2026, 6, 19),
            "rps120_base_date": date(2026, 3, 10),
            "rps250_base_date": date(2025, 9, 2),
        }
    )
    persist_rps_snapshot(snapshot, root=root, universe_path=universe_path)
    return root, universe_path


def test_bootstrap_dry_run_plans_partition_then_manifest_without_uploading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, universe_path = _build_local_dataset(tmp_path)
    release = {"assets": [], "upload_url": "unused"}
    monkeypatch.setattr(
        release_module,
        "get_release_metadata",
        lambda *_args, **_kwargs: release,
    )

    def unexpected_upload(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not upload RPS assets")

    monkeypatch.setattr(release_module, "upload_release_asset", unexpected_upload)

    result = publish_rps_release(
        repository="owner/repository",
        root=root,
        universe_path=universe_path,
        bootstrap=True,
        dry_run=True,
        client=GitHubClient(token="test-token"),
        environ={},
    )

    assert result["release_tag"] == "rpsData"
    assert result["planned_assets"] == [
        "rps-year-2026.parquet",
        RPS_RELEASE_MANIFEST_ASSET_NAME,
    ]
    assert result["dry_run"] is True


def test_bootstrap_requires_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, universe_path = _build_local_dataset(tmp_path)
    monkeypatch.setattr(
        release_module,
        "get_release_metadata",
        lambda *_args, **_kwargs: {"assets": [], "upload_url": "unused"},
    )

    with pytest.raises(RpsReleaseStorageError, match="confirm-bootstrap"):
        publish_rps_release(
            repository="owner/repository",
            root=root,
            universe_path=universe_path,
            bootstrap=True,
            confirm_bootstrap=False,
            dry_run=False,
            client=GitHubClient(token="test-token"),
            environ={},
        )


def test_bootstrap_uploads_year_partition_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, universe_path = _build_local_dataset(tmp_path)
    release = {"assets": [], "upload_url": "unused"}
    uploaded: list[str] = []
    monkeypatch.setattr(
        release_module,
        "get_release_metadata",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        release_module,
        "upload_release_asset",
        lambda *_args, **kwargs: uploaded.append(str(kwargs["asset_name"])),
    )
    monkeypatch.setattr(
        release_module,
        "_download_remote_manifest",
        lambda *_args, **_kwargs: load_rps_manifest(root),
    )

    result = publish_rps_release(
        repository="owner/repository",
        root=root,
        universe_path=universe_path,
        bootstrap=True,
        confirm_bootstrap=True,
        client=GitHubClient(token="test-token"),
        environ={},
    )

    assert uploaded == ["rps-year-2026.parquet", RPS_RELEASE_MANIFEST_ASSET_NAME]
    assert result["uploaded_assets"] == uploaded
    assert result["manifest_uploaded_last"] is True


@pytest.mark.parametrize(
    ("operation", "token_key", "injected"),
    [
        pytest.param(pull_rps_release, None, False, id="anonymous-pull"),
        pytest.param(check_rps_release, None, False, id="anonymous-check"),
        pytest.param(pull_rps_release, "GITHUB_TOKEN", False, id="env-token-pull"),
        pytest.param(check_rps_release, "GITHUB_TOKEN", True, id="injected-client"),
    ],
)
def test_read_authentication_uses_real_client_with_fake_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation,
    token_key: str | None,
    injected: bool,
) -> None:
    source, universe = _build_local_dataset(tmp_path)
    manifest = load_rps_manifest(source)
    base = "https://api.github.com/repos/owner/repository/releases"
    payloads = {f"{base}/assets/1": (source / "manifest.json").read_bytes()}
    assets = [{"name": RPS_RELEASE_MANIFEST_ASSET_NAME, "url": f"{base}/assets/1"}]
    for index, asset in enumerate(manifest["assets"].values(), start=2):
        url = f"{base}/assets/{index}"
        payloads[url] = (source / asset["local_path"]).read_bytes()
        assets.append({"name": asset["asset_name"], "url": url})
    payloads[f"{base}/tags/rpsData"] = json.dumps(
        {"assets": assets, "upload_url": "unused"}
    ).encode()
    requests: list[Request] = []

    def open_read(request: Request, timeout: float) -> io.BytesIO:
        assert request.get_method() == "GET"
        requests.append(request)
        return io.BytesIO(payloads[request.full_url])

    token = "test-read-token" if token_key else None
    github = GitHubClient(token=token, open_func=open_read)
    factory = Mock(return_value=github)
    resolver = Mock(wraps=release_module.resolve_github_token)
    monkeypatch.setattr(release_module, "GitHubClient", factory)
    monkeypatch.setattr(release_module, "resolve_github_token", resolver)
    destination = tmp_path / "pulled"
    kwargs = {"root": destination} if operation is pull_rps_release else {}
    result = operation(
        repository="owner/repository",
        universe_path=universe,
        client=github if injected else None,
        environ={} if injected or token_key is None else {token_key: token},
        **kwargs,
    )
    if injected:
        factory.assert_not_called()
        resolver.assert_not_called()
    else:
        factory.assert_called_once_with(token=token)
    assert result["success"] is True
    assert len(requests) == (3 if operation is pull_rps_release else 2)
    assert all(
        request.get_header("Authorization") == (f"Bearer {token}" if token else None)
        for request in requests
    )
    if operation is pull_rps_release:
        assert load_rps_manifest(destination) == manifest
        for asset in manifest["assets"].values():
            assert (destination / asset["local_path"]).read_bytes() == (
                source / asset["local_path"]
            ).read_bytes()


def test_anonymous_read_preserves_github_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(request: Request, timeout: float) -> io.BytesIO:
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    factory = Mock(return_value=GitHubClient(open_func=denied))
    monkeypatch.setattr(release_module, "GitHubClient", factory)
    with pytest.raises(ReleaseStorageError, match="HTTP 404.*private repository"):
        pull_rps_release(
            repository="owner/private", environ={}, root=tmp_path / "pulled"
        )
    factory.assert_called_once_with(token=None)
    assert not (tmp_path / "pulled").exists()


@pytest.mark.parametrize(
    ("bootstrap", "dry_run", "injected"),
    [
        pytest.param(False, False, False, id="publish"),
        pytest.param(True, False, False, id="bootstrap"),
        pytest.param(False, False, True, id="injected-anonymous-client"),
        pytest.param(True, True, False, id="bootstrap-dry-run"),
    ],
)
def test_publish_and_bootstrap_require_token_before_any_remote_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: bool,
    dry_run: bool,
    injected: bool,
) -> None:
    root, universe = _build_local_dataset(tmp_path)
    remote = Mock(side_effect=AssertionError("must reject before remote access"))
    upload = Mock(side_effect=AssertionError("must not upload without token"))
    monkeypatch.setattr(release_module, "get_release_metadata", remote)
    monkeypatch.setattr(release_module, "upload_release_asset", upload)
    with pytest.raises(
        RpsReleaseStorageError, match="GitHub token is required for publishing"
    ):
        publish_rps_release(
            repository="owner/repository",
            root=root,
            universe_path=universe,
            bootstrap=bootstrap,
            confirm_bootstrap=True,
            dry_run=dry_run,
            client=GitHubClient() if injected else None,
            environ={},
        )
    remote.assert_not_called()
    upload.assert_not_called()


def test_authenticated_publish_keeps_existing_upload_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, universe = _build_local_dataset(tmp_path)
    local = load_rps_manifest(root)
    release = {
        "assets": [{"name": RPS_RELEASE_MANIFEST_ASSET_NAME}],
        "upload_url": "unused",
    }
    remote = json.loads(json.dumps(local))
    remote["assets"]["2026"]["sha256"] = "0" * 64
    github = GitHubClient(token="test-write-token")
    factory = Mock(return_value=github)
    metadata = Mock(return_value=release)
    upload = Mock()
    monkeypatch.setattr(release_module, "GitHubClient", factory)
    monkeypatch.setattr(release_module, "get_release_metadata", metadata)
    monkeypatch.setattr(release_module, "upload_release_asset", upload)
    monkeypatch.setattr(
        release_module, "_download_remote_manifest", Mock(side_effect=[remote, local])
    )
    result = publish_rps_release(
        repository="owner/repository",
        root=root,
        universe_path=universe,
        environ={"GITHUB_TOKEN": "test-write-token"},
    )
    factory.assert_called_once_with(token="test-write-token")
    assert all(call.args[0] is github for call in metadata.call_args_list)
    expected = ["rps-year-2026.parquet", RPS_RELEASE_MANIFEST_ASSET_NAME]
    assert [call.kwargs["asset_name"] for call in upload.call_args_list] == expected
    assert all(call.args[0] is github for call in upload.call_args_list)
    assert result["uploaded_assets"] == expected
    assert result["success"] is True


def test_legacy_pull_and_migration_publish_require_explicit_preserved_source(
    tmp_path, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from momentum_screener.rps_storage import (
        LEGACY_RPS_LOOKBACKS,
        RpsStorageError,
        _manifest_for_partitions,
        rps_columns,
        rps_manifest_fingerprint,
        rps_schema,
    )
    from momentum_screener.storage_manifest import write_json_atomically

    root, universe = _build_local_dataset(tmp_path)
    local = load_rps_manifest(root)
    asset = local["assets"]["2026"]
    legacy_path = tmp_path / "legacy.parquet"
    rows = pq.read_table(root / asset["local_path"]).to_pandas()
    old = rows.loc[:, rps_columns(LEGACY_RPS_LOOKBACKS)]
    pq.write_table(
        pa.Table.from_pandas(
            old, schema=rps_schema(LEGACY_RPS_LOOKBACKS), preserve_index=False
        ),
        legacy_path,
    )
    remote = _manifest_for_partitions(
        universe=("AAA", "BBB"), partitions={2026: (legacy_path, len(old))}
    )
    remote.update(schema_version="rps_v1", lookbacks=list(LEGACY_RPS_LOOKBACKS))
    payloads = {
        RPS_RELEASE_MANIFEST_ASSET_NAME: json.dumps(remote).encode(),
        remote["assets"]["2026"]["asset_name"]: legacy_path.read_bytes(),
    }
    release = {
        "assets": [
            {"name": name, "size": len(payload)} for name, payload in payloads.items()
        ],
        "upload_url": "unused",
    }
    monkeypatch.setattr(release_module, "get_release_metadata", lambda *_: release)

    def download(_client, metadata, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[metadata["name"]])

    monkeypatch.setattr(release_module, "download_release_asset", download)
    common = {
        "repository": "owner/repo",
        "universe_path": universe,
        "client": GitHubClient(token="test-token"),
    }
    with pytest.raises(RpsStorageError, match="migration required"):
        check_rps_release(**common)
    assert (
        release_module.main(
            [
                "check",
                "--repository",
                "owner/repo",
                "--universe",
                str(universe),
            ]
        )
        == 1
    )
    imported = tmp_path / "imported-v1"
    pull_rps_release(root=imported, allow_legacy=True, **common)
    assert load_rps_manifest(imported, allow_legacy=True)["lookbacks"] == [50, 120, 250]
    with pytest.raises(
        release_module.RpsReleaseStorageError, match="exact preserved remote history"
    ):
        publish_rps_release(root=root, allow_migration=True, dry_run=True, **common)
    # Publication is gated on source identity recorded by the local migration.
    local["migration"] = {"source_manifest_sha256": rps_manifest_fingerprint(remote)}
    write_json_atomically(root / "manifest.json", local)
    plan = publish_rps_release(root=root, allow_migration=True, dry_run=True, **common)
    assert plan["planned_assets"][-1] == RPS_RELEASE_MANIFEST_ASSET_NAME
