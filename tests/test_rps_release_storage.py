from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.rps_release_storage as release_module
from momentum_screener.release_storage import GitHubClient
from momentum_screener.rps_release_storage import (
    DEFAULT_RPS_RELEASE_TAG,
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
            "rps50": [0.0, 100.0],
            "rps120": [100.0, 0.0],
            "rps250": [0.0, 100.0],
            "return_50": [0.1, 0.2],
            "return_120": [0.3, 0.4],
            "return_250": [np.nan, 0.5],
            "rps50_base_date": date(2026, 6, 19),
            "rps120_base_date": date(2026, 3, 10),
            "rps250_base_date": date(2025, 9, 2),
        }
    )
    persist_rps_snapshot(snapshot, root=root, universe_path=universe_path)
    return root, universe_path


def test_all_rps_release_operations_use_independent_case_sensitive_tag() -> None:
    assert DEFAULT_RPS_RELEASE_TAG == "rpsData"
    for operation in (check_rps_release, pull_rps_release, publish_rps_release):
        assert (
            inspect.signature(operation).parameters["release_tag"].default
            == "rpsData"
        )


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
        client=cast(GitHubClient, object()),
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
            client=cast(GitHubClient, object()),
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
        client=cast(GitHubClient, object()),
    )

    assert uploaded == ["rps-year-2026.parquet", RPS_RELEASE_MANIFEST_ASSET_NAME]
    assert result["uploaded_assets"] == uploaded
    assert result["manifest_uploaded_last"] is True
