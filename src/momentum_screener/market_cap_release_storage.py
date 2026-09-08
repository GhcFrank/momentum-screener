"""Release persistence for MarketCap observations using the shared GitHub client."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote

from momentum_screener.market_cap_storage import (
    DEFAULT_MARKET_CAP_ROOT,
    MARKET_CAP_MANIFEST_NAME,
    MarketCapStorageError,
    validate_market_cap_dataset,
    validate_market_cap_manifest,
)
from momentum_screener.prices import DEFAULT_UNIVERSE, load_universe, universe_sha256
from momentum_screener.release_storage import (
    GitHubClient,
    ReleaseStorageError,
    _release_asset_index,
    download_release_asset,
    get_release_metadata,
    resolve_github_token,
    resolve_repository,
    upload_release_asset,
)
from momentum_screener.storage_manifest import (
    calculate_sha256,
    remove_owned_tree,
    replace_files_transactionally,
    resolve_local_asset_path,
    write_json_atomically,
)

DEFAULT_MARKET_CAP_RELEASE_TAG = "marketCapData"
MARKET_CAP_RELEASE_MANIFEST = "market-cap-manifest.json"


def _optional_release(
    client: GitHubClient, repository: str, tag: str
) -> dict[str, Any] | None:
    try:
        return get_release_metadata(client, repository, tag)
    except ReleaseStorageError as exc:
        cause: BaseException | None = exc
        while cause is not None:
            if isinstance(cause, HTTPError) and cause.code == 404:
                # A private/inaccessible repository can also produce a 404.
                # Prove repository access before interpreting a tag 404 as absence.
                client.request_json(
                    "GET", f"{client.api_base}/repos/{quote(repository, safe='/')}"
                )
                return None
            cause = cause.__cause__
        raise


def _remote_manifest(
    client: GitHubClient,
    release: Mapping[str, Any],
    universe_path: Path,
    *,
    verify_assets: bool = True,
) -> dict[str, Any] | None:
    assets = _release_asset_index(release)
    metadata = assets.get(MARKET_CAP_RELEASE_MANIFEST)
    if metadata is None:
        if any(name.startswith("market-cap-year-") for name in assets):
            raise MarketCapStorageError(
                "MarketCap Release has partitions but no manifest; restore the "
                "committed manifest before pulling or bootstrapping history"
            )
        return None
    with tempfile.TemporaryDirectory(prefix="market-cap-manifest-") as temp:
        path = Path(temp) / MARKET_CAP_MANIFEST_NAME
        download_release_asset(client, metadata, path)
        manifest = validate_market_cap_manifest(json.loads(path.read_text()))
    universe = load_universe(universe_path)
    if manifest["universe_sha256"] != universe_sha256(universe) or manifest[
        "universe_ticker_count"
    ] != len(universe):
        raise MarketCapStorageError("Remote MarketCap Universe identity mismatch")
    for asset in manifest["assets"].values() if verify_assets else ():
        stored = assets.get(asset["asset_name"])
        if stored is None or stored.get("size") != asset["size_bytes"]:
            raise MarketCapStorageError(
                f"Remote MarketCap asset missing or wrong size: {asset['asset_name']}"
            )
    return manifest


def sync_market_cap_release(
    command: str,
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_MARKET_CAP_RELEASE_TAG,
    root: Path = DEFAULT_MARKET_CAP_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    allow_bootstrap: bool = False,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check, stage/pull, or publish changed assets then the manifest.

    Bootstrap is explicit for pull/publish. Only a confirmed absent tag (or an
    uninitialized Release) qualifies; other API, identity and data errors fail.
    """
    if command not in {"check", "pull", "publish"}:
        raise ValueError("Expected check, pull, or publish")
    local = (
        validate_market_cap_dataset(root, universe_path=universe_path)
        if command == "publish"
        else None
    )
    repository = resolve_repository(repository, environ=environ)
    github = client or GitHubClient(token=resolve_github_token(environ))
    if command == "publish" and not github.token:
        raise ReleaseStorageError("GitHub token required to publish MarketCap")
    release = _optional_release(github, repository, release_tag)
    remote = (
        _remote_manifest(
            github, release, universe_path, verify_assets=command != "publish"
        )
        if release is not None
        else None
    )
    result = {
        "repository": repository,
        "release_tag": release_tag,
        "remote_dataset_present": remote is not None,
        "bootstrap_required": remote is None,
        "success": True,
    }
    if command == "check":
        if remote:
            result.update(
                latest_session=remote["latest_session"],
                total_row_count=remote["total_row_count"],
            )
        return result
    if remote is None and not allow_bootstrap:
        raise MarketCapStorageError(
            "MarketCap Release is not initialized; use --allow-bootstrap for the first snapshot"
        )
    if command == "pull":
        if remote is None:
            if root.exists() and any(root.iterdir()):
                validate_market_cap_dataset(root, universe_path=universe_path)
            return {**result, "downloaded_partition_count": 0}
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".market-cap-pull-", dir=root.parent
        ) as temp:
            staging = Path(temp)
            assets = _release_asset_index(release)
            paths = []
            for asset in remote["assets"].values():
                destination = resolve_local_asset_path(staging, asset["local_path"])
                download_release_asset(github, assets[asset["asset_name"]], destination)
                if (
                    destination.stat().st_size != asset["size_bytes"]
                    or calculate_sha256(destination) != asset["sha256"]
                ):
                    raise MarketCapStorageError(
                        "Downloaded MarketCap asset size/hash mismatch"
                    )
                paths.append(asset["local_path"])
            write_json_atomically(staging / MARKET_CAP_MANIFEST_NAME, remote)
            validate_market_cap_dataset(staging, universe_path=universe_path)
            backup = root.parent / f".market-cap-pull-backup-{uuid.uuid4().hex}"
            replace_files_transactionally(
                root,
                staging,
                [*paths, MARKET_CAP_MANIFEST_NAME],
                backup_root=backup,
                validate_after=lambda: validate_market_cap_dataset(
                    root, universe_path=universe_path
                ),
            )
            remove_owned_tree(
                backup, parent=root.parent, prefix=".market-cap-pull-backup-"
            )
        return {
            **result,
            "downloaded_partition_count": len(remote["assets"]),
            "latest_session": remote["latest_session"],
        }

    if remote is not None:
        for session, info in remote["snapshots"].items():
            if session not in local["snapshots"] or (
                session != local["latest_session"]
                and local["snapshots"][session] != info
            ):
                raise MarketCapStorageError(
                    "Local MarketCap would lose/change prior snapshots; pull full remote history first"
                )
    changed = [
        year
        for year, asset in sorted(local["assets"].items())
        if remote is None
        or remote["assets"].get(year, {}).get("sha256") != asset["sha256"]
    ]
    planned = [
        *(local["assets"][year]["asset_name"] for year in changed),
        MARKET_CAP_RELEASE_MANIFEST,
    ]
    if dry_run:
        return {
            **result,
            "dry_run": True,
            "planned_assets": planned,
            "create_release": release is None,
        }
    if release is None:
        release = github.request_json(
            "POST",
            f"{github.api_base}/repos/{quote(repository, safe='/')}/releases",
            payload=json.dumps(
                {
                    "tag_name": release_tag,
                    "name": "Daily MarketCap snapshots",
                    "draft": False,
                    "prerelease": False,
                }
            ).encode(),
            content_type="application/json",
        )
    for year in changed:
        asset = local["assets"][year]
        upload_release_asset(
            github,
            repository=repository,
            release=release,
            asset_name=asset["asset_name"],
            path=resolve_local_asset_path(root, asset["local_path"]),
        )
        release = get_release_metadata(github, repository, release_tag)
    upload_release_asset(
        github,
        repository=repository,
        release=release,
        asset_name=MARKET_CAP_RELEASE_MANIFEST,
        path=root / MARKET_CAP_MANIFEST_NAME,
    )
    verified = _remote_manifest(
        github, get_release_metadata(github, repository, release_tag), universe_path
    )
    if verified != local:
        raise MarketCapStorageError(
            "MarketCap remote manifest differs after publication"
        )
    return {
        **result,
        "publish_success": True,
        "latest_session": local["latest_session"],
        "uploaded_assets": planned,
        "manifest_uploaded_last": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persist MarketCap history in the independent marketCapData Release."
    )
    parser.add_argument("command", choices=["check", "pull", "publish"])
    parser.add_argument("--repository")
    parser.add_argument("--release-tag", default=DEFAULT_MARKET_CAP_RELEASE_TAG)
    parser.add_argument("--root", type=Path, default=DEFAULT_MARKET_CAP_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="allow an absent dataset; publish creates the first Release",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run and args.command != "publish":
        parser.error("--dry-run applies only to publish")
    try:
        result = sync_market_cap_release(
            args.command,
            repository=args.repository,
            release_tag=args.release_tag,
            root=args.root,
            universe_path=args.universe,
            allow_bootstrap=args.allow_bootstrap,
            dry_run=args.dry_run,
        )
        if args.result_json:
            write_json_atomically(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        MarketCapStorageError,
        ReleaseStorageError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        parser.exit(1, f"MarketCap Release operation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
