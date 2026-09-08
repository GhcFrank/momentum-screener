"""Synchronize the independent RPS dataset through an ``rpsData`` Release."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from momentum_screener.prices import DEFAULT_UNIVERSE, load_universe, universe_sha256
from momentum_screener.release_storage import (
    GitHubClient,
    ReleaseStorageError,
    download_release_asset,
    get_release_metadata,
    resolve_github_token,
    resolve_repository,
    upload_release_asset,
)
from momentum_screener.rps_storage import (
    DEFAULT_RPS_ROOT,
    RPS_MANIFEST_NAME,
    RpsStorageError,
    rps_manifest_fingerprint,
    validate_rps_dataset,
    validate_rps_manifest,
    validate_rps_partition,
)
from momentum_screener.storage_manifest import (
    calculate_sha256,
    remove_owned_tree,
    replace_files_transactionally,
    write_json_atomically,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RPS_RELEASE_TAG: Final[str] = "rpsData"
RPS_RELEASE_MANIFEST_ASSET_NAME: Final[str] = "rps-manifest.json"


class RpsReleaseStorageError(RuntimeError):
    """Raised when RPS Release synchronization cannot finish safely."""


def _asset_index(release: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise RpsReleaseStorageError("RPS Release metadata has no asset list")
    for value in assets:
        if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
            raise RpsReleaseStorageError(
                "RPS Release contains malformed asset metadata"
            )
        asset = dict(value)
        name = str(asset["name"])
        if name in result:
            raise RpsReleaseStorageError(f"RPS Release has duplicate asset {name}")
        result[name] = asset
    return result


def _github_client(
    client: GitHubClient | None,
    *,
    environ: Mapping[str, str] | None,
) -> GitHubClient:
    if client is not None:
        return client
    return GitHubClient(token=resolve_github_token(environ))


def _require_identity(
    manifest: Mapping[str, Any],
    *,
    universe_path: Path,
) -> None:
    universe = load_universe(universe_path)
    if manifest.get("universe_sha256") != universe_sha256(universe):
        raise RpsReleaseStorageError("Remote RPS Universe hash does not match")
    if manifest.get("universe_ticker_count") != len(universe):
        raise RpsReleaseStorageError("Remote RPS Universe ticker count does not match")


def _download_remote_manifest(
    client: GitHubClient,
    assets: Mapping[str, Mapping[str, Any]],
    destination: Path,
    *,
    universe_path: Path,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    metadata = assets.get(RPS_RELEASE_MANIFEST_ASSET_NAME)
    if metadata is None:
        raise RpsReleaseStorageError(
            f"RPS Release is missing {RPS_RELEASE_MANIFEST_ASSET_NAME}"
        )
    download_release_asset(client, metadata, destination)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RpsReleaseStorageError("Remote RPS manifest is unreadable") from exc
    manifest = validate_rps_manifest(payload, allow_legacy=allow_legacy)
    _require_identity(manifest, universe_path=universe_path)
    return manifest


def check_rps_release(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RPS_RELEASE_TAG,
    universe_path: Path = DEFAULT_UNIVERSE,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read and validate the remote RPS manifest without changing local data."""

    resolved_repository = resolve_repository(repository, environ=environ)
    github = _github_client(client, environ=environ)
    release = get_release_metadata(github, resolved_repository, release_tag)
    assets = _asset_index(release)
    with tempfile.TemporaryDirectory(prefix="rps-release-check-") as temp:
        manifest = _download_remote_manifest(
            github,
            assets,
            Path(temp) / RPS_MANIFEST_NAME,
            universe_path=universe_path,
        )
    missing = sorted(
        str(asset["asset_name"])
        for asset in manifest["assets"].values()
        if str(asset["asset_name"]) not in assets
    )
    if missing:
        raise RpsReleaseStorageError(f"RPS Release is missing assets: {missing}")
    return {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "schema_version": manifest["schema_version"],
        "latest_session": manifest["latest_session"],
        "total_row_count": manifest["total_row_count"],
        "remote_asset_count": len(assets),
        "success": True,
    }


def pull_rps_release(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RPS_RELEASE_TAG,
    root: Path = DEFAULT_RPS_ROOT,
    allow_legacy: bool = False,
    universe_path: Path = DEFAULT_UNIVERSE,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Download and transactionally install the complete RPS dataset."""

    resolved_repository = resolve_repository(repository, environ=environ)
    github = _github_client(client, environ=environ)
    release = get_release_metadata(github, resolved_repository, release_tag)
    assets = _asset_index(release)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rps-release-pull-", dir=root.parent
    ) as temp:
        staging = Path(temp)
        manifest = _download_remote_manifest(
            github,
            assets,
            staging / RPS_MANIFEST_NAME,
            universe_path=universe_path,
            allow_legacy=allow_legacy,
        )
        relative_paths: list[str] = []
        for year, asset in sorted(manifest["assets"].items()):
            asset_name = str(asset["asset_name"])
            metadata = assets.get(asset_name)
            if metadata is None:
                raise RpsReleaseStorageError(f"RPS Release is missing {asset_name}")
            relative_path = str(asset["local_path"])
            destination = staging / relative_path
            download_release_asset(github, metadata, destination)
            if destination.stat().st_size != asset["size_bytes"]:
                raise RpsReleaseStorageError(
                    f"Downloaded RPS size mismatch: {asset_name}"
                )
            if calculate_sha256(destination) != asset["sha256"]:
                raise RpsReleaseStorageError(
                    f"Downloaded RPS hash mismatch: {asset_name}"
                )
            validate_rps_partition(
                destination, expected_year=int(year), lookbacks=manifest["lookbacks"]
            )
            relative_paths.append(relative_path)
        relative_paths.append(RPS_MANIFEST_NAME)
        backup = root.parent / f".rps-release-backup-{uuid.uuid4().hex}"

        def validate_installation() -> None:
            validate_rps_dataset(
                root, universe_path=universe_path, allow_legacy=allow_legacy
            )

        replace_files_transactionally(
            root,
            staging,
            relative_paths,
            backup_root=backup,
            validate_after=validate_installation,
        )
        remove_owned_tree(
            backup,
            parent=root.parent,
            prefix=".rps-release-backup-",
        )
    return {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "latest_session": manifest["latest_session"],
        "downloaded_partition_count": len(manifest["assets"]),
        "total_row_count": manifest["total_row_count"],
        "success": True,
    }


def publish_rps_release(
    *,
    repository: str | None = None,
    release_tag: str = DEFAULT_RPS_RELEASE_TAG,
    root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    bootstrap: bool = False,
    allow_migration: bool = False,
    confirm_bootstrap: bool = False,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Publish partitions then the manifest, requiring a token on the client."""

    local = validate_rps_dataset(root, universe_path=universe_path)
    resolved_repository = resolve_repository(repository, environ=environ)
    github = _github_client(client, environ=environ)
    if not github.token or not github.token.strip():
        raise RpsReleaseStorageError(
            "GitHub token is required for publishing RPS Release data; "
            "set GITHUB_TOKEN or GH_TOKEN, or inject an authenticated client"
        )
    release = get_release_metadata(github, resolved_repository, release_tag)
    assets = _asset_index(release)
    remote: dict[str, Any] | None = None
    if RPS_RELEASE_MANIFEST_ASSET_NAME in assets:
        with tempfile.TemporaryDirectory(prefix="rps-release-publish-") as temp:
            remote = _download_remote_manifest(
                github,
                assets,
                Path(temp) / RPS_MANIFEST_NAME,
                universe_path=universe_path,
                allow_legacy=allow_migration,
            )
    elif not bootstrap:
        raise RpsReleaseStorageError(
            "RPS Release has no manifest; run the explicit bootstrap command first"
        )
    elif not confirm_bootstrap and not dry_run:
        raise RpsReleaseStorageError("RPS bootstrap requires --confirm-bootstrap")

    if remote is not None and remote["schema_version"] != local["schema_version"]:
        provenance = local.get("migration", {})
        if (
            provenance.get("source_manifest_sha256") != rps_manifest_fingerprint(remote)
            or local["partition_row_counts"] != remote["partition_row_counts"]
            or local["actual_min_date"] != remote["actual_min_date"]
            or local["latest_session"] != remote["latest_session"]
        ):
            raise RpsReleaseStorageError(
                "Migration publication requires the exact preserved remote history; "
                "pull --allow-legacy and run rps_storage migrate before publishing"
            )

    changed_years = [
        year
        for year, asset in sorted(local["assets"].items())
        if remote is None
        or year not in remote["assets"]
        or remote["assets"][year]["sha256"] != asset["sha256"]
    ]
    planned_assets = [
        *(str(local["assets"][year]["asset_name"]) for year in changed_years),
        RPS_RELEASE_MANIFEST_ASSET_NAME,
    ]
    if dry_run:
        return {
            "repository": resolved_repository,
            "release_tag": release_tag,
            "latest_session": local["latest_session"],
            "planned_assets": planned_assets,
            "bootstrap": bootstrap,
            "dry_run": True,
            "success": True,
        }

    uploaded: list[str] = []
    for year in changed_years:
        asset = local["assets"][year]
        path = root / str(asset["local_path"])
        upload_release_asset(
            github,
            repository=resolved_repository,
            release=release,
            asset_name=str(asset["asset_name"]),
            path=path,
        )
        uploaded.append(str(asset["asset_name"]))
        release = get_release_metadata(github, resolved_repository, release_tag)
    upload_release_asset(
        github,
        repository=resolved_repository,
        release=release,
        asset_name=RPS_RELEASE_MANIFEST_ASSET_NAME,
        path=root / RPS_MANIFEST_NAME,
    )
    uploaded.append(RPS_RELEASE_MANIFEST_ASSET_NAME)
    release = get_release_metadata(github, resolved_repository, release_tag)
    with tempfile.TemporaryDirectory(prefix="rps-release-verify-") as temp:
        verified = _download_remote_manifest(
            github,
            _asset_index(release),
            Path(temp) / RPS_MANIFEST_NAME,
            universe_path=universe_path,
        )
    if verified != local:
        raise RpsReleaseStorageError("Remote RPS manifest differs after publication")
    return {
        "repository": resolved_repository,
        "release_tag": release_tag,
        "latest_session": local["latest_session"],
        "uploaded_assets": uploaded,
        "manifest_uploaded_last": True,
        "bootstrap": bootstrap,
        "success": True,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.rps_release_storage",
        description="Synchronize RPS history through the separate rpsData Release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "pull", "publish", "bootstrap"):
        command = subparsers.add_parser(name)
        command.add_argument("--repository")
        command.add_argument("--release-tag", default=DEFAULT_RPS_RELEASE_TAG)
        command.add_argument("--rps-root", type=Path, default=DEFAULT_RPS_ROOT)
        command.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
        command.add_argument("--result-json", type=Path)
        if name == "pull":
            command.add_argument(
                "--allow-legacy",
                action="store_true",
                help="download v1 only for explicit local migration",
            )
        if name == "publish":
            command.add_argument(
                "--allow-migration",
                action="store_true",
                help="publish an explicitly migrated v1 dataset",
            )
        if name in {"publish", "bootstrap"}:
            command.add_argument("--dry-run", action="store_true")
        if name == "bootstrap":
            command.add_argument("--confirm-bootstrap", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for remote RPS synchronization."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    common = {
        "repository": args.repository,
        "release_tag": args.release_tag,
        "universe_path": args.universe,
    }
    try:
        if args.command == "check":
            result = check_rps_release(**common)
        elif args.command == "pull":
            result = pull_rps_release(
                root=args.rps_root, allow_legacy=args.allow_legacy, **common
            )
        elif args.command == "publish":
            result = publish_rps_release(
                root=args.rps_root,
                allow_migration=args.allow_migration,
                dry_run=args.dry_run,
                **common,
            )
        elif args.command == "bootstrap":
            result = publish_rps_release(
                root=args.rps_root,
                bootstrap=True,
                confirm_bootstrap=args.confirm_bootstrap,
                dry_run=args.dry_run,
                **common,
            )
        else:
            parser.error(f"unsupported command: {args.command}")
        if args.result_json is not None:
            write_json_atomically(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        RpsReleaseStorageError,
        RpsStorageError,
        ReleaseStorageError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("RPS Release operation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
