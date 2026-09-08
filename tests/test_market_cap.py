"""Key contracts for observed MarketCap snapshots and first Release bootstrap."""

import json
from datetime import UTC, date, datetime
from urllib.error import HTTPError

import pytest

from momentum_screener import market_cap_storage as storage
from momentum_screener.market_cap_release_storage import sync_market_cap_release
from momentum_screener.prices import load_universe, universe_sha256
from momentum_screener.release_storage import ReleaseStorageError
from momentum_screener.universe import fetch_market_caps


def test_yahoo_caps_reuse_retry_parsing_and_fixed_membership():
    calls = []
    pages = [
        {"symbol": " brk.b ", "marketCap": "200.0"},
        {"symbol": "OUTSIDE", "marketCap": 99999},
        {"symbol": "NVDA", "lastclosemarketcap": {"lasttwelvemonths": 300}},
        {"symbol": "MISSING", "marketCap": 0},
    ]

    def screen(query, **kwargs):
        calls.append(kwargs["offset"])
        if len(calls) == 1:
            raise TimeoutError("temporary")
        offset = kwargs["offset"]
        return {"quotes": pages[offset : offset + kwargs["size"]]}

    result = fetch_market_caps(
        ("BRK-B", "NVDA", "MISSING"),
        screen_func=screen,
        sleep_func=lambda _: None,
        page_size=2,
        max_candidates=6,
    )
    assert result == {"BRK-B": 200, "NVDA": 300}
    assert calls == [0, 0, 2, 4]


@pytest.fixture
def cap_dataset(tmp_path, monkeypatch):
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker,market_cap\nAAA,123\nBBB,456\n")
    price_manifest = {
        "latest_session": "2026-09-08",
        "universe_ticker_count": 2,
        "universe_sha256": universe_sha256(load_universe(universe)),
    }
    monkeypatch.setattr(storage, "load_manifest", lambda _: price_manifest)
    root = tmp_path / "market_cap"
    return root, universe, price_manifest


def test_snapshot_upsert_missing_and_history_are_explicit(cap_dataset):
    root, universe, manifest = cap_dataset
    original_universe = universe.read_bytes()
    now = datetime(2026, 9, 8, 22, tzinfo=UTC)
    refresh = lambda caps: storage.refresh_market_cap_snapshot(
        root=root, universe_path=universe, now=now, fetch_func=lambda _: caps
    )
    result = refresh({"AAA": 1000, "BBB": 2000, "OUTSIDE": 999})
    assert (
        result["snapshot_date"],
        result["requested_ticker_count"],
        result["stored_ticker_count"],
        result["missing_ticker_count"],
    ) == ("2026-09-08", 2, 2, 0)
    refresh({"AAA": 1000, "BBB": 2000})
    assert len(storage.read_market_cap(root=root)) == 2
    result = refresh({"AAA": 1100, "BBB": 0})
    assert result["missing_ticker_count"] == 1
    assert storage.get_market_cap("aaa", date(2026, 9, 8), root=root) == 1100
    assert storage.get_market_cap("BBB", date(2026, 9, 8), root=root) is None
    assert "BBB" in (root / storage.MISSING_TICKERS_NAME).read_text()
    manifest["latest_session"] = "2026-09-09"
    result = storage.refresh_market_cap_snapshot(
        root=root,
        universe_path=universe,
        now=datetime(2026, 9, 9, 22, tzinfo=UTC),
        fetch_func=lambda _: {"AAA": False, "BBB": -1},
    )
    assert result["stored_ticker_count"] == 0 and result["missing_ticker_count"] == 2
    assert storage.get_market_cap("AAA", date(2026, 9, 8), root=root) == 1100
    assert storage.get_market_cap("AAA", date(2026, 9, 9), root=root) is None
    assert universe.read_bytes() == original_universe
    rows = storage.read_market_cap(root=root)
    assert not rows.duplicated(["date", "ticker"]).any()
    assert (rows["market_cap"] > 0).all()
    empty_root = root.parent / "first-empty-snapshot"
    result = storage.refresh_market_cap_snapshot(
        root=empty_root,
        universe_path=universe,
        now=datetime(2026, 9, 9, 22, tzinfo=UTC),
        fetch_func=lambda _: {},
    )
    assert result["stored_ticker_count"] == 0
    assert storage.read_market_cap(root=empty_root).empty
    with pytest.raises(storage.MarketCapStorageError, match="stale or unsettled"):
        refresh({"AAA": 999})


class MemoryGitHub:
    token = "test-token"
    api_base = "https://api.test"

    def __init__(self):
        self.release = None
        self.contents = {}
        self.writes = []
        self.failure = None

    def request_json(self, method, url, **kwargs):
        if self.failure or ("/tags/" in url and self.release is None):
            code = self.failure or 404
            try:
                raise HTTPError(url, code, "test failure", {}, None)
            except HTTPError as exc:
                raise ReleaseStorageError(f"HTTP {code}") from exc
        if method == "POST":
            self.writes.append("create")
            self.release = {"assets": [], "upload_url": "https://upload.test"}
        if "/tags/" in url or method == "POST":
            return self.release
        return {"id": 1}  # Confirm repository access independently of tag absence.

    def download_to(self, asset, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.contents[asset["name"]]
        destination.write_bytes(payload)
        return len(payload)

    def upload_file(self, upload_url, *, asset_name, path):
        payload = path.read_bytes()
        self.contents[asset_name] = payload
        asset = {
            "id": len(self.writes),
            "name": asset_name,
            "size": len(payload),
            "url": "unused",
        }
        self.release["assets"].append(asset)
        self.writes.append(asset_name)
        return asset

    def request_empty(self, method, url):
        asset_id = int(url.rsplit("/", 1)[-1])
        self.release["assets"] = [
            item for item in self.release["assets"] if item["id"] != asset_id
        ]


def test_release_absence_bootstrap_manifest_last_and_history_restore(
    cap_dataset, tmp_path
):
    root, universe, price_manifest = cap_dataset
    github = MemoryGitHub()
    common = {
        "repository": "owner/repo",
        "root": root,
        "universe_path": universe,
        "client": github,
    }
    assert sync_market_cap_release("check", **common)["bootstrap_required"]
    assert (
        sync_market_cap_release("pull", allow_bootstrap=True, **common)[
            "downloaded_partition_count"
        ]
        == 0
    )
    assert not root.exists() and github.writes == []
    storage.refresh_market_cap_snapshot(
        root=root,
        universe_path=universe,
        now=datetime(2026, 9, 8, 22, tzinfo=UTC),
        fetch_func=lambda _: {"AAA": 1000},
    )
    plan = sync_market_cap_release(
        "publish", allow_bootstrap=True, dry_run=True, **common
    )
    assert plan["create_release"] and github.writes == []
    result = sync_market_cap_release("publish", allow_bootstrap=True, **common)
    assert result["publish_success"]
    assert github.writes == [
        "create",
        "market-cap-year-2026.parquet",
        "market-cap-manifest.json",
    ]
    restored = tmp_path / "restored"
    sync_market_cap_release("pull", **{**common, "root": restored})
    assert storage.get_market_cap("AAA", date(2026, 9, 8), root=restored) == 1000
    assert (
        json.loads((restored / "manifest.json").read_text())["snapshots"]["2026-09-08"][
            "missing_ticker_count"
        ]
        == 1
    )
    # Simulate the next ephemeral runner: restore, add a day, publish, restore.
    price_manifest["latest_session"] = "2026-09-09"
    storage.refresh_market_cap_snapshot(
        root=restored,
        universe_path=universe,
        now=datetime(2026, 9, 9, 22, tzinfo=UTC),
        fetch_func=lambda _: {"BBB": 2000},
    )
    sync_market_cap_release("publish", **{**common, "root": restored})
    next_root = tmp_path / "next-runner"
    sync_market_cap_release("pull", **{**common, "root": next_root})
    assert storage.get_market_cap("AAA", date(2026, 9, 8), root=next_root) == 1000
    assert storage.get_market_cap("BBB", date(2026, 9, 9), root=next_root) == 2000
    github.release["assets"] = [
        item
        for item in github.release["assets"]
        if item["name"] != "market-cap-manifest.json"
    ]
    with pytest.raises(
        storage.MarketCapStorageError, match="partitions but no manifest"
    ):
        sync_market_cap_release("pull", allow_bootstrap=True, **common)
    github.failure = 403
    with pytest.raises(ReleaseStorageError, match="403"):
        sync_market_cap_release("check", **common)
