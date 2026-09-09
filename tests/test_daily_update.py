import json
from datetime import datetime

import pytest

from momentum_screener import daily_update as daily
from momentum_screener.release_storage import ReleaseStorageError
from momentum_screener.rps_notification import RenderedRpsEmail


class MemoryRelease:
    """Exercise real marker serialization/API ordering without a remote Release."""

    api_base = "https://example.test"

    def __init__(self):
        self.assets = {}
        self.bodies = {}
        self.fail_completion = False
        self.events = []

    def request_json(self, method, url, *, payload=None, **kwargs):
        if method == "GET":
            return {
                "assets": list(self.assets.values()),
                "upload_url": "https://example.test/upload",
            }
        assert method == "PATCH"
        self.events.append("complete")
        if self.fail_completion:
            raise ReleaseStorageError("completion acknowledgement unavailable")
        asset = next(item for item in self.assets.values() if item["url"] == url)
        asset.update(json.loads(payload))
        return asset

    def upload_file(self, url, *, asset_name, path):
        assert (
            asset_name not in self.assets
        )  # Never delete/clobber the send reservation.
        self.events.append("reserve")
        self.bodies[asset_name] = path.read_bytes()
        asset = {
            "id": len(self.assets) + 1,
            "name": asset_name,
            "label": None,
            "url": "https://example.test/asset/" + asset_name,
            "size": path.stat().st_size,
        }
        self.assets[asset_name] = asset
        return asset

    def download_to(self, asset, destination):
        destination.write_bytes(self.bodies[asset["name"]])


def markers(client):
    return daily.SessionMarkers(client, "owner/repo", "marketData")


def attempt():
    return daily.resolve_attempt(
        event_name="schedule",
        schedule=daily.FINAL_SCHEDULE,
        now=datetime.fromisoformat("2026-09-09T05:30:00-04:00"),
    )


def smtp_environment():
    return {
        "RPS_SMTP_HOST": "smtp.example.test",
        "RPS_SMTP_USERNAME": "sender@example.test",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.test",
        "RPS_EMAIL_TO": "recipient@example.test",
    }


def prepared_and_evidence(session):
    return (
        {
            "result": {"as_of_date": session, "dry_run": False},
            "email": {
                "subject": "screening",
                "text_body": "signals",
                "html_body": "<p>signals</p>",
            },
        },
        {
            "prices": {"workflow_ready": True, "remote_latest_session": session},
            "rps": {"success": True, "latest_session": session},
            "market_cap": {"publish_success": True, "latest_session": session},
        },
    )


def test_attempt_resolution_uses_scheduled_session_across_overnight_holidays_and_dst():
    cases = [
        ("30 21 * * 1-5", "2026-09-08T21:30:00-04:00", "2026-09-08"),
        ("30 0 * * 2-6", "2026-09-09T00:30:00-04:00", "2026-09-08"),
        (
            daily.FINAL_SCHEDULE,
            "2026-09-09T20:00:00-04:00",
            "2026-09-08",
        ),  # Delayed final.
        (daily.FINAL_SCHEDULE, "2026-09-05T05:30:00-04:00", "2026-09-04"),  # Saturday.
        ("30 18 * * 1-5", "2026-09-07T18:30:00-04:00", "2026-09-04"),  # Labor Day.
        (daily.FINAL_SCHEDULE, "2026-09-08T05:30:00-04:00", "2026-09-04"),
        ("30 21 * * 1-5", "2026-01-05T21:30:00-05:00", "2026-01-05"),
        ("30 18 * * 1-5", "2026-11-27T18:30:00-05:00", "2026-11-27"),  # Early close.
    ]
    for schedule, timestamp, session in cases:
        result = daily.resolve_attempt(
            event_name="schedule",
            schedule=schedule,
            now=datetime.fromisoformat(timestamp),
        )
        assert result["target_session"] == session
        assert daily.should_notify_failure(result, complete=False) is (
            schedule == daily.FINAL_SCHEDULE
        )
        assert not daily.should_notify_failure(result, complete=True)
    manual = daily.resolve_attempt(
        event_name="workflow_dispatch",
        schedule=daily.FINAL_SCHEDULE,
        now=datetime.fromisoformat("2026-09-09T05:30:00-04:00"),
    )
    assert manual["target_session"] == "2026-09-08"
    assert not daily.should_notify_failure(manual, complete=False)


def test_completion_skips_work_and_unconfirmed_send_never_resends(monkeypatch):
    client = MemoryRelease()
    store = markers(client)
    invocation = attempt()
    prepared, evidence = prepared_and_evidence(invocation["target_session"])
    monkeypatch.setattr(
        daily, "send_rps_email", lambda *args: client.events.append("send")
    )
    assert daily.preflight(invocation, store)["action"] == "run"
    evidence["rps"]["success"] = False
    with pytest.raises(ValueError, match="not verified"):
        daily.send_screening(
            invocation, store, prepared, evidence, environ=smtp_environment()
        )
    assert client.events == []
    evidence["rps"]["success"] = True
    daily.send_screening(
        invocation, store, prepared, evidence, environ=smtp_environment()
    )
    assert client.events == ["reserve", "send", "complete"]

    def unexpected(*args, **kwargs):
        raise AssertionError("completed sessions must not download or resend")

    monkeypatch.setattr(daily.prices, "run_update", unexpected)
    monkeypatch.setattr(daily, "send_rps_email", unexpected)
    assert daily.preflight(invocation, store)["action"] == "skip"
    assert (
        daily.send_screening(invocation, store, prepared, evidence)["action"] == "skip"
    )
    assert client.events == ["reserve", "send", "complete"]

    # SMTP succeeded, then the completion write failed: retain the reservation,
    # surface an error on retry, and never claim the entire session completed.
    uncertain = MemoryRelease()
    uncertain.fail_completion = True
    store = markers(uncertain)
    monkeypatch.setattr(
        daily, "send_rps_email", lambda *args: uncertain.events.append("send")
    )
    with pytest.raises(ReleaseStorageError, match="acknowledgement"):
        daily.send_screening(
            invocation, store, prepared, evidence, environ=smtp_environment()
        )
    with pytest.raises(daily.NotificationUncertainError):
        daily.preflight(invocation, store)
    with pytest.raises(daily.NotificationUncertainError):
        daily.send_screening(
            invocation, store, prepared, evidence, environ=smtp_environment()
        )
    assert uncertain.events.count("send") == 1
    assert store.read(invocation["target_session"])["state"] == "sending"


def test_only_final_failure_sends_once_with_bounded_coverage_diagnostics(monkeypatch):
    client = MemoryRelease()
    store = markers(client)
    invocation = attempt()
    report = {
        "status": "provider_not_settled",
        "ready": False,
        "failure_reason": "insufficient_target_coverage",
        "target_session_coverage_ratio": 1695 / 1996,
        "expected_active_ticker_count": 1996,
        "missing_ticker_count": 301,
        "unresolved_failure_count": 0,
    }
    messages: list[RenderedRpsEmail] = []
    monkeypatch.setattr(
        daily, "send_rps_email", lambda rendered, config: messages.append(rendered)
    )
    for key, value in smtp_environment().items():
        monkeypatch.setenv(key, value)
    nonfinal = {**invocation, "schedule": "30 21 * * 1-5", "final_attempt": False}
    assert daily.send_failure(nonfinal, store, report, {})["status"] == "not_final"
    assert not client.assets and not messages
    assert daily.send_failure(invocation, store, report, {})["status"] == "failure_sent"
    assert (
        daily.send_failure(invocation, store, report, {})["status"]
        == "failure_already_sent"
    )
    assert len(messages) == 1
    assert "Daily Update Failed — 2026-09-08" in messages[0].subject
    for text in (
        "final scheduled attempt",
        "84.92%",
        "97.00%",
        "1996",
        "301",
        "unresolved_failure_count: 0",
        "not advanced",
        "not generated",
    ):
        assert text in messages[0].text_body
    assert "Traceback" not in messages[0].text_body
