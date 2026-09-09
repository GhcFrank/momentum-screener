import re
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(
        Path(".github/workflows/update-daily-prices.yml").read_text(encoding="utf-8")
    )


def test_daily_workflow_preserves_critical_data_and_publication_order(
    workflow: dict,
) -> None:
    steps = workflow["jobs"]["update-daily-prices"]["steps"]
    operations = []
    for step in steps:
        script = step.get("run", "")
        if "validate_local_incremental_update_acceptance" in script:
            operations.append(("release_storage", "incremental-acceptance"))
        for module, command in re.findall(
            r"python\s+-m\s+momentum_screener\.(\w+)(?:\s+([a-z][\w-]*))?", script
        ):
            if module == "prices" and "--dry-run" in script:
                continue
            operations.append((module, command))

    def position(module, command):
        return operations.index((module, command))

    # Protect the dependency order, not an exact snapshot of every workflow step.
    assert position("daily_update", "preflight") < position(
        "release_storage", "pull-update-inputs"
    )
    assert position("daily_update", "prices") < position(
        "release_storage", "incremental-acceptance"
    )
    assert position("market_cap_storage", "refresh") < position(
        "daily_screening_notification", ""
    )
    assert position("daily_screening_notification", "") < position(
        "release_storage", "publish-update"
    )
    notify = position("daily_update", "notify")
    assert position("release_storage", "publish-update") < notify
    assert position("market_cap_release_storage", "publish") < notify
    assert position("rps_release_storage", "publish") < notify
    assert (
        max(
            i
            for i, item in enumerate(operations)
            if item == ("rps_release_storage", "check")
        )
        < notify
    )
    triggers = workflow.get("on", workflow.get(True))
    from momentum_screener.daily_update import SCHEDULES

    assert {entry["cron"] for entry in triggers["schedule"]} == set(SCHEDULES)
    assert all(
        entry["timezone"] == "America/New_York" for entry in triggers["schedule"]
    )
    assert "workflow_dispatch" in triggers


def test_daily_workflow_uses_release_storage_and_stops_notifications_on_failure(
    workflow: dict,
) -> None:
    job = workflow["jobs"]["update-daily-prices"]
    steps = job["steps"]
    scripts = "\n".join(step.get("run", "") for step in steps)
    assert not re.search(r"\bgit\s+(?:add|commit|push)\b", scripts)
    assert "momentum_screener.rps_notification" not in scripts
    assert "--allow-partial-session" not in scripts
    assert job["env"]["RELEASE_TAG"] == "marketData"
    assert job["env"]["RPS_RELEASE_TAG"] == "rpsData"
    assert job["env"]["MARKET_CAP_RELEASE_TAG"] == "marketCapData"
    assert "rps_storage migrate" not in scripts
    assert workflow["concurrency"]["cancel-in-progress"] is False
    notification = next(
        step
        for step in steps
        if "momentum_screener.daily_update notify" in step.get("run", "")
    )
    assert "success()" in notification["if"]
    assert "steps.refresh.outputs.ready == 'true'" in notification["if"]
    start = next(
        i
        for i, step in enumerate(steps)
        if "Validate incremental update acceptance" == step["name"]
    )
    end = steps.index(notification)
    for step in steps[start : end + 1]:
        assert "steps.refresh.outputs.ready == 'true'" in step["if"]
        assert "steps.preflight.outputs.action == 'run'" in step["if"]
    final = next(
        step for step in steps if "daily_update final-failure" in step.get("run", "")
    )
    assert "github.event_name == 'schedule'" in final["if"]
    assert "github.event.schedule == '30 5 * * 2-6'" in final["if"]
    assert notification.get("continue-on-error", False) is False
    assert job.get("continue-on-error", False) is False


def test_daily_workflow_wires_release_and_email_credentials(workflow: dict) -> None:
    job = workflow["jobs"]["update-daily-prices"]
    notification = next(
        step
        for step in job["steps"]
        if "momentum_screener.daily_update notify" in step.get("run", "")
    )
    environment = {**job.get("env", {}), **notification.get("env", {})}
    assert environment["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert environment["RPS_SMTP_USERNAME"] == "${{ secrets.GMAIL_USER }}"
    assert environment["RPS_SMTP_PASSWORD"] == "${{ secrets.GMAIL_APP_PASSWORD }}"
    assert environment["RPS_EMAIL_TO"] == "${{ secrets.EMAIL_TO }}"
