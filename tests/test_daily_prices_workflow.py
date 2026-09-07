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
    assert operations == [
        ("universe", "validate"),
        ("release_storage", "check"),
        ("release_storage", "pull-update-inputs"),
        ("rps_release_storage", "check"),
        ("rps_release_storage", "pull"),
        ("prices", "update"),
        ("release_storage", "incremental-acceptance"),
        ("daily_screening_notification", ""),
        ("release_storage", "publish-update"),
        ("release_storage", "check"),
        ("rps_release_storage", "publish"),
        ("rps_release_storage", "check"),
    ]


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
    assert workflow["concurrency"]["cancel-in-progress"] is False
    notification = next(
        step
        for step in steps
        if "momentum_screener.daily_screening_notification" in step.get("run", "")
    )
    assert notification.get("if", "success()") in {"success()", "${{ success() }}"}
    assert notification.get("continue-on-error", False) is False
    assert job.get("continue-on-error", False) is False


def test_daily_workflow_wires_release_and_email_credentials(workflow: dict) -> None:
    job = workflow["jobs"]["update-daily-prices"]
    notification = next(
        step
        for step in job["steps"]
        if "momentum_screener.daily_screening_notification" in step.get("run", "")
    )
    environment = {**job.get("env", {}), **notification.get("env", {})}
    assert environment["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert environment["RPS_SMTP_USERNAME"] == "${{ secrets.GMAIL_USER }}"
    assert environment["RPS_SMTP_PASSWORD"] == "${{ secrets.GMAIL_APP_PASSWORD }}"
    assert environment["RPS_EMAIL_TO"] == "${{ secrets.EMAIL_TO }}"
