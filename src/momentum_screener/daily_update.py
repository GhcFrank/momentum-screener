"""Small scheduled-attempt adapter; prices remain independent of cron and SMTP.

Release assets reserve a notification before SMTP. Their metadata label is changed
in place only after SMTP succeeds. A missing acknowledgement is deliberately an
error requiring reconciliation, never permission to resend or declare completion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from momentum_screener import prices
from momentum_screener.release_storage import (
    DEFAULT_RELEASE_TAG,
    GitHubClient,
    ReleaseStorageError,
    _release_asset_index,
    get_release_metadata,
    resolve_github_token,
    resolve_repository,
)
from momentum_screener.rps_notification import (
    RenderedRpsEmail,
    SmtpEmailConfig,
    send_rps_email,
)
from momentum_screener.storage_manifest import write_json_atomically

LOGGER = logging.getLogger(__name__)
SCHEDULES = {
    "30 18 * * 1-5": (18, range(5)),
    "30 21 * * 1-5": (21, range(5)),
    "30 0 * * 2-6": (0, range(1, 6)),
    "30 5 * * 2-6": (5, range(1, 6)),
}
FINAL_SCHEDULE = "30 5 * * 2-6"


class NotificationUncertainError(RuntimeError):
    """A durable send reservation exists without confirmed completion."""


def resolve_attempt(
    *, event_name: str, schedule: str = "", now: datetime | None = None
) -> dict[str, Any]:
    anchor = now or datetime.now(UTC)
    if anchor.tzinfo is None:
        raise ValueError("Attempt time must be timezone-aware")
    nominal = anchor
    if event_name == "schedule":
        if schedule not in SCHEDULES:
            raise ValueError(f"Unknown daily update schedule: {schedule}")
        hour, weekdays = SCHEDULES[schedule]
        local = anchor.astimezone(ZoneInfo(prices.DEFAULT_MARKET_TIMEZONE))
        # Use the triggering schedule, not the runner's current hour. This also
        # keeps a delayed overnight run aimed at the preceding trading session.
        for offset in range(8):
            day = local.date() - timedelta(days=offset)
            candidate = datetime.combine(day, time(hour, 30), tzinfo=local.tzinfo)
            if day.weekday() in weekdays and candidate <= local:
                nominal = candidate
                break
    session = prices.determine_target_session(now=nominal)
    return {
        "target_session": session.isoformat(),
        "event_name": event_name,
        "schedule": schedule,
        "attempt_time": nominal.astimezone(
            ZoneInfo(prices.DEFAULT_MARKET_TIMEZONE)
        ).isoformat(),
        "final_attempt": event_name == "schedule" and schedule == FINAL_SCHEDULE,
    }


class SessionMarkers:
    """Tiny per-session notification receipts on the existing marketData Release.

    Assets are never deleted/replaced by this adapter. GitHub rejects duplicate
    names, and a metadata PATCH makes completion visible without a deletion gap.
    The workflow's existing concurrency group serializes all attempts.
    """

    def __init__(self, client: GitHubClient, repository: str, release_tag: str):
        self.client = client
        self.repository = repository
        self.release_tag = release_tag

    @staticmethod
    def name(session: str, kind: str) -> str:
        date.fromisoformat(session)
        if kind not in {"screening", "failure"}:
            raise ValueError("Unknown notification kind")
        return f"daily-{kind}-{session}.json"

    def read(self, session: str, kind: str = "screening") -> dict[str, Any] | None:
        release = get_release_metadata(self.client, self.repository, self.release_tag)
        asset = _release_asset_index(release).get(self.name(session, kind))
        if asset is None:
            return None
        with tempfile.TemporaryDirectory(prefix="daily-marker-read-") as temp:
            path = Path(temp) / "marker.json"
            self.client.download_to(asset, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "version": 1,
            "target_session": session,
            "kind": kind,
            "repository": self.repository,
            "release_tag": self.release_tag,
        }
        if not isinstance(payload, dict) or any(
            payload.get(k) != v for k, v in expected.items()
        ):
            raise ReleaseStorageError("Daily notification marker identity is invalid")
        state = asset.get("label") or "sending"
        if state not in {"sending", "complete" if kind == "screening" else "sent"}:
            raise ReleaseStorageError("Daily notification marker state is invalid")
        if kind == "screening" and payload.get("pipeline_ready") is not True:
            raise ReleaseStorageError(
                "Screening marker has no completed pipeline evidence"
            )
        return {**payload, "state": state, "asset": asset}

    def claim(
        self, session: str, kind: str, evidence: dict[str, Any]
    ) -> dict[str, Any]:
        if self.read(session, kind) is not None:
            raise NotificationUncertainError(
                "Notification already reserved; refusing to resend"
            )
        release = get_release_metadata(self.client, self.repository, self.release_tag)
        payload = {
            **evidence,
            "version": 1,
            "target_session": session,
            "kind": kind,
            "repository": self.repository,
            "release_tag": self.release_tag,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        }
        with tempfile.TemporaryDirectory(prefix="daily-marker-write-") as temp:
            path = Path(temp) / "marker.json"
            write_json_atomically(path, payload)
            self.client.upload_file(
                release["upload_url"], asset_name=self.name(session, kind), path=path
            )
        marker = self.read(session, kind)
        if marker is None or marker["state"] != "sending":
            raise ReleaseStorageError("Cannot verify durable notification reservation")
        return marker

    def finish(self, marker: dict[str, Any]) -> None:
        state = "complete" if marker["kind"] == "screening" else "sent"
        asset = marker["asset"]
        result = self.client.request_json(
            "PATCH",
            asset["url"],
            payload=json.dumps({"label": state}).encode(),
            content_type="application/json",
        )
        if result.get("label") != state or result.get("id") != asset["id"]:
            raise ReleaseStorageError("Cannot confirm notification completion marker")


def preflight(attempt: dict[str, Any], markers: SessionMarkers) -> dict[str, Any]:
    marker = markers.read(attempt["target_session"])
    if marker is not None and marker["state"] != "complete":
        raise NotificationUncertainError(
            "Screening send was reserved but not confirmed. Reconcile the originating "
            "run/SMTP result before changing the marker; automatic resend is disabled."
        )
    return {**attempt, "action": "skip" if marker else "run"}


def run_price_attempt(target_session: date, **kwargs: Any) -> dict[str, Any]:
    """Translate price results for explicit workflow gating; never send email."""
    kwargs["minimum_coverage"] = prices.DEFAULT_MINIMUM_TARGET_COVERAGE
    kwargs["allow_partial_session"] = False
    try:
        result = prices.run_update(target_date=target_session, **kwargs)
        if result["status"] == "no_op":
            # A restored price session alone is not pipeline completion. Reuse it
            # without Yahoo, but still enforce canonical coverage before downstream.
            root = kwargs.get("prices_root", prices.DEFAULT_OUTPUT_ROOT)
            universe = prices.load_universe(
                kwargs.get("universe_path", prices.DEFAULT_UNIVERSE)
            )
            start = target_session - timedelta(days=40)
            rows = prices.read_affected_partitions(
                root,
                prices.affected_partition_years(start, target_session),
                tickers=universe,
            )
            expected = prices.expected_active_tickers(
                rows, universe=universe, target_session=target_session
            )
            if not expected:
                raise prices.PriceUpdateError(
                    "No active tickers available to verify coverage"
                )
            ratio, missing = prices.validate_target_coverage(
                rows, expected_active=expected, target_session=target_session
            )
            result.update(
                target_session_coverage_ratio=ratio,
                expected_active_ticker_count=len(expected),
                missing_ticker_count=len(missing),
                unresolved_failure_count=0,
            )
        ready = result["local_update_success"] is True and result["status"] in {
            "updated",
            "no_op",
        }
        return {
            **result,
            "ready": ready,
            "publish": ready and result["status"] == "updated",
        }
    except prices.ProviderNotSettledError as exc:
        LOGGER.info(
            "Target session %s is not yet provider-settled. Coverage: %.2f%%; required: %.2f%%. "
            "No canonical data published; downstream skipped. A later scheduled attempt may retry.",
            target_session,
            100 * exc.report["target_session_coverage_ratio"],
            100 * prices.DEFAULT_MINIMUM_TARGET_COVERAGE,
        )
        return {**exc.report, "ready": False, "publish": False}
    except Exception as exc:
        # Genuine errors retain a failing exit code and traceback in Actions logs.
        LOGGER.exception("Daily price attempt failed")
        report = getattr(exc, "report", {})
        status = report.get("status") or (
            "validation_failure"
            if isinstance(exc, (prices.PriceBackfillError, ValueError))
            else "unexpected_error"
        )
        return {
            **report,
            "status": status,
            "failure_reason": report.get("failure_reason", type(exc).__name__),
            "target_session": target_session.isoformat(),
            "ready": False,
            "publish": False,
            "local_update_success": False,
        }


def send_screening(
    attempt: dict[str, Any],
    markers: SessionMarkers,
    prepared: dict[str, Any],
    evidence: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    decision = preflight(attempt, markers)
    if decision["action"] == "skip":
        return decision
    session = attempt["target_session"]
    result = prepared["result"]
    if result["as_of_date"] != session or result["dry_run"] is not False:
        raise ValueError("Prepared screening does not match the target session")
    for key, flag, session_key in (
        ("prices", "workflow_ready", "remote_latest_session"),
        ("rps", "success", "latest_session"),
        ("market_cap", "publish_success", "latest_session"),
    ):
        check = evidence[key]
        if check.get(flag) is not True or check.get(session_key) != session:
            raise ValueError(f"{key} publication is not verified for {session}")
    rendered = RenderedRpsEmail(**prepared["email"])
    config = SmtpEmailConfig.from_environment(environ)
    marker = markers.claim(
        session,
        "screening",
        {
            "pipeline_ready": True,
            "screening_result": result,
            "publication": evidence,
        },
    )
    send_rps_email(rendered, config)
    markers.finish(marker)
    return {**result, "status": "complete", "notification_sent": True}


def should_notify_failure(attempt: dict[str, Any], *, complete: bool) -> bool:
    return (
        attempt["event_name"] == "schedule"
        and attempt["schedule"] == FINAL_SCHEDULE
        and attempt["final_attempt"] is True
        and not complete
    )


def render_failure_email(
    attempt: dict[str, Any], report: dict[str, Any], steps: dict[str, Any]
) -> RenderedRpsEmail:
    session = attempt["target_session"]
    failed_steps = [
        name for name, step in steps.items() if step.get("outcome") == "failure"
    ]
    reason = report.get("failure_reason") if report.get("ready") is False else None
    reason = reason or (
        ", ".join(failed_steps) if failed_steps else "pipeline_incomplete"
    )
    coverage = report.get("target_session_coverage_ratio")
    lines = [
        f"target_session: {session}",
        f"attempt time: {attempt['attempt_time']} (final scheduled attempt)",
        f"failure reason: {reason}",
        f"actual coverage: {coverage:.2%}"
        if coverage is not None
        else "actual coverage: N/A",
        f"required coverage: {prices.DEFAULT_MINIMUM_TARGET_COVERAGE:.2%}",
        *[
            f"{key}: {report.get(key, 'N/A')}"
            for key in (
                "expected_active_ticker_count",
                "missing_ticker_count",
                "unresolved_failure_count",
            )
        ],
    ]
    if report.get("status") == "provider_not_settled":
        lines.append(
            "Download succeeded, but complete target-session OHLC/Close remained unavailable."
        )
    if report.get("ready") is False:
        lines.extend(
            [
                "Canonical market dataset was not advanced by this attempt.",
                "RPS/signals were not generated for this target session by this attempt.",
            ]
        )
    else:
        lines.append(
            "Pipeline did not complete; earlier publication or screening may have succeeded. See Actions diagnostics."
        )
    subject = f"Momentum Screener — Daily Update Failed — {session}"
    body = "\n".join(lines) + "\n"
    return RenderedRpsEmail(
        subject=subject, text_body=body, html_body=f"<pre>{escape(body)}</pre>"
    )


def send_failure(
    attempt: dict[str, Any],
    markers: SessionMarkers,
    report: dict[str, Any],
    steps: dict[str, Any],
) -> dict[str, Any]:
    if not should_notify_failure(attempt, complete=False):
        return {"status": "not_final"}
    normal = markers.read(attempt["target_session"])
    if not should_notify_failure(
        attempt, complete=normal is not None and normal["state"] == "complete"
    ):
        return {"status": "skip_complete"}
    previous = markers.read(attempt["target_session"], "failure")
    if previous is not None:
        if previous["state"] != "sent":
            raise NotificationUncertainError(
                "Final failure email was reserved; delivery needs reconciliation"
            )
        return {"status": "failure_already_sent"}
    rendered = render_failure_email(attempt, report, steps)
    config = SmtpEmailConfig.from_environment()
    marker = markers.claim(
        attempt["target_session"],
        "failure",
        {"attempt": attempt, "diagnostics": report},
    )
    send_rps_email(rendered, config)
    markers.finish(marker)
    return {"status": "failure_sent"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(result: dict[str, Any]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with Path(path).open("a", encoding="utf-8") as output:
            for key in (
                "target_session",
                "final_attempt",
                "action",
                "ready",
                "publish",
                "local_update_success",
            ):
                if key in result:
                    value = (
                        str(result[key]).lower()
                        if isinstance(result[key], bool)
                        else str(result[key])
                    )
                    output.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "prices", "notify", "final-failure")
    )
    parser.add_argument("--repository")
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    work = args.work_dir
    attempt_path = work / "daily-attempt.json"
    result_path = work / f"daily-{args.command}-result.json"
    try:
        client = GitHubClient(token=resolve_github_token())
        repository = resolve_repository(args.repository)
        markers = SessionMarkers(client, repository, args.release_tag)
        if args.command == "preflight":
            anchor = datetime.now(UTC)
            run_id = os.environ.get("GITHUB_RUN_ID")
            if run_id:
                # created_at is stable across reruns/queued jobs, unlike wall time.
                if not run_id.isdigit():
                    raise ValueError("Invalid GitHub run ID")
                run = client.request_json(
                    "GET", f"{client.api_base}/repos/{repository}/actions/runs/{run_id}"
                )
                anchor = datetime.fromisoformat(run["created_at"])
            attempt = resolve_attempt(
                event_name=os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch"),
                schedule=os.environ.get("DAILY_SCHEDULE", ""),
                now=anchor,
            )
            write_json_atomically(attempt_path, attempt)
            result = preflight(attempt, markers)
        else:
            attempt = _read_json(attempt_path)
            if args.command == "prices":
                result = run_price_attempt(
                    date.fromisoformat(attempt["target_session"])
                )
            elif args.command == "notify":
                result = send_screening(
                    attempt,
                    markers,
                    _read_json(work / "prepared-screening.json"),
                    {
                        "prices": _read_json(work / "dataset-check-after.json"),
                        "rps": _read_json(work / "rps-check-after.json"),
                        "market_cap": _read_json(work / "market-cap-publish.json"),
                    },
                )
            else:
                report_path = work / "daily-prices-result.json"
                report = _read_json(report_path) if report_path.is_file() else {}
                result = send_failure(
                    attempt,
                    markers,
                    report,
                    json.loads(os.environ.get("DAILY_STEP_OUTCOMES", "{}")),
                )
        write_json_atomically(result_path, result)
        _outputs(result)
        LOGGER.info(
            "Daily %s: %s", args.command, result.get("status", result.get("action"))
        )
        if args.command == "prices":
            return (
                0
                if result.get("ready") or result.get("status") == "provider_not_settled"
                else 1
            )
        if args.command == "final-failure":
            return 0 if result["status"] in {"not_final", "skip_complete"} else 1
        return 0
    except Exception as exc:
        LOGGER.exception("Daily %s failed", args.command)
        write_json_atomically(
            result_path, {"status": "error", "error_type": type(exc).__name__}
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
