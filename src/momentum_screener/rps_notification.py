"""Screen an RPS snapshot and deliver the daily result by SMTP email."""

from __future__ import annotations

import argparse
import logging
import math
import os
import smtplib
import ssl
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any, Final, TextIO

import pandas as pd  # type: ignore[import-untyped]
from dotenv import load_dotenv

from momentum_screener.prices import DEFAULT_OUTPUT_ROOT, DEFAULT_UNIVERSE
from momentum_screener.rps import calculate_rps_snapshot
from momentum_screener.storage_manifest import load_manifest

LOGGER = logging.getLogger(__name__)

DEFAULT_RPS_EMAIL_THRESHOLD: Final[float] = 87.0
DEFAULT_GMAIL_SMTP_HOST: Final[str] = "smtp.gmail.com"
DEFAULT_SMTP_PORT: Final[int] = 587
DEFAULT_SMTP_TIMEOUT_SECONDS: Final[float] = 30.0


class RpsNotificationError(RuntimeError):
    """Base error for an RPS notification that cannot be delivered safely."""


class EmailConfigurationError(RpsNotificationError):
    """Raised when required SMTP email configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class RpsScreen:
    """The two independently screened and sorted RPS candidate sets."""

    threshold: float
    rps120_candidates: pd.DataFrame
    rps250_candidates: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RenderedRpsEmail:
    """Provider-independent content for one RPS email."""

    subject: str
    text_body: str
    html_body: str


@dataclass(frozen=True, slots=True)
class SmtpEmailConfig:
    """SMTP connection and message-address configuration."""

    host: str
    port: int
    username: str
    password: str = field(repr=False)
    sender: str
    recipients: tuple[str, ...]

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> SmtpEmailConfig:
        """Load canonical settings, with compatibility for local Gmail names."""

        environment = os.environ if environ is None else environ
        gmail_username = _first_configured(environment, "GMAIL_USER")
        gmail_password = _first_configured(environment, "GMAIL_APP_PASSWORD")
        username = _first_configured(
            environment,
            "RPS_SMTP_USERNAME",
            "SMTP_USERNAME",
            "GMAIL_USER",
        )
        password = _first_configured(
            environment,
            "RPS_SMTP_PASSWORD",
            "SMTP_PASSWORD",
            "GMAIL_APP_PASSWORD",
        )
        sender = _first_configured(
            environment,
            "RPS_EMAIL_FROM",
            "EMAIL_FROM",
        )
        if not sender and gmail_username:
            sender = gmail_username
        recipient_value = _first_configured(
            environment,
            "RPS_EMAIL_TO",
            "EMAIL_TO",
        )
        host = _first_configured(environment, "RPS_SMTP_HOST", "SMTP_HOST")
        if not host and (gmail_username or gmail_password):
            host = DEFAULT_GMAIL_SMTP_HOST

        required_values = (
            ("RPS_SMTP_HOST", host),
            ("RPS_SMTP_USERNAME", username),
            ("RPS_SMTP_PASSWORD", password),
            ("RPS_EMAIL_FROM", sender),
            ("RPS_EMAIL_TO", recipient_value),
        )
        missing = [name for name, value in required_values if not value]
        if missing:
            raise EmailConfigurationError(
                "Missing required email configuration: " + ", ".join(missing)
            )

        raw_port = _first_configured(environment, "RPS_SMTP_PORT", "SMTP_PORT")
        try:
            port = int(raw_port) if raw_port else DEFAULT_SMTP_PORT
        except ValueError as exc:
            raise EmailConfigurationError("RPS_SMTP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise EmailConfigurationError("RPS_SMTP_PORT must be between 1 and 65535")

        recipients = tuple(
            value.strip() for value in recipient_value.split(",") if value.strip()
        )
        if not recipients:
            raise EmailConfigurationError(
                "RPS_EMAIL_TO must contain at least one recipient"
            )
        header_values = (sender, *recipients)
        if any("\n" in value or "\r" in value for value in header_values):
            raise EmailConfigurationError("Email addresses cannot contain newlines")

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            sender=sender,
            recipients=recipients,
        )


@dataclass(frozen=True, slots=True)
class RpsNotificationResult:
    """Concise, non-secret outcome returned by the daily orchestration."""

    latest_session: date
    snapshot_ticker_count: int
    rps120_candidate_count: int
    rps250_candidate_count: int
    subject: str


def _first_configured(environment: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = environment.get(name, "").strip()
        if value:
            return value
    return ""


def _mask_recipient(value: str) -> str:
    local_part, separator, domain = value.partition("@")
    if separator and local_part and domain:
        return f"{local_part[0]}***@{domain}"
    return "<configured-recipient>"


def _validate_threshold(threshold: float) -> float:
    parsed = float(threshold)
    if not math.isfinite(parsed):
        raise ValueError("RPS threshold must be finite")
    return parsed


def build_rps_screen(
    snapshot: pd.DataFrame,
    threshold: float = DEFAULT_RPS_EMAIL_THRESHOLD,
) -> RpsScreen:
    """Apply two strict, independent RPS filters to a complete snapshot."""

    parsed_threshold = _validate_threshold(threshold)
    required_columns = {"ticker", "rps120", "rps250"}
    missing_columns = sorted(required_columns - set(snapshot.columns))
    if missing_columns:
        raise RpsNotificationError(
            "RPS snapshot is missing required columns: " + ", ".join(missing_columns)
        )

    numeric_120 = pd.to_numeric(snapshot["rps120"], errors="coerce")
    numeric_250 = pd.to_numeric(snapshot["rps250"], errors="coerce")

    rps120_candidates = snapshot.loc[numeric_120.gt(parsed_threshold)].copy()
    rps120_candidates["rps120"] = numeric_120.loc[rps120_candidates.index]
    rps120_candidates["rps250"] = numeric_250.loc[rps120_candidates.index]
    rps120_candidates = rps120_candidates.reset_index(drop=True).sort_values(
        ["rps120", "ticker"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )

    rps250_candidates = snapshot.loc[numeric_250.gt(parsed_threshold)].copy()
    rps250_candidates["rps120"] = numeric_120.loc[rps250_candidates.index]
    rps250_candidates["rps250"] = numeric_250.loc[rps250_candidates.index]
    rps250_candidates = rps250_candidates.reset_index(drop=True).sort_values(
        ["rps250", "ticker"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    return RpsScreen(
        threshold=parsed_threshold,
        rps120_candidates=rps120_candidates,
        rps250_candidates=rps250_candidates,
    )


def _format_threshold(threshold: float) -> str:
    return f"{threshold:g}"


def _format_rps(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{parsed:.2f}" if math.isfinite(parsed) else "N/A"


def _render_text_section(*, title: str, candidates: pd.DataFrame) -> list[str]:
    lines = [title, f"{len(candidates)} stocks"]
    if candidates.empty:
        lines.append("No stocks matched.")
        return lines

    lines.append(f"{'Ticker':<12} {'RPS120':>8} {'RPS250':>8}")
    for _, row in candidates.iterrows():
        lines.append(
            f"{row['ticker']!s:<12} "
            f"{_format_rps(row['rps120']):>8} "
            f"{_format_rps(row['rps250']):>8}"
        )
    return lines


def _render_html_section(*, title: str, candidates: pd.DataFrame) -> str:
    heading = escape(title)
    count = len(candidates)
    if candidates.empty:
        content = "<p>No stocks matched.</p>"
    else:
        rows = "".join(
            "<tr>"
            f"<td>{escape(str(row['ticker']))}</td>"
            f"<td>{_format_rps(row['rps120'])}</td>"
            f"<td>{_format_rps(row['rps250'])}</td>"
            "</tr>"
            for _, row in candidates.iterrows()
        )
        content = (
            '<table style="border-collapse:collapse">'
            "<thead><tr><th>Ticker</th><th>RPS120</th><th>RPS250</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return f"<section><h2>{heading}</h2><p>{count} stocks</p>{content}</section>"


def render_rps_email(
    *,
    as_of_date: date,
    rps120_candidates: pd.DataFrame,
    rps250_candidates: pd.DataFrame,
    threshold: float = DEFAULT_RPS_EMAIL_THRESHOLD,
) -> RenderedRpsEmail:
    """Render stable plain-text and HTML bodies without performing I/O."""

    parsed_threshold = _validate_threshold(threshold)
    threshold_text = _format_threshold(parsed_threshold)
    rps120_title = f"RPS120 > {threshold_text}"
    rps250_title = f"RPS250 > {threshold_text}"
    subject = f"Momentum Screener - RPS - {as_of_date.isoformat()}"

    text_lines = [
        "RPS Screen",
        f"Market session: {as_of_date.isoformat()}",
        "",
        f"{rps120_title}: {len(rps120_candidates)} stocks",
        f"{rps250_title}: {len(rps250_candidates)} stocks",
        "",
        *_render_text_section(title=rps120_title, candidates=rps120_candidates),
        "",
        *_render_text_section(title=rps250_title, candidates=rps250_candidates),
    ]
    html_body = (
        "<!doctype html><html><body>"
        "<h1>RPS Screen</h1>"
        f"<p>Market session: <strong>{as_of_date.isoformat()}</strong></p>"
        "<ul>"
        f"<li>{escape(rps120_title)}: {len(rps120_candidates)} stocks</li>"
        f"<li>{escape(rps250_title)}: {len(rps250_candidates)} stocks</li>"
        "</ul>"
        f"{_render_html_section(title=rps120_title, candidates=rps120_candidates)}"
        f"{_render_html_section(title=rps250_title, candidates=rps250_candidates)}"
        "</body></html>"
    )
    return RenderedRpsEmail(
        subject=subject,
        text_body="\n".join(text_lines) + "\n",
        html_body=html_body,
    )


def send_rps_email(
    rendered: RenderedRpsEmail,
    config: SmtpEmailConfig,
    *,
    timeout_seconds: float = DEFAULT_SMTP_TIMEOUT_SECONDS,
    smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP,
) -> None:
    """Send one multipart email over an authenticated STARTTLS connection."""

    message = EmailMessage()
    message["Subject"] = rendered.subject
    message["From"] = config.sender
    message["To"] = ", ".join(config.recipients)
    message.set_content(rendered.text_body)
    message.add_alternative(rendered.html_body, subtype="html")

    with smtp_factory(config.host, config.port, timeout=timeout_seconds) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def get_latest_dataset_session(
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
) -> date:
    """Read the validated all-market latest session from dataset metadata."""

    manifest = load_manifest(
        prices_root / "manifest.json",
        require_completed=True,
        require_assets=True,
    )
    return date.fromisoformat(str(manifest["latest_session"]))


def _coerce_as_of_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise TypeError("as_of_date must be a date or ISO YYYY-MM-DD string")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("as_of_date must use ISO YYYY-MM-DD format") from exc


def run_daily_rps_notification(
    *,
    as_of_date: date | str | None = None,
    threshold: float = DEFAULT_RPS_EMAIL_THRESHOLD,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
    preview_stream: TextIO | None = None,
) -> RpsNotificationResult:
    """Calculate the latest RPS screen and optionally send exactly one email."""

    config = None if dry_run else SmtpEmailConfig.from_environment(environ)

    latest_session = (
        get_latest_dataset_session(prices_root)
        if as_of_date is None
        else _coerce_as_of_date(as_of_date)
    )
    LOGGER.info("RPS notification latest_session=%s", latest_session.isoformat())

    snapshot = calculate_rps_snapshot(
        latest_session,
        prices_root=prices_root,
        universe_path=universe_path,
    )
    LOGGER.info("RPS snapshot ticker count=%d", len(snapshot))
    screen = build_rps_screen(snapshot, threshold=threshold)
    threshold_text = _format_threshold(screen.threshold)
    LOGGER.info(
        "RPS120 > %s count=%d",
        threshold_text,
        len(screen.rps120_candidates),
    )
    LOGGER.info(
        "RPS250 > %s count=%d",
        threshold_text,
        len(screen.rps250_candidates),
    )

    rendered = render_rps_email(
        as_of_date=latest_session,
        rps120_candidates=screen.rps120_candidates,
        rps250_candidates=screen.rps250_candidates,
        threshold=screen.threshold,
    )
    result = RpsNotificationResult(
        latest_session=latest_session,
        snapshot_ticker_count=len(snapshot),
        rps120_candidate_count=len(screen.rps120_candidates),
        rps250_candidate_count=len(screen.rps250_candidates),
        subject=rendered.subject,
    )
    if dry_run:
        LOGGER.info(
            "RPS notification dry run complete; SMTP was not contacted and no email "
            "was sent"
        )
        if preview_stream is not None:
            preview_stream.write(f"Subject: {rendered.subject}\n\n")
            preview_stream.write(rendered.text_body)
        return result

    if config is None:
        raise AssertionError("SMTP config must be available for a live send")
    recipient_summary = ", ".join(
        _mask_recipient(recipient) for recipient in config.recipients
    )
    LOGGER.info("Sending RPS email to %s", recipient_summary)
    try:
        send_rps_email(rendered, config)
    except Exception:
        LOGGER.exception(
            "RPS email send failed for session %s to %s",
            latest_session.isoformat(),
            recipient_summary,
        )
        raise
    LOGGER.info(
        "RPS email sent successfully for session %s to %s",
        latest_session.isoformat(),
        recipient_summary,
    )
    return result


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an ISO date in YYYY-MM-DD format"
        ) from exc


def _finite_float(value: str) -> float:
    try:
        return _validate_threshold(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.rps_notification",
        description="Calculate and email the daily full-Universe RPS screen.",
    )
    parser.add_argument(
        "--as-of-date",
        type=_parse_date,
        help="market session to screen (default: validated dataset latest_session)",
    )
    parser.add_argument(
        "--threshold",
        type=_finite_float,
        default=DEFAULT_RPS_EMAIL_THRESHOLD,
        help="strict RPS threshold (default: %(default)s)",
    )
    parser.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="calculate and render the email without connecting to SMTP",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for manual and workflow RPS email delivery."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_daily_rps_notification(
            as_of_date=args.as_of_date,
            threshold=args.threshold,
            prices_root=args.prices_root,
            universe_path=args.universe,
            dry_run=args.dry_run,
            preview_stream=sys.stdout if args.dry_run else None,
        )
    except Exception:
        LOGGER.exception("Daily RPS notification failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
