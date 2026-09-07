from __future__ import annotations

import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Self

import pytest

import momentum_screener.rps_notification as notification
from momentum_screener.rps_notification import (
    EmailConfigurationError,
    RenderedRpsEmail,
    SmtpEmailConfig,
    get_latest_dataset_session,
    send_rps_email,
)


def _email_environment() -> dict[str, str]:
    return {
        "RPS_SMTP_HOST": "smtp.example.com",
        "RPS_SMTP_PORT": "587",
        "RPS_SMTP_USERNAME": "sender@example.com",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.com",
        "RPS_EMAIL_TO": "first@example.com, second@example.com",
    }


def test_latest_session_comes_from_validated_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool, bool]] = []

    def fake_load_manifest(
        path: Path, *, require_completed: bool, require_assets: bool
    ) -> dict[str, Any]:
        calls.append((path, require_completed, require_assets))
        return {"latest_session": "2026-08-31"}

    monkeypatch.setattr(notification, "load_manifest", fake_load_manifest)

    assert get_latest_dataset_session(tmp_path) == date(2026, 8, 31)
    assert calls == [(tmp_path / "manifest.json", True, True)]


def test_email_configuration_supports_multiple_recipients_and_default_port() -> None:
    environment = _email_environment()
    environment["RPS_SMTP_PORT"] = ""

    config = SmtpEmailConfig.from_environment(environment)

    assert config.port == 587
    assert config.recipients == ("first@example.com", "second@example.com")


def test_email_configuration_accepts_local_gmail_names() -> None:
    config = SmtpEmailConfig.from_environment(
        {
            "GMAIL_USER": "sender@example.com",
            "GMAIL_APP_PASSWORD": "test-app-password",
            "EMAIL_TO": "recipient@example.com",
        }
    )

    assert config.host == "smtp.gmail.com"
    assert config.port == 587
    assert config.username == "sender@example.com"
    assert config.sender == "sender@example.com"
    assert config.recipients == ("recipient@example.com",)


@pytest.mark.parametrize(
    ("missing_key", "expected_name"),
    [
        ("RPS_SMTP_USERNAME", "RPS_SMTP_USERNAME"),
        ("RPS_SMTP_PASSWORD", "RPS_SMTP_PASSWORD"),
        ("RPS_EMAIL_TO", "RPS_EMAIL_TO"),
    ],
)
def test_email_configuration_rejects_missing_required_values_without_secrets(
    missing_key: str, expected_name: str
) -> None:
    environment = _email_environment()
    secret_value = environment["RPS_SMTP_PASSWORD"]
    del environment[missing_key]

    with pytest.raises(EmailConfigurationError) as raised:
        SmtpEmailConfig.from_environment(environment)

    assert expected_name in str(raised.value)
    assert secret_value not in str(raised.value)


def test_smtp_sender_uses_starttls_auth_and_one_message() -> None:
    events: list[object] = []

    class FakeSmtp:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            events.append(("connect", host, port, timeout))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            events.append("close")

        def ehlo(self) -> None:
            events.append("ehlo")

        def starttls(self, *, context: object) -> None:
            events.append(("starttls", context is not None))

        def login(self, username: str, password: str) -> None:
            events.append(("login", username, password))

        def send_message(self, message: EmailMessage) -> None:
            events.append(("send", message))

    config = SmtpEmailConfig.from_environment(_email_environment())
    rendered = RenderedRpsEmail("Subject", "plain", "<p>html</p>")
    send_rps_email(rendered, config, smtp_factory=FakeSmtp)  # type: ignore[arg-type]

    sent_messages = [
        event for event in events if isinstance(event, tuple) and event[0] == "send"
    ]
    assert len(sent_messages) == 1
    message = sent_messages[0][1]
    assert isinstance(message, EmailMessage)
    assert message["To"] == "first@example.com, second@example.com"
    assert events[:5] == [
        ("connect", "smtp.example.com", 587, 30.0),
        "ehlo",
        ("starttls", True),
        "ehlo",
        ("login", "sender@example.com", "test-password"),
    ]


@pytest.mark.parametrize("failure_stage", ["login", "send"])
def test_smtp_sender_propagates_authentication_and_send_failures(
    failure_stage: str,
) -> None:
    class FailingSmtp:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def ehlo(self) -> None:
            pass

        def starttls(self, *, context: object) -> None:
            pass

        def login(self, username: str, password: str) -> None:
            if failure_stage == "login":
                raise smtplib.SMTPAuthenticationError(535, b"authentication failed")

        def send_message(self, message: EmailMessage) -> None:
            if failure_stage == "send":
                raise smtplib.SMTPException("send failed")

    config = SmtpEmailConfig.from_environment(_email_environment())
    rendered = RenderedRpsEmail("Subject", "plain", "<p>html</p>")

    with pytest.raises(smtplib.SMTPException):
        send_rps_email(
            rendered,
            config,
            smtp_factory=FailingSmtp,  # type: ignore[arg-type]
        )
