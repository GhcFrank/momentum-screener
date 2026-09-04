from __future__ import annotations

import smtplib
from datetime import date
from email.message import EmailMessage
from io import StringIO
from pathlib import Path
from typing import Any, Self

import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.rps_notification as notification
from momentum_screener.rps_notification import (
    DEFAULT_RPS_EMAIL_THRESHOLD,
    EmailConfigurationError,
    RenderedRpsEmail,
    SmtpEmailConfig,
    build_rps_screen,
    get_latest_dataset_session,
    render_rps_email,
    run_daily_rps_notification,
    send_rps_email,
)


def _snapshot(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "rps120", "rps250"])


def _email_environment() -> dict[str, str]:
    return {
        "RPS_SMTP_HOST": "smtp.example.com",
        "RPS_SMTP_PORT": "587",
        "RPS_SMTP_USERNAME": "sender@example.com",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.com",
        "RPS_EMAIL_TO": "first@example.com, second@example.com",
    }


def test_rps120_screen_uses_strict_threshold() -> None:
    screen = build_rps_screen(
        _snapshot(
            [
                ("LOW", 86.99, 0.0),
                ("EQUAL", 87.0, 0.0),
                ("ABOVE", 87.01, 0.0),
                ("HIGH", 95.0, 0.0),
            ]
        )
    )

    assert list(screen.rps120_candidates["ticker"]) == ["HIGH", "ABOVE"]


def test_rps250_screen_uses_strict_threshold() -> None:
    screen = build_rps_screen(
        _snapshot(
            [
                ("LOW", 0.0, 86.99),
                ("EQUAL", 0.0, 87.0),
                ("ABOVE", 0.0, 87.01),
                ("HIGH", 0.0, 95.0),
            ]
        )
    )

    assert list(screen.rps250_candidates["ticker"]) == ["HIGH", "ABOVE"]


def test_invalid_rps_values_do_not_match() -> None:
    screen = build_rps_screen(
        _snapshot([("INVALID120", -1.0, 95.0), ("INVALID250", 95.0, -1.0)])
    )

    assert list(screen.rps120_candidates["ticker"]) == ["INVALID250"]
    assert list(screen.rps250_candidates["ticker"]) == ["INVALID120"]


def test_two_screens_are_independent_and_sorted_by_their_own_metric() -> None:
    screen = build_rps_screen(
        _snapshot(
            [
                ("AAA", 95.0, 70.0),
                ("BBB", 70.0, 96.0),
                ("CCC", 98.0, 99.0),
            ]
        )
    )

    assert list(screen.rps120_candidates["ticker"]) == ["CCC", "AAA"]
    assert list(screen.rps250_candidates["ticker"]) == ["CCC", "BBB"]


def test_screen_ignores_rps50_and_keeps_only_existing_email_sections() -> None:
    snapshot = _snapshot(
        [("ONLY50", 10.0, 20.0), ("RPS120", 95.0, 20.0), ("RPS250", 10.0, 96.0)]
    )
    snapshot["rps50"] = [100.0, 0.0, 0.0]

    screen = build_rps_screen(snapshot)
    rendered = render_rps_email(
        as_of_date=date(2026, 8, 31),
        rps120_candidates=screen.rps120_candidates,
        rps250_candidates=screen.rps250_candidates,
    )

    assert list(screen.rps120_candidates["ticker"]) == ["RPS120"]
    assert list(screen.rps250_candidates["ticker"]) == ["RPS250"]
    assert "RPS50" not in rendered.text_body
    assert "RPS50" not in rendered.html_body


def test_screen_accepts_production_snapshot_ticker_index_and_column() -> None:
    snapshot = _snapshot([("BBB", 95.0, 96.0), ("AAA", 95.0, 96.0)]).set_index(
        "ticker", drop=False
    )
    snapshot.index.name = "ticker"

    screen = build_rps_screen(snapshot)

    assert list(screen.rps120_candidates["ticker"]) == ["AAA", "BBB"]
    assert list(screen.rps250_candidates["ticker"]) == ["AAA", "BBB"]


def test_threshold_is_configurable() -> None:
    screen = build_rps_screen(
        _snapshot([("EQUAL", 90.0, 90.0), ("ABOVE", 90.01, 90.01)]),
        threshold=90.0,
    )

    assert screen.threshold == 90.0
    assert list(screen.rps120_candidates["ticker"]) == ["ABOVE"]
    assert list(screen.rps250_candidates["ticker"]) == ["ABOVE"]


def test_rendered_email_contains_session_counts_sections_and_tickers() -> None:
    screen = build_rps_screen(
        _snapshot([("AAA", 95.1234, 70.0), ("BBB", 70.0, 96.9876)])
    )
    rendered = render_rps_email(
        as_of_date=date(2026, 8, 31),
        rps120_candidates=screen.rps120_candidates,
        rps250_candidates=screen.rps250_candidates,
        threshold=screen.threshold,
    )

    assert rendered.subject == "Momentum Screener - RPS - 2026-08-31"
    assert "Market session: 2026-08-31" in rendered.text_body
    assert "RPS120 > 87: 1 stocks" in rendered.text_body
    assert "RPS250 > 87: 1 stocks" in rendered.text_body
    assert "95.12" in rendered.text_body
    assert "96.99" in rendered.text_body
    assert "AAA" in rendered.text_body and "BBB" in rendered.text_body
    assert "RPS120 &gt; 87" in rendered.html_body
    assert "RPS250 &gt; 87" in rendered.html_body


def test_rendered_email_keeps_empty_sections() -> None:
    empty = _snapshot([])
    rendered = render_rps_email(
        as_of_date=date(2026, 8, 31),
        rps120_candidates=empty,
        rps250_candidates=empty,
    )

    assert "RPS120 > 87\n0 stocks\nNo stocks matched." in rendered.text_body
    assert "RPS250 > 87\n0 stocks\nNo stocks matched." in rendered.text_body
    assert rendered.html_body.count("0 stocks") == 4
    assert rendered.html_body.count("No stocks matched.") == 2


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


def test_orchestration_reuses_full_snapshot_latest_session_and_sends_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    universe_path = tmp_path / "universe.csv"
    snapshot_calls: list[tuple[date, Path, Path]] = []
    sent: list[tuple[RenderedRpsEmail, SmtpEmailConfig]] = []

    monkeypatch.setattr(
        notification,
        "get_latest_dataset_session",
        lambda prices_root: date(2026, 8, 31),
    )

    def fake_calculate_rps_snapshot(
        as_of_date: date, *, prices_root: Path, universe_path: Path
    ) -> pd.DataFrame:
        snapshot_calls.append((as_of_date, prices_root, universe_path))
        return _snapshot(
            [("AAA", 95.0, 70.0), ("BBB", 70.0, 96.0), ("CCC", 98.0, 99.0)]
        )

    def fake_send(rendered: RenderedRpsEmail, config: SmtpEmailConfig) -> None:
        sent.append((rendered, config))

    monkeypatch.setattr(
        notification, "calculate_rps_snapshot", fake_calculate_rps_snapshot
    )
    monkeypatch.setattr(notification, "send_rps_email", fake_send)
    caplog.set_level("INFO")

    result = run_daily_rps_notification(
        prices_root=tmp_path,
        universe_path=universe_path,
        environ=_email_environment(),
    )

    assert snapshot_calls == [(date(2026, 8, 31), tmp_path, universe_path)]
    assert len(sent) == 1
    assert result.latest_session == date(2026, 8, 31)
    assert result.snapshot_ticker_count == 3
    assert result.rps120_candidate_count == 2
    assert result.rps250_candidate_count == 2
    assert "first@example.com" not in caplog.text
    assert "second@example.com" not in caplog.text
    assert "f***@example.com, s***@example.com" in caplog.text


def test_manual_date_still_calls_complete_snapshot_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called_with: list[date] = []

    def fake_calculate_rps_snapshot(
        as_of_date: date, *, prices_root: Path, universe_path: Path
    ) -> pd.DataFrame:
        called_with.append(as_of_date)
        return _snapshot([])

    monkeypatch.setattr(
        notification, "calculate_rps_snapshot", fake_calculate_rps_snapshot
    )
    monkeypatch.setattr(notification, "send_rps_email", lambda rendered, config: None)

    result = run_daily_rps_notification(
        as_of_date="2026-08-31",
        prices_root=tmp_path,
        universe_path=tmp_path / "universe.csv",
        environ=_email_environment(),
    )

    assert called_with == [date(2026, 8, 31)]
    assert result.snapshot_ticker_count == 0


def test_email_send_failure_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        notification,
        "calculate_rps_snapshot",
        lambda as_of_date, **kwargs: _snapshot([]),
    )

    def fail_send(rendered: RenderedRpsEmail, config: SmtpEmailConfig) -> None:
        raise smtplib.SMTPException("delivery rejected")

    monkeypatch.setattr(notification, "send_rps_email", fail_send)

    with pytest.raises(smtplib.SMTPException, match="delivery rejected"):
        run_daily_rps_notification(
            as_of_date=date(2026, 8, 31),
            prices_root=tmp_path,
            universe_path=tmp_path / "universe.csv",
            environ=_email_environment(),
        )


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


def test_live_orchestration_validates_config_before_calculating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calculated = False

    def fake_calculate(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calculated
        calculated = True
        return _snapshot([])

    monkeypatch.setattr(notification, "calculate_rps_snapshot", fake_calculate)

    with pytest.raises(EmailConfigurationError, match="RPS_SMTP_USERNAME"):
        run_daily_rps_notification(
            as_of_date=date(2026, 8, 31),
            prices_root=tmp_path,
            universe_path=tmp_path / "universe.csv",
            environ={},
        )

    assert calculated is False


def test_dry_run_renders_preview_without_loading_config_or_sending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = StringIO()
    monkeypatch.setattr(
        notification,
        "calculate_rps_snapshot",
        lambda as_of_date, **kwargs: _snapshot([("AAA", 95.0, 96.0)]),
    )

    def unexpected_send(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not send")

    monkeypatch.setattr(notification, "send_rps_email", unexpected_send)

    result = run_daily_rps_notification(
        as_of_date=date(2026, 8, 31),
        prices_root=tmp_path,
        universe_path=tmp_path / "universe.csv",
        environ={},
        dry_run=True,
        preview_stream=output,
    )

    assert result.rps120_candidate_count == 1
    assert result.rps250_candidate_count == 1
    assert "Subject: Momentum Screener - RPS - 2026-08-31" in output.getvalue()
    assert "RPS120 > 87: 1 stocks" in output.getvalue()
    assert "RPS250 > 87: 1 stocks" in output.getvalue()


def test_cli_loads_root_dotenv_without_overriding_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_calls: list[tuple[Path, bool]] = []
    run_calls: list[dict[str, object]] = []

    def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
        dotenv_calls.append((dotenv_path, override))
        return True

    def fake_run(**kwargs: object) -> object:
        run_calls.append(kwargs)
        return object()

    monkeypatch.setattr(notification, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(notification, "run_daily_rps_notification", fake_run)

    assert notification.main(["--dry-run"]) == 0
    assert dotenv_calls == [(Path(".env"), False)]
    assert run_calls[0]["dry_run"] is True


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


def test_default_threshold_is_defined_once_for_public_apis() -> None:
    assert DEFAULT_RPS_EMAIL_THRESHOLD == 87.0
