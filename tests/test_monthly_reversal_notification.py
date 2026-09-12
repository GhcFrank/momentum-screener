from __future__ import annotations

from datetime import date
from email.message import EmailMessage
from io import StringIO
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.monthly_reversal_notification as notification
from momentum_screener.monthly_reversal_notification import (
    render_monthly_reversal_email,
    run_daily_monthly_reversal_notification,
)
from momentum_screener.rps_notification import RenderedRpsEmail, SmtpEmailConfig


def _email_environment() -> dict[str, str]:
    return {
        "RPS_SMTP_HOST": "smtp.example.com",
        "RPS_SMTP_PORT": "587",
        "RPS_SMTP_USERNAME": "sender@example.com",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.com",
        "RPS_EMAIL_TO": "recipient@example.com",
    }


def _screen_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "rps50": 95.0,
                "rps120": 91.0,
                "rps250": 20.0,
                "adj_close": 230.5,
                "yxfz": True,
                "signal": True,
            },
            {
                "ticker": "MSFT",
                "rps50": 92.0,
                "rps120": 99.0,
                "rps250": 99.0,
                "adj_close": 510.0,
                "yxfz": True,
                "signal": False,
            },
            {
                "ticker": "NVDA",
                "rps50": 99.0,
                "rps120": 99.0,
                "rps250": 99.0,
                "adj_close": 180.0,
                "yxfz": False,
                "signal": False,
            },
        ]
    )


def test_email_contains_only_signal_true_not_yxfz_or_high_rps_only() -> None:
    rows = _screen_rows()
    rows.loc[rows["ticker"].eq("AAPL"), "industry"] = "Consumer Electronics"
    rendered = render_monthly_reversal_email(
        as_of_date=date(2026, 9, 3),
        screen_rows=rows,
    )

    assert rendered.subject == (
        "Momentum Screener — Monthly Reversal — 2026-09-03 — 1 signals"
    )
    assert "Signal count: 1" in rendered.text_body
    assert "AAPL" in rendered.text_body
    assert "95.00" in rendered.text_body
    assert "91.00" in rendered.text_body
    assert "230.50" in rendered.text_body
    assert "Industry" in rendered.text_body
    assert "Consumer Electronics" in rendered.text_body
    assert "Consumer Electronics" in rendered.html_body
    assert "MSFT" not in rendered.text_body
    assert "NVDA" not in rendered.text_body
    assert "RPS250" not in rendered.text_body
    assert "RPS120 >" not in rendered.text_body
    assert "RPS250 >" not in rendered.text_body
    assert "MSFT" not in rendered.html_body
    assert "NVDA" not in rendered.html_body


def test_daily_orchestration_calculates_once_persists_before_screen_and_sends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    snapshot = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "rps50": [95.0, 20.0],
            "rps120": [91.0, 99.0],
            "rps250": [20.0, 99.0],
        }
    )

    def fake_calculate(as_of_date: date, **kwargs: object) -> pd.DataFrame:
        events.append(("calculate", as_of_date, kwargs["lookbacks"]))
        return snapshot

    def fake_persist(rows: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        events.append(("persist", rows))
        assert "rps250" in rows
        return {"rows_persisted": len(rows)}

    screen = _screen_rows().iloc[[0]].copy()
    screen.attrs.update(
        {
            "universe_count": 2,
            "fyx1_candidate_count": 1,
            "yxfz_count": 1,
        }
    )

    def fake_screen(as_of_date: date, **kwargs: object) -> pd.DataFrame:
        events.append(("screen", as_of_date, kwargs["rps_snapshots"]))
        return screen

    def fake_send(rendered: RenderedRpsEmail, config: SmtpEmailConfig) -> None:
        events.append(("send", rendered, config))

    monkeypatch.setattr(notification, "calculate_rps_snapshot", fake_calculate)
    monkeypatch.setattr(notification, "persist_rps_snapshot", fake_persist)
    monkeypatch.setattr(notification, "screen_monthly_reversal", fake_screen)
    monkeypatch.setattr(notification, "send_rps_email", fake_send)

    result = run_daily_monthly_reversal_notification(
        as_of_date=date(2026, 9, 3),
        prices_root=tmp_path / "prices",
        rps_root=tmp_path / "rps",
        universe_path=tmp_path / "universe.csv",
        environ=_email_environment(),
    )

    assert [event[0] for event in events] == ["calculate", "persist", "screen", "send"]
    assert events[0][2] == (20, 50, 120, 250)
    assert events[2][2] is snapshot
    assert result.rps_row_count == 2
    assert result.rps_rows_persisted == 2
    assert result.signal_count == 1
    assert result.candidate_tickers == ("AAPL",)


def test_persistence_failure_stops_before_screen_or_email(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        notification,
        "calculate_rps_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"ticker": ["AAA"], "rps250": [99.0]}),
    )

    def fail_persist(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("persist")
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(notification, "persist_rps_snapshot", fail_persist)
    monkeypatch.setattr(
        notification,
        "screen_monthly_reversal",
        lambda *args, **kwargs: events.append("screen"),
    )
    monkeypatch.setattr(
        notification,
        "send_rps_email",
        lambda *args, **kwargs: events.append("send"),
    )

    with pytest.raises(RuntimeError, match="persistence failed"):
        run_daily_monthly_reversal_notification(
            as_of_date=date(2026, 9, 3),
            prices_root=tmp_path / "prices",
            rps_root=tmp_path / "rps",
            universe_path=tmp_path / "universe.csv",
            environ=_email_environment(),
        )

    assert events == ["persist"]


def test_dry_run_calculates_and_renders_without_persisting_or_smtp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = StringIO()
    snapshot = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "rps50": [95.0],
            "rps120": [91.0],
            "rps250": [20.0],
        }
    )
    screen = _screen_rows().iloc[[0]].copy()
    screen.attrs.update(
        {
            "universe_count": 1,
            "fyx1_candidate_count": 1,
            "yxfz_count": 1,
        }
    )
    monkeypatch.setattr(
        notification, "calculate_rps_snapshot", lambda *args, **kwargs: snapshot
    )
    monkeypatch.setattr(
        notification, "screen_monthly_reversal", lambda *args, **kwargs: screen
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not persist or send")

    monkeypatch.setattr(notification, "persist_rps_snapshot", unexpected)
    monkeypatch.setattr(notification, "send_rps_email", unexpected)

    result = run_daily_monthly_reversal_notification(
        as_of_date=date(2026, 9, 3),
        prices_root=tmp_path / "prices",
        rps_root=tmp_path / "rps",
        universe_path=tmp_path / "universe.csv",
        environ={},
        dry_run=True,
        preview_stream=output,
    )

    assert result.rps_rows_persisted == 0
    assert "AAPL" in output.getvalue()
    assert "RPS250" not in output.getvalue()


def test_zero_signal_live_run_still_sends_one_email(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = pd.DataFrame(
        {"ticker": ["AAA"], "rps50": [1.0], "rps120": [1.0], "rps250": [1.0]}
    )
    screen = pd.DataFrame(columns=["ticker", "rps50", "rps120", "adj_close", "signal"])
    screen.attrs.update(
        {
            "universe_count": 1,
            "fyx1_candidate_count": 0,
            "yxfz_count": 0,
        }
    )
    sent: list[EmailMessage | RenderedRpsEmail] = []
    monkeypatch.setattr(
        notification, "calculate_rps_snapshot", lambda *args, **kwargs: snapshot
    )
    monkeypatch.setattr(
        notification,
        "persist_rps_snapshot",
        lambda *args, **kwargs: {"rows_persisted": 1},
    )
    monkeypatch.setattr(
        notification, "screen_monthly_reversal", lambda *args, **kwargs: screen
    )
    monkeypatch.setattr(
        notification,
        "send_rps_email",
        lambda rendered, config: sent.append(rendered),
    )

    result = run_daily_monthly_reversal_notification(
        as_of_date=date(2026, 9, 3),
        prices_root=tmp_path / "prices",
        rps_root=tmp_path / "rps",
        universe_path=tmp_path / "universe.csv",
        environ=_email_environment(),
    )

    assert result.signal_count == 0
    assert len(sent) == 1
    assert sent[0].subject.endswith("0 signals")
    message = "No new monthly reversal signals for 2026-09-03."
    assert message in sent[0].text_body  # type: ignore[union-attr]
    assert message in sent[0].html_body  # type: ignore[union-attr]
