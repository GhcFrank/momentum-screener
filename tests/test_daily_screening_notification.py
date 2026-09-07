from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.daily_screening_notification as notification
import momentum_screener.monthly_reversal_notification as legacy
from momentum_screener.rps_notification import RenderedRpsEmail, SmtpEmailConfig


def screen_rows(ticker: str, *, signal: bool = True) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "ticker": [ticker],
            "rps50": [95.0],
            "rps120": [96.0],
            "rps250": [97.0],
            "adj_close": [123.0],
            "signal": [signal],
        }
    )
    rows.attrs.update(
        {
            "universe_count": 2,
            "fyx1_candidate_count": 1,
            "yxfz_count": 1,
            "momentum_candidate_count": 1,
        }
    )
    return rows


@pytest.mark.parametrize(
    ("monthly_signal", "trend_signal"),
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_email_renders_independent_sections_and_explicit_empty_results(
    monthly_signal: bool,
    trend_signal: bool,
) -> None:
    result = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("MONTHLY", signal=monthly_signal),
        trend_reacceleration_rows=screen_rows("TREND", signal=trend_signal),
    )
    assert result.subject == "Momentum Screener — 2026-09-03"
    for body in (result.text_body, result.html_body):
        assert "Monthly Reversal 6.2" in body
        assert "顺向火车2" in body
        assert "Strong Momentum + Healthy Pullback + Trend Re-acceleration" in body
        assert ("MONTHLY" in body) is monthly_signal
        assert ("TREND" in body) is trend_signal
        if not monthly_signal:
            assert "No new monthly reversal signals" in body
        if not trend_signal:
            assert "No matches" in body
        assert "RPS120 >" not in body
        assert "RPS250 >" not in body
    assert result.html_body.count("<html>") == 1
    assert result.html_body.count("<body>") == 1


def test_html_escapes_tickers() -> None:
    result = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("AAA"),
        trend_reacceleration_rows=screen_rows("<unsafe>"),
    )
    assert "&lt;unsafe&gt;" in result.html_body
    assert "<unsafe>" not in result.html_body


def smtp_environment() -> dict[str, str]:
    return {
        "RPS_SMTP_HOST": "smtp.example.com",
        "RPS_SMTP_USERNAME": "sender@example.com",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.com",
        "RPS_EMAIL_TO": "recipient@example.com",
    }


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("empty_strategy", ["monthly", "trend", "both", "neither"])
def test_orchestration_shares_rps_once_persists_once_and_sends_once(
    dry_run: bool,
    empty_strategy: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = date(2026, 9, 3)
    events: list[str] = []
    shared = pd.DataFrame(
        {
            "date": [date(2026, 9, 2), session, session],
            "ticker": ["AAA", "AAA", "BBB"],
            "rps50": [95.0] * 3,
            "rps120": [96.0] * 3,
            "rps250": [97.0] * 3,
        }
    )
    monthly = screen_rows("AAA")
    trend = screen_rows("BBB")
    if empty_strategy in {"monthly", "both"}:
        monthly = monthly.iloc[:0].copy()
    if empty_strategy in {"trend", "both"}:
        trend = trend.iloc[:0].copy()

    def prepare(sessions: tuple[date, ...], **kwargs: object) -> pd.DataFrame:
        events.append("prepare")
        assert sessions[-1] == session
        assert len(sessions) == 15
        assert kwargs["lookbacks"] == (50, 120, 250)
        return shared

    def persist(rows: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        assert not dry_run
        events.append("persist")
        assert rows["date"].tolist() == [session, session]
        assert rows["rps250"].tolist() == [97.0, 97.0]
        return {"rows_persisted": len(rows)}

    def screen_monthly(as_of_date: date, **kwargs: object) -> pd.DataFrame:
        events.append("monthly")
        assert kwargs["rps_snapshots"] is shared
        assert kwargs["rps_root"] is None
        return monthly

    def screen_trend(as_of_date: date, **kwargs: object) -> pd.DataFrame:
        events.append("trend")
        assert kwargs["rps_snapshots"] is shared
        assert kwargs["rps_root"] is None
        return trend

    def send(rendered: RenderedRpsEmail, config: SmtpEmailConfig) -> None:
        assert not dry_run
        events.append("send")
        assert "Monthly Reversal 6.2" in rendered.text_body
        assert "顺向火车2" in rendered.text_body

    monkeypatch.setattr(
        notification, "get_latest_dataset_session", lambda root: session
    )
    monkeypatch.setattr(notification, "load_or_calculate_rps", prepare)
    monkeypatch.setattr(notification, "persist_rps_snapshot", persist)
    monkeypatch.setattr(notification, "screen_monthly_reversal", screen_monthly)
    monkeypatch.setattr(notification, "screen_trend_reacceleration", screen_trend)
    monkeypatch.setattr(notification, "send_rps_email", send)
    if dry_run:

        def no_config(*args: object, **kwargs: object) -> None:
            raise AssertionError("dry run must not load SMTP configuration")

        monkeypatch.setattr(SmtpEmailConfig, "from_environment", no_config)
    preview = StringIO()
    result = notification.run_daily_screening_notification(
        prices_root=tmp_path / "prices",
        rps_root=tmp_path / "rps",
        universe_path=tmp_path / "universe.csv",
        environ={} if dry_run else smtp_environment(),
        dry_run=dry_run,
        preview_stream=preview,
    )
    assert events == (
        ["prepare", "monthly", "trend"]
        if dry_run
        else ["prepare", "persist", "monthly", "trend", "send"]
    )
    assert result.rps_row_count == 2
    assert result.rps_rows_persisted == (0 if dry_run else 2)
    assert result.signal_count == len(monthly)
    assert result.trend_signal_count == len(trend)
    assert result.as_dict()["trend_candidate_tickers"] == trend["ticker"].tolist()
    if dry_run:
        assert "顺向火车2" in preview.getvalue()


def test_persistence_failure_prevents_both_screens_and_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = screen_rows("AAA").assign(date=date(2026, 9, 3))
    monkeypatch.setattr(
        notification, "load_or_calculate_rps", lambda *args, **kwargs: rows
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("persistence failed")

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("persistence must finish before screening/sending")

    monkeypatch.setattr(notification, "persist_rps_snapshot", fail)
    monkeypatch.setattr(notification, "screen_monthly_reversal", unexpected)
    monkeypatch.setattr(notification, "screen_trend_reacceleration", unexpected)
    monkeypatch.setattr(notification, "send_rps_email", unexpected)
    with pytest.raises(RuntimeError, match="persistence failed"):
        notification.run_daily_screening_notification(
            as_of_date="2026-09-03", environ=smtp_environment()
        )


def test_legacy_cli_delegates_with_unchanged_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = ["--dry-run", "--as-of-date", "2026-09-03"]
    seen: list[object] = []

    def main(argv: object) -> int:
        seen.append(argv)
        return 7

    monkeypatch.setattr(notification, "main", main)
    assert legacy.main(arguments) == 7
    assert seen == [arguments]


def test_cli_preserves_dotenv_precedence_and_returns_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fail(**kwargs: object) -> None:
        assert kwargs["dry_run"] is True
        raise RuntimeError("screening failed")

    monkeypatch.setattr(
        notification, "load_dotenv", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(notification, "run_daily_screening_notification", fail)
    assert notification.main(["--dry-run"]) == 1
    assert calls == [{"dotenv_path": Path(".env"), "override": False}]
