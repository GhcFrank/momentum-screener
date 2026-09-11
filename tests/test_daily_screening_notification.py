from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

import momentum_screener.daily_screening_notification as notification
import momentum_screener.monthly_reversal_notification as legacy
import momentum_screener.strategy_data as data
from momentum_screener.rps_notification import RenderedRpsEmail, SmtpEmailConfig


def screen_rows(
    ticker: str, *, signal: bool = True, turnover: float = 0.0084
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "date": [date(2026, 9, 3)],
            "ticker": [ticker],
            "rps20": [94.0],
            "rps50": [95.0],
            "rps120": [96.0],
            "rps250": [97.0],
            "adj_close": [123.0],
            "turnover": [turnover],
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


def test_email_renders_three_sections_turnover_and_explicit_empty_results() -> None:
    result = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("MONTHLY", turnover=0.0084),
        trend_reacceleration_rows=screen_rows("TREND", turnover=0.0127),
        blue_diamond_rows=screen_rows("BLUE", turnover=0.1234),
    )
    assert result.subject == "Momentum Screener — 2026-09-03"
    for body in (result.text_body, result.html_body):
        assert "Monthly Reversal 6.2" in body
        assert "顺向火车2" in body
        assert "Strong Momentum + Healthy Pullback + Trend Re-acceleration" in body
        assert "Blue Diamond / 蓝色钻石" in body
        assert "Extreme Momentum + Strong Trend Structure" in body
        assert all(ticker in body for ticker in ("MONTHLY", "TREND", "BLUE"))
        assert body.count("Turnover") == 3
        assert all(value in body for value in ("0.84%", "1.27%", "12.34%"))
        assert "RPS120 >" not in body
        assert "RPS250 >" not in body
    assert result.html_body.count("<html>") == 1
    assert result.html_body.count("<body>") == 1

    empty = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("MONTHLY", signal=False),
        trend_reacceleration_rows=screen_rows("TREND", signal=False),
        blue_diamond_rows=screen_rows("BLUE", signal=False),
    )
    assert empty.text_body.count("Signal count: 0") == 3
    assert empty.html_body.count("Signal count: <strong>0</strong>") == 3
    for body in (empty.text_body, empty.html_body):
        assert "No new monthly reversal signals" in body
        assert body.count("No matches.") == 2

    missing = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("KEPT", turnover=float("nan")),
        trend_reacceleration_rows=screen_rows("TREND", signal=False),
        blue_diamond_rows=screen_rows("BLUE", signal=False),
    )
    for body in (missing.text_body, missing.html_body):
        assert "KEPT" in body
        assert "N/A" in body


def test_html_escapes_tickers() -> None:
    result = notification.render_daily_screening_email(
        as_of_date=date(2026, 9, 3),
        monthly_reversal_rows=screen_rows("AAA"),
        trend_reacceleration_rows=screen_rows("<unsafe>"),
    )
    assert "&lt;unsafe&gt;" in result.html_body
    assert "<unsafe>" not in result.html_body


def test_turnover_enrichment_uses_exact_raw_close_and_retains_missing_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = date(2026, 9, 3)
    prices = pd.DataFrame(
        {
            "date": [session, session],
            "ticker": ["AAA", "BBB"],
            "close": [100.0, 200.0],
            "adj_close": [50.0, 100.0],
            "volume": [1_000_000, 500_000],
        }
    )
    caps = pd.DataFrame(
        {
            "date": [session, date(2026, 9, 2)],
            "ticker": ["AAA", "BBB"],
            "market_cap": [10_000_000_000.0, 1.0],
        }
    )

    def load_prices(**kwargs: object) -> tuple[pd.DataFrame, int]:
        assert kwargs["start_date"] == kwargs["end_date"] == session
        assert kwargs["tickers"] == ("AAA", "BBB")
        return prices, 1

    monkeypatch.setattr(data, "load_strategy_price_history", load_prices)
    turnover = data.load_session_turnover(
        session,
        ("AAA", "BBB", "AAA"),
        market_cap_rows=caps,
    )
    assert turnover["ticker"].tolist() == ["AAA", "BBB"]
    assert turnover.loc[
        turnover["ticker"].eq("AAA"), "turnover"
    ].item() == pytest.approx(0.01)
    assert turnover.loc[turnover["ticker"].eq("BBB"), "turnover"].isna().all()

    signals = pd.DataFrame(
        {"date": [session, session], "ticker": ["AAA", "BBB"], "signal": True}
    )
    signals.attrs["preserved"] = "yes"
    enriched = data.enrich_signal_turnover(signals, turnover)
    assert enriched["ticker"].tolist() == ["AAA", "BBB"]
    assert enriched["signal"].tolist() == [True, True]
    assert enriched["turnover"].iloc[0] == pytest.approx(0.01)
    assert pd.isna(enriched["turnover"].iloc[1])
    assert enriched.attrs["preserved"] == "yes"


def smtp_environment() -> dict[str, str]:
    return {
        "RPS_SMTP_HOST": "smtp.example.com",
        "RPS_SMTP_USERNAME": "sender@example.com",
        "RPS_SMTP_PASSWORD": "test-password",
        "RPS_EMAIL_FROM": "sender@example.com",
        "RPS_EMAIL_TO": "recipient@example.com",
    }


# Per-strategy empty rendering is covered above. Orchestration needs a normal
# send, an all-empty send, and a dry run, not every render state in both modes.
@pytest.mark.parametrize(
    ("dry_run", "empty_strategy", "prepare_only"),
    [
        pytest.param(False, "neither", False, id="live-matches"),
        pytest.param(False, "all", False, id="live-zero-signals"),
        pytest.param(True, "neither", False, id="dry-run"),
        pytest.param(False, "neither", True, id="prepare-before-publication"),
    ],
)
def test_orchestration_shares_rps_once_persists_once_and_sends_once(
    dry_run: bool,
    empty_strategy: str,
    prepare_only: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = date(2026, 9, 3)
    events: list[str] = []
    shared = pd.DataFrame(
        {
            "date": [date(2026, 9, 2), session, session],
            "ticker": ["AAA", "AAA", "BBB"],
            "rps20": [94.0] * 3,
            "rps50": [95.0] * 3,
            "rps120": [96.0] * 3,
            "rps250": [97.0] * 3,
        }
    )
    monthly = screen_rows("AAA")
    trend = screen_rows("BBB", turnover=0.0127)
    blue = screen_rows("CCC", turnover=0.1234)
    if empty_strategy in {"monthly", "all"}:
        monthly = monthly.iloc[:0].copy()
    if empty_strategy in {"trend", "all"}:
        trend = trend.iloc[:0].copy()
    if empty_strategy in {"blue", "all"}:
        blue = blue.iloc[:0].copy()
    caps = pd.DataFrame(
        {
            "date": [session] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "market_cap": [1_000_000_000.0] * 3,
        }
    )
    caps.attrs["session_counts"] = {
        session.isoformat(): {
            "market_cap_available_count": 3,
            "market_cap_missing_ticker_count": 0,
        }
    }

    def load_caps(sessions: tuple[date, ...], **kwargs: object) -> pd.DataFrame:
        events.append("caps")
        assert sessions == (session,)
        return caps

    def prepare(sessions: tuple[date, ...], **kwargs: object) -> pd.DataFrame:
        events.append("prepare")
        assert sessions[-1] == session
        assert len(sessions) == 15
        assert kwargs["lookbacks"] == (20, 50, 120, 250)
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

    def screen_blue(as_of_date: date, **kwargs: object) -> pd.DataFrame:
        events.append("blue")
        assert kwargs["signal_only"] is True
        assert kwargs["rps_snapshots"] is shared
        assert kwargs["rps_root"] is None
        assert kwargs["market_cap_rows"] is caps
        assert "config" not in kwargs
        return blue

    def load_turnover(
        as_of_date: date, tickers: tuple[str, ...], **kwargs: object
    ) -> pd.DataFrame:
        events.append("turnover")
        assert as_of_date == session
        assert kwargs["market_cap_rows"] is caps
        expected = () if empty_strategy == "all" else ("AAA", "BBB", "CCC")
        assert tickers == expected
        values = {"AAA": 0.0084, "BBB": 0.0127, "CCC": 0.1234}
        return pd.DataFrame(
            {
                "date": [session] * len(tickers),
                "ticker": list(tickers),
                "turnover": [values[ticker] for ticker in tickers],
            }
        )

    def send(rendered: RenderedRpsEmail, config: SmtpEmailConfig) -> None:
        assert not dry_run
        events.append("send")
        assert "Monthly Reversal 6.2" in rendered.text_body
        assert "顺向火车2" in rendered.text_body
        assert "Blue Diamond / 蓝色钻石" in rendered.text_body
        assert rendered.text_body.count("Turnover") == (
            0 if empty_strategy == "all" else 3
        )

    monkeypatch.setattr(
        notification, "get_latest_dataset_session", lambda root: session
    )
    monkeypatch.setattr(notification, "load_or_calculate_rps", prepare)
    monkeypatch.setattr(notification, "load_strategy_market_cap", load_caps)
    monkeypatch.setattr(notification, "persist_rps_snapshot", persist)
    monkeypatch.setattr(notification, "screen_monthly_reversal", screen_monthly)
    monkeypatch.setattr(notification, "screen_trend_reacceleration", screen_trend)
    monkeypatch.setattr(notification, "screen_blue_diamond", screen_blue)
    monkeypatch.setattr(notification, "load_session_turnover", load_turnover)
    monkeypatch.setattr(notification, "send_rps_email", send)
    if dry_run or prepare_only:

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
        prepared_email_path=tmp_path / "prepared.json" if prepare_only else None,
    )
    assert events == (
        ["caps", "prepare", "monthly", "trend", "blue", "turnover"]
        if dry_run
        else ["caps", "prepare", "persist", "monthly", "trend", "blue", "turnover"]
        + ([] if prepare_only else ["send"])
    )
    assert result.rps_row_count == 2
    assert result.rps_rows_persisted == (0 if dry_run else 2)
    assert result.signal_count == len(monthly)
    assert result.trend_signal_count == len(trend)
    assert result.blue_diamond_signal_count == len(blue)
    assert result.as_dict()["trend_candidate_tickers"] == trend["ticker"].tolist()
    assert result.as_dict()["blue_diamond_candidate_tickers"] == blue["ticker"].tolist()
    if dry_run:
        assert "顺向火车2" in preview.getvalue()
        assert "Blue Diamond / 蓝色钻石" in preview.getvalue()
    if prepare_only:
        import json

        payload = json.loads((tmp_path / "prepared.json").read_text())
        assert payload["result"] == result.as_dict()
        assert "顺向火车2" in payload["email"]["text_body"]
        assert "Blue Diamond / 蓝色钻石" in payload["email"]["text_body"]


def test_persistence_failure_prevents_all_screens_and_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = screen_rows("AAA").assign(date=date(2026, 9, 3))
    monkeypatch.setattr(
        notification, "load_or_calculate_rps", lambda *args, **kwargs: rows
    )
    monkeypatch.setattr(
        notification,
        "load_strategy_market_cap",
        lambda *args, **kwargs: pd.DataFrame(
            {"date": [date(2026, 9, 3)], "ticker": ["AAA"], "market_cap": [1.0]}
        ),
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("persistence failed")

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("persistence must finish before screening/sending")

    monkeypatch.setattr(notification, "persist_rps_snapshot", fail)
    monkeypatch.setattr(notification, "screen_monthly_reversal", unexpected)
    monkeypatch.setattr(notification, "screen_trend_reacceleration", unexpected)
    monkeypatch.setattr(notification, "screen_blue_diamond", unexpected)
    monkeypatch.setattr(notification, "load_session_turnover", unexpected)
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
