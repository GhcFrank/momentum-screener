"""Legacy Monthly Reversal APIs; CLI delegates to the combined daily email."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any, TextIO

import pandas as pd  # type: ignore[import-untyped]

from momentum_screener.monthly_reversal import screen_monthly_reversal
from momentum_screener.prices import DEFAULT_OUTPUT_ROOT, DEFAULT_UNIVERSE
from momentum_screener.rps import RPS_LOOKBACKS, calculate_rps_snapshot
from momentum_screener.rps_notification import (
    RenderedRpsEmail,
    SmtpEmailConfig,
    _mask_recipient,
    get_latest_dataset_session,
    send_rps_email,
)
from momentum_screener.rps_storage import (
    DEFAULT_RPS_ROOT,
    persist_rps_snapshot,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MonthlyReversalNotificationResult:
    """Non-secret outcome of one daily RPS/persist/screen/email pipeline."""

    as_of_date: date
    universe_count: int
    rps_row_count: int
    rps_rows_persisted: int
    fyx1_candidate_count: int
    yxfz_count: int
    signal_count: int
    candidate_tickers: tuple[str, ...]
    subject: str
    dry_run: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "universe_count": self.universe_count,
            "rps_row_count": self.rps_row_count,
            "rps_rows_persisted": self.rps_rows_persisted,
            "fyx1_candidate_count": self.fyx1_candidate_count,
            "yxfz_count": self.yxfz_count,
            "signal_count": self.signal_count,
            "candidate_tickers": list(self.candidate_tickers),
            "subject": self.subject,
            "dry_run": self.dry_run,
        }


def _format_number(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{parsed:.2f}" if math.isfinite(parsed) else "N/A"


def _signal_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "rps50", "rps120", "signal"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Monthly Reversal email rows are missing columns: {missing}")
    signal = rows["signal"].fillna(False).astype("bool")
    selected = rows.loc[signal].copy()
    if "adj_close" not in selected:
        selected["adj_close"] = float("nan")
    return selected.sort_values("ticker", kind="mergesort", ignore_index=True)


def render_monthly_reversal_email(
    *,
    as_of_date: date,
    screen_rows: pd.DataFrame,
) -> RenderedRpsEmail:
    """Render a concise signal-only email without performing any I/O."""

    signals = _signal_rows(screen_rows)
    count = len(signals)
    subject = (
        "Momentum Screener — Monthly Reversal — "
        f"{as_of_date.isoformat()} — {count} signals"
    )
    text_lines = [
        "Monthly Reversal 6.2",
        f"Market session: {as_of_date.isoformat()}",
        f"Signal count: {count}",
        "",
    ]
    if signals.empty:
        text_lines.append(
            f"No new monthly reversal signals for {as_of_date.isoformat()}."
        )
        html_content = (
            f"<p>No new monthly reversal signals for {as_of_date.isoformat()}.</p>"
        )
    else:
        text_lines.append(
            f"{'Ticker':<12} {'RPS50':>8} {'RPS120':>8} {'Adj Close':>12}"
        )
        for _, row in signals.iterrows():
            text_lines.append(
                f"{row['ticker']!s:<12} "
                f"{_format_number(row['rps50']):>8} "
                f"{_format_number(row['rps120']):>8} "
                f"{_format_number(row['adj_close']):>12}"
            )
        html_rows = "".join(
            "<tr>"
            f"<td>{escape(str(row['ticker']))}</td>"
            f"<td>{_format_number(row['rps50'])}</td>"
            f"<td>{_format_number(row['rps120'])}</td>"
            f"<td>{_format_number(row['adj_close'])}</td>"
            "</tr>"
            for _, row in signals.iterrows()
        )
        html_content = (
            '<table style="border-collapse:collapse">'
            "<thead><tr><th>Ticker</th><th>RPS50</th><th>RPS120</th>"
            f"<th>Adj Close</th></tr></thead><tbody>{html_rows}</tbody></table>"
        )
    html_body = (
        "<!doctype html><html><body>"
        "<h1>Monthly Reversal 6.2</h1>"
        f"<p>Market session: <strong>{as_of_date.isoformat()}</strong></p>"
        f"<p>Signal count: <strong>{count}</strong></p>"
        f"{html_content}</body></html>"
    )
    return RenderedRpsEmail(
        subject=subject,
        text_body="\n".join(text_lines) + "\n",
        html_body=html_body,
    )


def _coerce_as_of_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise TypeError("as_of_date must be a date or ISO YYYY-MM-DD string")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("as_of_date must use ISO YYYY-MM-DD format") from exc


def run_daily_monthly_reversal_notification(
    *,
    as_of_date: date | str | None = None,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    rps_root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
    preview_stream: TextIO | None = None,
) -> MonthlyReversalNotificationResult:
    """Run calculate-once RPS, persist, signal screen, render, and send."""

    config = None if dry_run else SmtpEmailConfig.from_environment(environ)
    session = (
        get_latest_dataset_session(prices_root)
        if as_of_date is None
        else _coerce_as_of_date(as_of_date)
    )
    LOGGER.info("Monthly Reversal daily session=%s", session.isoformat())

    snapshot = calculate_rps_snapshot(
        session,
        lookbacks=RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
    )
    LOGGER.info("Calculated RPS lookbacks=%s rows=%d", RPS_LOOKBACKS, len(snapshot))
    if dry_run:
        persisted_count = 0
    else:
        persistence = persist_rps_snapshot(
            snapshot,
            root=rps_root,
            universe_path=universe_path,
        )
        persisted_count = int(persistence["rows_persisted"])
        LOGGER.info("Persisted RPS rows=%d", persisted_count)

    screen = screen_monthly_reversal(
        session,
        signal_only=True,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=rps_root,
        rps_snapshots=snapshot,
    )
    rendered = render_monthly_reversal_email(
        as_of_date=session,
        screen_rows=screen,
    )
    candidate_tickers = tuple(str(value) for value in screen["ticker"])
    result = MonthlyReversalNotificationResult(
        as_of_date=session,
        universe_count=int(screen.attrs["universe_count"]),
        rps_row_count=len(snapshot),
        rps_rows_persisted=persisted_count,
        fyx1_candidate_count=int(screen.attrs["fyx1_candidate_count"]),
        yxfz_count=int(screen.attrs["yxfz_count"]),
        signal_count=len(screen),
        candidate_tickers=candidate_tickers,
        subject=rendered.subject,
        dry_run=dry_run,
    )
    if dry_run:
        LOGGER.info(
            "Monthly Reversal notification dry run complete; RPS was not persisted, "
            "SMTP was not contacted, and no email was sent"
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
    LOGGER.info(
        "Sending Monthly Reversal email with %d signals to %s",
        result.signal_count,
        recipient_summary,
    )
    send_rps_email(rendered, config)
    LOGGER.info(
        "Monthly Reversal email sent for %s to %s",
        session.isoformat(),
        recipient_summary,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Compatibility CLI for the combined daily screening notification.

    The legacy programmatic renderer/runner above retain their existing
    Monthly Reversal-only contracts for callers that still import them.
    """

    from momentum_screener.daily_screening_notification import main as daily_main

    return daily_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
