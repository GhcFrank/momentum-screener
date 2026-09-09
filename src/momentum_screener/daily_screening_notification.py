"""Share daily RPS preparation and send one email containing both strategies."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import TextIO

import pandas as pd  # type: ignore[import-untyped]
from dotenv import load_dotenv

from momentum_screener.monthly_reversal import (
    MONTHLY_REVERSAL_SIGNAL_WINDOW,
    screen_monthly_reversal,
)
from momentum_screener.monthly_reversal_notification import (
    MonthlyReversalNotificationResult,
    _coerce_as_of_date,
    _format_number,
    render_monthly_reversal_email,
)
from momentum_screener.prices import DEFAULT_OUTPUT_ROOT, DEFAULT_UNIVERSE
from momentum_screener.rps import RPS_LOOKBACKS
from momentum_screener.rps_notification import (
    RenderedRpsEmail,
    SmtpEmailConfig,
    _mask_recipient,
    get_latest_dataset_session,
    send_rps_email,
)
from momentum_screener.rps_storage import DEFAULT_RPS_ROOT, persist_rps_snapshot
from momentum_screener.storage_manifest import write_json_atomically
from momentum_screener.strategy_data import (
    coerce_session_date,
    load_or_calculate_rps,
    resolve_strategy_sessions,
)
from momentum_screener.trend_reacceleration import (
    DEFAULT_CONFIG,
    STRATEGY_DESCRIPTION,
    STRATEGY_NAME,
    TrendReaccelerationConfig,
    screen_trend_reacceleration,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DailyScreeningNotificationResult(MonthlyReversalNotificationResult):
    """Keep Monthly Reversal summary keys and add explicit second-strategy keys."""

    trend_momentum_candidate_count: int
    trend_signal_count: int
    trend_candidate_tickers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        result = MonthlyReversalNotificationResult.as_dict(self)
        result.update(
            {
                "trend_momentum_candidate_count": self.trend_momentum_candidate_count,
                "trend_signal_count": self.trend_signal_count,
                "trend_candidate_tickers": list(self.trend_candidate_tickers),
            }
        )
        return result


def render_daily_screening_email(
    *,
    as_of_date: date,
    monthly_reversal_rows: pd.DataFrame,
    trend_reacceleration_rows: pd.DataFrame,
) -> RenderedRpsEmail:
    """Render independent signal sections; never suppress yesterday's tickers."""

    monthly = render_monthly_reversal_email(
        as_of_date=as_of_date, screen_rows=monthly_reversal_rows
    )
    required = {"ticker", "rps120", "rps250", "signal"}
    missing = sorted(required.difference(trend_reacceleration_rows.columns))
    if missing:
        raise ValueError(
            f"Trend Re-acceleration email rows are missing columns: {missing}"
        )
    rows = trend_reacceleration_rows.loc[
        trend_reacceleration_rows["signal"].fillna(False).astype("bool")
    ].sort_values("ticker", kind="mergesort")
    title = f"Momentum Screener — {as_of_date.isoformat()}"
    text_lines = [
        title,
        "",
        monthly.text_body.rstrip(),
        "",
        STRATEGY_NAME,
        STRATEGY_DESCRIPTION,
        f"Signal count: {len(rows)}",
        "",
    ]
    trend_html = (
        f"<h2>{STRATEGY_NAME}</h2><p>{STRATEGY_DESCRIPTION}</p>"
        f"<p>Signal count: <strong>{len(rows)}</strong></p>"
    )
    if rows.empty:
        text_lines.append("No matches.")
        trend_html += "<p>No matches.</p>"
    else:
        text_lines.append(
            f"{'Ticker':<12} {'RPS120':>8} {'RPS250':>8} {'Adj Close':>12}"
        )
        table_rows: list[str] = []
        for _, row in rows.iterrows():
            values = [
                str(row["ticker"]),
                _format_number(row["rps120"]),
                _format_number(row["rps250"]),
                _format_number(row.get("adj_close")),
            ]
            text_lines.append(
                f"{values[0]:<12} {values[1]:>8} {values[2]:>8} {values[3]:>12}"
            )
            table_rows.append(
                "<tr>"
                + "".join(f"<td>{escape(value)}</td>" for value in values)
                + "</tr>"
            )
        trend_html += (
            '<table style="border-collapse:collapse"><thead><tr>'
            "<th>Ticker</th><th>RPS120</th><th>RPS250</th><th>Adj Close</th>"
            "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
        )
    monthly_html = monthly.html_body.partition("<body>")[2].rpartition("</body>")[0]
    monthly_html = monthly_html.replace("<h1>", "<h2>").replace("</h1>", "</h2>")
    return RenderedRpsEmail(
        subject=title,
        text_body="\n".join(text_lines) + "\n",
        html_body=(
            "<!doctype html><html><body>"
            + f"<h1>{title}</h1>"
            + monthly_html
            + trend_html
            + "</body></html>"
        ),
    )


def run_daily_screening_notification(
    *,
    as_of_date: date | str | None = None,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    rps_root: Path = DEFAULT_RPS_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
    preview_stream: TextIO | None = None,
    trend_config: TrendReaccelerationConfig = DEFAULT_CONFIG,
    prepared_email_path: Path | None = None,
) -> DailyScreeningNotificationResult:
    """Load/calculate shared RPS once, persist today's rows once, send one email.

    The 15-session batch warms Monthly Reversal's existing first-occurrence
    logic; Trend Re-acceleration uses only today's RPS from the same batch.
    dry_run renders without loading SMTP config, contacting SMTP or persisting.
    """

    if dry_run and prepared_email_path is not None:
        raise ValueError("dry_run cannot prepare a persisted notification")
    smtp_config = (
        None
        if dry_run or prepared_email_path is not None
        else SmtpEmailConfig.from_environment(environ)
    )
    session = coerce_session_date(
        get_latest_dataset_session(prices_root)
        if as_of_date is None
        else _coerce_as_of_date(as_of_date)
    )
    LOGGER.info("Daily screening session=%s", session.isoformat())
    sessions = resolve_strategy_sessions(session, MONTHLY_REVERSAL_SIGNAL_WINDOW)
    shared_rps = load_or_calculate_rps(
        sessions,
        lookbacks=RPS_LOOKBACKS,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=rps_root,
    )
    snapshot = shared_rps.loc[shared_rps["date"].eq(session)].copy()
    LOGGER.info(
        "Prepared shared RPS50/RPS120/RPS250 sessions=%d current_rows=%d",
        len(sessions),
        len(snapshot),
    )
    persisted_count = 0
    if not dry_run:
        persistence = persist_rps_snapshot(
            snapshot, root=rps_root, universe_path=universe_path
        )
        persisted_count = int(persistence["rows_persisted"])
        LOGGER.info("Persisted RPS rows=%d", persisted_count)

    monthly = screen_monthly_reversal(
        session,
        signal_only=True,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=None,
        rps_snapshots=shared_rps,
    )
    trend = screen_trend_reacceleration(
        session,
        signal_only=True,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=None,
        rps_snapshots=shared_rps,
        config=trend_config,
    )
    rendered = render_daily_screening_email(
        as_of_date=session,
        monthly_reversal_rows=monthly,
        trend_reacceleration_rows=trend,
    )
    result = DailyScreeningNotificationResult(
        as_of_date=session,
        universe_count=int(monthly.attrs["universe_count"]),
        rps_row_count=len(snapshot),
        rps_rows_persisted=persisted_count,
        fyx1_candidate_count=int(monthly.attrs["fyx1_candidate_count"]),
        yxfz_count=int(monthly.attrs["yxfz_count"]),
        signal_count=len(monthly),
        candidate_tickers=tuple(str(value) for value in monthly["ticker"]),
        trend_momentum_candidate_count=int(trend.attrs["momentum_candidate_count"]),
        trend_signal_count=len(trend),
        trend_candidate_tickers=tuple(str(value) for value in trend["ticker"]),
        subject=rendered.subject,
        dry_run=dry_run,
    )
    LOGGER.info(
        "Monthly Reversal signals=%d; Trend Re-acceleration signals=%d",
        len(monthly),
        len(trend),
    )
    if dry_run:
        if preview_stream is not None:
            preview_stream.write(f"Subject: {rendered.subject}\n\n{rendered.text_body}")
        LOGGER.info("Daily screening dry run complete; no persistence or SMTP contact")
        return result
    if prepared_email_path is not None:
        # The daily workflow publishes and verifies all datasets before sending
        # this exact rendering; no second RPS calculation or strategy execution.
        write_json_atomically(
            prepared_email_path,
            {"result": result.as_dict(), "email": asdict(rendered)},
        )
        LOGGER.info("Daily screening prepared; SMTP deferred until publication")
        return result
    if smtp_config is None:
        raise AssertionError("SMTP config must be available for a live send")
    recipient_summary = ", ".join(
        _mask_recipient(value) for value in smtp_config.recipients
    )
    LOGGER.info("Sending daily screening email to %s", recipient_summary)
    send_rps_email(rendered, smtp_config)
    LOGGER.info(
        "Daily screening email sent for %s to %s",
        session.isoformat(),
        recipient_summary,
    )
    return result


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Production CLI, retaining the previous notification options."""

    parser = argparse.ArgumentParser(
        prog="python -m momentum_screener.daily_screening_notification",
        description="Persist daily RPS and email Monthly Reversal and 顺向火车2 signals.",
    )
    parser.add_argument("--as-of-date", type=_parse_date)
    parser.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rps-root", type=Path, default=DEFAULT_RPS_ROOT)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--prepare-email",
        type=Path,
        help="persist RPS and save the rendered email without contacting SMTP",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render without persistence or SMTP contact",
    )
    args = parser.parse_args(argv)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run_daily_screening_notification(
            as_of_date=args.as_of_date,
            prices_root=args.prices_root,
            rps_root=args.rps_root,
            universe_path=args.universe,
            dry_run=args.dry_run,
            preview_stream=sys.stdout if args.dry_run else None,
            prepared_email_path=args.prepare_email,
        )
        if args.result_json is not None:
            write_json_atomically(args.result_json, result.as_dict())
        return 0
    except Exception:
        LOGGER.exception("Daily screening notification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
