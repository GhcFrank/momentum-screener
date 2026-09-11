"""Share daily inputs and send one email containing all production strategies."""

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

from momentum_screener.blue_diamond import (
    STRATEGY_DESCRIPTION as BLUE_DIAMOND_DESCRIPTION,
)
from momentum_screener.blue_diamond import STRATEGY_NAME as BLUE_DIAMOND_NAME
from momentum_screener.blue_diamond import screen_blue_diamond
from momentum_screener.daily_watch_3 import (
    STRATEGY_DESCRIPTION as DAILY_WATCH_3_DESCRIPTION,
)
from momentum_screener.daily_watch_3 import STRATEGY_NAME as DAILY_WATCH_3_NAME
from momentum_screener.daily_watch_3 import screen_daily_watch_3
from momentum_screener.market_cap_storage import DEFAULT_MARKET_CAP_ROOT
from momentum_screener.monthly_reversal import (
    MONTHLY_REVERSAL_SIGNAL_WINDOW,
    screen_monthly_reversal,
)
from momentum_screener.monthly_reversal_notification import (
    MonthlyReversalNotificationResult,
    _coerce_as_of_date,
    _format_number,
    _format_percentage,
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
    enrich_signal_turnover,
    load_or_calculate_rps,
    load_session_turnover,
    load_strategy_market_cap,
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
    """Keep Monthly Reversal summary keys and add the other daily strategies."""

    trend_momentum_candidate_count: int
    trend_signal_count: int
    trend_candidate_tickers: tuple[str, ...]
    blue_diamond_signal_count: int
    blue_diamond_candidate_tickers: tuple[str, ...]
    daily_watch_3_rps_candidate_count: int
    daily_watch_3_signal_count: int
    daily_watch_3_candidate_tickers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        result = MonthlyReversalNotificationResult.as_dict(self)
        result.update(
            {
                "trend_momentum_candidate_count": self.trend_momentum_candidate_count,
                "trend_signal_count": self.trend_signal_count,
                "trend_candidate_tickers": list(self.trend_candidate_tickers),
                "blue_diamond_signal_count": self.blue_diamond_signal_count,
                "blue_diamond_candidate_tickers": list(
                    self.blue_diamond_candidate_tickers
                ),
                "daily_watch_3_rps_candidate_count": (
                    self.daily_watch_3_rps_candidate_count
                ),
                "daily_watch_3_signal_count": self.daily_watch_3_signal_count,
                "daily_watch_3_candidate_tickers": list(
                    self.daily_watch_3_candidate_tickers
                ),
            }
        )
        return result


def _render_strategy_section(
    *,
    rows: pd.DataFrame,
    name: str,
    description: str,
    columns: tuple[tuple[str, str, int], ...],
) -> tuple[list[str], str]:
    """Render one already-selected strategy without changing its signal mask."""

    required = {"ticker", "signal"} | {
        key for key, _, _ in columns if key not in {"adj_close", "turnover"}
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"{name} email rows are missing columns: {missing}")
    selected = rows.loc[rows["signal"].fillna(False).astype("bool")].sort_values(
        "ticker", kind="mergesort"
    )
    text_lines = [name, description, f"Signal count: {len(selected)}", ""]
    html = (
        f"<h2>{escape(name)}</h2><p>{escape(description)}</p>"
        f"<p>Signal count: <strong>{len(selected)}</strong></p>"
    )
    if selected.empty:
        text_lines.append("No matches.")
        return text_lines, html + "<p>No matches.</p>"

    headers = (("ticker", "Ticker", 12), *columns)
    text_lines.append(
        " ".join(
            f"{label:<{width}}" if key == "ticker" else f"{label:>{width}}"
            for key, label, width in headers
        )
    )
    html_rows: list[str] = []
    for _, row in selected.iterrows():
        values = [str(row["ticker"])]
        for key, _, _ in columns:
            value = row.get(key)
            values.append(
                _format_percentage(value)
                if key in {"turnover", "turnover_market_cap_proxy"}
                else _format_number(value)
            )
        text_lines.append(
            " ".join(
                f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
                for index, (value, (_, _, width)) in enumerate(zip(values, headers))
            )
        )
        html_rows.append(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"
        )
    html += (
        '<table style="border-collapse:collapse"><thead><tr>'
        + "".join(f"<th>{escape(label)}</th>" for _, label, _ in headers)
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )
    return text_lines, html


def render_daily_screening_email(
    *,
    as_of_date: date,
    monthly_reversal_rows: pd.DataFrame,
    trend_reacceleration_rows: pd.DataFrame,
    blue_diamond_rows: pd.DataFrame | None = None,
    daily_watch_3_rows: pd.DataFrame | None = None,
) -> RenderedRpsEmail:
    """Render all production signal sections with presentation turnover."""

    monthly_rows = monthly_reversal_rows.copy()
    if "turnover" not in monthly_rows:
        monthly_rows["turnover"] = float("nan")
    monthly = render_monthly_reversal_email(
        as_of_date=as_of_date, screen_rows=monthly_rows
    )
    trend_text, trend_html = _render_strategy_section(
        rows=trend_reacceleration_rows,
        name=STRATEGY_NAME,
        description=STRATEGY_DESCRIPTION,
        columns=(
            ("rps120", "RPS120", 8),
            ("rps250", "RPS250", 8),
            ("adj_close", "Adj Close", 12),
            ("turnover", "Turnover", 10),
        ),
    )
    if blue_diamond_rows is None:
        blue_diamond_rows = pd.DataFrame(
            columns=("ticker", "rps20", "rps50", "adj_close", "signal")
        )
    blue_text, blue_html = _render_strategy_section(
        rows=blue_diamond_rows,
        name=f"Blue Diamond / {BLUE_DIAMOND_NAME}",
        description=BLUE_DIAMOND_DESCRIPTION,
        columns=(
            ("rps20", "RPS20", 8),
            ("rps50", "RPS50", 8),
            ("adj_close", "Adj Close", 12),
            ("turnover", "Turnover", 10),
        ),
    )
    if daily_watch_3_rows is None:
        daily_watch_3_rows = pd.DataFrame(
            columns=(
                "ticker",
                "rps50",
                "rps120",
                "rps250",
                "adj_close",
                "turnover_market_cap_proxy",
                "signal",
            )
        )
    daily_watch_3_text, daily_watch_3_html = _render_strategy_section(
        rows=daily_watch_3_rows,
        name=f"Daily Watch 3 / {DAILY_WATCH_3_NAME}",
        description=DAILY_WATCH_3_DESCRIPTION,
        columns=(
            ("rps50", "RPS50", 8),
            ("rps120", "RPS120", 8),
            ("rps250", "RPS250", 8),
            ("adj_close", "Adj Close", 12),
            ("turnover_market_cap_proxy", "Turnover", 10),
        ),
    )
    title = f"Momentum Screener — {as_of_date.isoformat()}"
    text_lines = [
        title,
        "",
        monthly.text_body.rstrip(),
        "",
        *trend_text,
        "",
        *blue_text,
        "",
        *daily_watch_3_text,
    ]
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
            + blue_html
            + daily_watch_3_html
            + "</body></html>"
        ),
    )


def run_daily_screening_notification(
    *,
    as_of_date: date | str | None = None,
    prices_root: Path = DEFAULT_OUTPUT_ROOT,
    rps_root: Path = DEFAULT_RPS_ROOT,
    market_cap_root: Path = DEFAULT_MARKET_CAP_ROOT,
    universe_path: Path = DEFAULT_UNIVERSE,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
    preview_stream: TextIO | None = None,
    trend_config: TrendReaccelerationConfig = DEFAULT_CONFIG,
    prepared_email_path: Path | None = None,
) -> DailyScreeningNotificationResult:
    """Load shared point-in-time inputs once and send one screening email.

    The 15-session batch warms Monthly Reversal's existing first-occurrence
    logic; Trend, Blue Diamond and Daily Watch 3 use today's RPS from the same batch.
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
    market_cap_rows = load_strategy_market_cap(
        (session,), root=market_cap_root, universe_path=universe_path
    )
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
        "Prepared shared RPS20/RPS50/RPS120/RPS250 sessions=%d current_rows=%d",
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
    blue_diamond = screen_blue_diamond(
        session,
        signal_only=True,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=None,
        rps_snapshots=shared_rps,
        market_cap_root=market_cap_root,
        market_cap_rows=market_cap_rows,
    )
    daily_watch_3 = screen_daily_watch_3(
        session,
        signal_only=True,
        prices_root=prices_root,
        universe_path=universe_path,
        rps_root=None,
        rps_snapshots=shared_rps,
        market_cap_root=market_cap_root,
        market_cap_rows=market_cap_rows,
    )
    signal_tickers = tuple(
        dict.fromkeys(
            str(ticker)
            for rows in (monthly, trend, blue_diamond)
            for ticker in rows["ticker"]
        )
    )
    turnover_rows = load_session_turnover(
        session,
        signal_tickers,
        prices_root=prices_root,
        market_cap_root=market_cap_root,
        universe_path=universe_path,
        market_cap_rows=market_cap_rows,
    )
    monthly_email_rows = enrich_signal_turnover(monthly, turnover_rows)
    trend_email_rows = enrich_signal_turnover(trend, turnover_rows)
    blue_diamond_email_rows = enrich_signal_turnover(blue_diamond, turnover_rows)
    rendered = render_daily_screening_email(
        as_of_date=session,
        monthly_reversal_rows=monthly_email_rows,
        trend_reacceleration_rows=trend_email_rows,
        blue_diamond_rows=blue_diamond_email_rows,
        daily_watch_3_rows=daily_watch_3,
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
        blue_diamond_signal_count=len(blue_diamond),
        blue_diamond_candidate_tickers=tuple(
            str(value) for value in blue_diamond["ticker"]
        ),
        daily_watch_3_rps_candidate_count=int(
            daily_watch_3.attrs["rps_candidate_count"]
        ),
        daily_watch_3_signal_count=len(daily_watch_3),
        daily_watch_3_candidate_tickers=tuple(
            str(value) for value in daily_watch_3["ticker"]
        ),
        subject=rendered.subject,
        dry_run=dry_run,
    )
    LOGGER.info(
        "Monthly Reversal signals=%d; Trend Re-acceleration signals=%d; "
        "Blue Diamond signals=%d; Daily Watch 3 signals=%d",
        len(monthly),
        len(trend),
        len(blue_diamond),
        len(daily_watch_3),
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
        description="Persist daily RPS and email all production screening signals.",
    )
    parser.add_argument("--as-of-date", type=_parse_date)
    parser.add_argument("--prices-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rps-root", type=Path, default=DEFAULT_RPS_ROOT)
    parser.add_argument("--market-cap-root", type=Path, default=DEFAULT_MARKET_CAP_ROOT)
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
            market_cap_root=args.market_cap_root,
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
