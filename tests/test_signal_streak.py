from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from momentum_screener.signal_store import (
    DATA_MODE,
    export_signal_csv,
    replace_signal_range,
)
from momentum_screener.signal_streak import (
    SignalStreakContextUnavailable,
    read_signal_streaks,
    refresh_signal_streak_range,
)
from momentum_screener.strategy_data import sessions_in_range

STRATEGY = "example_strategy"
GENERATED = datetime(2026, 5, 4, 22, tzinfo=UTC)


def _commit(
    root: Path,
    days: list[tuple[date, str, str, tuple[str, ...]]],
) -> None:
    coverage_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []
    for session, version, config_hash, tickers in days:
        metadata = {
            "session": session,
            "strategy_id": STRATEGY,
            "strategy_version": version,
            "generated_at": GENERATED,
            "config_hash": config_hash,
            "config_json": "{}",
            "universe_hash": "c" * 64,
            "input_fingerprint": None,
            "data_mode": DATA_MODE,
        }
        coverage_records.append(
            {**metadata, "status": "complete", "signal_count": len(tickers)}
        )
        signal_records.extend(
            {**metadata, "ticker": ticker, "signal": True} for ticker in tickers
        )
    signals = pd.DataFrame(
        signal_records,
        columns=[
            "session",
            "strategy_id",
            "strategy_version",
            "generated_at",
            "config_hash",
            "config_json",
            "universe_hash",
            "input_fingerprint",
            "data_mode",
            "ticker",
            "signal",
        ],
    )
    signals["signal"] = signals["signal"].astype("bool")
    replace_signal_range({STRATEGY: signals}, pd.DataFrame(coverage_records), root=root)


def test_xnys_streak_weekend_and_explicit_false_reset(tmp_path: Path) -> None:
    days = [
        (date(2026, 4, 30), "1", "a" * 64, ()),
        (date(2026, 5, 1), "1", "a" * 64, ("AAA",)),
        (date(2026, 5, 4), "1", "a" * 64, ("AAA",)),
        (date(2026, 5, 5), "1", "a" * 64, ()),
        (date(2026, 5, 6), "1", "a" * 64, ("AAA",)),
    ]
    _commit(tmp_path, days)

    refresh_signal_streak_range(
        days[0][0], days[-1][0], strategies=(STRATEGY,), signal_root=tmp_path
    )

    rows = read_signal_streaks(STRATEGY, days[0][0], days[-1][0], signal_root=tmp_path)
    assert rows[["session", "signal_streak"]].to_dict("records") == [
        {"session": date(2026, 5, 1), "signal_streak": 1},
        {"session": date(2026, 5, 4), "signal_streak": 2},
        {"session": date(2026, 5, 6), "signal_streak": 1},
    ]


def test_mid_streak_range_refresh_and_csv_keep_true_count(tmp_path: Path) -> None:
    sessions = sessions_in_range(date(2026, 4, 24), date(2026, 5, 1))
    _commit(
        tmp_path,
        [
            (session, "1", "a" * 64, () if index == 0 else ("AAA",))
            for index, session in enumerate(sessions)
        ],
    )
    refresh_signal_streak_range(
        sessions[0], sessions[-1], strategies=(STRATEGY,), signal_root=tmp_path
    )
    outside_before = read_signal_streaks(
        STRATEGY, sessions[1], sessions[-1], signal_root=tmp_path
    ).loc[lambda rows: rows["session"].isin((sessions[1], sessions[-1]))]

    refresh_signal_streak_range(
        sessions[2], sessions[-2], strategies=(STRATEGY,), signal_root=tmp_path
    )

    outside_after = read_signal_streaks(
        STRATEGY, sessions[1], sessions[-1], signal_root=tmp_path
    ).loc[lambda rows: rows["session"].isin((sessions[1], sessions[-1]))]
    pd.testing.assert_frame_equal(
        outside_after.reset_index(drop=True), outside_before.reset_index(drop=True)
    )
    current = read_signal_streaks(
        STRATEGY, sessions[-1], sessions[-1], signal_root=tmp_path
    )
    assert current["signal_streak"].tolist() == [5]
    destination = tmp_path / "export.csv"
    export_signal_csv(
        destination,
        sessions[-1],
        sessions[-1],
        strategies=(STRATEGY,),
        root=tmp_path,
    )
    exported = pd.read_csv(destination)
    assert exported["signal_streak"].tolist() == [5]


def test_changed_identity_recomputes_context_instead_of_continuing_old(
    tmp_path: Path,
) -> None:
    old_sessions = sessions_in_range(date(2026, 4, 24), date(2026, 4, 30))
    current = date(2026, 5, 1)
    _commit(
        tmp_path,
        [(session, "1", "a" * 64, ("AAA",)) for session in old_sessions]
        + [(current, "2", "b" * 64, ("AAA",))],
    )
    calls: list[tuple[date, date]] = []

    def recompute(
        start: date, end: date, strategy_ids: tuple[str, ...]
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        calls.append((start, end))
        assert strategy_ids == (STRATEGY,)
        expected = sessions_in_range(start, end)
        coverage = pd.DataFrame(
            {
                "session": expected,
                "strategy_id": STRATEGY,
                "strategy_version": "2",
                "config_hash": "b" * 64,
            }
        )
        signals = pd.DataFrame({"session": [expected[-1]], "ticker": ["AAA"]})
        return {STRATEGY: signals}, coverage

    refresh_signal_streak_range(
        current,
        current,
        strategies=(STRATEGY,),
        signal_root=tmp_path,
        context_recompute=recompute,
    )

    rows = read_signal_streaks(STRATEGY, current, current, signal_root=tmp_path)
    assert calls and rows["signal_streak"].tolist() == [2]


def test_coverage_gap_never_silently_continues(tmp_path: Path) -> None:
    _commit(
        tmp_path,
        [
            (date(2026, 4, 27), "1", "a" * 64, ("AAA",)),
            (date(2026, 4, 29), "1", "a" * 64, ("AAA",)),
            (date(2026, 4, 30), "1", "a" * 64, ("AAA",)),
            (date(2026, 5, 1), "1", "a" * 64, ("AAA",)),
        ],
    )
    called = False

    def unavailable(
        start: date, end: date, strategy_ids: tuple[str, ...]
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        nonlocal called
        called = True
        raise SignalStreakContextUnavailable("fixture has no recompute inputs")

    refresh_signal_streak_range(
        date(2026, 5, 1),
        date(2026, 5, 1),
        strategies=(STRATEGY,),
        signal_root=tmp_path,
        context_recompute=unavailable,
    )

    rows = read_signal_streaks(
        STRATEGY, date(2026, 5, 1), date(2026, 5, 1), signal_root=tmp_path
    )
    assert called and rows["signal_streak"].isna().all()
