"""Focused local persistence tests for the Signal Research Notebook."""

from datetime import UTC, date, datetime

from momentum_screener.signal_notebook import (
    NOTEBOOK_COLUMNS,
    notebook_path,
    read_signal_notebook,
    replace_notebook_context,
)

DISPLAY_NAMES = {
    "daily_watch_3": "每日观察选股3",
    "trend_reacceleration": "顺向火车2",
}
SAVED_AT = datetime(2026, 10, 2, 19, 20, tzinfo=UTC)


def save(path, session, strategies, tickers):
    return replace_notebook_context(
        path,
        signal_date=session,
        strategy_ids=strategies,
        display_names=DISPLAY_NAMES,
        tickers=tickers,
        saved_at=SAVED_AT,
    )


def test_new_notebook_uses_isolated_subdirectory_and_creates_on_first_save(tmp_path):
    path = notebook_path(tmp_path / "research")

    empty = read_signal_notebook(path)
    assert list(empty.columns) == NOTEBOOK_COLUMNS
    assert empty.empty
    assert path == (tmp_path / "research" / "notebook" / "signal_notebook.csv")

    saved = save(path, date(2026, 10, 1), ["trend_reacceleration"], ["B", "A"])
    assert path.is_file()
    assert saved["ticker"].tolist() == ["A", "B"]


def test_save_replaces_same_context_without_duplicates(tmp_path):
    path = notebook_path(tmp_path)
    save(path, date(2026, 10, 1), ["trend_reacceleration"], ["A", "B", "C"])

    updated = save(
        path, date(2026, 10, 1), ["trend_reacceleration"], ["A", "C", "C"]
    )

    assert updated[["signal_date", "strategy_ids", "ticker"]].to_dict("records") == [
        {
            "signal_date": "2026-10-01",
            "strategy_ids": "trend_reacceleration",
            "ticker": "A",
        },
        {
            "signal_date": "2026-10-01",
            "strategy_ids": "trend_reacceleration",
            "ticker": "C",
        },
    ]
    assert read_signal_notebook(path).equals(updated)


def test_save_keeps_date_and_strategy_contexts_isolated(tmp_path):
    path = notebook_path(tmp_path)
    save(path, date(2026, 10, 1), ["trend_reacceleration"], ["A"])
    save(path, date(2026, 10, 2), ["trend_reacceleration"], ["B"])
    updated = save(path, date(2026, 10, 1), ["daily_watch_3"], ["C"])

    assert set(
        updated[["signal_date", "strategy_ids", "ticker"]].itertuples(
            index=False, name=None
        )
    ) == {
        ("2026-10-01", "trend_reacceleration", "A"),
        ("2026-10-02", "trend_reacceleration", "B"),
        ("2026-10-01", "daily_watch_3", "C"),
    }


def test_multi_strategy_order_is_one_replace_context(tmp_path):
    path = notebook_path(tmp_path)
    save(
        path,
        date(2026, 10, 1),
        ["trend_reacceleration", "daily_watch_3"],
        ["A", "B"],
    )
    updated = save(
        path,
        date(2026, 10, 1),
        ["daily_watch_3", "trend_reacceleration"],
        ["C"],
    )

    assert updated[["strategy_ids", "strategy_name", "ticker"]].to_dict(
        "records"
    ) == [
        {
            "strategy_ids": "daily_watch_3|trend_reacceleration",
            "strategy_name": "每日观察选股3 + 顺向火车2",
            "ticker": "C",
        }
    ]
