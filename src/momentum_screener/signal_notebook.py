"""Local persistence for confirmed Historical Signal Research selections."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

NOTEBOOK_COLUMNS = [
    "signal_date",
    "strategy_ids",
    "strategy_name",
    "ticker",
    "saved_at",
]
NOTEBOOK_DIRECTORY = "notebook"
NOTEBOOK_FILENAME = "signal_notebook.csv"


def notebook_path(research_root: str | Path) -> Path:
    """Return the notebook path below, never directly inside, a research root."""

    return (
        Path(research_root).expanduser().resolve()
        / NOTEBOOK_DIRECTORY
        / NOTEBOOK_FILENAME
    )


def empty_signal_notebook() -> pd.DataFrame:
    """Return an empty notebook with the persisted schema."""

    return pd.DataFrame(
        {column: pd.Series(dtype="string") for column in NOTEBOOK_COLUMNS}
    )


def _canonical_strategy_ids(strategy_ids: Sequence[str] | str) -> tuple[str, ...]:
    values = strategy_ids.split("|") if isinstance(strategy_ids, str) else strategy_ids
    canonical = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if not canonical:
        raise ValueError("Notebook context requires at least one strategy id.")
    return canonical


def normalize_notebook_context(
    signal_date: date | str,
    strategy_ids: Sequence[str] | str,
    display_names: Mapping[str, str],
) -> tuple[str, str, str]:
    """Return canonical date, sorted strategy identity, and display label."""

    session = (
        signal_date
        if isinstance(signal_date, date)
        else date.fromisoformat(str(signal_date))
    )
    canonical_ids = _canonical_strategy_ids(strategy_ids)
    return (
        session.isoformat(),
        "|".join(canonical_ids),
        " + ".join(display_names.get(strategy_id, strategy_id) for strategy_id in canonical_ids),
    )


def _sort_notebook(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.sort_values(
        ["signal_date", "strategy_ids", "ticker"],
        ascending=[False, True, True],
        ignore_index=True,
    ).loc[:, NOTEBOOK_COLUMNS]
    return sorted_frame.astype(
        {column: "string" for column in NOTEBOOK_COLUMNS}
    )


def _normalize_notebook(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in NOTEBOOK_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            f"Notebook is missing required column(s) {missing}: {path}"
        )
    unexpected = [column for column in frame if column not in NOTEBOOK_COLUMNS]
    if unexpected:
        raise ValueError(f"Notebook has unexpected column(s) {unexpected}: {path}")

    normalized = frame.loc[:, NOTEBOOK_COLUMNS].copy()
    for column in NOTEBOOK_COLUMNS:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or normalized[column].eq("").any():
            raise ValueError(f"Notebook contains an empty {column} value: {path}")

    parsed_dates = pd.to_datetime(
        normalized["signal_date"], format="%Y-%m-%d", errors="coerce"
    )
    if parsed_dates.isna().any():
        raise ValueError(f"Notebook contains an invalid signal_date: {path}")
    normalized["signal_date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    try:
        normalized["strategy_ids"] = normalized["strategy_ids"].map(
            lambda value: "|".join(_canonical_strategy_ids(value))
        )
    except ValueError as exc:
        raise ValueError(f"Notebook contains an invalid strategy_ids value: {path}") from exc

    normalized["ticker"] = normalized["ticker"].str.upper()
    timestamps = pd.to_datetime(normalized["saved_at"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"Notebook contains an invalid saved_at value: {path}")
    duplicated = normalized.duplicated(
        ["signal_date", "strategy_ids", "ticker"], keep=False
    )
    if duplicated.any():
        raise ValueError(f"Notebook contains duplicate ticker rows: {path}")
    return _sort_notebook(normalized)


def read_signal_notebook(path: str | Path) -> pd.DataFrame:
    """Read and validate a notebook; a missing file is an empty notebook."""

    source = Path(path).expanduser()
    if not source.exists():
        return empty_signal_notebook()
    frame = pd.read_csv(
        source,
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    return _normalize_notebook(frame, source)


def _write_signal_notebook_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            frame.to_csv(output, index=False, lineterminator="\n")
            output.flush()
            os.fsync(output.fileno())
        read_signal_notebook(temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_notebook_context(
    path: str | Path,
    *,
    signal_date: date | str,
    strategy_ids: Sequence[str],
    display_names: Mapping[str, str],
    tickers: Sequence[str],
    saved_at: datetime | None = None,
) -> pd.DataFrame:
    """Atomically replace all ticker rows for one date and strategy context."""

    destination = Path(path).expanduser()
    existing = read_signal_notebook(destination)
    session, canonical_ids, strategy_name = normalize_notebook_context(
        signal_date, strategy_ids, display_names
    )
    visible_tickers = sorted(
        {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
    )
    if not visible_tickers:
        raise ValueError("Notebook save requires at least one visible ticker.")

    timestamp = saved_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("saved_at must be timezone-aware.")
    saved_at_text = (
        timestamp.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    keep = ~(
        existing["signal_date"].eq(session)
        & existing["strategy_ids"].eq(canonical_ids)
    )
    replacement = pd.DataFrame(
        {
            "signal_date": session,
            "strategy_ids": canonical_ids,
            "strategy_name": strategy_name,
            "ticker": visible_tickers,
            "saved_at": saved_at_text,
        }
    ).astype("string")
    updated = _sort_notebook(pd.concat([existing.loc[keep], replacement]))
    _write_signal_notebook_atomically(updated, destination)
    return updated
