"""Launch the offline Historical Signal Research UI with python -m."""

from __future__ import annotations

import errno
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
DEFAULT_SIGNAL_PATH = "/home/gooder/momentum-screener-research/"
STRATEGY_DISPLAY_NAMES = {
    "blue_diamond": "蓝色钻石",
    "blue_diamond_core": "蓝色钻石 Core",
    "daily_watch_3": "每日观察选股3",
    "daily_watch_3_core": "每日观察选股3 Core",
    "trend_reacceleration": "顺向火车2",
    "trend_reacceleration_entry": "顺向火车2 · 首次触发（实验）",
}


def render_app() -> None:
    import pandas as pd
    import plotly.graph_objects as go
    import pyarrow as pa
    import streamlit as st

    from momentum_screener.company_metadata import DEFAULT_METADATA_PATH
    from momentum_screener.forward_performance import (
        calculate_forward_performance_for_signals,
    )
    from momentum_screener.local_price_data import price_files_in_range
    from momentum_screener.market_cap_storage import MarketCapStorageError
    from momentum_screener.rps_storage import RpsStorageError
    from momentum_screener.signal_notebook import (
        normalize_notebook_context,
        notebook_path,
        read_signal_notebook,
        replace_notebook_context,
    )
    from momentum_screener.signal_ui_data import (
        SELECTION_CLEAR_MARKER_COLUMN,
        LocalFile,
        apply_hide_selection,
        build_signal_streak_table,
        build_ticker_rps_table,
        clip_price_window,
        combine_signals,
        consume_selection_clear_pending,
        discover_signal_csvs,
        enrich_ticker_table_with_metadata,
        filter_hidden_tickers,
        filter_tickers_by_strategies,
        hidden_ticker_context,
        load_company_metadata_for_ui,
        load_rps_for_session,
        load_turnover_for_session,
        local_company_metadata_file,
        local_market_cap_files,
        local_price_files,
        local_rps_files,
        read_local_prices,
        read_signal_csv,
        ticker_table_context,
        ticker_table_key,
        ticker_table_selection_clear_key,
    )
    from momentum_screener.storage_manifest import ManifestError

    st.set_page_config(page_title="Signal Research", layout="wide")
    st.title("Historical Signal Research")
    st.caption("Local CSV signals and local marketData · Adjusted Close")
    # All cache arguments participate in hashing, including path and nanosecond
    # file timestamps. Widget reruns keep the loaded signal frame in the session.
    cached_csv = st.cache_data(read_signal_csv, show_spinner=False, max_entries=32)
    cached_prices = st.cache_data(read_local_prices, show_spinner=False, max_entries=64)
    cached_rps = st.cache_data(load_rps_for_session, show_spinner=False, max_entries=32)
    cached_metadata = st.cache_data(
        load_company_metadata_for_ui, show_spinner=False, max_entries=4
    )
    cached_turnover = st.cache_data(
        load_turnover_for_session, show_spinner=False, max_entries=32
    )
    cached_forward = st.cache_data(
        calculate_forward_performance_for_signals, show_spinner=False, max_entries=32
    )

    with st.sidebar:
        if "signal_paths_input" not in st.session_state:
            st.session_state["signal_paths_input"] = DEFAULT_SIGNAL_PATH
        paths_text = st.text_area(
            "Signal CSV files or folders",
            key="signal_paths_input",
            height=160,
            placeholder="/path/to/signals_folder\n/path/to/signal.csv",
            help=(
                "One local CSV file or directory per line. Directories load only their "
                "immediate CSV files; subdirectories are not scanned. "
                "Relative paths use the launch directory."
            ),
        )
        load = st.button("Load Signals", type="primary")
        if load:
            paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
            csv_paths, discovery_warnings = discover_signal_csvs(paths)
            frames, metadata = [], []
            reports = [("warning", warning) for warning in discovery_warnings]
            for path in csv_paths:
                try:
                    source = LocalFile.inspect(path)
                    frame, warnings = cached_csv(source)
                    frames.append(frame)
                    metadata.append(source)
                    reports.append(("success", f"Loaded: {path} ({len(frame)} rows)"))
                    reports.extend(
                        ("warning", f"{path}: {warning}") for warning in warnings
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    # One unreadable file must not make other input files unusable.
                    reports.append(("error", f"Error: {path}: {exc}"))
            combined, warnings = combine_signals(frames)
            reports.extend(("warning", warning) for warning in warnings)
            reports.insert(0, ("info", f"Loaded {len(metadata)} CSV files"))
            if not paths:
                reports.append(
                    ("warning", "Enter at least one local CSV file or directory path.")
                )
            st.session_state["signals"] = combined
            st.session_state["signal_files"] = metadata
            st.session_state["load_reports"] = reports
            if metadata:
                research_root = Path(metadata[0].path).parent.resolve()
                signal_notebook_path = notebook_path(research_root)
                st.session_state["research_root"] = research_root
                st.session_state["signal_notebook_path"] = signal_notebook_path
                try:
                    st.session_state["signal_notebook"] = read_signal_notebook(
                        signal_notebook_path
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    st.session_state["signal_notebook"] = None
                    st.session_state["signal_notebook_error"] = str(exc)
                else:
                    st.session_state.pop("signal_notebook_error", None)
            else:
                for key in (
                    "research_root",
                    "signal_notebook_path",
                    "signal_notebook",
                    "signal_notebook_error",
                ):
                    st.session_state.pop(key, None)
            st.session_state["signal_load_revision"] = (
                st.session_state.get("signal_load_revision", 0) + 1
            )
            # Reset date/row selection on reload; preserve only strategies that
            # still exist in the new collection, including an empty selection.
            st.session_state["selected_strategies"] = [
                item
                for item in st.session_state.get("selected_strategies", [])
                if item in set(combined["strategy_id"])
            ]
            for key in (
                "signal_date",
                "ticker_table_context",
                "ticker_table_revision",
                "hidden_tickers_by_context",
            ):
                st.session_state.pop(key, None)
        for level, message in st.session_state.get("load_reports", []):
            getattr(st, level)(message)

    signals = st.session_state.get("signals")
    if signals is None or signals.empty:
        st.info("Load one or more signal CSV files to begin.")
        return

    def render_research_notebook() -> None:
        notebook = st.session_state.get("signal_notebook")
        error = st.session_state.get("signal_notebook_error")
        path = st.session_state.get("signal_notebook_path")
        with st.expander("Research Notebook"):
            if path is not None:
                st.caption(str(path))
            if error:
                st.error(f"Unable to read Research Notebook: {error}")
                return
            if notebook is None:
                st.info("Load signals from a research directory to view its notebook.")
                return

            if notebook.empty:
                st.info("Research Notebook is empty.")
            else:
                summary = (
                    notebook.groupby(
                        ["signal_date", "strategy_ids", "strategy_name"],
                        as_index=False,
                        sort=False,
                    )
                    .size()
                    .rename(
                        columns={
                            "signal_date": "Signal Date",
                            "strategy_name": "Strategy",
                            "size": "Tickers",
                        }
                    )
                    .sort_values(
                        ["Signal Date", "Strategy"],
                        ascending=[False, True],
                        ignore_index=True,
                    )
                )
                st.dataframe(
                    summary.loc[:, ["Signal Date", "Strategy", "Tickers"]],
                    hide_index=True,
                    width="stretch",
                )
                details = notebook.rename(
                    columns={
                        "signal_date": "Signal Date",
                        "strategy_name": "Strategy",
                        "ticker": "Ticker",
                        "saved_at": "Saved At",
                    }
                )
                st.dataframe(
                    details.loc[
                        :, ["Signal Date", "Strategy", "Ticker", "Saved At"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            st.download_button(
                "Download Notebook CSV",
                data=notebook.to_csv(index=False, lineterminator="\n").encode(
                    "utf-8"
                ),
                file_name="signal_notebook.csv",
                mime="text/csv",
                disabled=notebook.empty,
                key=f"download_notebook:{path}",
            )

    first_date = signals["session"].min().date()
    last_date = signals["session"].max().date()
    signal_date = st.date_input(
        "Signal Date",
        value=last_date,
        min_value=first_date,
        max_value=last_date,
        key="signal_date",
    )
    selected_strategies = st.pills(
        "Strategies",
        sorted(signals["strategy_id"].unique()),
        format_func=lambda strategy_id: STRATEGY_DISPLAY_NAMES.get(
            strategy_id, strategy_id
        ),
        selection_mode="multi",
        key="selected_strategies",
    )
    matching_tickers = filter_tickers_by_strategies(
        signals, signal_date, selected_strategies
    )
    if not selected_strategies:
        st.info("Select at least one strategy.")
        render_research_notebook()
        return
    if not signals["session"].eq(pd.Timestamp(signal_date)).any():
        st.info("No signals for this date.")
        render_research_notebook()
        return
    if not matching_tickers:
        st.info(f"No matching signals for the selected strategies on {signal_date}.")
        render_research_notebook()
        return

    table_context = ticker_table_context(
        st.session_state.get("signal_load_revision", 0),
        signal_date,
        selected_strategies,
    )
    table_widget_key = ticker_table_key(table_context)
    selection_clear_pending_key = ticker_table_selection_clear_key(table_context)
    st.session_state["ticker_table_context"] = table_context
    cleared_selection = consume_selection_clear_pending(
        st.session_state, selection_clear_pending_key
    )
    if cleared_selection is not None:
        # This must happen before st.dataframe receives the filtered rows.
        st.session_state[table_widget_key] = cleared_selection

    context_key = hidden_ticker_context(signal_date, selected_strategies)
    if "hidden_tickers_by_context" not in st.session_state:
        st.session_state["hidden_tickers_by_context"] = {}
    hidden_by_context = st.session_state["hidden_tickers_by_context"]
    hidden = set(hidden_by_context.get(context_key, ())).intersection(matching_tickers)
    tickers = filter_hidden_tickers(
        matching_tickers,
        hidden_by_context,
        signal_date,
        selected_strategies,
    )
    # Always reserve this element position. Conditionally inserting the button
    # would shift the dataframe's delta path after the first hide and remount
    # the frontend component even though its explicit key stayed unchanged.
    reset_hidden_slot = st.empty()
    if hidden and reset_hidden_slot.button(
        "Reset hidden",
        key=f"reset_hidden:{context_key[0]}:{'|'.join(context_key[1])}",
    ):
        hidden_by_context.pop(context_key, None)
        st.session_state[selection_clear_pending_key] = True
        st.rerun()
    if hidden:
        st.caption(f"{len(tickers)} visible · {len(hidden)} hidden")
    else:
        st.caption(f"{len(tickers)} matching tickers")

    notebook_file = st.session_state.get("signal_notebook_path")
    notebook = st.session_state.get("signal_notebook")
    notebook_error = st.session_state.get("signal_notebook_error")
    notebook_session, notebook_strategy_ids, notebook_strategy_name = (
        normalize_notebook_context(
            signal_date, selected_strategies, STRATEGY_DISPLAY_NAMES
        )
    )
    save_notebook = st.button(
        "Save / Update Notebook",
        disabled=(
            not tickers
            or notebook_file is None
            or notebook is None
            or notebook_error is not None
        ),
        key=f"save_notebook:{notebook_session}:{notebook_strategy_ids}",
    )
    notebook_status_slot = st.empty()
    notebook_feedback_slot = st.empty()
    if save_notebook:
        try:
            notebook = replace_notebook_context(
                notebook_file,
                signal_date=signal_date,
                strategy_ids=selected_strategies,
                display_names=STRATEGY_DISPLAY_NAMES,
                tickers=tickers,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            st.session_state["signal_notebook"] = None
            st.session_state["signal_notebook_error"] = str(exc)
            notebook_error = str(exc)
            notebook_feedback_slot.error(
                f"Unable to save Research Notebook: {exc}"
            )
        else:
            st.session_state["signal_notebook"] = notebook
            st.session_state.pop("signal_notebook_error", None)
            notebook_feedback_slot.success(
                f"Saved {len(tickers)} tickers\n\n"
                f"{notebook_session} · {notebook_strategy_name}"
            )
    if notebook_error:
        notebook_status_slot.error(
            f"Unable to read Research Notebook: {notebook_error}"
        )
    elif not tickers:
        notebook_status_slot.caption("No visible tickers to save.")
    elif notebook is not None:
        saved_count = int(
            (
                notebook["signal_date"].eq(notebook_session)
                & notebook["strategy_ids"].eq(notebook_strategy_ids)
            ).sum()
        )
        if saved_count:
            notebook_status_slot.caption(
                f"Notebook: {saved_count} saved tickers"
            )
    if not tickers:
        st.info("All matching tickers are hidden for this selection.")
        render_research_notebook()
        return

    try:
        snapshot = cached_rps(signal_date, local_rps_files(signal_date))
        if snapshot.empty:
            st.caption(f"No local RPS snapshot for {signal_date}; showing N/A.")
    except (
        OSError,
        ValueError,
        RpsStorageError,
        ManifestError,
        pa.ArrowException,
    ) as exc:
        st.warning(f"Local RPS unavailable; showing N/A. {exc}")
        snapshot = pd.DataFrame()
    table = build_ticker_rps_table(tickers, snapshot)
    try:
        metadata = cached_metadata(local_company_metadata_file(DEFAULT_METADATA_PATH))
    except (OSError, UnicodeError, ValueError) as exc:
        st.warning(f"Company metadata unavailable; showing N/A. {exc}")
        metadata = pd.DataFrame()
    table = enrich_ticker_table_with_metadata(table, metadata)
    streaks = build_signal_streak_table(
        tickers,
        signals,
        signal_date,
        selected_strategies,
        STRATEGY_DISPLAY_NAMES,
    )
    table = table.merge(streaks, on="Ticker", how="left", validate="one_to_one")
    streak_columns = [column for column in streaks if column != "Ticker"]
    other_columns = [
        column
        for column in table
        if column not in {"Ticker", "Sector", "Industry", *streak_columns}
    ]
    table = table.loc[
        :, ["Ticker", "Sector", "Industry", *streak_columns, *other_columns]
    ]
    try:
        turnover = cached_turnover(
            signal_date,
            tuple(tickers),
            price_files_in_range(signal_date, signal_date),
            local_market_cap_files(signal_date),
        ).set_index("ticker")
    except (
        OSError,
        ValueError,
        TypeError,
        MarketCapStorageError,
        ManifestError,
        pa.ArrowException,
    ) as exc:
        st.warning(f"Turnover unavailable; showing N/A. {exc}")
        turnover = pd.DataFrame(columns=["turnover"])
    table.insert(
        table.columns.get_loc("RPS250") + 1,
        "Turnover",
        pd.to_numeric(
            turnover["turnover"].reindex(table["Ticker"]), errors="coerce"
        ).to_numpy(dtype="float64")
        * 100,
    )
    forward_columns = {
        "forward_40d_max_drawdown": "40D DD",
        "forward_40d_max_gain": "40D Gain",
        "forward_120d_max_drawdown": "120D DD",
        "forward_120d_max_gain": "120D Gain",
    }
    try:
        performance = cached_forward(
            pd.DataFrame({"ticker": tickers, "session": signal_date}),
            files=price_files_in_range(signal_date),
        ).set_index("ticker")
    except (OSError, ValueError, TypeError, ManifestError, pa.ArrowException) as exc:
        st.warning(f"Forward performance unavailable; showing N/A. {exc}")
        performance = pd.DataFrame(columns=list(forward_columns))
    for metric, label in forward_columns.items():
        # Presentation-only conversion to percent; domain returns decimal ratios.
        table[label] = (
            performance[metric].reindex(table["Ticker"]).to_numpy(dtype="float64") * 100
        )
    st.caption(
        "Forward performance uses raw signal Close and future High/Low; incomplete 40/120-session windows use available data."
    )
    st.caption(
        "Use row checkboxes to hide tickers; select any data cell to view its price chart."
    )

    dataframe_table = table.assign(**{SELECTION_CLEAR_MARKER_COLUMN: False})
    dataframe_column_config = {
        name: st.column_config.NumberColumn(
            name,
            format=(
                "%d"
                if name == "Streak" or name.endswith(" Streak")
                else "%+.1f%%"
                if name in forward_columns.values()
                else "%.1f%%"
                if name == "Turnover"
                else "%.1f"
            ),
        )
        for name in table.columns
        if name not in {"Ticker", "Sector", "Industry"}
    }
    dataframe_column_config[SELECTION_CLEAR_MARKER_COLUMN] = None

    selection = st.dataframe(
        dataframe_table,
        hide_index=True,
        width="stretch",
        placeholder="N/A",
        column_config=dataframe_column_config,
        on_select="rerun",
        selection_mode=["multi-row", "single-cell"],
        key=table_widget_key,
    )
    rows = selection.selection.rows
    if rows and apply_hide_selection(
        hidden_by_context,
        context_key,
        table["Ticker"].astype(str).tolist(),
        rows,
    ):
        st.session_state[selection_clear_pending_key] = True
        st.rerun()
    render_research_notebook()
    cells = [
        cell
        for cell in selection.selection.cells
        if cell[1] != SELECTION_CLEAR_MARKER_COLUMN
    ]
    if not cells or not 0 <= cells[0][0] < len(table):
        st.info("Select a ticker row to view its price chart.")
        return
    # Streamlit returns original integer row positions even after client sorting.
    selected_row = table.iloc[cells[0][0]]
    ticker = selected_row["Ticker"]
    strategy = " + ".join(selected_strategies)
    st.text(f"Ticker: {ticker}    Signal Date: {signal_date}    Strategy: {strategy}")
    st.caption(f"{selected_row['Sector']} · {selected_row['Industry']}")
    details = signals.loc[
        signals["session"].eq(pd.Timestamp(signal_date))
        & signals["ticker"].eq(ticker)
        & signals["strategy_id"].isin(selected_strategies)
    ]
    versions = [
        f"{row.strategy_id}: {row.strategy_version}"
        for row in details.itertuples()
        if pd.notna(row.strategy_version)
    ]
    if versions:
        st.caption("Strategy Versions: " + " · ".join(versions))

    try:
        with st.spinner("Reading local price data…"):
            files = local_price_files(signal_date)
            prices = cached_prices(ticker, signal_date, files)
    except FileNotFoundError as exc:
        st.info(f"No local price data available for {ticker}.")
        st.caption(str(exc))
        return
    except (OSError, ValueError, TypeError, pa.ArrowException, ManifestError) as exc:
        st.error(f"Unable to read local price data for {ticker}: {exc}")
        return
    if prices.empty:
        st.info(f"No local price data available for {ticker}.")
        st.caption(
            "The ticker is absent or has no prices in the permitted -2 year / +1 year window."
        )
        return

    window = clip_price_window(
        signal_date, prices["date"].min().date(), prices["date"].max().date()
    )
    if window is None:
        st.info(f"No local price data available for {ticker} in this date range.")
        return
    st.caption(f"Available price data: {window.start} → {window.end}")
    if window.viewport_fallback:
        st.caption(
            "No price span inside the default viewport; showing the available window."
        )
    fig = go.Figure(
        go.Scatter(
            x=prices["date"],
            y=prices["adjusted_close"],
            mode="lines",
            name="Adjusted Close",
            line={"color": "red"},
            hovertemplate="%{x|%Y-%m-%d}<br>Adjusted Close: %{y:.2f}<extra></extra>",
        )
    )
    # A paper-height shape marks the exact calendar date even without a price
    # row that day, and remains attached to the date axis through zoom/pan.
    fig.add_vline(
        x=signal_date.isoformat(),
        line_color="blue",
        line_dash="dash",
        line_width=2,
    )
    axis = {
        "type": "date",
        "title": "Date",
        "rangeslider": {"visible": True, "range": [window.start, window.end]},
    }
    if window.start < window.end:
        axis.update(
            range=[window.viewport_start, window.viewport_end],
            minallowed=window.start,
            maxallowed=window.end,
        )
    fig.update_layout(
        xaxis=axis,
        yaxis_title="Adjusted Close",
        height=560,
        dragmode="zoom",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        key=f"price_chart:{signal_date}:{strategy}:{ticker}",
    )
    with st.expander("Price Data"):
        st.dataframe(prices, hide_index=True, width="stretch")


def _open_when_ready(process: subprocess.Popen, stopped: threading.Event) -> None:
    deadline = time.monotonic() + 30
    while (
        process.poll() is None and time.monotonic() < deadline and not stopped.is_set()
    ):
        try:
            with socket.create_connection((HOST, PORT), timeout=0.2):
                pass
        except OSError:
            stopped.wait(0.1)
            continue
        try:
            if not webbrowser.open(URL):
                print(f"Open {URL} in your browser.", flush=True)
        except (webbrowser.Error, OSError):
            print(f"Open {URL} in your browser.", flush=True)
        return


def main() -> int:
    """Run a fixed-port child and forward Ctrl+C only to that owned process."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((HOST, PORT))
    except OSError as exc:
        message = (
            f"Port {PORT} is already in use."
            if exc.errno == errno.EADDRINUSE
            else f"Cannot listen on {HOST}:{PORT}: {exc}"
        )
        print(message, file=sys.stderr)
        return 1

    print(f"Momentum Screener Signal UI\n{URL}\n\nPress Ctrl+C to stop.", flush=True)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--server.showEmailPrompt",
        "false",
        "--server.fileWatcherType",
        "none",
        "--server.baseUrlPath",
        "",
        "--browser.serverAddress",
        HOST,
        "--browser.serverPort",
        str(PORT),
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
        "--client.toolbarMode",
        "minimal",
        "--client.showErrorLinks",
        "false",
        "--logger.level",
        "warning",
        "--",
        "--signal-ui-app",
    ]
    process = subprocess.Popen(command, start_new_session=True)
    stopped = threading.Event()
    threading.Thread(
        target=_open_when_ready, args=(process, stopped), daemon=True
    ).start()
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        stopped.set()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    if "--signal-ui-app" in sys.argv:
        render_app()
    else:
        raise SystemExit(main())
