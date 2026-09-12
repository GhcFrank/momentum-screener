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
    from momentum_screener.signal_ui_data import (
        LocalFile,
        build_ticker_rps_table,
        clip_price_window,
        combine_signals,
        discover_signal_csvs,
        enrich_ticker_table_with_metadata,
        filter_tickers_by_strategies,
        load_company_metadata_for_ui,
        load_rps_for_session,
        load_turnover_for_session,
        local_company_metadata_file,
        local_market_cap_files,
        local_price_files,
        local_rps_files,
        read_local_prices,
        read_signal_csv,
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
            # Reset date/row selection on reload; preserve only strategies that
            # still exist in the new collection, including an empty selection.
            st.session_state["selected_strategies"] = [
                item
                for item in st.session_state.get("selected_strategies", [])
                if item in set(combined["strategy_id"])
            ]
            for key in ("signal_date", "ticker_table_context"):
                st.session_state.pop(key, None)
        for level, message in st.session_state.get("load_reports", []):
            getattr(st, level)(message)

    signals = st.session_state.get("signals")
    if signals is None or signals.empty:
        st.info("Load one or more signal CSV files to begin.")
        return

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
    tickers = filter_tickers_by_strategies(signals, signal_date, selected_strategies)
    table_context = (signal_date, tuple(sorted(selected_strategies)), tuple(tickers))
    if st.session_state.get("ticker_table_context") != table_context:
        # Dataframe selection state is read-only. A new widget key discards old
        # row positions when date, strategies, results, or loaded CSVs change.
        st.session_state["ticker_table_context"] = table_context
        st.session_state["ticker_table_revision"] = (
            st.session_state.get("ticker_table_revision", 0) + 1
        )
    if not selected_strategies:
        st.info("Select at least one strategy.")
        return
    if not signals["session"].eq(pd.Timestamp(signal_date)).any():
        st.info("No signals for this date.")
        return
    if not tickers:
        st.info(f"No matching signals for the selected strategies on {signal_date}.")
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
    st.caption(f"{len(table)} matching tickers")
    st.caption(
        "Forward performance uses raw signal Close and future High/Low; incomplete 40/120-session windows use available data."
    )
    selection = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        placeholder="N/A",
        column_config={
            name: st.column_config.NumberColumn(
                name,
                format=(
                    "%+.1f%%"
                    if name in forward_columns.values()
                    else "%.1f%%"
                    if name == "Turnover"
                    else "%.1f"
                ),
            )
            for name in table.columns
            if name not in {"Ticker", "Sector", "Industry"}
        },
        on_select="rerun",
        selection_mode="single-row",
        key=f"ticker_results:{st.session_state['ticker_table_revision']}",
    )
    rows = selection.selection.rows
    if not rows or not 0 <= rows[0] < len(table):
        st.info("Select a ticker row to view its price chart.")
        return
    # Streamlit returns original integer row positions even after client sorting.
    selected_row = table.iloc[rows[0]]
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
