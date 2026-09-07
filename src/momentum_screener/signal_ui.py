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


def render_app() -> None:
    import pandas as pd
    import plotly.graph_objects as go
    import pyarrow as pa
    import streamlit as st

    from momentum_screener.signal_ui_data import (
        LocalFile,
        clip_price_window,
        combine_signals,
        discover_signal_csvs,
        local_price_files,
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

    with st.sidebar:
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
            # Reset dependent choices when a new collection is loaded.
            for key in ("signal_date", "strategy", "ticker"):
                st.session_state.pop(key, None)
        for level, message in st.session_state.get("load_reports", []):
            getattr(st, level)(message)

    signals = st.session_state.get("signals")
    if signals is None or signals.empty:
        st.info("Load one or more signal CSV files to begin.")
        return

    first_date = signals["session"].min().date()
    last_date = signals["session"].max().date()
    date_column, strategy_column, ticker_column = st.columns(3)
    with date_column:
        signal_date = st.date_input(
            "Signal Date",
            value=last_date,
            min_value=first_date,
            max_value=last_date,
            key="signal_date",
        )
    with strategy_column:
        strategy = st.selectbox(
            "Strategy", sorted(signals["strategy_id"].unique()), key="strategy"
        )
    day = signals.loc[signals["session"].eq(pd.Timestamp(signal_date))]
    if day.empty:
        st.info("No signals for this date.")
        return
    matches = day.loc[day["strategy_id"].eq(strategy)]
    if matches.empty:
        st.info("No signals for this strategy on this date.")
        return
    tickers = sorted(matches["ticker"].unique())
    if st.session_state.get("ticker") not in tickers:
        st.session_state.pop("ticker", None)
    with ticker_column:
        ticker = st.selectbox("Ticker", tickers, key="ticker")
    st.caption(f"{len(tickers)} ticker(s) for this date and strategy")
    st.text(f"Ticker: {ticker}    Signal Date: {signal_date}    Strategy: {strategy}")
    version = matches.loc[matches["ticker"].eq(ticker), "strategy_version"].iloc[0]
    if pd.notna(version):
        st.caption(f"Strategy Version: {version}")

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
