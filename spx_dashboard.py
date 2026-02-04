import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPY Paper Trading Dashboard", layout="wide")
st.title("📈 SPY Paper Trading + 15-Min Dashboard")

# --- SIDEBAR ---
time_range = st.sidebar.selectbox("Select Time Range", ["6M", "3M", "1M", "2W", "1W", "24H"])
show_ema = st.sidebar.checkbox("Show EMAs", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 60, 30)

placeholder = st.empty()

# --- PAPER TRADING DATAFRAME ---
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=[
        "Entry Time", "Direction", "Entry", "Target", "Stop", "Exit Time", "Exit Price", "P&L"
    ])

# --- HELPER FUNCTIONS ---
def get_period(range_str):
    mapping = {"6M":"6mo","3M":"3mo","1M":"1mo","2W":"1mo","1W":"1mo","24H":"1mo"}
    return mapping.get(range_str,"1mo")

def simulate_15min(df, bars_per_day=26):
    all_bars = []
    for _, row in df.iterrows():
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        bar_open = np.linspace(o, c, bars_per_day)
        bar_high = np.linspace(o, h, bars_per_day)
        bar_low = np.linspace(o, l, bars_per_day)
        noise = np.random.uniform(-0.1, 0.1, bars_per_day)
        bar_close = (bar_open + noise).flatten()
        times = pd.date_range(start=row.name, periods=bars_per_day, freq="15T")
        day_bars = pd.DataFrame({
            "Open": bar_open.flatten(),
            "High": bar_high.flatten(),
            "Low": bar_low.flatten(),
            "Close": bar_close
        }, index=times)
        all_bars.append(day_bars)
    return pd.concat(all_bars)

def calculate_signal(latest_price, ema_fast, ema_slow, recent_return, atr):
    do_not_trade = False
    if atr / latest_price > 0.008 or abs(recent_return) < 0.0005:
        return "🚫 DO NOT TRADE", "N/A", True
    score = 0
    score += 1 if ema_fast > ema_slow else -1
    score += 1 if recent_return > 0 else -1
    if score >= 2:
        return "LONG", "High", False
    elif score == 1:
        return "LONG (Cautious)", "Medium", False
    elif score == -1:
        return "SHORT (Cautious)", "Medium", False
    else:
        return "SHORT", "High", False

def execute_trade(signal, latest_price, atr, current_time):
    entry = latest_price
    target = entry + atr if "LONG" in signal else entry - atr
    stop = entry - atr*0.75 if "LONG" in signal else entry + atr*0.75
    st.session_state.trades.loc[len(st.session_state.trades)] = [
        current_time, signal, entry, target, stop, None, None, None
    ]

def update_trades(latest_price, current_time):
    trades = st.session_state.trades
    for idx, row in trades.iterrows():
        if pd.notna(row["Exit Price"]):
            continue
        if "LONG" in row["Direction"]:
            if latest_price >= row["Target"]:
                trades.at[idx, "Exit Price"] = row["Target"]
                trades.at[idx, "Exit Time"] = current_time
                trades.at[idx, "P&L"] = row["Target"] - row["Entry"]
            elif latest_price <= row["Stop"]:
                trades.at[idx, "Exit Price"] = row["Stop"]
                trades.at[idx, "Exit Time"] = current_time
                trades.at[idx, "P&L"] = row["Stop"] - row["Entry"]
        elif "SHORT" in row["Direction"]:
            if latest_price <= row["Target"]:
                trades.at[idx, "Exit Price"] = row["Target"]
                trades.at[idx, "Exit Time"] = current_time
                trades.at[idx, "P&L"] = row["Entry"] - row["Target"]
            elif latest_price >= row["Stop"]:
                trades.at[idx, "Exit Price"] = row["Stop"]
                trades.at[idx, "Exit Time"] = current_time
                trades.at[idx, "P&L"] = row["Entry"] - row["Stop"]

# --- MAIN LOOP ---
while True:
    with placeholder.container():
        period = get_period(time_range)
        spy_daily = yf.download("SPY", period=period, interval="1d", progress=False)
        if spy_daily.empty:
            st.warning("No data yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        if time_range == "2W": spy_daily = spy_daily.tail(14)
        elif time_range == "1W": spy_daily = spy_daily.tail(7)
        elif time_range == "24H": spy_daily = spy_daily.tail(1)

        spy = simulate_15min(spy_daily)
        spy = spy.dropna(subset=["Close"])
        spy = spy.tail(100)

        spy["EMA_9"] = spy["Close"].ewm(span=9, adjust=False).mean()
        spy["EMA_21"] = spy["Close"].ewm(span=21, adjust=False).mean()
        spy["ATR"] = spy["Close"].diff().abs().rolling(14).mean()

        latest_price = float(spy["Close"].iloc[-1])
        ema_fast = float(spy["EMA_9"].iloc[-1])
        ema_slow = float(spy["EMA_21"].iloc[-1])
        atr = float(spy["ATR"].iloc[-1]) if pd.notna(spy["ATR"].iloc[-1]) else 0.0
        recent_return = (latest_price - float(spy["Close"].iloc[-2])) / float(spy["Close"].iloc[-2])

        signal, confidence, do_not_trade = calculate_signal(latest_price, ema_fast, ema_slow, recent_return, atr)

        # --- Safe check for last trade ---
        last_trade_closed = True
        if not st.session_state.trades.empty:
            last_trade_closed = pd.notna(st.session_state.trades.iloc[-1]["Exit Price"])

        if not do_not_trade and last_trade_closed:
            execute_trade(signal, latest_price, atr, spy.index[-1])

        # Update trades
        update_trades(latest_price, spy.index[-1])

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("Signal", signal)
        c3.metric("Confidence", confidence)
        c4.metric("ATR", f"{atr:.4f}")

        # P&L
        st.subheader("Paper P&L")
        trades_display = st.session_state.trades.copy()
        trades_display["P&L"] = trades_display["P&L"].fillna("Open")
        st.dataframe(trades_display)

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy.index, y=spy["Close"], mode="lines+markers", name="Close"))
        if show_ema:
            fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_9"], mode="lines", name="EMA 9"))
            fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_21"], mode="lines", name="EMA 21"))

        # Add entry/exit markers
        for _, t in trades_display.iterrows():
            if pd.notna(t["Entry"]):
                fig.add_trace(go.Scatter(
                    x=[t["Entry Time"]], y=[t["Entry"]],
                    mode="markers+text",
                    marker=dict(color="green" if "LONG" in t["Direction"] else "red", size=12),
                    text=["Entry"], textposition="top center", name="Entry"
                ))
            if pd.notna(t["Exit Price"]):
                fig.add_trace(go.Scatter(
                    x=[t["Exit Time"]], y=[t["Exit Price"]],
                    mode="markers+text",
                    marker=dict(color="gold" if isinstance(t["P&L"], (int,float)) and t["P&L"]>0 else "black", size=12),
                    text=[f"P&L: {t['P&L']}"], textposition="bottom center", name="Exit"
                ))

        fig.update_layout(title=f"SPY Simulated 15-Min Close + EMA ({time_range} View)",
                          xaxis_title="Time", yaxis_title="Price", height=600, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)
