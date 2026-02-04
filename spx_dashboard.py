import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPY Simulated 15-Min Dashboard", layout="wide")
st.title("📈 SPY Simulated 15-Min Dashboard")

placeholder = st.empty()
refresh_interval = 30  # seconds

def simulate_15min(df):
    """Split daily OHLC into 15-min pseudo bars (26 bars per day)"""
    all_bars = []
    for _, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        bar_open = np.linspace(o, c, 26)  # linear open -> close
        bar_high = np.linspace(o, h, 26)
        bar_low = np.linspace(o, l, 26)
        bar_close = bar_open + np.random.uniform(-0.1, 0.1, 26)  # small noise
        times = pd.date_range(start=row.name, periods=26, freq="15T")
        day_bars = pd.DataFrame({"Open": bar_open, "High": bar_high, "Low": bar_low, "Close": bar_close}, index=times)
        all_bars.append(day_bars)
    return pd.concat(all_bars)

while True:
    with placeholder.container():
        # Download daily SPY data (guaranteed columns)
        spy_daily = yf.download("SPY", period="60d", interval="1d", progress=False)
        if spy_daily is None or spy_daily.empty:
            st.warning("No data yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        # Ensure columns exist
        for col in ["Open", "High", "Low", "Close"]:
            if col not in spy_daily.columns:
                st.warning(f"Missing column: {col}. Retrying...")
                time.sleep(refresh_interval)
                continue

        # Simulate 15-min bars
        spy = simulate_15min(spy_daily)
        spy["Close"] = pd.to_numeric(spy["Close"], errors="coerce")
        spy = spy.dropna(subset=["Close"])

        # Use last 100 simulated bars
        spy = spy.tail(100)

        # Indicators
        spy["EMA_9"] = spy["Close"].ewm(span=9, adjust=False).mean()
        spy["EMA_21"] = spy["Close"].ewm(span=21, adjust=False).mean()
        spy["ATR"] = spy["Close"].diff().abs().rolling(14).mean()

        latest_price = float(spy["Close"].iloc[-1])
        ema_fast = float(spy["EMA_9"].iloc[-1])
        ema_slow = float(spy["EMA_21"].iloc[-1])
        atr = float(spy["ATR"].iloc[-1]) if pd.notna(spy["ATR"].iloc[-1]) else 0.0

        # Trade logic
        do_not_trade = False
        recent_return = (latest_price - float(spy["Close"].iloc[-2])) / float(spy["Close"].iloc[-2])
        if atr / latest_price > 0.008 or abs(recent_return) < 0.0005:
            do_not_trade = True
            signal = "🚫 DO NOT TRADE"
            confidence = "N/A"
        else:
            score = 0
            score += 1 if ema_fast > ema_slow else -1
            score += 1 if recent_return > 0 else -1
            if score >= 2:
                signal = "🟢 LONG"
                confidence = "High"
            elif score == 1:
                signal = "🟡 LONG (Cautious)"
                confidence = "Medium"
            elif score == -1:
                signal = "🟠 SHORT (Cautious)"
                confidence = "Medium"
            else:
                signal = "🔴 SHORT"
                confidence = "High"

        entry = latest_price
        target = entry + atr if "LONG" in signal else entry - atr
        invalidation = entry - atr * 0.75 if "LONG" in signal else entry + atr * 0.75

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("Signal", signal)
        c3.metric("Confidence", confidence)
        c4.metric("ATR", f"{atr:.4f}")

        st.subheader("Trade Levels (Simulated 15-min)")
        st.markdown(f"- Entry: {entry:.2f}\n- Target: {target:.2f}\n- Invalidation: {invalidation:.2f}")
        if do_not_trade:
            st.warning("DO NOT TRADE: Market conditions not ideal")

        # Plot line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy.index, y=spy["Close"], mode="lines+markers", name="Close"))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_9"], mode="lines", name="EMA 9"))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_21"], mode="lines", name="EMA 21"))

        fig.update_layout(
            title="SPY Simulated 15-Min Close + EMA (Last 100 Bars)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)
