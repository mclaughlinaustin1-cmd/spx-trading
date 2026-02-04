import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="SPX 15-Min Live Dashboard", layout="wide")
st.title("📈 SPX Live 15-Min Candle Dashboard")

# -------------------------------
# Placeholder for live update
# -------------------------------
placeholder = st.empty()
refresh_interval = 30  # seconds

# -------------------------------
# Live update loop
# -------------------------------
for _ in range(1000):  # prevent infinite loop issues
    with placeholder.container():
        # -------------------------------
        # Load last 2 15-min candles
        # -------------------------------
        spx = yf.download("^GSPC", period="2d", interval="15m", progress=False)
        if spx.empty:
            st.warning("Data unavailable. Try again shortly.")
            continue
        spx = spx.tail(2)  # only last 2 candles

        # -------------------------------
        # Indicators
        # -------------------------------
        spx["EMA_9"] = spx["Close"].ewm(span=9, adjust=False).mean()
        spx["EMA_21"] = spx["Close"].ewm(span=21, adjust=False).mean()

        latest_price = spx["Close"].iloc[-1]
        ema_fast = spx["EMA_9"].iloc[-1]
        ema_slow = spx["EMA_21"].iloc[-1]

        # -------------------------------
        # Simple trade signal
        # -------------------------------
        if ema_fast > ema_slow:
            signal = "🟢 LONG"
        elif ema_fast < ema_slow:
            signal = "🔴 SHORT"
        else:
            signal = "⚪ NEUTRAL"

        # -------------------------------
        # Metrics display
        # -------------------------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("EMA 9", f"{ema_fast:.2f}")
        c3.metric("EMA 21", f"{ema_slow:.2f}")
        st.subheader(f"Trade Signal: {signal}")

        # -------------------------------
        # Candlestick chart of last 2 candles
        # -------------------------------
        fig = go.Figure(data=[go.Candlestick(
            x=spx.index,
            open=spx['Open'],
            high=spx['High'],
            low=spx['Low'],
            close=spx['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name="SPX"
        )])

        # Add EMAs
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_9"], line=dict(color='blue', width=2), name="EMA 9"))
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_21"], line=dict(color='orange', width=2), name="EMA 21"))

        fig.update_layout(
            title="SPX 15-Min Candles (Last 2 Bars)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Wait before next refresh
    time.sleep(refresh_interval)
