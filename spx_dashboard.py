import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPX Live Dashboard", layout="wide")
st.title("📊 SPX Live Trade Signals (15-Min Horizon)")

# Placeholder container for live data
placeholder = st.empty()

# Refresh loop: Streamlit Cloud safe
refresh_interval = 30  # seconds
max_iterations = 1000  # prevent infinite loop in dev

for _ in range(max_iterations):
    with placeholder.container():
        # -------------------------------
        # Load Market Data
        # -------------------------------
        spx = yf.download("^GSPC", period="5d", interval="15m", progress=False)
        vix = yf.download("^VIX", period="5d", interval="15m", progress=False)

        if spx.empty or vix.empty:
            st.warning("Data unavailable. Try again shortly.")
            continue

        # -------------------------------
        # Indicators
        # -------------------------------
        spx["EMA_9"] = spx["Close"].ewm(span=9).mean()
        spx["EMA_21"] = spx["Close"].ewm(span=21).mean()
        spx["ATR"] = (spx["High"] - spx["Low"]).rolling(14).mean()
        spx = spx.dropna()

        price = float(spx["Close"].iloc[-1])
        ema_fast = float(spx["EMA_9"].iloc[-1])
        ema_slow = float(spx["EMA_21"].iloc[-1])
        atr = float(spx["ATR"].iloc[-1])
        vix_level = float(vix["Close"].iloc[-1])
        recent_return = float((spx["Close"].iloc[-1] - spx["Close"].iloc[-3]) / spx["Close"].iloc[-3])

        # -------------------------------
        # Trade Logic
        # -------------------------------
        do_not_trade = False
        do_not_trade_reason = ""
        if atr / price > 0.008:
            do_not_trade = True
            do_not_trade_reason = "High volatility"
        if vix_level > 30:
            do_not_trade = True
            do_not_trade_reason = "Extreme volatility"
        if abs(recent_return) < 0.0005:
            do_not_trade = True
            do_not_trade_reason = "Choppy market"

        score = 0
        score += 1 if ema_fast > ema_slow else -1
        score += 1 if recent_return > 0 else -1
        if vix_level > 25:
            score -= 1

        if do_not_trade:
            signal = "🚫 DO NOT TRADE"
            confidence = "N/A"
        elif score >= 2:
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

        entry = price
        target = price + atr if "LONG" in signal else price - atr
        invalidation = price - atr * 0.75 if "LONG" in signal else price + atr * 0.75

        # -------------------------------
        # Display Metrics
        # -------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SPX Price", f"{price:.2f}")
        c2.metric("VIX", f"{vix_level:.2f}")
        c3.metric("Signal", signal)
        c4.metric("Confidence", confidence)

        st.subheader("Trade Levels (15-Min Horizon)")
        st.markdown(f"- Entry: {entry:.2f}\n- Target: {target:.2f}\n- Invalidation: {invalidation:.2f}")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spx.index, y=spx["Close"], name="SPX"))
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_9"], name="EMA 9"))
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_21"], name="EMA 21"))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Wait before refreshing
    time.sleep(refresh_interval)
