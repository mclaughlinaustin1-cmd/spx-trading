import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPX 15-Min Live Dashboard", layout="wide")
st.title("📈 SPX Live 15-Min Candles Dashboard")

placeholder = st.empty()
refresh_interval = 30  # seconds

for _ in range(1000):
    with placeholder.container():
        # Fetch last 2 days of 15-min candles
        spx = yf.download("^GSPC", period="2d", interval="15m", progress=False)

        # Ensure spx is a DataFrame
        if spx is None or not isinstance(spx, pd.DataFrame) or spx.empty:
            st.warning("No data returned from yfinance. Waiting for next refresh...")
            time.sleep(refresh_interval)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(spx.columns, pd.MultiIndex):
            spx.columns = [col[1] if col[1] else col[0] for col in spx.columns]

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in spx.columns for col in required_cols):
            st.warning("Missing required columns. Waiting for next refresh...")
            time.sleep(refresh_interval)
            continue

        # Keep only required columns and convert to numeric
        spx = spx[required_cols].apply(pd.to_numeric, errors='coerce')
        spx = spx.dropna(subset=['Close'])

        if len(spx) < 5:
            st.warning("Not enough valid data yet. Waiting for next refresh...")
            time.sleep(refresh_interval)
            continue

        spx = spx.tail(5)

        # Indicators
        spx["EMA_9"] = spx["Close"].ewm(span=9, adjust=False).mean()
        spx["EMA_21"] = spx["Close"].ewm(span=21, adjust=False).mean()
        spx["ATR"] = (spx["High"] - spx["Low"]).rolling(14).mean()

        # Safely get latest scalar values
        latest_price = float(spx["Close"].iloc[-1])
        ema_fast = float(spx["EMA_9"].iloc[-1])
        ema_slow = float(spx["EMA_21"].iloc[-1])
        atr = float(spx["ATR"].iloc[-1]) if pd.notna(spx["ATR"].iloc[-1]) else 0.0

        # Trade signal logic
        do_not_trade = False
        do_not_trade_reason = ""
        recent_return = (latest_price - float(spx["Close"].iloc[-2])) / float(spx["Close"].iloc[-2])

        if atr / latest_price > 0.008:
            do_not_trade = True
            do_not_trade_reason = "High volatility"

        if abs(recent_return) < 0.0005:
            do_not_trade = True
            do_not_trade_reason = "Choppy market"

        score = 0
        score += 1 if ema_fast > ema_slow else -1
        score += 1 if recent_return > 0 else -1

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

        # Entry / Target / Invalidation
        entry = latest_price
        target = entry + atr if "LONG" in signal else entry - atr
        invalidation = entry - atr*0.75 if "LONG" in signal else entry + atr*0.75

        # Display metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("EMA 9", f"{ema_fast:.2f}")
        c3.metric("EMA 21", f"{ema_slow:.2f}")
        c4.metric("Signal", signal)
        c5.metric("Confidence", confidence)

        st.subheader("Trade Levels (15-Min Horizon)")
        st.markdown(f"- Entry: {entry:.2f}\n- Target: {target:.2f}\n- Invalidation: {invalidation:.2f}")
        if do_not_trade:
            st.warning(f"DO NOT TRADE: {do_not_trade_reason}")

        # Candlestick chart
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

        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_9"], line=dict(color='blue', width=2), name="EMA 9"))
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_21"], line=dict(color='orange', width=2), name="EMA 21"))

        fig.update_layout(
            title="SPX 15-Min Candles (Last 5 Bars)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)

