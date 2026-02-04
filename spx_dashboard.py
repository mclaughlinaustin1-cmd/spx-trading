import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPY 15-Min Live Dashboard", layout="wide")
st.title("📈 SPY 15-Min Live Dashboard")

placeholder = st.empty()
refresh_interval = 30  # seconds

while True:
    with placeholder.container():
        # Download last 7 days of 15-min SPY data
        spy = yf.download("SPY", period="7d", interval="15m", progress=False)

        if spy is None or spy.empty:
            st.warning("No data yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        # Flatten MultiIndex if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [col[1] if col[1] else col[0] for col in spy.columns]

        # Use Adj Close as Close
        if "Close" not in spy.columns and "Adj Close" in spy.columns:
            spy.rename(columns={"Adj Close": "Close"}, inplace=True)

        if "Close" not in spy.columns:
            st.warning("No Close/Adj Close column available. Retrying...")
            time.sleep(refresh_interval)
            continue

        spy["Close"] = pd.to_numeric(spy["Close"], errors="coerce")
        spy = spy.dropna(subset=["Close"])

        if len(spy) < 5:
            st.warning("Not enough 15-min bars yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        spy = spy.tail(5)  # Last 5 bars

        # Indicators
        spy["EMA_9"] = spy["Close"].ewm(span=9, adjust=False).mean()
        spy["EMA_21"] = spy["Close"].ewm(span=21, adjust=False).mean()
        spy["ATR"] = spy["Close"].diff().abs().rolling(14).mean()

        latest_price = float(spy["Close"].iloc[-1])
        ema_fast = float(spy["EMA_9"].iloc[-1])
        ema_slow = float(spy["EMA_21"].iloc[-1])
        atr = float(spy["ATR"].iloc[-1]) if pd.notna(spy["ATR"].iloc[-1]) else 0.0

        # Simple signal
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

        # Entry / target / invalidation
        entry = latest_price
        target = entry + atr if "LONG" in signal else entry - atr
        invalidation = entry - atr * 0.75 if "LONG" in signal else entry + atr * 0.75

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("EMA 9", f"{ema_fast:.2f}")
        c3.metric("EMA 21", f"{ema_slow:.2f}")
        c4.metric("Signal", signal)
        c5.metric("Confidence", confidence)

        st.subheader("Trade Levels (15-Min Horizon)")
        st.markdown(f"- Entry: {entry:.2f}\n- Target: {target:.2f}\n- Invalidation: {invalidation:.2f}")

        if do_not_trade:
            st.warning("DO NOT TRADE: Market conditions not ideal")

        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy.index, y=spy["Close"], mode="lines+markers", name="Close", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_9"], mode="lines", name="EMA 9", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_21"], mode="lines", name="EMA 21", line=dict(color="green")))

        fig.update_layout(
            title="SPY 15-Min Close + EMA (Last 5 Bars)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)
