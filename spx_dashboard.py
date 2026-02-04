import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="SPY Daily Dashboard", layout="wide")
st.title("📈 SPY Daily Dashboard")

placeholder = st.empty()
refresh_interval = 30  # seconds

while True:
    with placeholder.container():
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)

        if spy is None or spy.empty:
            st.warning("No data yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [col[1] if col[1] else col[0] for col in spy.columns]

        # Ensure Close exists
        if "Close" not in spy.columns:
            if "Adj Close" in spy.columns:
                spy["Close"] = spy["Adj Close"]
            else:
                st.warning("No Close/Adj Close column available. Retrying...")
                time.sleep(refresh_interval)
                continue

        # Make sure Close is a 1D numeric Series
        close_series = pd.to_numeric(spy["Close"].squeeze(), errors="coerce")
        spy["Close"] = close_series
        spy = spy.dropna(subset=["Close"])

        if len(spy) < 5:
            st.warning("Not enough bars yet. Retrying...")
            time.sleep(refresh_interval)
            continue

        spy = spy.tail(30)  # Last 30 daily bars

        # Indicators
        spy["EMA_9"] = spy["Close"].ewm(span=9, adjust=False).mean()
        spy["EMA_21"] = spy["Close"].ewm(span=21, adjust=False).mean()
        spy["ATR"] = spy["Close"].diff().abs().rolling(14).mean()

        latest_price = float(spy["Close"].iloc[-1])
        ema_fast = float(spy["EMA_9"].iloc[-1])
        ema_slow = float(spy["EMA_21"].iloc[-1])
        atr = float(spy["ATR"].iloc[-1]) if pd.notna(spy["ATR"].iloc[-1]) else 0.0

        # Simple signal
        score = 0
        score += 1 if ema_fast > ema_slow else -1
        recent_return = (latest_price - float(spy["Close"].iloc[-2])) / float(spy["Close"].iloc[-2])
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

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Price", f"{latest_price:.2f}")
        c2.metric("Signal", signal)
        c3.metric("Confidence", confidence)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy.index, y=spy["Close"], mode="lines+markers", name="Close"))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_9"], mode="lines", name="EMA 9"))
        fig.add_trace(go.Scatter(x=spy.index, y=spy["EMA_21"], mode="lines", name="EMA 21"))

        fig.update_layout(
            title="SPY Daily Close + EMA (Last 30 Bars)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)
