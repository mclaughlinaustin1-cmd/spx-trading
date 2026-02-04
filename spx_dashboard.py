# ==========================================
# SIMPLE SPX LIVE DASHBOARD (ROBUST VERSION)
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="SPX Dashboard", layout="wide")
st.title("📈 SPX Live Market Dashboard")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(ttl=300)
def load_data():
    spx = yf.download("^GSPC", period="5d", interval="15m", progress=False)
    vix = yf.download("^VIX", period="5d", interval="15m", progress=False)
    return spx, vix

spx, vix = load_data()

# -----------------------------
# Safety check
# -----------------------------
if spx.empty or vix.empty:
    st.warning("Market data not available. Please refresh in a moment.")
    st.stop()

# -----------------------------
# Indicators (simple & safe)
# -----------------------------
spx["EMA_9"] = spx["Close"].ewm(span=9).mean()
spx["EMA_21"] = spx["Close"].ewm(span=21).mean()

# Drop early NaNs
spx = spx.dropna()

latest_price = float(spx["Close"].iloc[-1])
ema_fast = float(spx["EMA_9"].iloc[-1])
ema_slow = float(spx["EMA_21"].iloc[-1])
latest_vix = float(vix["Close"].iloc[-1])

# -----------------------------
# Simple signal logic
# -----------------------------
if ema_fast > ema_slow:
    bias = "🟢 BULLISH"
elif ema_fast < ema_slow:
    bias = "🔴 BEARISH"
else:
    bias = "⚪ NEUTRAL"

# Volatility warning
if latest_vix > 25:
    bias += " (High Volatility)"

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("SPX Price", f"{latest_price:.2f}")
c2.metric("VIX", f"{latest_vix:.2f}")
c3.metric("Market Bias", bias)

# -----------------------------
# Chart
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spx.index,
    y=spx["Close"],
    name="SPX",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=spx.index,
    y=spx["EMA_9"],
    name="EMA 9"
))

fig.add_trace(go.Scatter(
    x=spx.index,
    y=spx["EMA_21"],
    name="EMA 21"
))

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
