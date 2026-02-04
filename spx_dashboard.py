# ==========================================
# SPX Live Predictive Trading Dashboard
# Streamlit Cloud Safe Version
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="SPX Live Trading Dashboard",
    layout="wide"
)

st.title("📈 SPX Live Predictive Trading Dashboard")

# ------------------------------------------
# Load Market Data
# ------------------------------------------
@st.cache_data(ttl=300)
def load_data():
    spx = yf.download("^GSPC", period="5d", interval="15m", progress=False)
    vix = yf.download("^VIX", period="5d", interval="15m", progress=False)
    return spx, vix

spx, vix = load_data()

# ------------------------------------------
# SAFETY CHECK (prevents crashes)
# ------------------------------------------
if spx.empty or vix.empty:
    st.warning("⚠️ Market data temporarily unavailable. Please refresh in a moment.")
    st.stop()

# ------------------------------------------
# Feature Engineering
# ------------------------------------------
spx["ema_8"] = spx["Close"].ewm(span=8).mean()
spx["ema_21"] = spx["Close"].ewm(span=21).mean()
spx["ema_55"] = spx["Close"].ewm(span=55).mean()

spx["return"] = spx["Close"].pct_change()
spx["volatility"] = spx["return"].rolling(20).std()

spx["zscore"] = (
    (spx["Close"] - spx["Close"].rolling(20).mean())
    / spx["Close"].rolling(20).std()
)

spx = spx.dropna()

# ------------------------------------------
# Latest Values
# ------------------------------------------
latest = spx.iloc[-1]
latest_vix = vix["Close"].iloc[-1]

# ------------------------------------------
# Signal Logic
# ------------------------------------------
momentum = 0

# Mean reversion signal
if latest["zscore"] < -1:
    momentum += 1

# Trend confirmation
if latest["ema_8"] > latest["ema_21"] > latest["ema_55"]:
    momentum += 1

momentum = momentum / 2  # normalize

# Placeholder predictive models
regression_pred = latest["return"]
lstm_pred = latest["return"] * 0.9

# Ensemble signal
ensemble_signal = (
    0.25 * momentum
    + 0.35 * regression_pred
    + 0.40 * lstm_pred
)

# Volatility regime adjustment
if latest_vix > 20:
    ensemble_signal *= 0.6

# ------------------------------------------
# Dashboard Metrics
# ------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("SPX Price", f"{latest['Close']:.2f}")
col2.metric("VIX", f"{latest_vix:.2f}")
col3.metric("Momentum Score", round(momentum, 2))
col4.metric("Ensemble Signal", round(ensemble_signal, 3))

# ------------------------------------------
# Price Chart
# ------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spx.index,
    y=spx["Close"],
    name="SPX",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_8"], name="EMA 8"))
fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_21"], name="EMA 21"))
fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_55"], name="EMA 55"))

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# Trade Bias
# ------------------------------------------
if ensemble_signal > 0.25:
    bias = "🟢 LONG"
elif ensemble_signal < -0.25:
    bias = "🔴 SHORT"
else:
    bias = "⚪ FLAT"

st.subheader(f"Current Bias: {bias}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
