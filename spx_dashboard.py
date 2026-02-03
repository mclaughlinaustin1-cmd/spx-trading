# SPX Live Predictive Trading Dashboard - Streamlit Cloud Version

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="SPX Live Dashboard")
st.title("📈 SPX Live Predictive Trading Dashboard")

# Load data
@st.cache_data(ttl=300)
def load_data():
    spx = yf.download("^GSPC", period="5d", interval="15m")
    vix = yf.download("^VIX", period="5d", interval="15m")
    return spx.dropna(), vix.dropna()

spx, vix = load_data()

# Feature engineering
spx["ema_8"] = spx["Close"].ewm(span=8).mean()
spx["ema_21"] = spx["Close"].ewm(span=21).mean()
spx["ema_55"] = spx["Close"].ewm(span=55).mean()
spx["return"] = spx["Close"].pct_change()
spx["volatility"] = spx["return"].rolling(20).std()
spx["zscore"] = (spx["Close"] - spx["Close"].rolling(20).mean()) / spx["Close"].rolling(20).std()

latest = spx.iloc[-1]
latest_vix = vix["Close"].iloc[-1]

# Signals
momentum = 0
if latest["zscore"] < -1:
    momentum += 1
if latest["ema_8"] > latest["ema_21"] > latest["ema_55"]:
    momentum += 1
momentum = momentum / 2

regression_pred = latest["return"]
lstm_pred = latest["return"] * 0.9
ensemble_signal = 0.25*momentum + 0.35*regression_pred + 0.4*lstm_pred
if latest_vix > 20:
    ensemble_signal *= 0.6

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("SPX Price", f"{latest['Close']:.2f}")
col2.metric("VIX", f"{latest_vix:.2f}")
col3.metric("Momentum", round(momentum,2))
col4.metric("Ensemble Signal", round(ensemble_signal,3))

# Price chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=spx.index, y=spx["Close"], name="SPX", line=dict(width=2)))
fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_8"], name="EMA 8"))
fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_21"], name="EMA 21"))
fig.add_trace(go.Scatter(x=spx.index, y=spx["ema_55"], name="EMA 55"))
fig.update_layout(height=500, xaxis_title="Time", yaxis_title="Price", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Bias
if ensemble_signal > 0.25:
    bias = "🟢 LONG"
elif ensemble_signal < -0.25:
    bias = "🔴 SHORT"
else:
    bias = "⚪ FLAT"

st.subheader(f"Current Bias: {bias}")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
