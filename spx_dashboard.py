# =========================================================
# SPX INTRADAY TRADE SIGNAL DASHBOARD (ROBUST VERSION)
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="SPX Trade Signals", layout="wide")
st.title("📊 SPX Intraday Trade Signal Engine (15-Min Horizon)")

# -----------------------------
# Load Market Data
# -----------------------------
@st.cache_data(ttl=300)
def load_data():
    spx = yf.download("^GSPC", period="5d", interval="15m", progress=False)
    vix = yf.download("^VIX", period="5d", interval="15m", progress=False)
    return spx, vix

spx, vix = load_data()

if spx.empty or vix.empty:
    st.warning("Market data unavailable. Refresh shortly.")
    st.stop()

# -----------------------------
# Indicators (safe & explicit)
# -----------------------------
spx["EMA_9"] = spx["Close"].ewm(span=9).mean()
spx["EMA_21"] = spx["Close"].ewm(span=21).mean()
spx["ATR"] = (spx["High"] - spx["Low"]).rolling(14).mean()

spx = spx.dropna()

price = float(spx["Close"].iloc[-1])
ema_fast = float(spx["EMA_9"].iloc[-1])
ema_slow = float(spx["EMA_21"].iloc[-1])
atr = float(spx["ATR"].iloc[-1])
vix_level = float(vix["Close"].iloc[-1])

# -----------------------------
# Recent Return (safe float)
# -----------------------------
recent_return = float(
    (spx["Close"].iloc[-1] - spx["Close"].iloc[-3])
    / spx["Close"].iloc[-3]
)

# -----------------------------
# 0DTE / Volatility Regime
# -----------------------------
if vix_level < 18:
    regime = "LOW VOL (Trend Friendly)"
elif vix_level < 25:
    regime = "NORMAL VOL"
else:
    regime = "HIGH VOL (0DTE Risk)"

# -----------------------------
# DO NOT TRADE CONDITIONS
# -----------------------------
do_not_trade = False
do_not_trade_reason = ""

if atr / price > 0.008:
    do_not_trade = True
    do_not_trade_reason = "Excessive intraday volatility"

if vix_level > 30:
    do_not_trade = True
    do_not_trade_reason = "Extreme volatility regime"

if abs(recent_return) < 0.0005:
    do_not_trade = True
    do_not_trade_reason = "No momentum / chop zone"

# -----------------------------
# Signal Scoring
# -----------------------------
score = 0

# Trend
score += 1 if ema_fast > ema_slow else -1

# Momentum
score += 1 if recent_return > 0 else -1

# Volatility penalty
if vix_level > 25:
    score -= 1

# -----------------------------
# Trade Signal Logic
# -----------------------------
if do_not_trade:
    signal = "🚫 DO NOT TRADE"
    confidence = "N/A"
elif score >= 2:
    signal = "🟢 LONG"
    confidence = "High (65–70%)"
elif score == 1:
    signal = "🟡 LONG (Cautious)"
    confidence = "Medium (55–60%)"
elif score == -1:
    signal = "🟠 SHORT (Cautious)"
    confidence = "Medium (55–60%)"
else:
    signal = "🔴 SHORT"
    confidence = "High (65–70%)"

# -----------------------------
# Entry / Exit / Invalidation Zones
# -----------------------------
entry = price
target = price + atr if "LONG" in signal else price - atr
invalidation = price - atr * 0.75 if "LONG" in signal else price + atr * 0.75

# -----------------------------
# Paper P&L Logging
# -----------------------------
st.session_state.setdefault("trades", [])

if st.button("📌 Log Trade (Paper)"):
    st.session_state.trades.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Signal": signal,
        "Entry": round(entry, 2),
        "Target": round(target, 2),
        "Invalidation": round(invalidation, 2)
    })

trades_df = pd.DataFrame(st.session_state.trades)

# -----------------------------
# Display Metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("SPX Price", f"{price:.2f}")
c2.metric("VIX", f"{vix_level:.2f}")
c3.metric("Vol Regime", regime)
c4.metric("Signal", signal)

st.markdown(f"**Confidence:** {confidence}")

if do_not_trade:
    st.error(f"DO NOT TRADE: {do_not_trade_reason}")

# -----------------------------
# Trade Plan Display
# -----------------------------
st.subheader("📍 Trade Levels (15-Min Horizon)")
st.markdown(f"""
- **Entry:** {entry:.2f}  
- **Target:** {target:.2f}  
- **Invalidation:** {invalidation:.2f}  
- **Risk Type:** ATR-based  
""")

# -----------------------------
# Chart
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=spx.index, y=spx["Close"], name="SPX", line=dict(width=2)))
fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA_21"], name="EMA 21"))
fig.update_layout(height=500, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Paper Trade Log
# -----------------------------
st.subheader("🧾 Paper Trade Log")
st.dataframe(trades_df, use_container_width=True)

st.caption(
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Rule-based signals. Educational / informational use only."
)
