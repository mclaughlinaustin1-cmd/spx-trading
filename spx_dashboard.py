import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf  # Fallback data source
try:
    from polygon import RESTClient  # For real-time data; assumes API key in env
except ImportError:
    RESTClient = None  # Handle if not installed
from prophet import Prophet  # For simple ML forecasting
import ta  # Technical Analysis library for RSI/MACD (pip install ta)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SPX Advanced Predictive Dashboard", layout="wide")
st.title("📈 SPX Live 15-Min Predictive Trading Dashboard")

# User-configurable params
with st.sidebar:
    st.header("Settings")
    refresh_interval = st.slider("Refresh Interval (sec)", 10, 300, 30)
    ema_fast_period = st.slider("Fast EMA Period", 5, 20, 9)
    ema_slow_period = st.slider("Slow EMA Period", 10, 50, 21)
    atr_period = st.slider("ATR Period", 5, 20, 14)
    rsi_period = st.slider("RSI Period", 5, 20, 14)
    risk_tolerance = st.slider("Risk Tolerance (ATR multiplier)", 0.5, 2.0, 1.0)
    api_key = st.text_input("Polygon API Key (optional for premium data)", type="password")  # Secure input

# Initialize Polygon client if available
client = None
if api_key and RESTClient:
    try:
        client = RESTClient(api_key)
    except Exception as e:
        st.warning(f"Failed to initialize Polygon client: {e}. Falling back to yfinance.")

placeholder = st.empty()
tabs = st.tabs(["Dashboard", "Backtest", "Predictions"])

def fetch_data(use_polygon=True):
    if use_polygon and client:
        try:
            end = datetime.now()
            start = end - timedelta(days=5)  # More history for better indicators/ML
            bars = client.get_aggs("^GSPC", 15, "minute", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if not bars:
                raise ValueError("No data from Polygon")
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Polygon fetch error: {e}")
            st.warning("Polygon data fetch failed. Falling back to yfinance.")
    
    # Fallback to yfinance
    try:
        df = yf.download("^GSPC", period="5d", interval="15m", progress=False)
        if df.empty:
            raise ValueError("No data from yfinance")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        logger.error(f"yfinance fetch error: {e}")
        return None

def compute_indicators(df):
    df["EMA_Fast"] = df["Close"].ewm(span=ema_fast_period, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=ema_slow_period, adjust=False).mean()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=atr_period)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_period)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    return df

def generate_signals(df, forecast=None):
    latest = df.iloc[-1]
    recent_return = (latest["Close"] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
    
    # Enhanced checks
    volatility_high = latest["ATR"] / latest["Close"] > 0.005  # Tighter threshold
    overbought = latest["RSI"] > 70
    oversold = latest["RSI"] < 30
    macd_bullish = latest["MACD"] > 0
    
    do_not_trade = volatility_high or abs(recent_return) < 0.0003
    if do_not_trade:
        return "🚫 DO NOT TRADE", "High volatility or choppy", None, None, None
    
    score = 0
    score += 1 if latest["EMA_Fast"] > latest["EMA_Slow"] else -1
    score += 1 if recent_return > 0 else -1
    score += 1 if macd_bullish else -1
    score += 1 if forecast and forecast['yhat'].iloc[-1] > latest["Close"] else -1  # ML integration
    
    if score >= 3:
        signal, conf = "🟢 LONG", "High"
    elif score >= 1:
        signal, conf = "🟡 LONG (Cautious)", "Medium"
    elif score <= -3:
        signal, conf = "🔴 SHORT", "High"
    else:
        signal, conf = "🟠 SHORT (Cautious)", "Medium"
    
    entry = latest["Close"]
    atr_adj = latest["ATR"] * risk_tolerance
    target = entry + atr_adj if "LONG" in signal else entry - atr_adj
    invalidation = entry - atr_adj * 0.5 if "LONG" in signal else entry + atr_adj * 0.5  # Tighter stop
    return signal, conf, entry, target, invalidation

def forecast_future(df):
    try:
        prophet_df = df.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=4, freq='15min')  # Predict next hour
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return None

def backtest_strategy(df):
    # Simple backtest: Buy on long signal, sell on short/invalidation
    df['Signal'] = 0
    for i in range(1, len(df)):
        signal, _, _, _, _ = generate_signals(df.iloc[:i])
        df.loc[df.index[i], 'Signal'] = 1 if "LONG" in signal else -1 if "SHORT" in signal else 0
    df['Returns'] = df['Close'].pct_change() * df['Signal'].shift()
    total_return = df['Returns'].sum()
    sharpe = df['Returns'].mean() / df['Returns'].std() * (252 * 4 * 4) ** 0.5  # Annualized, assuming 4x4=16 trades/day
    return total_return, sharpe

# Main loop
for _ in range(1000):  # Infinite loop with break conditions in prod
    with placeholder.container():
        df = fetch_data(use_polygon=bool(client))
        if df is None or df.empty:
            st.warning("No data. Retrying...")
            time.sleep(refresh_interval)
            continue
        
        df = compute_indicators(df)
        forecast = forecast_future(df.tail(100))  # Last day for speed
        signal, conf, entry, target, invalidation = generate_signals(df.tail(5), forecast)
        
        # Dashboard Tab
        with tabs[0]:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Latest Price", f"{df['Close'].iloc[-1]:.2f}")
            c2.metric(f"EMA {ema_fast_period}", f"{df['EMA_Fast'].iloc[-1]:.2f}")
            c3.metric(f"EMA {ema_slow_period}", f"{df['EMA_Slow'].iloc[-1]:.2f}")
            c4.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            c5.metric("Signal", signal)
            c6.metric("Confidence", conf)
            
            st.subheader("Trade Levels (15-Min Horizon)")
            if entry:
                st.markdown(f"- Entry: {entry:.2f}\n- Target: {target:.2f}\n- Invalidation: {invalidation:.2f}")
            
            # Chart with forecast
            fig = go.Figure(data=[go.Candlestick(x=df.index[-20:], open=df['Open'][-20:], high=df['High'][-20:], low=df['Low'][-20:], close=df['Close'][-20:])])
            fig.add_trace(go.Scatter(x=df.index[-20:], y=df["EMA_Fast"][-20:], name="EMA Fast"))
            fig.add_trace(go.Scatter(x=df.index[-20:], y=df["EMA_Slow"][-20:], name="EMA Slow"))
            if forecast is not None:
                fig.add_trace(go.Scatter(x=forecast['ds'][-4:], y=forecast['yhat'][-4:], name="Forecast", line=dict(dash='dot')))
            fig.update_layout(title="SPX 15-Min Chart + Forecast", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Backtest Tab
        with tabs[1]:
            if st.button("Run Backtest"):
                total_ret, sharpe = backtest_strategy(df)
                st.metric("Total Return", f"{total_ret:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.line_chart(df['Returns'].cumsum())
        
        # Predictions Tab
        with tabs[2]:
            if forecast is not None:
                st.subheader("Next 1-Hour Forecast")
                st.table(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4))
            else:
                st.info("Forecast unavailable.")
        
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    time.sleep(refresh_interval)
