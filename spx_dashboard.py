import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf
import ta  # pip install ta
from statsmodels.tsa.arima.model import ARIMA  # Built-in to statsmodels (reliable install)
import warnings
import logging

warnings.filterwarnings("ignore")  # ARIMA convergence warnings are noisy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SPX Advanced Predictive Dashboard", layout="wide")
st.title("📈 SPX Live 15-Min Predictive Trading Dashboard")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    refresh_interval = st.slider("Refresh Interval (sec)", 10, 300, 30)
    ema_fast_period = st.slider("Fast EMA Period", 5, 20, 9)
    ema_slow_period = st.slider("Slow EMA Period", 10, 50, 21)
    atr_period = st.slider("ATR Period", 5, 20, 14)
    rsi_period = st.slider("RSI Period", 5, 20, 14)
    risk_tolerance = st.slider("Risk Tolerance (ATR multiplier)", 0.5, 2.0, 1.0)

placeholder = st.empty()
tabs = st.tabs(["Dashboard", "Backtest", "Predictions"])

def fetch_data():
    try:
        df = yf.download("^GSPC", period="5d", interval="15m", progress=False)
        if df.empty:
            raise ValueError("No data from yfinance")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        logger.error(f"yfinance error: {e}")
        return None

def compute_indicators(df):
    df["EMA_Fast"] = df["Close"].ewm(span=ema_fast_period, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=ema_slow_period, adjust=False).mean()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=atr_period)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_period)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    return df

def forecast_future_arima(df, steps=4):
    try:
        series = df['Close'].tail(100)  # Use recent data for speed & relevance
        model = ARIMA(series, order=(1,1,1))  # Simple ARIMA(1,1,1) – good baseline
        model_fit = model.fit()
        forecast_obj = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        
        # Build forecast DataFrame
        last_time = series.index[-1]
        future_times = pd.date_range(start=last_time + timedelta(minutes=15), periods=steps, freq='15min')
        forecast_df = pd.DataFrame({
            'ds': future_times,
            'yhat': forecast_mean.values,
            'yhat_lower': conf_int.iloc[:,0].values,
            'yhat_upper': conf_int.iloc[:,1].values
        })
        return forecast_df
    except Exception as e:
        logger.error(f"ARIMA forecast error: {e}")
        return None

def generate_signals(df, forecast=None):
    latest = df.iloc[-1]
    recent_return = (latest["Close"] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
    
    volatility_high = latest["ATR"] / latest["Close"] > 0.005
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
    score += 1 if forecast is not None and forecast['yhat'].iloc[-1] > latest["Close"] else -1
    
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
    invalidation = entry - atr_adj * 0.5 if "LONG" in signal else entry + atr_adj * 0.5
    
    return signal, conf, entry, target, invalidation

def backtest_strategy(df):
    df['Signal'] = 0
    for i in range(1, len(df)):
        signal, _, _, _, _ = generate_signals(df.iloc[:i])
        df.loc[df.index[i], 'Signal'] = 1 if "LONG" in signal else -1 if "SHORT" in signal else 0
    df['Returns'] = df['Close'].pct_change() * df['Signal'].shift()
    total_return = df['Returns'].sum()
    if df['Returns'].std() == 0:
        sharpe = 0
    else:
        sharpe = df['Returns'].mean() / df['Returns'].std() * (252 * 4 * 4) ** 0.5
    return total_return, sharpe

# Main refresh loop
for _ in range(1000):
    with placeholder.container():
        df = fetch_data()
        if df is None or df.empty or len(df) < 20:
            st.warning("Not enough data yet. Retrying...")
            time.sleep(refresh_interval)
            continue
        
        df = compute_indicators(df)
        forecast = forecast_future_arima(df)
        signal, conf, entry, target, invalidation = generate_signals(df.tail(5), forecast)
        
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
                st.markdown(f"- **Entry**: {entry:.2f}\n- **Target**: {target:.2f}\n- **Invalidation**: {invalidation:.2f}")
            
            fig = go.Figure(data=[go.Candlestick(
                x=df.index[-20:],
                open=df['Open'][-20:],
                high=df['High'][-20:],
                low=df['Low'][-20:],
                close=df['Close'][-20:],
                name="SPX"
            )])
            fig.add_trace(go.Scatter(x=df.index[-20:], y=df["EMA_Fast"][-20:], name=f"EMA {ema_fast_period}", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index[-20:], y=df["EMA_Slow"][-20:], name=f"EMA {ema_slow_period}", line=dict(color='orange')))
            if forecast is not None:
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='purple', dash='dot')))
            fig.update_layout(title="SPX 15-Min Chart + Forecast (Last 20 Bars)", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            if st.button("Run Backtest on Available Data"):
                total_ret, sharpe = backtest_strategy(df)
                st.metric("Total Return (backtest)", f"{total_ret:.2%}")
                st.metric("Sharpe Ratio (annualized approx)", f"{sharpe:.2f}")
                if 'Returns' in df.columns:
                    st.line_chart(df['Returns'].cumsum().dropna())
        
        with tabs[2]:
            if forecast is not None:
                st.subheader("Next ~1 Hour Forecast (ARIMA)")
                st.dataframe(forecast.style.format({"yhat": "{:.2f}", "yhat_lower": "{:.2f}", "yhat_upper": "{:.2f}"}))
            else:
                st.info("Forecast unavailable – not enough data or model error.")
        
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: yfinance | Model: ARIMA(1,1,1)")
    
    time.sleep(refresh_interval)
