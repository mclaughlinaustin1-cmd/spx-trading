import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf
import ta  # Technical indicators: rsi, macd_diff, average_true_range
from statsmodels.tsa.arima.model import ARIMA
import warnings
import logging

warnings.filterwarnings("ignore")  # Suppress ARIMA convergence warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SPX 15-Min Predictive Dashboard", layout="wide")
st.title("📈 SPX Live 15-Min Predictive Trading Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 30)
    ema_fast_period = st.slider("Fast EMA Period", 5, 20, 9)
    ema_slow_period = st.slider("Slow EMA Period", 10, 50, 21)
    atr_period = st.slider("ATR Period", 5, 20, 14)
    rsi_period = st.slider("RSI Period", 5, 20, 14)
    risk_tolerance = st.slider("Risk Tolerance (ATR ×)", 0.5, 2.0, 1.0)

placeholder = st.empty()
tabs = st.tabs(["Dashboard", "Backtest", "Forecast"])

def fetch_data():
    try:
        df = yf.download("^GSPC", period="5d", interval="15m", progress=False)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        return df
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return None

def add_indicators(df):
    df["EMA_Fast"] = df["Close"].ewm(span=ema_fast_period, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=ema_slow_period, adjust=False).mean()
    df["ATR"] = ta.volatility.average_true_range(
        high=df["High"], low=df["Low"], close=df["Close"], window=atr_period
    )
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_period)
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    return df

def forecast_arima(df, steps=4):
    try:
        series = df['Close'].tail(120)  # More data = better model, still fast
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()
        forecast_obj = fitted.get_forecast(steps=steps)
        mean = forecast_obj.predicted_mean
        conf = forecast_obj.conf_int()
        
        last_ts = series.index[-1]
        future_index = pd.date_range(start=last_ts + timedelta(minutes=15), periods=steps, freq='15T')
        
        fc_df = pd.DataFrame({
            'ds': future_index,
            'yhat': mean.values,
            'yhat_lower': conf.iloc[:, 0].values,
            'yhat_upper': conf.iloc[:, 1].values
        })
        return fc_df
    except Exception as e:
        logger.error(f"ARIMA error: {e}")
        return None

def get_signal(df, forecast=None):
    latest = df.iloc[-1]
    prev_close = df["Close"].iloc[-2]
    recent_return = (latest["Close"] - prev_close) / prev_close
    
    vol_high = (latest["ATR"] / latest["Close"]) > 0.005
    choppy = abs(recent_return) < 0.0003
    overbought = latest["RSI"] > 70
    oversold = latest["RSI"] < 30
    macd_bull = latest["MACD_diff"] > 0
    
    if vol_high or choppy:
        return "🚫 DO NOT TRADE", "Volatility or choppy conditions", None, None, None
    
    score = 0
    score += 1 if latest["EMA_Fast"] > latest["EMA_Slow"] else -1
    score += 1 if recent_return > 0 else -1
    score += 1 if macd_bull else -1
    score += 1 if forecast is not None and forecast['yhat'].iloc[-1] > latest["Close"] else -1
    
    if score >= 3:
        sig, conf = "🟢 LONG", "High"
    elif score >= 1:
        sig, conf = "🟡 LONG (Cautious)", "Medium"
    elif score <= -3:
        sig, conf = "🔴 SHORT", "High"
    else:
        sig, conf = "🟠 SHORT (Cautious)", "Medium"
    
    entry = latest["Close"]
    atr_adj = latest["ATR"] * risk_tolerance
    target = entry + atr_adj if "LONG" in sig else entry - atr_adj
    stop = entry - atr_adj * 0.5 if "LONG" in sig else entry + atr_adj * 0.5
    
    return sig, conf, entry, target, stop

def simple_backtest(df):
    df = df.copy()
    df['Signal'] = 0
    for i in range(1, len(df)):
        sig, _, _, _, _ = get_signal(df.iloc[:i])
        if "LONG" in sig:
            df.loc[df.index[i], 'Signal'] = 1
        elif "SHORT" in sig:
            df.loc[df.index[i], 'Signal'] = -1
    df['Returns'] = df['Close'].pct_change() * df['Signal'].shift(1)
    total_ret = df['Returns'].sum()
    std = df['Returns'].std()
    sharpe = (df['Returns'].mean() / std * (252 * 16**0.5)) if std != 0 else 0  # ~16 15-min bars/day
    return total_ret, sharpe, df['Returns'].cumsum()

# Live loop
for _ in range(1000):
    with placeholder.container():
        df = fetch_data()
        if df is None or len(df) < 30:
            st.warning("Insufficient data — retrying soon...")
            time.sleep(refresh_interval)
            continue
        
        df = add_indicators(df)
        forecast = forecast_arima(df)
        signal, conf_level, entry, target, stop = get_signal(df.tail(5), forecast)
        
        # Dashboard tab
        with tabs[0]:
            cols = st.columns(6)
            cols[0].metric("Latest Price", f"{df['Close'].iloc[-1]:.2f}")
            cols[1].metric(f"EMA {ema_fast_period}", f"{df['EMA_Fast'].iloc[-1]:.2f}")
            cols[2].metric(f"EMA {ema_slow_period}", f"{df['EMA_Slow'].iloc[-1]:.2f}")
            cols[3].metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            cols[4].metric("Signal", signal)
            cols[5].metric("Confidence", conf_level)
            
            st.subheader("Trade Plan (15-min outlook)")
            if entry is not None:
                st.markdown(f"""
                - **Entry**: {entry:.2f}
                - **Target**: {target:.2f}
                - **Stop / Invalidation**: {stop:.2f}
                """)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index[-25:],
                open=df['Open'][-25:], high=df['High'][-25:],
                low=df['Low'][-25:], close=df['Close'][-25:],
                name="SPX"
            ))
            fig.add_trace(go.Scatter(x=df.index[-25:], y=df['EMA_Fast'][-25:], name="EMA Fast", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index[-25:], y=df['EMA_Slow'][-25:], name="EMA Slow", line=dict(color='orange')))
            if forecast is not None:
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    name="ARIMA Forecast", line=dict(color='purple', dash='dot')
                ))
            fig.update_layout(
                title="SPX 15-min Candles + Indicators + Forecast (last 25 bars)",
                height=600,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Backtest tab
        with tabs[1]:
            if st.button("Run Quick Backtest"):
                total_ret, sharpe, cum_ret = simple_backtest(df)
                st.metric("Total Return", f"{total_ret:.2%}")
                st.metric("Approx Sharpe (ann.)", f"{sharpe:.2f}")
                st.line_chart(cum_ret.dropna())
        
        # Forecast tab
        with tabs[2]:
            if forecast is not None:
                st.subheader("Next ~1 Hour ARIMA Forecast")
                st.dataframe(
                    forecast.style.format(precision=2)
                )
            else:
                st.info("Forecast not available (data/model issue)")
        
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: yfinance | Indicators: ta | Forecast: ARIMA(1,1,1)")
    
    time.sleep(refresh_interval)
