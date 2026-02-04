import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. USER INPUT ---
ticker = input("Enter a stock ticker (e.g., AAPL, MSFT): ").upper()
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# --- 2. DOWNLOAD DATA ---
print(f"\nDownloading data for {ticker} from {start_date.date()} to {end_date.date()}...")
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print("Error: No data found for this ticker.")
    exit()

# --- 3. CLEAN DATA ---
data = data.dropna()
data['Date'] = data.index
data['Date_ordinal'] = pd.to_datetime(data['Date']).map(datetime.toordinal)

# Features and target
X = data[['Date_ordinal']]
y = data['Close']

# --- 4. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 5. TRAIN MODELS ---
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest (Non-linear regression)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# --- 6. TEST MODELS ---
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("\nModel Accuracy (R² score):")
print(f"Linear Regression:      {r2_score(y_test, y_pred_lr):.4f}")
print(f"Decision Tree:          {r2_score(y_test, y_pred_dt):.4f}")
print(f"Random Forest:          {r2_score(y_test, y_pred_rf):.4f}")

# --- 7. USER PREDICTION ---
future_date_str = input("\nEnter a date to predict the stock price (YYYY-MM-DD): ")
try:
    future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
    future_ordinal = np.array([[future_date.toordinal()]])
    
    pred_lr = lr_model.predict(future_ordinal)[0]
    pred_dt = dt_model.predict(future_ordinal)[0]
    pred_rf = rf_model.predict(future_ordinal)[0]
    
    print(f"\nPredicted {ticker} Close Price on {future_date.date()}:")
    print(f"Linear Regression:  ${pred_lr:.2f}")
    print(f"Decision Tree:      ${pred_dt:.2f}")
    print(f"Random Forest:      ${pred_rf:.2f}")
except:
    print("Invalid date format. Please enter as YYYY-MM-DD.")
