import os
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Alpaca setup ---
FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true
API_KEY = os.environ["APCA_API_KEY_ID"]
SECRET_KEY = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Parameters ---
SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN"]  # example watchlist
LOOKBACK = 5  # days for momentum calculation
STOP_LOSS_PCT = 0.02  # 2% stop loss

# --- Helper functions ---
def get_momentum(symbol, days=LOOKBACK):
    df = yf.download(symbol, period=f"{days+1}d", interval="1d")
    if df.empty or len(df) < 2:
        return 0
    momentum = df['Close'][-1] / df['Close'][0] - 1
    return momentum

def get_cash():
    account = api.get_account()
    return float(account.cash)

# --- Check market status ---
clock = api.get_clock()
if not clock.is_open:
    print("Market is closed.")
    exit()

# --- Calculate momentum ---
momentum_scores = {s: get_momentum(s) for s in SYMBOLS}
best_symbol = max(momentum_scores, key=momentum_scores.get)
print(f"Best momentum: {best_symbol} ({momentum_scores[best_symbol]:.2%})")

# --- Determine position size ---
cash = get_cash()
price = api.get_last_trade(best_symbol).price
qty = int((cash * 0.95) / price)  # use 95% of cash
if qty <= 0:
    print("Insufficient funds to buy.")
    exit()

# --- Submit order ---
try:
    api.submit_order(
        symbol=best_symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )
    print(f"Market buy submitted for {qty} shares of {best_symbol}")
except Exception as e:
    print(f"Order failed: {e}")

# --- Apply stop-loss (bracket order) ---
try:
    stop_price = price * (1 - STOP_LOSS_PCT)
    api.submit_order(
        symbol=best_symbol,
        qty=qty,
        side='sell',
        type='stop',
        stop_price=round(stop_price, 2),
        time_in_force='day'
    )
    print(f"Stop-loss set at ${stop_price:.2f}")
except Exception as e:
    print(f"Stop-loss setup failed: {e}")

if __name__ == "__main__":
    run_bot()
