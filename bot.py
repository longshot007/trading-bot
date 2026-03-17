# =========================
# Alpaca Paper Trading Bot with Backtesting & Risk Controls
# =========================

import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd

# --- API setup ---
API_KEY = os.environ["APCA_API_KEY_ID"]
SECRET_KEY = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Bot settings ---
SYMBOLS = ["AAPL", "MSFT", "GOOG"]  # Symbols to trade
INVESTMENT_PERCENTAGE = 0.25         # Max cash allocation per bot run
RISK_PER_TRADE = 0.1                 # Fraction of cash per trade
STOP_LOSS_PERCENT = 0.03             # 3% stop-loss
TAKE_PROFIT_PERCENT = 0.05           # 5% take-profit
MAX_POSITIONS = 2                     # Max simultaneous open positions
BACKTEST_DAYS = 30                    # Historical days for backtest

# =========================
# 1️⃣ Backtesting module
# =========================
print("=== Starting backtest ===")
backtest_results = []

for symbol in SYMBOLS:
    # Fetch historical daily bars
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date.isoformat(), end=end_date.isoformat()).df
    bars = bars.sort_index()  # oldest to newest

    # Simple momentum: daily % change
    bars['pct_change'] = bars['close'].pct_change()
    bars.dropna(inplace=True)

    # Backtest: long if daily % change positive, sell next day
    profit = 0
    for i in range(len(bars)-1):
        if bars['pct_change'].iloc[i] > 0:
            entry = bars['close'].iloc[i+1]  # next day open
            exit_price = bars['close'].iloc[i+1] * (1 + TAKE_PROFIT_PERCENT)
            stop_price = bars['close'].iloc[i+1] * (1 - STOP_LOSS_PERCENT)
            # Simple simulation: assume take-profit reached
            trade_profit = (exit_price - entry) / entry
            profit += trade_profit
    backtest_results.append((symbol, round(profit*100,2)))

print("Backtest results (% profit over last {} days):".format(BACKTEST_DAYS))
for symbol, pct in backtest_results:
    print(f"{symbol}: {pct}%")

import os
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pandas as pd

# API credentials from GitHub Secrets
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

# Connect to Alpaca
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Trading settings
SYMBOLS = [
    "AAPL","TSLA","NVDA","AMD","META",
    "AMZN","MSFT","GOOGL","SPY","QQQ"
]

TIMEFRAME = "5Min"
LOOKBACK_DAYS = 1
TRADE_QTY = 1
MOMENTUM_THRESHOLD = 0.002  # 0.2%

def get_rfc3339_time(days_back=1):
    """Return properly formatted RFC3339 timestamp"""
    return (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

def market_is_open():
    clock = api.get_clock()
    return clock.is_open

def get_momentum(symbol):
    try:
        start = get_rfc3339_time(LOOKBACK_DAYS)

        bars = api.get_bars(
            symbol,
            TIMEFRAME,
            start=start,
            limit=50
        ).df

        if len(bars) < 2:
            return None

        last_price = bars.close.iloc[-1]
        prev_price = bars.close.iloc[-2]

        momentum = (last_price - prev_price) / prev_price

        return momentum

    except Exception as e:
        print(f"Momentum error for {symbol}: {e}")
        return None

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def place_trade(symbol):
    try:
        print(f"Placing trade for {symbol}")

        api.submit_order(
            symbol=symbol,
            qty=TRADE_QTY,
            side="buy",
            type="market",
            time_in_force="day"
        )

    except Exception as e:
        print(f"Trade failed for {symbol}: {e}")

def run_bot():

    print("Bot started")

    if not market_is_open():
        print("Market closed")
        return

    for symbol in SYMBOLS:

        momentum = get_momentum(symbol)

        if momentum is None:
            continue

        print(f"{symbol} momentum: {momentum}")

        if momentum > MOMENTUM_THRESHOLD:

            if not position_exists(symbol):
                place_trade(symbol)

    print("Bot finished run")

if __name__ == "__main__":
    run_bot()
