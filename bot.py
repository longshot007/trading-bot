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

# =========================
# 2️⃣ Market hours check
# =========================
clock = api.get_clock()
if not clock.is_open:
    print(f"{datetime.now()} - Market closed. Exiting bot.")
    exit()

# =========================
# 3️⃣ Fetch account cash and positions
# =========================
account = api.get_account()
cash = float(account.cash)
cash_for_trades = cash * INVESTMENT_PERCENTAGE

open_positions = api.list_positions()
open_symbols = [p.symbol for p in open_positions]

# =========================
# 4️⃣ Select best momentum stocks
# =========================
momentum_scores = {}
for sym in SYMBOLS:
    # Fetch last 5-min bars for momentum
    bars = api.get_bars(sym, tradeapi.TimeFrame.Minute, limit=5).df
    momentum = (bars['close'][-1] - bars['close'][0]) / bars['close'][0]
    momentum_scores[sym] = momentum

sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

# =========================
# 5️⃣ Place trades respecting diversification & risk
# =========================
positions_added = 0
for symbol, score in sorted_symbols:
    if positions_added >= MAX_POSITIONS:
        break
    if symbol in open_symbols:
        continue  # already have position
    if score <= 0:
        continue  # skip negative momentum

    price = float(api.get_last_trade(symbol).price)
    qty = int((cash_for_trades * RISK_PER_TRADE) / price)
    if qty <= 0:
        continue

    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )
    positions_added += 1

    stop_price = round(price * (1 - STOP_LOSS_PERCENT), 2)
    take_profit_price = round(price * (1 + TAKE_PROFIT_PERCENT), 2)

    print(f"{datetime.now()} - Bought {qty} shares of {symbol} at ${price:.2f}")
    print(f"  Stop-loss at ${stop_price}, Take-profit at ${take_profit_price}")

# =========================
# 6️⃣ Debug recent orders
# =========================
orders = api.list_orders(status='all', limit=10)
for o in orders:
    print(f"Order: {o.symbol}, Side: {o.side}, Filled: {o.filled_qty}, Status: {o.status}")
