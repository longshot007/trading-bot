# =========================
# Alpaca Paper Trading Bot - Full Turnkey Version
# =========================

import os
import alpaca_trade_api as tradeapi
from datetime import datetime

# --- API setup ---
API_KEY = os.environ["APCA_API_KEY_ID"]
SECRET_KEY = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Bot settings ---
SYMBOLS = ["AAPL", "MSFT", "GOOG"]   # Symbols to trade
INVESTMENT_PERCENTAGE = 0.25         # Max 25% of total cash for this bot run
RISK_PER_TRADE = 0.1                 # Fraction of cash to risk per trade (10%)

# --- Check if market is open ---
clock = api.get_clock()
if not clock.is_open:
    print(f"{datetime.now()} - Market is closed. Exiting bot.")
    exit()

# --- Fetch account cash ---
account = api.get_account()
cash = float(account.cash)
cash_for_trades = cash * INVESTMENT_PERCENTAGE

# --- Placeholder for momentum logic ---
# Replace this with your real calculation for momentum
momentum_scores = {s: 0 for s in SYMBOLS}   # Example: all zeros
best_symbol = max(momentum_scores, key=momentum_scores.get)

# --- Get latest price for selected symbol ---
trade_info = api.get_last_trade(best_symbol)
price = float(trade_info.price)

# --- Calculate quantity based on fractional risk ---
qty = int((cash_for_trades * RISK_PER_TRADE) / price)

# --- Debug prints ---
print(f"{datetime.now()} - Cash available: ${cash:,.2f}")
print(f"{datetime.now()} - Max cash for this bot run (25%): ${cash_for_trades:,.2f}")
print(f"{datetime.now()} - Best symbol: {best_symbol}, Price: ${price:.2f}")
print(f"{datetime.now()} - Fractional risk per trade: {RISK_PER_TRADE*100:.1f}%")
print(f"{datetime.now()} - Quantity to buy: {qty}")

# --- Submit order (paper only) ---
if qty > 0:
    order = api.submit_order(
        symbol=best_symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )
    print(f"{datetime.now()} - Order submitted for {qty} shares of {best_symbol}")
else:
    print(f"{datetime.now()} - Qty calculated as 0, skipping order.")

# --- List recent orders for verification ---
orders = api.list_orders(status='all', limit=5)
for o in orders:
    print(f"Order: {o.symbol}, Side: {o.side}, Filled: {o.filled_qty}, Status: {o.status}")

if __name__ == "__main__":
    run_bot()
