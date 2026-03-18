import os
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, time
import pytz

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

RISK_PER_TRADE = 0.01
MAX_TRADES_PER_DAY = 5

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "META"]

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

trades_today = 0
open_positions = {}

# =========================
# TIME FILTER
# =========================
def valid_trading_time():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est).time()

    return (
        time(9, 35) <= now <= time(11, 30)
        or time(15, 0) <= now <= time(15, 50)
    )

# =========================
# MARKET TREND FILTER
# =========================
def market_trending():
    bars = api.get_bars("SPY", "5Min", limit=20).df
    ema = bars['close'].ewm(span=9).mean()
    return bars['close'].iloc[-1] > ema.iloc[-1]

# =========================
# SIGNAL
# =========================
def get_signal(symbol):
    bars = api.get_bars(symbol, "5Min", limit=30).df

    if len(bars) < 20:
        return None

    high = bars['high'].rolling(20).max().iloc[-2]
    price = bars['close'].iloc[-1]

    avg_vol = bars['volume'].rolling(20).mean().iloc[-2]
    vol = bars['volume'].iloc[-1]

    rel_vol = vol / avg_vol if avg_vol > 0 else 0

    if price > high and rel_vol > 1.5:
        return price

    return None

# =========================
# POSITION SIZE
# =========================
def position_size(price):
    equity = float(api.get_account().equity)
    risk = equity * RISK_PER_TRADE

    stop_distance = price * 0.05  # 5% stop
    qty = int(risk / stop_distance)

    return max(qty, 1)

# =========================
# ENTER TRADE (WITH HARD STOP)
# =========================
def enter_trade(symbol, price):
    global trades_today

    if trades_today >= MAX_TRADES_PER_DAY:
        return

    qty = position_size(price)

    stop_loss = round(price * 0.95, 2)   # 5% visible stop
    take_profit = round(price * 1.10, 2) # wide TP (bot exits earlier anyway)

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day',
            order_class='bracket',
            stop_loss={'stop_price': stop_loss},
            take_profit={'limit_price': take_profit}
        )

        open_positions[symbol] = {
            "entry": price,
            "time": datetime.now(),
        }

        trades_today += 1
        print(f"ENTERED: {symbol} @ {price}")

    except Exception as e:
        print(f"ENTRY ERROR: {e}")

# =========================
# SMART EXIT (HIDDEN)
# =========================
def check_exit(symbol):
    bars = api.get_bars(symbol, "5Min", limit=10).df

    if symbol not in open_positions:
        return False

    entry = open_positions[symbol]["entry"]
    price = bars['close'].iloc[-1]

    ema = bars['close'].ewm(span=5).mean().iloc[-1]

    # 1) Momentum loss
    if price < ema:
        return True

    # 2) Profit capture (~3–5%)
    if price > entry * 1.04:
        return True

    # 3) Time decay
    held_minutes = (datetime.now() - open_positions[symbol]["time"]).seconds / 60
    if held_minutes > 30:
        return True

    return False

# =========================
# EXIT TRADE
# =========================
def exit_trade(symbol):
    try:
        position = api.get_position(symbol)

        api.submit_order(
            symbol=symbol,
            qty=position.qty,
            side='sell',
            type='market',
            time_in_force='day'
        )

        if symbol in open_positions:
            del open_positions[symbol]

        print(f"EXITED EARLY: {symbol}")

    except Exception as e:
        print(f"EXIT ERROR: {e}")

# =========================
# MAIN
# =========================
def run_bot():
    if not valid_trading_time():
        print("Outside trading window")
        return

    if not market_trending():
        print("Market not trending")
        return

    # Manage positions
    for symbol in list(open_positions.keys()):
        if check_exit(symbol):
            exit_trade(symbol)

    # New trades
    for symbol in SYMBOLS:
        if symbol not in open_positions:
            signal = get_signal(symbol)
            if signal:
                enter_trade(symbol, signal)

if __name__ == "__main__":
    run_bot()
run_bot()
print("=== PHASE-4 PROFESSIONAL BOT COMPLETE ===")
