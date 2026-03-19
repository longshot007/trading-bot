

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

MAX_POSITIONS = 5
POSITION_SIZE = 0.2  # 20% per trade
MAX_HOLD_MINUTES = 45
PROFIT_TARGET = 0.01  # 1%
STOP_LOSS = -0.007   # -0.7%
MIN_VOLATILITY = 0.003

# ----------------------
# Helpers
# ----------------------

def get_account():
    r = requests.get(f"{BASE_URL}/v2/account", headers=HEADERS)
    return r.json()


def get_positions():
    r = requests.get(f"{BASE_URL}/v2/positions", headers=HEADERS)
    return r.json()


def get_bars(symbol):
    end = datetime.utcnow()
    start = end - timedelta(hours=6)

    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z",
        "timeframe": "5Min"
    }

    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()

    if "bars" not in data:
        return None

    df = pd.DataFrame(data["bars"])
    return df


def compute_volatility(df):
    df['return'] = df['c'].pct_change()
    return df['return'].std()


def compute_momentum(df):
    return (df['c'].iloc[-1] - df['c'].iloc[0]) / df['c'].iloc[0]


def submit_order(symbol, qty, side):
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    requests.post(f"{BASE_URL}/v2/orders", json=order, headers=HEADERS)


# ----------------------
# Stock Selection
# ----------------------

def get_watchlist():
    return [
        "AAPL", "TSLA", "NVDA", "AMD", "MSFT",
        "META", "AMZN", "GOOGL", "NFLX", "SPY"
    ]


def pick_stocks():
    candidates = []

    for symbol in get_watchlist():
        df = get_bars(symbol)
        if df is None or len(df) < 10:
            continue

        vol = compute_volatility(df)
        mom = compute_momentum(df)

        if vol < MIN_VOLATILITY:
            continue

        candidates.append((symbol, mom, vol))

    # Sort by momentum
    candidates.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in candidates[:MAX_POSITIONS]]


# ----------------------
# Trading Logic
# ----------------------

def get_last_trade_price(symbol):
    r = requests.get(f"{DATA_URL}/v2/stocks/{symbol}/trades/latest", headers=HEADERS)
    return r.json()["trade"]["p"]


def calculate_qty(cash, price):
    return int((cash * POSITION_SIZE) / price)


def should_exit(position):
    entry_price = float(position['avg_entry_price'])
    current_price = float(position['current_price'])
    qty = float(position['qty'])

    pnl = (current_price - entry_price) / entry_price

    # Time-based exit (using placeholder since Alpaca doesn't store entry time directly)
    # In production you'd track this externally

    if pnl >= PROFIT_TARGET:
        return True

    if pnl <= STOP_LOSS:
        return True

    return False


def run_bot():
    account = get_account()
    cash = float(account['cash'])

    positions = get_positions()
    held_symbols = [p['symbol'] for p in positions]

    # SELL LOGIC
    for p in positions:
        if should_exit(p):
            print(f"Selling {p['symbol']}")
            submit_order(p['symbol'], p['qty'], "sell")

    # BUY LOGIC
    if len(positions) >= MAX_POSITIONS:
        return

    picks = pick_stocks()

    for symbol in picks:
        if symbol in held_symbols:
            continue

        price = get_last_trade_price(symbol)
        qty = calculate_qty(cash, price)

        if qty > 0:
            print(f"Buying {symbol}")
            submit_order(symbol, qty, "buy")


if __name__ == "__main__":
    run_bot()

