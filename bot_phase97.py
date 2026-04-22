#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

# ================= CONFIG =================
MAX_POSITION_PCT   = 0.05
BUYING_POWER_CAP   = 0.25
MAX_HOLD_DAYS      = 4
DAILY_LOSS_LIMIT   = 20_000

SEC_FEE_RATE       = 0.0000206  # REQUIRED

TIMEZONE_ET        = pytz.timezone("US/Eastern")

LOG_FILE           = "/tmp/bot97_log.json"
STATE_FILE         = "/tmp/positions_state.json"

# ================= ENV =================
api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    os.environ["APCA_API_BASE_URL"],
    api_version="v2"
)

# ================= LOGGING =================
def log(data):
    data["ts"] = datetime.utcnow().isoformat()
    print(json.dumps(data), flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except:
        pass

# ================= TIME =================
def market_is_open():
    return api.get_clock().is_open

def now_et():
    return datetime.now(TIMEZONE_ET)

def entry_window():
    t = now_et()
    return (9,30) <= (t.hour, t.minute) <= (14,30)

# ================= ACCOUNT =================
def get_account():
    a = api.get_account()
    return float(a.equity), float(a.buying_power)

def get_positions():
    return {p.symbol: float(p.qty) for p in api.list_positions()}

def get_open_orders():
    return {o.symbol for o in api.list_orders(status="open")}

# ================= SCANNER =================
def get_universe():
    log({"event": "SCAN_START"})

    try:
        assets = api.list_assets(status='active')
    except Exception as e:
        log({"event": "SCAN_FAIL", "err": str(e)})
        return []

    symbols = [a.symbol for a in assets if a.tradable and a.shortable]

    # limit for runtime safety
    symbols = symbols[:300]

    log({"event": "UNIVERSE_SIZE", "count": len(symbols)})
    return symbols

# ================= DATA =================
def get_bars(symbol):
    try:
        bars = api.get_bars(symbol, "1Min", limit=100).df
        if bars.empty or len(bars) < 30:
            return None
        return bars
    except:
        return None

# ================= INDICATORS =================
def add_indicators(df):
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_g = gain.ewm(alpha=1/14).mean()
    avg_l = loss.ewm(alpha=1/14).mean()

    rs = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

# ================= SIGNAL =================
def generate_signal(df):
    r = df.iloc[-1]

    score = 0

    if r["ema9"] > r["ema20"]:
        score += 1
    else:
        score -= 1

    if r["rsi"] < 35:
        score += 1
    elif r["rsi"] > 65:
        score -= 1

    if score >= 1:
        return "BUY"
    elif score <= -1:
        return "SELL"

    return None

# ================= FILTER =================
def passes_filters(df):
    r = df.iloc[-1]

    if r["close"] < 5:
        return False

    if df["volume"].mean() < 50000:
        return False

    return True

# ================= RISK =================
def position_size(equity, price):
    return int((equity * MAX_POSITION_PCT) // price)

# ================= EXECUTION =================
def place_order(symbol, side, qty, price):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

        sec_fee = qty * price * SEC_FEE_RATE if side == "sell" else 0

        log({
            "event": "ORDER",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "sec_fee": sec_fee
        })

    except Exception as e:
        log({"event": "ORDER_FAIL", "symbol": symbol, "err": str(e)})

# ================= MAIN =================
def run():
    log({"event": "RUN_START"})

    if not market_is_open():
        log({"event": "MARKET_CLOSED"})
        return

    equity, bp = get_account()
    log({"event": "ACCOUNT", "equity": equity, "bp": bp})

    if not entry_window():
        log({"event": "OUTSIDE_ENTRY_WINDOW"})
        return

    positions   = get_positions()
    open_orders = get_open_orders()

    universe = get_universe()

    candidates = []

    for symbol in universe:
        if symbol in positions or symbol in open_orders:
            continue

        bars = get_bars(symbol)
        if bars is None:
            continue

        df = add_indicators(bars)

        if not passes_filters(df):
            continue

        signal = generate_signal(df)

        log({"event": "SIGNAL_CHECK", "symbol": symbol, "signal": signal})

        if not signal:
            continue

        candidates.append((symbol, signal, df))

    log({"event": "CANDIDATE_COUNT", "count": len(candidates)})

    # limit trades per run
    for symbol, signal, df in candidates[:5]:

        price = float(df["close"].iloc[-1])
        qty   = position_size(equity, price)

        if qty <= 0:
            continue

        if qty * price > equity * BUYING_POWER_CAP:
            continue

        side = "buy" if signal == "BUY" else "sell"

        place_order(symbol, side, qty, price)

    log({"event": "RUN_END"})


if __name__ == "__main__":
    run()
