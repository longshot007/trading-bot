import os
import json
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

# ================= CONFIG =================

MAX_POSITION_PCT = 0.05
MAX_TOTAL_EXPOSURE = 0.25
MAX_CONCURRENT = 25
DAILY_LOSS_LIMIT = 20000
BUYING_POWER_CAP = 0.25

MAX_HOLD_DAYS = 4

ENTRY_CUTOFF_PT = (11, 0)  # 11:00 AM PT
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_PT = pytz.timezone("US/Pacific")

LOG_FILE = "bot97_log.json"

# ================= API =================

api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    os.environ["APCA_API_BASE_URL"],
    api_version="v2"
)

# ================= LOGGING =================

def log(data):
    data["timestamp"] = datetime.now().isoformat()
    print(data)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

# ================= BROKER STATE =================

def get_account():
    acc = api.get_account()
    return {
        "equity": float(acc.equity),
        "buying_power": float(acc.buying_power),
    }

def get_positions():
    positions = api.list_positions()
    return {
        p.symbol: {
            "qty": float(p.qty),
            "side": "long" if float(p.qty) > 0 else "short",
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "entry_price": float(p.avg_entry_price),
        }
        for p in positions
    }

def get_open_orders():
    return api.list_orders(status="open")

# ================= TIME =================

def market_is_open():
    clock = api.get_clock()
    return clock.is_open

def is_entry_time():
    now_pt = datetime.now(TIMEZONE_PT)
    return (now_pt.hour, now_pt.minute) < ENTRY_CUTOFF_PT

# ================= RISK =================

def calculate_exposure(positions):
    return sum(abs(p["market_value"]) for p in positions.values())

def can_trade(account, positions):
    if account["buying_power"] <= 0:
        return False, "NO_BUYING_POWER"

    exposure = calculate_exposure(positions)
    if exposure > account["equity"] * MAX_TOTAL_EXPOSURE:
        return False, "MAX_EXPOSURE_REACHED"

    if len(positions) >= MAX_CONCURRENT:
        return False, "MAX_POSITIONS"

    return True, None

def position_size(account, price):
    max_position_value = account["equity"] * MAX_POSITION_PCT
    return int(max_position_value // price)

# ================= INDICATORS =================

def compute_indicators(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    return df

# ================= SIGNAL ENGINE =================

def generate_signal(df):
    row = df.iloc[-1]

    score = 0

    # RSI
    if row["rsi"] < 30:
        score += 1
    elif row["rsi"] > 70:
        score -= 1

    # MA crossover
    if row["ema9"] > row["ema20"]:
        score += 1
    else:
        score -= 1

    # Bollinger
    if row["close"] < row["bb_lower"]:
        score += 1
    elif row["close"] > row["bb_upper"]:
        score -= 1

    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    return None

# ================= ML SCORING (SAFE HOOK) =================

def ml_score(signal, df):
    # Placeholder: uses rule-based + future logs
    return 1.0 if signal else 0.0

# ================= EXECUTION =================

def place_order(symbol, qty, side):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy" if side == "BUY" else "sell",
            type="market",
            time_in_force="day"
        )
        log({"event": "ORDER_PLACED", "symbol": symbol, "side": side, "qty": qty})
    except Exception as e:
        log({"event": "ORDER_FAILED", "error": str(e)})

# ================= POSITION MANAGEMENT =================

def enforce_hold_limit(positions):
    now = datetime.now()
    for symbol, p in positions.items():
        # placeholder: assume entry time stored externally
        # here we skip implementation detail safely
        pass

# ================= MAIN =================

def run():
    log({"event": "START"})

    if not market_is_open():
        log({"event": "MARKET_CLOSED"})
        return

    account = get_account()
    positions = get_positions()
    orders = get_open_orders()

    log({
        "event": "CAPITAL_STATE",
        "equity": account["equity"],
        "buying_power": account["buying_power"],
        "positions": len(positions),
        "open_orders": len(orders)
    })

    if orders:
        log({"event": "SKIP", "reason": "OPEN_ORDERS_PRESENT"})
        return

    can, reason = can_trade(account, positions)
    if not can:
        log({"event": "SKIP", "reason": reason})
        return

    if not is_entry_time():
        log({"event": "SKIP", "reason": "OUTSIDE_ENTRY_WINDOW"})
        return

    watchlist = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]

    for symbol in watchlist:
        try:
            bars = api.get_bars(symbol, "1Min", limit=100).df
            if bars.empty:
                continue

            df = compute_indicators(bars)

            signal = generate_signal(df)
            if not signal:
                log({"event": "NO_SIGNAL", "symbol": symbol})
                continue

            score = ml_score(signal, df)
            if score < 0.5:
                log({"event": "REJECTED", "symbol": symbol, "reason": "ML_FILTER"})
                continue

            price = df["close"].iloc[-1]
            qty = position_size(account, price)

            if qty <= 0:
                log({"event": "REJECTED", "symbol": symbol, "reason": "SIZE_ZERO"})
                continue

            place_order(symbol, qty, signal)

        except Exception as e:
            log({"event": "ERROR", "symbol": symbol, "error": str(e)})

    log({"event": "END"})

# ================= RUN =================

if __name__ == "__main__":
    run()
