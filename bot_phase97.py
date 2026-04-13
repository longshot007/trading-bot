import os, json, time
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

ENTRY_CUTOFF_PT = (11, 0)
TIMEZONE_PT = pytz.timezone("US/Pacific")

LOG_FILE = "bot97_log.json"
STATE_FILE = "positions_state.json"

# ================= API =================
api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    os.environ["APCA_API_BASE_URL"],
    api_version="v2"
)

# ================= LOGGING =================
def log(data):
    data["ts"] = datetime.utcnow().isoformat()
    print(data)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

def rotate_logs():
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 5_000_000:
        os.rename(LOG_FILE, f"{LOG_FILE}.{int(time.time())}")

# ================= STATE =================
def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"))

# ================= BROKER =================
def get_account():
    a = api.get_account()
    return float(a.equity), float(a.buying_power)

def get_positions():
    return {p.symbol: float(p.qty) for p in api.list_positions()}

def get_orders():
    orders = api.list_orders(status="open")
    return {o.symbol: True for o in orders}

# ================= TIME =================
def market_open():
    return api.get_clock().is_open

def entry_ok():
    now = datetime.now(TIMEZONE_PT)
    return (now.hour, now.minute) < ENTRY_CUTOFF_PT

# ================= INDICATORS =================
def indicators(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100/(1+rs))

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2*std
    df["bb_lower"] = mid - 2*std
    return df

# ================= SIGNAL =================
def signal(df):
    r = df.iloc[-1]
    score = 0
    if r["rsi"] < 30: score += 1
    if r["rsi"] > 70: score -= 1
    if r["ema9"] > r["ema20"]: score += 1
    else: score -= 1
    if r["close"] < r["bb_lower"]: score += 1
    if r["close"] > r["bb_upper"]: score -= 1

    if score >= 2: return "BUY"
    if score <= -2: return "SELL"
    return None

def ml_score(df):
    r = df.iloc[-1]
    score = 0
    score += (r["ema9"] > r["ema20"])
    score += (30 < r["rsi"] < 70)
    return score / 2

# ================= RISK =================
def exposure(positions, prices):
    return sum(abs(q * prices.get(s,0)) for s,q in positions.items())

def size(equity, price):
    return int((equity * MAX_POSITION_PCT) // price)

# ================= HOLD LIMIT =================
def enforce_hold(positions, state):
    now = datetime.utcnow()
    for s in list(positions):
        if s not in state:
            state[s] = {"entry": now.isoformat()}
        entry = datetime.fromisoformat(state[s]["entry"])
        if (now - entry).days >= MAX_HOLD_DAYS:
            api.submit_order(symbol=s, qty=abs(positions[s]),
                             side="sell" if positions[s]>0 else "buy",
                             type="market", time_in_force="day")
            log({"event":"FORCE_EXIT", "symbol":s})
            del state[s]
    save_state(state)

# ================= MAIN =================
def run():
    rotate_logs()

    if not market_open():
        log({"event":"MARKET_CLOSED"})
        return

    equity, bp = get_account()
    positions = get_positions()
    orders = get_orders()
    state = load_state()

    # Daily loss enforcement
    unrealized = sum([float(p.unrealized_pl) for p in api.list_positions()])
    if unrealized <= -DAILY_LOSS_LIMIT:
        for s,q in positions.items():
            api.submit_order(symbol=s, qty=abs(q),
                             side="sell" if q>0 else "buy",
                             type="market", time_in_force="day")
        log({"event":"KILL_SWITCH"})
        return

    enforce_hold(positions, state)

    watchlist = ["AAPL","TSLA","NVDA","AMD","SPY"]

    for s in watchlist:
        if not entry_ok():
            break

        if s in positions or s in orders:
            continue

        bars = api.get_bars(s, "1Min", limit=100).df
        if bars.empty:
            continue

        df = indicators(bars)
        sig = signal(df)
        if not sig:
            continue

        if ml_score(df) < 0.5:
            log({"event":"REJECT", "symbol":s, "reason":"ML"})
            continue

        price = df["close"].iloc[-1]
        qty = size(equity, price)
        if qty <= 0:
            continue

        # Buying power cap enforcement
        if (qty * price) > equity * BUYING_POWER_CAP:
            continue

        side = "buy" if sig=="BUY" else "sell"
        api.submit_order(symbol=s, qty=qty, side=side,
                         type="market", time_in_force="day")

        log({"event":"ORDER", "symbol":s, "side":side, "qty":qty})

        # refresh state after each order (CRITICAL FIX)
        equity, bp = get_account()
        positions = get_positions()

    log({"event":"END"})

if __name__ == "__main__":
    run()
