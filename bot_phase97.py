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

ENTRY_CUTOFF_PT    = (11, 0)
TIMEZONE_PT        = pytz.timezone("US/Pacific")

# GitHub Actions ephemeral runner — write to /tmp so paths always exist
LOG_FILE           = "/tmp/bot97_log.json"
STATE_FILE         = "/tmp/positions_state.json"

WATCHLIST          = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]

# ================= API =================
try:
    api = tradeapi.REST(
        os.environ["APCA_API_KEY_ID"],
        os.environ["APCA_API_SECRET_KEY"],
        os.environ["APCA_API_BASE_URL"],
        api_version="v2"
    )
except KeyError as e:
    raise SystemExit(f"[FATAL] Missing environment variable: {e}")

# ================= LOGGING =================
def log(data: dict):
    data["ts"] = datetime.utcnow().isoformat()
    print(json.dumps(data), flush=True)          # stdout captured by Actions
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except OSError as e:
        print(json.dumps({"event": "LOG_WRITE_ERROR", "err": str(e)}), flush=True)

def rotate_logs():
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 5_000_000:
            os.rename(LOG_FILE, f"{LOG_FILE}.{int(time.time())}")
    except OSError:
        pass

# ================= STATE =================
def load_state() -> dict:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log({"event": "STATE_LOAD_ERROR", "err": str(e)})
    return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError as e:
        log({"event": "STATE_SAVE_ERROR", "err": str(e)})

# ================= BROKER =================
def get_account():
    a = api.get_account()
    return float(a.equity), float(a.buying_power)

def get_positions() -> dict:
    return {p.symbol: float(p.qty) for p in api.list_positions()}

def get_open_orders() -> set:
    return {o.symbol for o in api.list_orders(status="open")}

# ================= TIME =================
def market_is_open() -> bool:
    return api.get_clock().is_open

def entry_allowed() -> bool:
    now = datetime.now(TIMEZONE_PT)
    return (now.hour, now.minute) < ENTRY_CUTOFF_PT

# ================= INDICATORS =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    return df

# ================= SIGNAL =================
def get_signal(df: pd.DataFrame):
    """Returns 'BUY', 'SELL', or None."""
    if len(df) < 20:
        return None
    r = df.iloc[-1]
    if pd.isna(r["rsi"]) or pd.isna(r["bb_upper"]):
        return None

    score = 0
    if r["rsi"] < 30:              score += 1
    if r["rsi"] > 70:              score -= 1
    if r["ema9"] > r["ema20"]:     score += 1
    else:                          score -= 1
    if r["close"] < r["bb_lower"]: score += 1
    if r["close"] > r["bb_upper"]: score -= 1

    if score >= 2:  return "BUY"
    if score <= -2: return "SELL"
    return None

def quality_score(df: pd.DataFrame) -> float:
    """Simple quality gate: 0.0–1.0. Requires score >= 0.5 to trade."""
    r = df.iloc[-1]
    score = 0
    score += int(r["ema9"] > r["ema20"])
    score += int(30 < r["rsi"] < 70)
    return score / 2

# ================= RISK =================
def position_size(equity: float, price: float) -> int:
    return max(0, int((equity * MAX_POSITION_PCT) // price))

# ================= HOLD ENFORCEMENT =================
def enforce_hold_limits(positions: dict, state: dict):
    now = datetime.utcnow()
    for symbol in list(positions):
        if symbol not in state:
            state[symbol] = {"entry": now.isoformat()}
        try:
            entry_dt = datetime.fromisoformat(state[symbol]["entry"])
        except (KeyError, ValueError):
            state[symbol] = {"entry": now.isoformat()}
            continue

        if (now - entry_dt).days >= MAX_HOLD_DAYS:
            qty  = abs(positions[symbol])
            side = "sell" if positions[symbol] > 0 else "buy"
            try:
                api.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type="market", time_in_force="day"
                )
                log({"event": "FORCE_EXIT", "symbol": symbol,
                     "held_days": (now - entry_dt).days})
            except Exception as e:
                log({"event": "FORCE_EXIT_ERROR", "symbol": symbol, "err": str(e)})
            state.pop(symbol, None)

    save_state(state)

# ================= KILL SWITCH =================
def check_kill_switch(positions: dict) -> bool:
    try:
        unrealized = sum(float(p.unrealized_pl) for p in api.list_positions())
    except Exception as e:
        log({"event": "PNL_CHECK_ERROR", "err": str(e)})
        return False

    if unrealized <= -DAILY_LOSS_LIMIT:
        log({"event": "KILL_SWITCH", "unrealized_pl": unrealized})
        for symbol, qty in positions.items():
            side = "sell" if qty > 0 else "buy"
            try:
                api.submit_order(
                    symbol=symbol, qty=abs(qty), side=side,
                    type="market", time_in_force="day"
                )
            except Exception as e:
                log({"event": "LIQUIDATE_ERROR", "symbol": symbol, "err": str(e)})
        return True
    return False

# ================= MAIN =================
def run():
    rotate_logs()
    log({"event": "RUN_START"})

    if not market_is_open():
        log({"event": "MARKET_CLOSED"})
        return

    try:
        equity, bp = get_account()
    except Exception as e:
        log({"event": "ACCOUNT_ERROR", "err": str(e)})
        return

    log({"event": "ACCOUNT", "equity": equity, "buying_power": bp})

    try:
        positions   = get_positions()
        open_orders = get_open_orders()
    except Exception as e:
        log({"event": "POSITIONS_ERROR", "err": str(e)})
        return

    if check_kill_switch(positions):
        return

    state = load_state()
    enforce_hold_limits(positions, state)

    # Refresh after hold exits
    positions   = get_positions()
    open_orders = get_open_orders()

    if not entry_allowed():
        log({"event": "PAST_ENTRY_CUTOFF"})
        log({"event": "RUN_END"})
        return

    for symbol in WATCHLIST:
        if symbol in positions or symbol in open_orders:
            log({"event": "SKIP", "symbol": symbol, "reason": "already_held_or_ordered"})
            continue

        try:
            bars = api.get_bars(symbol, "1Min", limit=100).df
        except Exception as e:
            log({"event": "BARS_ERROR", "symbol": symbol, "err": str(e)})
            continue

        if bars.empty or len(bars) < 20:
            log({"event": "SKIP", "symbol": symbol, "reason": "insufficient_data"})
            continue

        df  = add_indicators(bars)
        sig = get_signal(df)

        if not sig:
            log({"event": "NO_SIGNAL", "symbol": symbol})
            continue

        if quality_score(df) < 0.5:
            log({"event": "REJECT", "symbol": symbol, "reason": "quality_gate"})
            continue

        price = float(df["close"].iloc[-1])
        qty   = position_size(equity, price)

        if qty <= 0:
            log({"event": "SKIP", "symbol": symbol, "reason": "qty_zero"})
            continue

        order_value = qty * price
        if order_value > equity * BUYING_POWER_CAP:
            log({"event": "SKIP", "symbol": symbol,
                 "reason": "buying_power_cap", "order_value": order_value})
            continue

        side = "buy" if sig == "BUY" else "sell"
        try:
            api.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="market", time_in_force="day"
            )
            log({"event": "ORDER", "symbol": symbol, "side": side,
                 "qty": qty, "price": price, "value": order_value})
            state[symbol] = {"entry": datetime.utcnow().isoformat()}
            save_state(state)
        except Exception as e:
            log({"event": "ORDER_ERROR", "symbol": symbol, "err": str(e)})

        # Refresh account after each order
        try:
            equity, bp = get_account()
        except Exception as e:
            log({"event": "ACCOUNT_REFRESH_ERROR", "err": str(e)})
            break

    log({"event": "RUN_END"})


if __name__ == "__main__":
    run()
