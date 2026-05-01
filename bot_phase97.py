#!/usr/bin/env python3

"""
Institutional-Style Alpaca Momentum Bot
---------------------------------------

Key Features:
- Long-only architecture
- Hidden/internal stop losses (NOT sent to broker)
- ATR volatility-adjusted sizing
- Multi-stage filtering pipeline
- Timed exits
- Daily loss circuit breaker
- SEC fee accounting
- Persistent state management
- Cooldown system
- Portfolio exposure control
- Structured logging for future ML training

WARNING:
This is still a trading system and carries real financial risk.
Paper trade extensively before production deployment.
"""

import os
import json
import time
from datetime import datetime, timedelta

import pytz
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi


# =========================================================
# CONFIG
# =========================================================

MAX_POSITION_RISK_PCT     = 0.005      # 0.5% account risk per trade
MAX_TOTAL_EXPOSURE_PCT    = 0.25       # 25% deployed capital max
MAX_TRADES_PER_RUN        = 3

MIN_PRICE                 = 5
MIN_DOLLAR_VOLUME         = 2_000_000
MIN_AVG_VOLUME            = 200_000

ATR_STOP_MULTIPLIER       = 1.8
TAKE_PROFIT_MULTIPLIER    = 2.5

MAX_HOLD_DAYS             = 4

DAILY_LOSS_LIMIT          = 20_000

MAX_SPREAD_PCT            = 0.0035

COOLDOWN_MINUTES          = 60

SEC_FEE_RATE              = 0.0000206

TIMEZONE_ET               = pytz.timezone("US/Eastern")

LOG_FILE                  = "/tmp/institutional_bot_log.json"
STATE_FILE                = "/tmp/institutional_bot_state.json"


# =========================================================
# API
# =========================================================

api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    os.environ["APCA_API_BASE_URL"],
    api_version="v2"
)


# =========================================================
# LOGGING
# =========================================================

def log(data):
    data["ts"] = datetime.utcnow().isoformat()

    print(json.dumps(data), flush=True)

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass


# =========================================================
# STATE
# =========================================================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {
            "positions": {},
            "closed_pnl_today": 0,
            "cooldowns": {}
        }

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "positions": {},
            "closed_pnl_today": 0,
            "cooldowns": {}
        }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =========================================================
# TIME
# =========================================================

def now_et():
    return datetime.now(TIMEZONE_ET)


def market_is_open():
    return api.get_clock().is_open


def entry_window():

    t = now_et()

    return (9, 35) <= (t.hour, t.minute) <= (14, 0)


# =========================================================
# ACCOUNT
# =========================================================

def get_account():

    acct = api.get_account()

    return {
        "equity": float(acct.equity),
        "buying_power": float(acct.buying_power)
    }


def get_live_positions():

    out = {}

    for p in api.list_positions():

        out[p.symbol] = {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "unrealized_pl": float(p.unrealized_pl)
        }

    return out


# =========================================================
# SCANNER
# =========================================================

def get_universe():

    log({"event": "SCAN_START"})

    try:
        assets = api.list_assets(status="active")
    except Exception as e:
        log({"event": "SCAN_FAIL", "error": str(e)})
        return []

    symbols = []

    for a in assets:

        if not a.tradable:
            continue

        if "." in a.symbol:
            continue

        symbols.append(a.symbol)

    return symbols[:400]


# =========================================================
# DATA
# =========================================================

def get_bars(symbol):

    try:

        bars = api.get_bars(
            symbol,
            "1Min",
            limit=120
        ).df

        if bars.empty or len(bars) < 60:
            return None

        return bars

    except Exception:
        return None


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14).mean()
    avg_loss = loss.ewm(alpha=1/14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift())
    tr3 = abs(df["low"] - df["close"].shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["avg_volume"] = df["volume"].rolling(20).mean()

    return df


# =========================================================
# FILTERS
# =========================================================

def passes_filters(df):

    r = df.iloc[-1]

    price = float(r["close"])

    if price < MIN_PRICE:
        return False

    avg_volume = float(df["avg_volume"].iloc[-1])

    if avg_volume < MIN_AVG_VOLUME:
        return False

    dollar_volume = avg_volume * price

    if dollar_volume < MIN_DOLLAR_VOLUME:
        return False

    spread_pct = (r["high"] - r["low"]) / price

    if spread_pct > MAX_SPREAD_PCT:
        return False

    return True


# =========================================================
# SIGNALS
# =========================================================

def generate_signal(df):

    r = df.iloc[-1]

    volume_confirmed = (
        r["volume"] >
        df["volume"].rolling(20).mean().iloc[-1]
    )

    trend_confirmed = (
        r["ema9"] > r["ema20"]
    )

    momentum_confirmed = (
        55 < r["rsi"] < 75
    )

    if (
        trend_confirmed and
        momentum_confirmed and
        volume_confirmed
    ):
        return "BUY"

    return None


# =========================================================
# RISK
# =========================================================

def calculate_position_size(
    equity,
    entry_price,
    stop_price
):

    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return 0

    dollar_risk = equity * MAX_POSITION_RISK_PCT

    shares = int(dollar_risk / risk_per_share)

    return max(shares, 0)


# =========================================================
# EXECUTION
# =========================================================

def submit_buy(symbol, qty):

    try:

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )

        log({
            "event": "BUY_ORDER",
            "symbol": symbol,
            "qty": qty
        })

        return True

    except Exception as e:

        log({
            "event": "BUY_FAIL",
            "symbol": symbol,
            "error": str(e)
        })

        return False


def submit_sell(symbol, qty, reason, est_price):

    try:

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day"
        )

        sec_fee = qty * est_price * SEC_FEE_RATE

        log({
            "event": "SELL_ORDER",
            "symbol": symbol,
            "qty": qty,
            "reason": reason,
            "sec_fee": sec_fee
        })

        return sec_fee

    except Exception as e:

        log({
            "event": "SELL_FAIL",
            "symbol": symbol,
            "error": str(e)
        })

        return 0


# =========================================================
# EXIT ENGINE
# =========================================================

def manage_positions(state):

    live_positions = get_live_positions()

    for symbol, meta in list(state["positions"].items()):

        if symbol not in live_positions:
            continue

        qty = live_positions[symbol]["qty"]

        bars = get_bars(symbol)

        if bars is None:
            continue

        price = float(bars["close"].iloc[-1])

        entry_price = meta["entry_price"]

        stop_price = meta["stop_price"]

        target_price = meta["target_price"]

        entry_time = datetime.fromisoformat(meta["entry_time"])

        age = now_et() - entry_time

        exit_reason = None

        # hidden stop
        if price <= stop_price:
            exit_reason = "STOP"

        # take profit
        elif price >= target_price:
            exit_reason = "TARGET"

        # timed exit
        elif age > timedelta(days=MAX_HOLD_DAYS):
            exit_reason = "TIME_EXIT"

        if exit_reason:

            sec_fee = submit_sell(
                symbol,
                qty,
                exit_reason,
                price
            )

            pnl = (
                (price - entry_price) * qty
            ) - sec_fee

            state["closed_pnl_today"] += pnl

            del state["positions"][symbol]

            state["cooldowns"][symbol] = (
                now_et() + timedelta(
                    minutes=COOLDOWN_MINUTES
                )
            ).isoformat()

            log({
                "event": "POSITION_EXIT",
                "symbol": symbol,
                "reason": exit_reason,
                "pnl": pnl
            })


# =========================================================
# DAILY RISK CONTROL
# =========================================================

def risk_circuit_breaker(state):

    if state["closed_pnl_today"] <= -DAILY_LOSS_LIMIT:

        log({
            "event": "DAILY_STOP_TRIGGERED",
            "loss": state["closed_pnl_today"]
        })

        return True

    return False


# =========================================================
# MAIN
# =========================================================

def run():

    log({"event": "RUN_START"})

    if not market_is_open():

        log({"event": "MARKET_CLOSED"})
        return

    state = load_state()

    manage_positions(state)

    if risk_circuit_breaker(state):

        save_state(state)
        return

    if not entry_window():

        save_state(state)

        log({"event": "OUTSIDE_ENTRY_WINDOW"})
        return

    acct = get_account()

    equity = acct["equity"]

    live_positions = get_live_positions()

    current_exposure = sum(
        p["market_value"]
        for p in live_positions.values()
    )

    if (
        current_exposure >=
        equity * MAX_TOTAL_EXPOSURE_PCT
    ):

        log({"event": "MAX_EXPOSURE_REACHED"})

        save_state(state)
        return

    universe = get_universe()

    candidates = []

    for symbol in universe:

        if symbol in live_positions:
            continue

        cooldown = state["cooldowns"].get(symbol)

        if cooldown:

            if now_et() < datetime.fromisoformat(cooldown):
                continue

        bars = get_bars(symbol)

        if bars is None:
            continue

        df = add_indicators(bars)

        if not passes_filters(df):
            continue

        signal = generate_signal(df)

        if signal != "BUY":
            continue

        candidates.append((symbol, df))

    log({
        "event": "CANDIDATES",
        "count": len(candidates)
    })

    for symbol, df in candidates[:MAX_TRADES_PER_RUN]:

        r = df.iloc[-1]

        entry_price = float(r["close"])

        atr = float(r["atr"])

        if np.isnan(atr) or atr <= 0:
            continue

        stop_price = (
            entry_price -
            (atr * ATR_STOP_MULTIPLIER)
        )

        target_price = (
            entry_price +
            (
                atr *
                TAKE_PROFIT_MULTIPLIER
            )
        )

        qty = calculate_position_size(
            equity,
            entry_price,
            stop_price
        )

        if qty <= 0:
            continue

        estimated_cost = qty * entry_price

        if (
            current_exposure +
            estimated_cost
        ) > (
            equity *
            MAX_TOTAL_EXPOSURE_PCT
        ):
            continue

        ok = submit_buy(symbol, qty)

        if not ok:
            continue

        state["positions"][symbol] = {

            "entry_price": entry_price,

            "stop_price": stop_price,

            "target_price": target_price,

            "qty": qty,

            "entry_time": now_et().isoformat()
        }

        current_exposure += estimated_cost

        log({
            "event": "POSITION_OPENED",
            "symbol": symbol,
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "qty": qty
        })

    save_state(state)

    log({"event": "RUN_END"})


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    run()
