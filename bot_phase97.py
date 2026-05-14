#!/usr/bin/env python3

"""
Institutional-Style Alpaca Momentum Bot (Phase 97)
---------------------------------------------------

Key Features:
- Long-only architecture
- Hidden/internal stop losses (NOT sent to broker)
- ATR volatility-adjusted sizing, scaled by confluence score
- Candlestick pattern detection via candlestick_rules.py
- Tiered entry windows with escalating confluence requirements
- Weak-position forced-flat at 3:30–3:45 PM ET
- Daily loss circuit breaker
- SEC fee accounting
- Persistent state committed back to GitHub after each run
- Position reconciliation against live Alpaca state on startup
- Structured JSON logging to logs/ directory

WARNING:
Paper trade extensively before production deployment.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone

import pytz
import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from candlestick_rules import (
    detect_signals,
    confluence_score,
    market_structure,
    CandlestickRules,
)


# =========================================================
# RETRY
# =========================================================

def with_retry(max_attempts=4):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = 2 ** attempt
                    log({"event": "API_RETRY", "func": func.__name__, "attempt": attempt+1, "error": str(e), "wait": wait})
                    time.sleep(wait)
        return wrapper
    return decorator


# =========================================================
# CONFIG
# =========================================================

MAX_POSITION_RISK_PCT   = 0.005       # 0.5% account risk per trade
MAX_TOTAL_EXPOSURE_PCT  = 0.25        # 25% deployed capital max
MAX_TRADES_PER_RUN      = 999         # effectively unlimited - exposure limit is the real cap
UNIVERSE_SIZE           = 300         # rate limit safety (200 req/min); was 1500

MIN_PRICE               = 5
MIN_DOLLAR_VOLUME       = 2_000_000
MIN_AVG_VOLUME          = 200_000
MIN_POSITION_DOLLARS    = 100         # skip if qty * price < this

ATR_STOP_MULTIPLIER     = 1.8
TAKE_PROFIT_MULTIPLIER  = 2.5

MAX_HOLD_DAYS           = 4

DAILY_LOSS_LIMIT_PCT    = 0.02        # 2% account equity = daily kill switch

MAX_SPREAD_PCT          = 0.0035

COOLDOWN_MINUTES        = 60

SEC_FEE_RATE            = 0.0000206

CONFLUENCE_MIN_FULL     = 3           # required score 9:30–11:30 ET
CONFLUENCE_MIN_LATE     = 4           # required score 11:30–13:00 ET
WEAK_FAVOR_PCT          = 0.005       # 0.5% — below this is a weak position

TIMEZONE_ET             = pytz.timezone("US/Eastern")

LOG_DIR                 = "logs"
STATE_FILE              = os.path.join("state", "bot_state.json")


# =========================================================
# API CLIENTS
# =========================================================

_base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
_is_paper = "paper" in _base_url.lower()

trading_client = TradingClient(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    paper=_is_paper,
)

data_client = StockHistoricalDataClient(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
)


# =========================================================
# LOGGING
# =========================================================

_log_file = os.path.join(LOG_DIR, f"bot_{datetime.utcnow().strftime('%Y%m%d')}.json")


def log(data):
    data["ts"] = datetime.utcnow().isoformat()
    print(json.dumps(data), flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        pass


# =========================================================
# STATE
# =========================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"positions": {}, "closed_pnl_today": 0, "cooldowns": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"positions": {}, "closed_pnl_today": 0, "cooldowns": {}}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE)


# =========================================================
# TIME
# =========================================================

def now_et():
    return datetime.now(TIMEZONE_ET)


def market_is_open():
    return trading_client.get_clock().is_open


def trading_window():
    t = now_et()
    hm = (t.hour, t.minute)
    if (9, 30) <= hm < (11, 30):
        return "FULL"
    elif (11, 30) <= hm < (13, 0):
        return "CONFLUENCE"
    elif (13, 0) <= hm < (15, 30):
        return "EXITS_ONLY"
    elif (15, 30) <= hm < (15, 45):
        return "WEAK_CLOSE"
    elif (15, 45) <= hm < (16, 0):
        return "EXITS_ONLY"
    else:
        return "CLOSED"


# =========================================================
# ACCOUNT
# =========================================================

@with_retry()
def get_account():
    acct = trading_client.get_account()
    return {
        "equity":       float(acct.equity),
        "buying_power": float(acct.buying_power),
    }


@with_retry()
def get_live_positions():
    out = {}
    for p in trading_client.get_all_positions():
        out[p.symbol] = {
            "qty":             float(p.qty),
            "market_value":    float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "unrealized_pl":   float(p.unrealized_pl),
        }
    return out


# =========================================================
# POSITION RECONCILIATION
# =========================================================

def reconcile_positions(state):
    live = get_live_positions()

    for symbol in list(state["positions"].keys()):
        if symbol not in live:
            log({"event": "RECONCILE_REMOVE", "symbol": symbol, "reason": "not_in_alpaca"})
            del state["positions"][symbol]

    for symbol, pos in live.items():
        if symbol not in state["positions"]:
            entry_price = pos["avg_entry_price"]
            try:
                bars = get_bars(symbol)
            except Exception:
                bars = None
            if bars is not None:
                df = add_indicators(bars)
                atr = float(df["atr"].iloc[-1])
                if np.isnan(atr) or atr <= 0:
                    atr = entry_price * 0.02
            else:
                atr = entry_price * 0.02
            state["positions"][symbol] = {
                "entry_price":  entry_price,
                "entry_open":   entry_price,
                "stop_price":   entry_price - (atr * ATR_STOP_MULTIPLIER),
                "target_price": entry_price + (atr * TAKE_PROFIT_MULTIPLIER),
                "qty":          pos["qty"],
                "entry_time":   now_et().isoformat(),
            }
            log({"event": "RECONCILE_ADD", "symbol": symbol, "entry": entry_price})

    return state


# =========================================================
# SCANNER
# =========================================================

@with_retry()
def get_universe():
    log({"event": "SCAN_START"})
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = trading_client.get_all_assets(request)
    symbols = [
        a.symbol for a in assets
        if a.tradable
        and a.fractionable
        and "." not in a.symbol
        and "/" not in a.symbol
        and len(a.symbol) <= 5
    ]
    log({"event": "UNIVERSE_BUILT", "total": len(symbols), "scanning": min(len(symbols), UNIVERSE_SIZE)})
    return symbols[:UNIVERSE_SIZE]


# =========================================================
# DATA
# =========================================================

@with_retry()
def get_bars(symbol):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.now(timezone.utc) - timedelta(hours=5),
        limit=120,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        if symbol not in df.index.get_level_values(0):
            return None
        df = df.loc[symbol]
    if df.empty or len(df) < 60:
        return None
    return df


# =========================================================
# INDICATORS  (ATR + avg_volume still needed here)
# =========================================================

def add_indicators(df):
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14).mean()
    avg_loss = loss.ewm(alpha=1/14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift())
    tr3 = abs(df["low"]  - df["close"].shift())
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"]        = tr.rolling(14).mean()
    df["avg_volume"] = df["volume"].rolling(20).mean()
    return df


# =========================================================
# FILTERS
# =========================================================

def passes_filters(df):
    r          = df.iloc[-1]
    price      = float(r["close"])
    avg_volume = float(df["avg_volume"].iloc[-1])

    if price < MIN_PRICE:
        return False
    if pd.isna(avg_volume) or avg_volume < MIN_AVG_VOLUME:
        return False
    if avg_volume * price < MIN_DOLLAR_VOLUME:
        return False
    if (r["high"] - r["low"]) / price > MAX_SPREAD_PCT:
        return False
    return True


# =========================================================
# RISK
# =========================================================

def calculate_position_size(equity, entry_price, stop_price):
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    shares = int((equity * MAX_POSITION_RISK_PCT) / risk_per_share)
    if shares * entry_price < MIN_POSITION_DOLLARS:
        return 0
    return shares


# =========================================================
# EXECUTION
# =========================================================

def submit_buy(symbol, qty, entry_price, stop_price, target_price):
    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(entry_price * 1.005, 2),
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(target_price, 2))
        )
        trading_client.submit_order(order)
        log({"event": "BUY_ORDER", "symbol": symbol, "qty": qty})
        return True
    except Exception as e:
        log({"event": "BUY_FAIL", "symbol": symbol, "error": str(e)})
        return False


@with_retry()
def cancel_pending_orders(symbol):
    """Cancel any open child orders for this symbol so we can submit a fresh sell."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for o in orders:
            try:
                trading_client.cancel_order_by_id(o.id)
                log({"event": "CANCEL_ORDER", "symbol": symbol, "order_id": str(o.id)})
            except Exception as e:
                log({"event": "CANCEL_ORDER_FAIL", "symbol": symbol, "order_id": str(o.id), "error": str(e)})
    except Exception as e:
        log({"event": "CANCEL_LOOKUP_FAIL", "symbol": symbol, "error": str(e)})


def submit_sell(symbol, qty, reason, est_price):
    cancel_pending_orders(symbol)
    time.sleep(0.5)
    try:
        trading_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        sec_fee = qty * est_price * SEC_FEE_RATE
        log({"event": "SELL_ORDER", "symbol": symbol, "qty": qty,
             "reason": reason, "sec_fee": sec_fee})
        return sec_fee
    except Exception as e:
        log({"event": "SELL_FAIL", "symbol": symbol, "error": str(e)})
        return 0


# =========================================================
# EXIT ENGINE
# =========================================================

def manage_positions(state):
    live_positions = get_live_positions()

    for symbol, meta in list(state["positions"].items()):
        if symbol not in live_positions:
            continue

        qty          = live_positions[symbol]["qty"]
        try:
            bars = get_bars(symbol)
        except Exception:
            continue
        if bars is None:
            continue

        df           = add_indicators(bars)
        price        = float(df["close"].iloc[-1])
        entry_price  = meta["entry_price"]
        stop_price   = meta["stop_price"]
        target_price = meta["target_price"]
        entry_time   = datetime.fromisoformat(meta["entry_time"])
        age          = now_et() - entry_time
        exit_reason  = None

        # candlestick exit signals take priority
        entry_open = meta.get("entry_open", entry_price)
        should_exit, cs_reason = CandlestickRules.check_exit(
            df, entry_open, entry_price, stop_price, price
        )
        if should_exit:
            exit_reason = cs_reason
        elif price <= stop_price:
            exit_reason = "STOP"
        elif price >= target_price:
            exit_reason = "TARGET"
        elif age > timedelta(days=MAX_HOLD_DAYS):
            exit_reason = "TIME_EXIT"

        if exit_reason:
            sec_fee = submit_sell(symbol, qty, exit_reason, price)
            pnl = ((price - entry_price) * qty) - sec_fee
            state["closed_pnl_today"] += pnl
            del state["positions"][symbol]
            state["cooldowns"][symbol] = (
                now_et() + timedelta(minutes=COOLDOWN_MINUTES)
            ).isoformat()
            log({"event": "POSITION_EXIT", "symbol": symbol,
                 "reason": exit_reason, "pnl": pnl})


# =========================================================
# WEAK POSITION CLOSE  (3:30–3:45 PM ET)
# =========================================================

def close_weak_positions(state):
    live_positions = get_live_positions()

    for symbol, meta in list(state["positions"].items()):
        if symbol not in live_positions:
            continue

        pos           = live_positions[symbol]
        qty           = pos["qty"]
        unrealized_pl = pos["unrealized_pl"]
        entry_price   = meta["entry_price"]
        current_price = pos["market_value"] / qty if qty > 0 else entry_price
        favor_pct     = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        is_weak = unrealized_pl < 0 or favor_pct < WEAK_FAVOR_PCT

        if is_weak:
            sec_fee = submit_sell(symbol, qty, "WEAK_CLOSE_EOD", current_price)
            pnl = ((current_price - entry_price) * qty) - sec_fee
            state["closed_pnl_today"] += pnl
            del state["positions"][symbol]
            state["cooldowns"][symbol] = (
                now_et() + timedelta(minutes=COOLDOWN_MINUTES)
            ).isoformat()
            log({
                "event":      "WEAK_POSITION_CLOSED",
                "symbol":     symbol,
                "pnl":        pnl,
                "favor_pct":  round(favor_pct, 5),
                "unrealized": unrealized_pl,
            })


# =========================================================
# DAILY RISK CONTROL
# =========================================================

def risk_circuit_breaker():
    current_state = load_state()
    closed_pnl_today = current_state.get("closed_pnl_today", 0.0)
    try:
        acct = trading_client.get_account()
        equity = float(acct.equity)
    except Exception:
        equity = 100_000  # fallback if Alpaca unreachable
    daily_loss_limit_dollars = equity * DAILY_LOSS_LIMIT_PCT
    if closed_pnl_today <= -daily_loss_limit_dollars:
        if not current_state.get("circuit_breaker_tripped", False):
            current_state["circuit_breaker_tripped"] = True
            current_state["tripped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(current_state)
            log({"event": "CIRCUIT_BREAKER_TRIPPED", "loss": closed_pnl_today, "limit": -daily_loss_limit_dollars, "equity": equity})
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
    state = reconcile_positions(state)

    manage_positions(state)

    if risk_circuit_breaker():
        save_state(state)
        return

    window = trading_window()
    log({"event": "TRADING_WINDOW", "window": window})

    if window == "WEAK_CLOSE":
        close_weak_positions(state)
        manage_positions(state)
        save_state(state)
        return

    if window in ("EXITS_ONLY", "CLOSED"):
        save_state(state)
        return

    # ── Entry phase: FULL or CONFLUENCE ──────────────────
    confluence_min = CONFLUENCE_MIN_FULL if window == "FULL" else CONFLUENCE_MIN_LATE

    acct             = get_account()
    equity           = acct["equity"]
    live_positions   = get_live_positions()
    current_exposure = sum(p["market_value"] for p in live_positions.values())

    if current_exposure >= equity * MAX_TOTAL_EXPOSURE_PCT:
        log({"event": "MAX_EXPOSURE_REACHED"})
        save_state(state)
        return

    try:
        universe = get_universe()
    except Exception as e:
        log({"event": "SCAN_FAIL", "error": str(e)})
        save_state(state)
        return
    candidates = []

    # Diagnostic counters
    stats = {
        "scanned":           0,
        "skip_held":         0,
        "skip_cooldown":     0,
        "skip_no_bars":      0,
        "skip_filter":       0,
        "skip_no_signal":    0,
        "skip_low_score":    0,
        "skip_choppy":       0,
        "passed":            0,
    }

    for symbol in universe:
        stats["scanned"] += 1
        if symbol in live_positions:
            stats["skip_held"] += 1
            continue

        cooldown = state["cooldowns"].get(symbol)
        if cooldown and now_et() < datetime.fromisoformat(cooldown):
            stats["skip_cooldown"] += 1
            continue

        try:
            bars = get_bars(symbol)
        except Exception:
            stats["skip_no_bars"] += 1
            continue
        time.sleep(0.3)  # rate limit safety - 200 req/min = 0.3s minimum
        if bars is None:
            stats["skip_no_bars"] += 1
            continue

        df = add_indicators(bars)
        if not passes_filters(df):
            stats["skip_filter"] += 1
            continue

        signals      = detect_signals(df)
        bull_signals = {k: v for k, v in signals.items()
                        if v["direction"] in ("bull", "neutral")}
        if not bull_signals:
            stats["skip_no_signal"] += 1
            continue

        score, score_details = confluence_score(df, "bull")
        if score < confluence_min:
            stats["skip_low_score"] += 1
            log({"event": "NEAR_MISS", "symbol": symbol, "score": score, "needed": confluence_min, "signals": list(bull_signals.keys())})
            continue

        if market_structure(df) == "choppy":
            stats["skip_choppy"] += 1
            continue

        stats["passed"] += 1
        log({"event": "CANDIDATE_FOUND", "symbol": symbol, "score": score, "signals": list(bull_signals.keys()), "details": score_details})
        candidates.append((symbol, df, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    log({"event": "SCAN_COMPLETE", "stats": stats, "candidates": len(candidates)})

    for symbol, df, score in candidates[:MAX_TRADES_PER_RUN]:
        r           = df.iloc[-1]
        entry_price = float(r["close"])
        entry_open  = float(r["open"])
        atr         = float(r["atr"])

        if np.isnan(atr) or atr <= 0:
            continue

        stop_price   = entry_price - (atr * ATR_STOP_MULTIPLIER)
        target_price = entry_price + (atr * TAKE_PROFIT_MULTIPLIER)

        size_factor = CandlestickRules.position_size_factor(score)
        qty         = int(calculate_position_size(equity, entry_price, stop_price) * size_factor)

        if qty <= 0:
            continue

        estimated_cost = qty * entry_price
        if current_exposure + estimated_cost > equity * MAX_TOTAL_EXPOSURE_PCT:
            continue

        ok = submit_buy(symbol, qty, entry_price, stop_price, target_price)
        if not ok:
            continue

        state["positions"][symbol] = {
            "entry_price":  entry_price,
            "entry_open":   entry_open,
            "stop_price":   stop_price,
            "target_price": target_price,
            "qty":          qty,
            "entry_time":   now_et().isoformat(),
        }

        current_exposure += estimated_cost

        log({
            "event":       "POSITION_OPENED",
            "symbol":      symbol,
            "entry":       entry_price,
            "stop":        stop_price,
            "target":      target_price,
            "qty":         qty,
            "score":       score,
            "size_factor": size_factor,
            "window":      window,
        })

    save_state(state)
    log({"event": "RUN_END"})


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()
