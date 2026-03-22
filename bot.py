import os
import math
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi


# ============================================================
# PHASE BOT — CORRECTED / GITHUB ACTIONS SAFE / TURNKEY
# - No local model files
# - No sklearn/joblib dependency
# - Market-hours guard
# - Existing-position guard
# - Capital allocation guard
# - SEC Section 31 fee integrated in execution logging
# - Lightweight candlestick + momentum scoring
# - Safe for scheduled GitHub Actions runs
# ============================================================


# =========================
# ENV / API
# =========================
API_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
BASE_URL = os.getenv("APCA_API_BASE_URL", "").strip()

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials in environment variables.")

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


# =========================
# CONFIG
# =========================
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META",
    "AMZN", "GOOGL", "TSLA", "SPY", "QQQ"
]

TIMEFRAME = tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute)
LOOKBACK_BARS = 120

MAX_NEW_POSITIONS_PER_RUN = 2
MAX_TOTAL_POSITIONS = 4

RISK_PER_TRADE = 0.20          # 20% of available cash max per new entry
MIN_CASH_RESERVE = 25.00       # leave some idle cash
MIN_SHARE_PRICE = 5.00
MAX_SHARE_PRICE = 1000.00

STOP_LOSS_PCT = 0.020          # 2.0%
TAKE_PROFIT_PCT = 0.030        # 3.0%
TRAIL_ARM_PCT = 0.015          # once gain exceeds 1.5%, arm a trailing-style exit
TRAIL_GIVEBACK_PCT = 0.008     # exit if gain gives back below peak by 0.8%

MIN_ENTRY_SCORE = 5.0
ENTRY_COOLDOWN_MINUTES = 10

SEC_FEE_RATE = 0.0000206       # user-required SEC Section 31 fee rate on sells

STATE_FILE = "bot_state.json"
TRADE_LOG_FILE = "trade_log.csv"


# =========================
# UTILITIES
# =========================
def now_utc():
    return datetime.now(timezone.utc)


def iso_now():
    return now_utc().isoformat()


def log(msg):
    print(f"[{iso_now()}] {msg}", flush=True)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_buys": {}, "peaks": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"last_buys": {}, "peaks": {}}
        data.setdefault("last_buys", {})
        data.setdefault("peaks", {})
        return data
    except Exception as e:
        log(f"State load failed, using blank state: {e}")
        return {"last_buys": {}, "peaks": {}}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"State save failed: {e}")


def ensure_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        header = (
            "timestamp,action,symbol,qty,price,notional,sec_fee,reason,"
            "entry_price,current_price,pnl_dollars,pnl_pct\n"
        )
        with open(TRADE_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)


def append_trade_log(
    action,
    symbol,
    qty,
    price,
    notional,
    sec_fee,
    reason,
    entry_price="",
    current_price="",
    pnl_dollars="",
    pnl_pct=""
):
    ensure_trade_log()
    row = (
        f"{iso_now()},{action},{symbol},{qty},{price:.4f},{notional:.4f},"
        f"{sec_fee:.6f},{reason},{entry_price},{current_price},{pnl_dollars},{pnl_pct}\n"
    )
    with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(row)


# =========================
# MARKET / ACCOUNT
# =========================
def get_clock():
    return api.get_clock()


def market_is_open():
    try:
        return bool(get_clock().is_open)
    except Exception as e:
        log(f"Clock check failed: {e}")
        return False


def get_account():
    return api.get_account()


def get_cash():
    return safe_float(get_account().cash)


def get_buying_power():
    account = get_account()
    bp = safe_float(getattr(account, "buying_power", 0.0))
    if bp <= 0:
        bp = safe_float(getattr(account, "cash", 0.0))
    return bp


def list_positions_map():
    positions = {}
    try:
        for p in api.list_positions():
            positions[p.symbol] = p
    except Exception as e:
        log(f"Failed to fetch positions: {e}")
    return positions


def list_open_orders_symbols():
    symbols = set()
    try:
        for o in api.list_orders(status="open"):
            symbols.add(o.symbol)
    except Exception as e:
        log(f"Failed to fetch open orders: {e}")
    return symbols


# =========================
# DATA
# =========================
def get_bars_df(symbol, limit=LOOKBACK_BARS):
    try:
        raw = api.get_bars(symbol, TIMEFRAME, limit=limit).df
        if raw is None or raw.empty:
            return None

        df = raw.copy()

        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()

        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0)
            except Exception:
                pass

        if df.empty:
            return None

        df = df.sort_index().copy()
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None

        return df
    except Exception as e:
        log(f"{symbol}: bar fetch failed: {e}")
        return None


def add_features(df):
    df = df.copy()

    df["ret1"] = df["close"].pct_change()
    df["ret3"] = df["close"].pct_change(3)
    df["ret10"] = df["close"].pct_change(10)

    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = df["close"] - df["open"]
    df["body_abs"] = df["body"].abs()

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_pct_range"] = df["body_abs"] / df["range"]
    df["close_pos_in_range"] = (df["close"] - df["low"]) / df["range"]

    df["vol_ma10"] = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma10"]

    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["bullish_engulfing"] = (
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] > df["open"]) &
        (df["open"] <= df["close"].shift(1)) &
        (df["close"] >= df["open"].shift(1))
    ).astype(int)

    df["bearish_engulfing"] = (
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"] < df["open"]) &
        (df["open"] >= df["close"].shift(1)) &
        (df["close"] <= df["open"].shift(1))
    ).astype(int)

    df["hammer"] = (
        (df["lower_wick"] >= 2 * df["body_abs"]) &
        (df["upper_wick"] <= df["body_abs"]) &
        (df["close"] > df["open"])
    ).astype(int)

    df["shooting_star"] = (
        (df["upper_wick"] >= 2 * df["body_abs"]) &
        (df["lower_wick"] <= df["body_abs"]) &
        (df["close"] < df["open"])
    ).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# =========================
# SIGNALS
# =========================
def entry_score(df):
    if df is None or len(df) < 30:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    reasons = []

    # Trend / momentum
    if row["close"] > row["ema8"] > row["ema21"]:
        score += 2.0
        reasons.append("trend_up")

    if row["ret3"] > 0:
        score += 1.0
        reasons.append("ret3_up")

    if row["ret10"] > 0:
        score += 1.0
        reasons.append("ret10_up")

    # Volume confirmation
    if row["vol_ratio"] >= 1.20:
        score += 1.0
        reasons.append("volume_confirm")

    # Candle structure
    if row["close"] > row["open"]:
        score += 0.5
        reasons.append("green_candle")

    if row["body_pct_range"] >= 0.50:
        score += 0.5
        reasons.append("strong_body")

    if row["close_pos_in_range"] >= 0.70:
        score += 0.5
        reasons.append("strong_close")

    # Candlestick patterns
    if row["bullish_engulfing"] == 1:
        score += 1.5
        reasons.append("bullish_engulfing")

    if row["hammer"] == 1:
        score += 1.0
        reasons.append("hammer")

    # RSI sweet spot
    if 50 <= row["rsi14"] <= 68:
        score += 1.0
        reasons.append("rsi_ok")

    # Avoid obvious overextension
    if row["rsi14"] > 74:
        score -= 2.0
        reasons.append("rsi_overbought")

    # Avoid bearish reversal
    if row["bearish_engulfing"] == 1:
        score -= 2.0
        reasons.append("bearish_engulfing")

    if row["shooting_star"] == 1:
        score -= 1.5
        reasons.append("shooting_star")

    # Require bar-to-bar improvement
    if row["close"] > prev["close"]:
        score += 0.5
        reasons.append("close_gt_prev_close")

    return {
        "score": round(score, 4),
        "price": float(row["close"]),
        "reasons": reasons,
    }


# =========================
# RISK / POSITION SIZING
# =========================
def allowed_new_positions(current_positions_count):
    remaining = MAX_TOTAL_POSITIONS - current_positions_count
    return max(0, min(MAX_NEW_POSITIONS_PER_RUN, remaining))


def compute_order_qty(price, available_cash):
    if price <= 0:
        return 0

    usable_cash = max(0.0, available_cash - MIN_CASH_RESERVE)
    budget = usable_cash * RISK_PER_TRADE

    if budget < price:
        return 0

    qty = int(math.floor(budget / price))
    return max(0, qty)


def sec_fee_for_sell(notional):
    return float(notional) * SEC_FEE_RATE


def can_reenter_symbol(state, symbol):
    ts = state.get("last_buys", {}).get(symbol)
    if not ts:
        return True
    try:
        prior = datetime.fromisoformat(ts)
        return now_utc() - prior >= timedelta(minutes=ENTRY_COOLDOWN_MINUTES)
    except Exception:
        return True


# =========================
# ORDER HELPERS
# =========================
def submit_market_buy(symbol, qty, reason):
    if qty <= 0:
        return False

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        notional_est = 0.0
        try:
            notional_est = safe_float(getattr(order, "notional", 0.0))
        except Exception:
            pass
        append_trade_log(
            action="BUY",
            symbol=symbol,
            qty=qty,
            price=0.0,
            notional=notional_est,
            sec_fee=0.0,
            reason=reason
        )
        log(f"BUY submitted: {symbol} qty={qty} reason={reason}")
        return True
    except Exception as e:
        log(f"BUY failed: {symbol} qty={qty} error={e}")
        return False


def submit_market_sell(symbol, qty, current_price, entry_price, reason):
    if qty <= 0:
        return False

    notional = qty * current_price
    sec_fee = sec_fee_for_sell(notional)
    pnl_dollars = (current_price - entry_price) * qty - sec_fee
    pnl_pct = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0.0

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day"
        )
        append_trade_log(
            action="SELL",
            symbol=symbol,
            qty=qty,
            price=current_price,
            notional=notional,
            sec_fee=sec_fee,
            reason=reason,
            entry_price=round(entry_price, 4),
            current_price=round(current_price, 4),
            pnl_dollars=round(pnl_dollars, 4),
            pnl_pct=round(pnl_pct, 6),
        )
        log(
            f"SELL submitted: {symbol} qty={qty} price={current_price:.4f} "
            f"entry={entry_price:.4f} pnl={pnl_dollars:.4f} fee={sec_fee:.6f} reason={reason}"
        )
        return True
    except Exception as e:
        log(f"SELL failed: {symbol} qty={qty} error={e}")
        return False


# =========================
# POSITION MANAGEMENT
# =========================
def manage_positions(state):
    positions = list_positions_map()
    open_order_symbols = list_open_orders_symbols()

    for symbol, p in positions.items():
        if symbol in open_order_symbols:
            log(f"{symbol}: skipping manage, open order exists")
            continue

        qty = safe_int(p.qty)
        entry_price = safe_float(p.avg_entry_price)
        current_price = safe_float(p.current_price)

        if qty <= 0 or entry_price <= 0 or current_price <= 0:
            continue

        pnl_pct = (current_price - entry_price) / entry_price

        peaks = state.setdefault("peaks", {})
        peak = peaks.get(symbol, pnl_pct)
        if pnl_pct > peak:
            peak = pnl_pct
            peaks[symbol] = peak

        # Hard stop
        if pnl_pct <= -STOP_LOSS_PCT:
            submit_market_sell(
                symbol=symbol,
                qty=qty,
                current_price=current_price,
                entry_price=entry_price,
                reason="stop_loss"
            )
            peaks.pop(symbol, None)
            continue

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            submit_market_sell(
                symbol=symbol,
                qty=qty,
                current_price=current_price,
                entry_price=entry_price,
                reason="take_profit"
            )
            peaks.pop(symbol, None)
            continue

        # Trailing-style giveback once enough profit exists
        if peak >= TRAIL_ARM_PCT and pnl_pct <= (peak - TRAIL_GIVEBACK_PCT):
            submit_market_sell(
                symbol=symbol,
                qty=qty,
                current_price=current_price,
                entry_price=entry_price,
                reason="trail_giveback"
            )
            peaks.pop(symbol, None)
            continue

    return state


# =========================
# ENTRY SCAN / EXECUTION
# =========================
def find_candidates(state):
    positions = list_positions_map()
    open_order_symbols = list_open_orders_symbols()
    candidates = []

    for symbol in SYMBOLS:
        if symbol in positions:
            log(f"{symbol}: skip, already held")
            continue

        if symbol in open_order_symbols:
            log(f"{symbol}: skip, open order exists")
            continue

        if not can_reenter_symbol(state, symbol):
            log(f"{symbol}: skip, cooldown active")
            continue

        df = get_bars_df(symbol)
        if df is None or len(df) < 30:
            log(f"{symbol}: skip, insufficient data")
            continue

        df = add_features(df)
        if df is None or len(df) < 30:
            log(f"{symbol}: skip, insufficient feature data")
            continue

        sig = entry_score(df)
        if sig is None:
            log(f"{symbol}: skip, no signal")
            continue

        price = sig["price"]
        score = sig["score"]

        if price < MIN_SHARE_PRICE or price > MAX_SHARE_PRICE:
            log(f"{symbol}: skip, price filter failed price={price:.2f}")
            continue

        if score >= MIN_ENTRY_SCORE:
            candidates.append({
                "symbol": symbol,
                "score": score,
                "price": price,
                "reasons": sig["reasons"],
            })
            log(f"{symbol}: candidate score={score} reasons={','.join(sig['reasons'])}")
        else:
            log(f"{symbol}: reject score={score}")

    candidates.sort(key=lambda x: (x["score"], x["price"]), reverse=True)
    return candidates


def place_entries(state):
    positions = list_positions_map()
    slots = allowed_new_positions(len(positions))
    if slots <= 0:
        log("No available slots for new positions.")
        return state

    candidates = find_candidates(state)
    if not candidates:
        log("No valid entry candidates.")
        return state

    buys_done = 0
    available_cash = min(get_cash(), get_buying_power())

    for c in candidates:
        if buys_done >= slots:
            break

        symbol = c["symbol"]
        price = c["price"]
        score = c["score"]
        reason = "|".join(c["reasons"][:8])

        qty = compute_order_qty(price, available_cash)
        if qty <= 0:
            log(f"{symbol}: qty computed as 0")
            continue

        estimated_cost = qty * price
        if estimated_cost > max(0.0, available_cash - MIN_CASH_RESERVE):
            log(f"{symbol}: insufficient available cash for qty={qty}")
            continue

        ok = submit_market_buy(
            symbol=symbol,
            qty=qty,
            reason=f"entry_score={score}|{reason}"
        )
        if ok:
            available_cash -= estimated_cost
            state.setdefault("last_buys", {})[symbol] = iso_now()
            buys_done += 1

    if buys_done == 0:
        log("No entries placed.")
    else:
        log(f"Entries placed: {buys_done}")

    return state


# =========================
# MAIN
# =========================
def run():
    log("Bot starting.")
    state = load_state()

    if not market_is_open():
        log("Market is closed. Exiting cleanly.")
        save_state(state)
        return

    state = manage_positions(state)
    time.sleep(1)
    state = place_entries(state)
    save_state(state)
    log("Bot finished cleanly.")


if __name__ == "__main__":
    run()
