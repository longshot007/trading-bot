import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

SEC_FEE_RATE = 0.0000206

MAX_POSITIONS = 6
MAX_RISK_PER_TRADE = 0.02   # 2% equity risk
MIN_DOLLAR_VOLUME = 5_000_000

SLEEP_INTERVAL = 60

# =========================
# API LAYER
# =========================
def api_get(url, params=None):
    return requests.get(url, headers=HEADERS, params=params).json()

def api_post(url, payload):
    return requests.post(url, json=payload, headers=HEADERS).json()

def get_account():
    return api_get(f"{BASE_URL}/v2/account")

def get_positions():
    return api_get(f"{BASE_URL}/v2/positions")

def submit_order(symbol, qty, side):
    return api_post(f"{BASE_URL}/v2/orders", {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    })

def get_bars(symbol, timeframe, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timeframe": timeframe
    }

    data = api_get(f"{BASE_URL}/v2/stocks/{symbol}/bars", params)

    if "bars" not in data:
        return None

    return pd.DataFrame(data["bars"])

# =========================
# CORE CALCULATIONS
# =========================
def calculate_sec_fee(value):
    return value * SEC_FEE_RATE

def atr(df, period=14):
    high = df["h"]
    low = df["l"]
    close = df["c"]

    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return tr.rolling(period).mean().iloc[-1]

# =========================
# MARKET REGIME
# =========================
def market_regime():
    spy = get_bars("SPY", "1Day", 60)
    if spy is None or len(spy) < 50:
        return "neutral"

    ma50 = spy["c"].rolling(50).mean().iloc[-1]
    price = spy["c"].iloc[-1]

    if price > ma50:
        return "bull"
    elif price < ma50:
        return "bear"
    return "neutral"

# =========================
# DYNAMIC SCANNER
# =========================
def get_tradeable_universe():
    assets = api_get(f"{BASE_URL}/v2/assets")

    tradable = [
        a["symbol"] for a in assets
        if a["tradable"] and a["status"] == "active"
    ]

    return tradable[:200]  # cap for performance

def liquidity_filter(symbol):
    df = get_bars(symbol, "1Day", 5)
    if df is None:
        return False

    dollar_volume = (df["c"] * df["v"]).mean()

    return dollar_volume > MIN_DOLLAR_VOLUME

# =========================
# MULTI-TIMEFRAME SCORING
# =========================
def score_symbol(symbol):
    df_5m = get_bars(symbol, "5Min", 1)
    df_1h = get_bars(symbol, "1Hour", 5)
    df_d = get_bars(symbol, "1Day", 20)

    if df_5m is None or df_1h is None or df_d is None:
        return None

    # momentum across frames
    m1 = df_5m["c"].pct_change().tail(10).mean()
    m2 = df_1h["c"].pct_change().tail(10).mean()
    m3 = df_d["c"].pct_change().tail(5).mean()

    # alignment score
    alignment = (m1 > 0) + (m2 > 0) + (m3 > 0)

    if alignment < 2:
        return None

    # volatility penalty
    vol = df_1h["c"].pct_change().std()

    return (m1 * 2 + m2 + m3) - vol

# =========================
# POSITION SIZING
# =========================
def position_size(equity, symbol):
    df = get_bars(symbol, "1Hour", 5)
    if df is None:
        return 0

    current_price = df["c"].iloc[-1]
    atr_value = atr(df)

    if atr_value == 0 or np.isnan(atr_value):
        return 0

    risk_per_share = atr_value
    max_risk = equity * MAX_RISK_PER_TRADE

    qty = int(max_risk / risk_per_share)

    return max(qty, 0)

# =========================
# SELL LOGIC
# =========================
def should_sell(position):
    symbol = position["symbol"]
    df = get_bars(symbol, "5Min", 1)

    if df is None:
        return False

    trend = df["c"].pct_change().tail(5).mean()

    if trend < -0.003:
        return True

    entry = float(position["avg_entry_price"])
    current = float(position["current_price"])

    if current >= entry * 1.03:
        return True

    return False

# =========================
# MAIN LOOP
# =========================
def run_bot():
    print("Phase 4 Bot Running...")

    while True:
        try:
            account = get_account()
            equity = float(account["equity"])

            positions = get_positions()
            held = [p["symbol"] for p in positions]

            regime = market_regime()
            print(f"\n[{datetime.now()}] Regime: {regime} | Equity: ${equity:.2f}")

            # ===== SELL =====
            for p in positions:
                if should_sell(p):
                    symbol = p["symbol"]
                    qty = float(p["qty"])
                    price = float(p["current_price"])

                    value = qty * price
                    sec_fee = calculate_sec_fee(value)
                    net = value - sec_fee
                    cost = float(p["avg_entry_price"]) * qty
                    profit = net - cost

                    submit_order(symbol, qty, "sell")

                    print(f"\nSELL {symbol}")
                    print(f"Gross: ${value:.2f}")
                    print(f"SEC Fee: ${sec_fee:.4f}")
                    print(f"Net: ${net:.2f}")
                    print(f"Profit: ${profit:.2f}")

            # ===== BUY =====
            if len(positions) < MAX_POSITIONS:
                universe = get_tradeable_universe()

                candidates = []

                for symbol in universe:
                    if symbol in held:
                        continue

                    if not liquidity_filter(symbol):
                        continue

                    score = score_symbol(symbol)
                    if score:
                        candidates.append((symbol, score))

                ranked = sorted(candidates, key=lambda x: x[1], reverse=True)

                for symbol, _ in ranked[:10]:
                    qty = position_size(equity, symbol)
                    if qty <= 0:
                        continue

                    submit_order(symbol, qty, "buy")

                    print(f"\nBUY {symbol} | Qty: {qty}")
                    break

        except Exception as e:
            print("ERROR:", e)

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    run_bot()
