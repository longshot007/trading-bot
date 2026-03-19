# ==========================
# Phase 5 Quant Trading Bot (Telegram Removed, Local Logging Only)
# ==========================
# ADDED:
# - Real-time PnL tracking
# - Auto nightly retraining
# - Symbol rotation (stop trading weak tickers)
# - Performance-based strategy weighting
# - Local logging instead of Telegram notifications
# ==========================

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
import joblib
import os
import threading
import time

# ==========================
# CONFIG
# ==========================
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD"]
performance_scores = {s:1.0 for s in SYMBOLS}
MARKET_SYMBOL = "SPY"
TIMEFRAME = "1Min"
LOOKBACK = 1000
MODEL_PATH = "model.pkl"

RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
MAX_TRADES_PER_DAY = 10
PROB_THRESHOLD = 0.65

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

trade_count = 0
start_equity = None
pnl_tracker = {}

# ==========================
# LOGGING
# ==========================
def log_message(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("trades.log", "a") as f:
        f.write(f"{timestamp} {msg}\n")
    print(msg)

# ==========================
# DATA
# ==========================
def get_data(symbol):
    bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df
    return bars[bars['symbol'] == symbol]

# ==========================
# FEATURES
# ==========================
def compute_features(df):
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']

    df['body_pct'] = df['body'] / df['range'].replace(0,1)
    df['upper_wick_pct'] = df['upper_wick'] / df['range'].replace(0,1)
    df['lower_wick_pct'] = df['lower_wick'] / df['range'].replace(0,1)

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    df['vwap'] = (df['volume'] * (df['high']+df['low']+df['close'])/3).cumsum() / df['volume'].cumsum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

    df['momentum'] = df['close'] - df['close'].shift(3)
    df['acceleration'] = df['momentum'] - df['momentum'].shift(1)

    return df.dropna()

# ==========================
# TARGET
# ==========================
def create_target(df):
    future_max = df['close'].rolling(3).max().shift(-3)
    future_min = df['close'].rolling(3).min().shift(-3)

    up_move = (future_max - df['close']) / df['close']
    down_move = (df['close'] - future_min) / df['close']

    df['target'] = (up_move > 0.0015) & (down_move < 0.0015)
    return df.dropna()

# ==========================
# TRAIN
# ==========================
def train_model():
    dfs = []
    for s in SYMBOLS:
        df = create_target(compute_features(get_data(s)))
        dfs.append(df)

    data = pd.concat(dfs)
    features = ['body_pct','upper_wick_pct','lower_wick_pct','volatility','volume_ratio','vwap_dist','momentum','acceleration']

    model = XGBClassifier(n_estimators=300, max_depth=6)
    model.fit(data[features], data['target'])

    joblib.dump(model, MODEL_PATH)
    log_message("Model retrained successfully")

# ==========================
# AUTO RETRAIN
# ==========================
def retrain_scheduler():
    while True:
        now = datetime.now()
        if now.hour == 2 and now.minute == 0:
            train_model()
        time.sleep(60)

# ==========================
# LOAD
# ==========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)

# ==========================
# SYMBOL ROTATION
# ==========================
def get_active_symbols():
    sorted_symbols = sorted(performance_scores, key=performance_scores.get, reverse=True)
    return sorted_symbols[:3]

# ==========================
# STRATEGY
# ==========================
def strategy_vote(df, model, symbol):
    latest = df.iloc[-1:]
    features = ['body_pct','upper_wick_pct','lower_wick_pct','volatility','volume_ratio','vwap_dist','momentum','acceleration']

    ml_prob = model.predict_proba(latest[features])[0][1]
    momentum_signal = latest['momentum'].values[0] > 0
    vwap_signal = latest['vwap_dist'].values[0] > 0

    weight = performance_scores[symbol]
    score = (ml_prob * weight) + momentum_signal + vwap_signal

    return score, ml_prob

# ==========================
# EXECUTION + PnL TRACK
# ==========================
def place_trade(symbol, price, volatility):
    qty = 1
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    pnl_tracker[symbol] = price
    log_message(f"BUY {symbol} at {price}")

# ==========================
# CHECK POSITIONS
# ==========================
def check_positions():
    positions = api.list_positions()
    for pos in positions:
        symbol = pos.symbol
        entry = pnl_tracker.get(symbol, float(pos.avg_entry_price))
        current = float(pos.current_price)
        pnl = (current - entry) / entry

        if pnl > 0.002 or pnl < -0.002:
            api.close_position(symbol)
            if pnl > 0:
                performance_scores[symbol] += 0.1
            else:
                performance_scores[symbol] -= 0.1
            log_message(f"CLOSE {symbol} PnL: {pnl:.4f}")

# ==========================
# MAIN LOOP
# ==========================
def run_bot():
    model = load_model()
    symbols = get_active_symbols()

    for symbol in symbols:
        df = compute_features(get_data(symbol))
        if len(df) < 100:
            continue

        score, prob = strategy_vote(df, model, symbol)
        if score > 2:
            price = df['close'].iloc[-1]
            vol = df['volatility'].iloc[-1]
            place_trade(symbol, price, vol)

    check_positions()

# ==========================
# ENTRY
# ==========================
if __name__ == "__main__":
    threading.Thread(target=retrain_scheduler).start()
    while True:
        run_bot()
        time.sleep(60)
