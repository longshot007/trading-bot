import os
import csv
import time
from datetime import datetime
import alpaca_trade_api as tradeapi

print("=== PHASE-4 PROFESSIONAL BOT START ===")

# ==========================
# Alpaca API Config
# ==========================
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Change to live URL for live trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ==========================
# Bot Config
# ==========================
TICKERS = [
    # Top 200 tickers, example subset here
    "AAPL","TSLA","NVDA","AMD","META","AMZN","MSFT","GOOGL",
    "SPY","QQQ","INTC","F","GM","NFLX","BA","SHOP","SQ","PYPL",
    "CRM","ORCL","IBM","V","MA","JNJ","PFE","MRNA","BNTX",
    "UBER","LYFT","SNAP","TWTR","DIS","NKE","SBUX","MCD",
    "COST","WMT","T","VZ","XOM","CVX","BP","TOT","RDS-A",
    "CAT","DE","GS","JPM","BAC"
]

RISK_PER_TRADE = 0.01           # Max 1% equity per trade
MOMENTUM_THRESHOLD = 0.003      # 0.3% momentum
VOLUME_SURGE = 1.5              # 50% above avg volume
VOLATILITY_THRESHOLD = 0.02     # Max 2% intrabar volatility
STOP_LOSS = 0.97                 # 3% stop
TRAILING_STOP = 0.98             # 2% trailing stop
TAKE_PROFIT = 1.05               # 5% profit
TIMEFRAME = "5Min"
DAILY_RISK_LIMIT = 0.05          # Max 5% equity loss per day
MAX_TRADES_PER_DAY = 10
LOG_FILE = "phase4_trades_log.csv"

# ==========================
# Utility Functions
# ==========================
def market_open():
    try:
        return api.get_clock().is_open
    except Exception as e:
        print("Clock error:", e)
        return False

def account_equity():
    try:
        return float(api.get_account().equity)
    except Exception as e:
        print("Account error:", e)
        return 10000.0  # fallback

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def get_latest_price(symbol):
    try:
        bar = api.get_latest_bar(symbol)
        return float(bar.c)
    except Exception as e:
        print(f"Price error {symbol}:", e)
        return None

def get_previous_price(symbol):
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=2).df
        if len(bars) < 2:
            return None
        return float(bars.close.iloc[-2])
    except Exception as e:
        print(f"Previous price error {symbol}:", e)
        return None

def get_avg_volume(symbol, lookback=20):
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=lookback).df
        return bars.volume.mean()
    except Exception as e:
        print(f"Volume error {symbol}:", e)
        return None

def calculate_qty(symbol):
    equity = account_equity()
    price = get_latest_price(symbol)
    if price is None or price == 0:
        return 0
    risk_amount = equity * RISK_PER_TRADE
    qty = int(risk_amount / price)
    return max(qty, 1)

def log_trade(symbol, action, price, qty):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), symbol, action, price, qty])

# ==========================
# Strategy Checks
# ==========================
def check_stop_take_trailing():
    try:
        positions = api.list_positions()
        for p in positions:
            entry = float(p.avg_entry_price)
            current = float(p.current_price)
            qty = p.qty
            # Stop-loss
            if current < entry * STOP_LOSS:
                print(f"STOP LOSS {p.symbol}")
                api.submit_order(symbol=p.symbol, qty=qty, side="sell", type="market", time_in_force="day")
                log_trade(p.symbol, "SELL_STOP", current, qty)
            # Take-profit
            elif current > entry * TAKE_PROFIT:
                print(f"TAKE PROFIT {p.symbol}")
                api.submit_order(symbol=p.symbol, qty=qty, side="sell", type="market", time_in_force="day")
                log_trade(p.symbol, "SELL_PROFIT", current, qty)
            # Trailing stop
            elif current < entry * TRAILING_STOP:
                print(f"TRAILING STOP {p.symbol}")
                api.submit_order(symbol=p.symbol, qty=qty, side="sell", type="market", time_in_force="day")
                log_trade(p.symbol, "SELL_TRAIL", current, qty)
    except Exception as e:
        print("Stop/take check failed:", e)

def check_momentum_volume(symbol):
    price_now = get_latest_price(symbol)
    price_prev = get_previous_price(symbol)
    avg_vol = get_avg_volume(symbol)
    if None in (price_now, price_prev, avg_vol):
        return False
    momentum = (price_now - price_prev) / price_prev
    vol_bar = api.get_bars(symbol, TIMEFRAME, limit=1).df.volume.iloc[-1]
    # Volatility filter
    if abs(momentum) > VOLATILITY_THRESHOLD:
        return False
    if momentum > MOMENTUM_THRESHOLD and vol_bar > avg_vol * VOLUME_SURGE:
        print(symbol, "momentum:", round(momentum,4), "volume surge:", round(vol_bar/avg_vol,2))
        return True
    return False

# ==========================
# Main Bot Loop
# ==========================
def run_bot():
    if not market_open():
        print("Market closed")
        return
    check_stop_take_trailing()
    trades_today = 0
    for symbol in TICKERS:
        if trades_today >= MAX_TRADES_PER_DAY:
            print("Max trades reached today")
            break
        try:
            if position_exists(symbol):
                continue
            if check_momentum_volume(symbol):
                qty = calculate_qty(symbol)
                if qty > 0:
                    print(f"BUYING {symbol} qty {qty}")
                    api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
                    log_trade(symbol, "BUY", get_latest_price(symbol), qty)
                    trades_today += 1
        except Exception as e:
            print(f"Error scanning {symbol}:", e)
            time.sleep(1)

run_bot()
print("=== PHASE-4 PROFESSIONAL BOT COMPLETE ===")
