import os
import alpaca_trade_api as tradeapi
import pandas as pd
import yfinance as yf
from datetime import datetime

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

# ----- SETTINGS -----

WATCHLIST = [
    "AAPL","NVDA","TSLA","AMD","META",
    "MSFT","AMZN","GOOGL","SPY","QQQ"
]

MOMENTUM_LOOKBACK = 5
STOP_LOSS = 0.03
TAKE_PROFIT = 0.06
RISK_PER_TRADE = 0.05

# --------------------

def market_is_open():
    clock = api.get_clock()
    return clock.is_open

def get_account_balance():
    account = api.get_account()
    return float(account.cash)

def get_positions():
    try:
        return api.list_positions()
    except:
        return []

def get_momentum_score(ticker):
    data = yf.download(ticker, period="10d", interval="1d", progress=False)

    if len(data) < MOMENTUM_LOOKBACK:
        return 0

    recent = data["Close"].iloc[-MOMENTUM_LOOKBACK:]
    momentum = (recent[-1] - recent[0]) / recent[0]

    return momentum

def find_best_stock():

    scores = {}

    for ticker in WATCHLIST:
        try:
            score = get_momentum_score(ticker)
            scores[ticker] = score
        except:
            continue

    if not scores:
        return None

    best = max(scores, key=scores.get)

    if scores[best] > 0:
        return best

    return None

def calculate_position_size(price):

    cash = get_account_balance()

    risk_amount = cash * RISK_PER_TRADE

    shares = int(risk_amount / price)

    return max(shares, 1)

def place_trade(ticker):

    price = yf.download(ticker, period="1d", interval="1m", progress=False)["Close"].iloc[-1]

    shares = calculate_position_size(price)

    stop_price = round(price * (1 - STOP_LOSS), 2)
    take_price = round(price * (1 + TAKE_PROFIT), 2)

    api.submit_order(
        symbol=ticker,
        qty=shares,
        side="buy",
        type="market",
        time_in_force="gtc",
        order_class="bracket",
        stop_loss={"stop_price": stop_price},
        take_profit={"limit_price": take_price}
    )

    print(f"BUY {shares} {ticker}")
    print(f"Entry: {price}")
    print(f"Stop Loss: {stop_price}")
    print(f"Take Profit: {take_price}")

def already_holding(ticker):

    positions = get_positions()

    for p in positions:
        if p.symbol == ticker:
            return True

    return False

def run_bot():

    print("Bot started:", datetime.now())

    if not market_is_open():
        print("Market closed")
        return

    best_stock = find_best_stock()

    if best_stock is None:
        print("No momentum trade found")
        return

    if already_holding(best_stock):
        print("Already holding", best_stock)
        return

    place_trade(best_stock)

if __name__ == "__main__":
    run_bot()
