import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas as pd

import os

API_KEY = os.environ.get("ALPACA_API_KEY_ID")
SECRET_KEY = os.environ.get("ALPACA_API_SECRET_KEY")

BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

stocks = [
"AAPL","MSFT","NVDA","AMZN","GOOGL",
"META","TSLA","AMD","AVGO","NFLX"
]

def momentum(stock):

    data = yf.download(stock, period="30d", interval="1d")

    start = data["Close"].iloc[0]
    end = data["Close"].iloc[-1]

    return (end/start)-1

scores = {}

for s in stocks:

    try:
        scores[s] = momentum(s)
    except:
        pass

top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

for stock in top:

    api.submit_order(
        symbol=stock[0],
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc"
    )

    print("BUY", stock[0])
