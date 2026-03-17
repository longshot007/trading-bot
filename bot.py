import os
import alpaca_trade_api as tradeapi

print("=== BOT START ===")

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

SYMBOLS = ["AAPL", "TSLA", "NVDA"]
TRADE_QTY = 1


def market_open():
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        print("Clock error:", e)
        return False


def latest_price(symbol):
    try:
        bar = api.get_latest_bar(symbol)
        return bar.c
    except Exception as e:
        print("Price error:", e)
        return None


def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False


def place_trade(symbol):
    try:
        print("Buying", symbol)

        api.submit_order(
            symbol=symbol,
            qty=TRADE_QTY,
            side="buy",
            type="market",
            time_in_force="day"
        )

    except Exception as e:
        print("Trade failed:", e)


def run_bot():

    if not market_open():
        print("Market closed")
        return

    for symbol in SYMBOLS:

        price = latest_price(symbol)

        if price is None:
            continue

        print(symbol, "price:", price)

        if not position_exists(symbol):
            place_trade(symbol)

    print("Run complete")


run_bot()

print("=== BOT END ===")

if __name__ == "__main__":
    run_bot()
