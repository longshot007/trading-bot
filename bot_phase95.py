import os
import json
import math
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi


BOT_NAME = "bot_phase95.py"

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")

LOG_FILE = "phase95_run_log.csv"
TRADE_LOG_FILE = "phase95_trade_log.csv"
SUMMARY_FILE = "phase95_run_summary.json"
STATE_FILE = "phase95_state.json"

SEC_FEE_RATE = 0.0000206  # estimated SEC Section 31 fee on sells

# Session controls
MARKET_OPEN_ET = (9, 30)
ENTRY_START_ET = (9, 40)
ENTRY_END_ET = (13, 0)
FORCE_FLAT_ET = (15, 45)

# Risk controls
RISK_PER_TRADE = 0.01
MAX_TOTAL_POSITIONS = 4
MAX_NEW_ORDERS_PER_RUN = 2
MAX_NOTIONAL_PER_POSITION_PCT = 0.20
MIN_ORDER_NOTIONAL = 250.0

# Scanner controls
MIN_PRICE = 5.0
MAX_PRICE = 250.0
MIN_RVOL = 1.05
MIN_DOLLAR_VOL_1M = 200000
SIGNAL_THRESHOLD = 3.2
PROFIT_TARGET_R = 2.0
TRAIL_TRIGGER_R = 1.0
TRAIL_STOP_R_MULT = 0.30  # once >1R, trail to lock ~0.3R
MAX_BARS = 120  # enough to cover session start through early afternoon

# Longable / shortable universes separated
LONG_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "PLTR",
    "MU", "AVGO", "QCOM", "INTC", "SMCI", "CRM", "UBER", "SHOP", "COIN", "HOOD",
    "ADBE", "PANW", "SNOW", "CRWD", "ARM", "TSM", "BABA", "JPM", "BAC", "WFC",
    "XOM", "CVX", "OXY", "SLB", "HAL", "PFE", "LLY", "MRK", "UNH", "ABBV",
    "SPY", "QQQ", "IWM", "DIA", "SOXL", "TQQQ", "SQQQ", "MARA", "RIOT", "MSTR",
    "DIS", "BA", "CAT", "DE", "NKE", "COST", "WMT", "HD", "LOW", "ORCL"
]

SHORT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "PLTR",
    "MU", "AVGO", "QCOM", "INTC", "SMCI", "CRM", "UBER", "SHOP", "COIN", "HOOD",
    "ADBE", "PANW", "SNOW", "CRWD", "ARM", "TSM", "BABA", "JPM", "BAC", "WFC",
    "XOM", "CVX", "OXY", "SLB", "HAL", "PFE", "LLY", "MRK", "UNH", "ABBV",
    "SPY", "QQQ", "IWM", "DIA", "MARA", "RIOT", "MSTR",
    "DIS", "BA", "CAT", "DE", "NKE", "COST", "WMT", "HD", "LOW", "ORCL"
]

BASE_UNIVERSE = sorted(set(LONG_UNIVERSE) | set(SHORT_UNIVERSE))


def now_utc() -> datetime:
    return datetime.now(UTC)


def now_et() -> datetime:
    return now_utc().astimezone(ET)


def now_pt() -> datetime:
    return now_utc().astimezone(PT)


def stamp() -> str:
    return f"[ET {now_et().strftime('%Y-%m-%d %H:%M:%S')} | PT {now_pt().strftime('%H:%M:%S')}]"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_round_qty(qty: float) -> str:
    qty = float(qty)
    if abs(qty - round(qty)) < 1e-9:
        return str(int(round(qty)))
    return f"{qty:.6f}".rstrip("0").rstrip(".")


def append_csv(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def log(message: str) -> None:
    line = f"{stamp()} {message}"
    print(line)
    append_csv(
        LOG_FILE,
        {
            "utc_time": now_utc().isoformat(),
            "et_time": now_et().isoformat(),
            "pt_time": now_pt().isoformat(),
            "message": message,
        },
    )


def write_summary(summary: dict) -> None:
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_state() -> dict:
    today = now_et().date().isoformat()
    default_state = {
        "session_date": today,
        "traded_today": [],
        "entry_meta": {}
    }

    if not os.path.exists(STATE_FILE):
        return default_state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return default_state

    if state.get("session_date") != today:
        return default_state

    if "traded_today" not in state or not isinstance(state["traded_today"], list):
        state["traded_today"] = []
    if "entry_meta" not in state or not isinstance(state["entry_meta"], dict):
        state["entry_meta"] = {}

    return state


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reset_state_if_new_day() -> dict:
    state = load_state()
    state["session_date"] = now_et().date().isoformat()
    state.setdefault("traded_today", [])
    state.setdefault("entry_meta", {})
    return state


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_api() -> tradeapi.REST:
    key = env_required("APCA_API_KEY_ID")
    secret = env_required("APCA_API_SECRET_KEY")
    base_url = env_required("APCA_API_BASE_URL")
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    return tradeapi.REST(key, secret, base_url, api_version="v2")


def market_is_open(api: tradeapi.REST) -> bool:
    return bool(api.get_clock().is_open)


def get_session_open_et(dt_et: datetime) -> datetime:
    return dt_et.replace(
        hour=MARKET_OPEN_ET[0],
        minute=MARKET_OPEN_ET[1],
        second=0,
        microsecond=0
    )


def in_entry_window(dt_et: datetime) -> bool:
    start = dt_et.replace(hour=ENTRY_START_ET[0], minute=ENTRY_START_ET[1], second=0, microsecond=0)
    end = dt_et.replace(hour=ENTRY_END_ET[0], minute=ENTRY_END_ET[1], second=0, microsecond=0)
    return start <= dt_et <= end


def should_force_flat(dt_et: datetime) -> bool:
    cutoff = dt_et.replace(hour=FORCE_FLAT_ET[0], minute=FORCE_FLAT_ET[1], second=0, microsecond=0)
    return dt_et >= cutoff


def get_equity(api: tradeapi.REST) -> float:
    return safe_float(api.get_account().equity)


def get_buying_power(api: tradeapi.REST) -> float:
    return safe_float(api.get_account().buying_power)


def get_positions(api: tradeapi.REST) -> list:
    try:
        return list(api.list_positions())
    except Exception:
        return []


def get_open_orders(api: tradeapi.REST) -> list:
    try:
        return list(api.list_orders(status="open", limit=500))
    except Exception:
        return []


def symbols_with_exposure(api: tradeapi.REST) -> set:
    out = set()
    for p in get_positions(api):
        out.add(p.symbol)
    for o in get_open_orders(api):
        out.add(o.symbol)
    return out


def cancel_orders_for_symbol(api: tradeapi.REST, symbol: str) -> None:
    for order in get_open_orders(api):
        if getattr(order, "symbol", "") == symbol:
            try:
                api.cancel_order(order.id)
                log(f"Cancelled open order for {symbol} | order_id={order.id}")
            except Exception as exc:
                log(f"Cancel failed for {symbol}: {exc}")


def close_position_market(api: tradeapi.REST, position, reason: str) -> None:
    symbol = position.symbol
    qty_abs = abs(safe_float(position.qty))
    if qty_abs <= 0:
        return

    side = "sell" if safe_float(position.qty) > 0 else "buy"

    try:
        cancel_orders_for_symbol(api, symbol)
        api.submit_order(
            symbol=symbol,
            qty=safe_round_qty(qty_abs),
            side=side,
            type="market",
            time_in_force="day",
        )
        append_csv(
            TRADE_LOG_FILE,
            {
                "utc_time": now_utc().isoformat(),
                "et_time": now_et().isoformat(),
                "pt_time": now_pt().isoformat(),
                "symbol": symbol,
                "action": "force_close",
                "qty": qty_abs,
                "side": side,
                "reason": reason,
                "market_value": safe_float(getattr(position, "market_value", 0)),
                "unrealized_pl": safe_float(getattr(position, "unrealized_pl", 0)),
            },
        )
        log(f"Flattened {symbol} qty={qty_abs} reason={reason}")
    except Exception as exc:
        log(f"Failed to flatten {symbol}: {exc}")


def get_latest_trade_price(api: tradeapi.REST, symbol: str) -> float:
    try:
        return safe_float(api.get_latest_trade(symbol).price)
    except Exception:
        return 0.0


def get_order_map_by_symbol(api: tradeapi.REST) -> dict:
    order_map = {}
    for order in get_open_orders(api):
        order_map.setdefault(order.symbol, []).append(order)
    return order_map


def cancel_order_by_id(api: tradeapi.REST, order_id: str) -> None:
    try:
        api.cancel_order(order_id)
    except Exception as exc:
        log(f"Cancel order failed {order_id}: {exc}")


def submit_trailing_bracket_replacement(
    api: tradeapi.REST,
    symbol: str,
    position_qty_abs: float,
    strategy_side: str,
    target_price: float,
    new_stop_price: float,
) -> bool:
    try:
        close_side = "sell" if strategy_side == "long" else "buy"
        api.submit_order(
            symbol=symbol,
            qty=safe_round_qty(position_qty_abs),
            side=close_side,
            type="market",
            time_in_force="day",
            order_class="oto",
            stop_loss={"stop_price": round(new_stop_price, 2)},
        )
        log(f"Submitted replacement protective stop for {symbol} at {new_stop_price:.2f}")
        return True
    except Exception as exc:
        log(f"Failed replacement stop for {symbol}: {exc}")
        return False


def manage_open_positions(api: tradeapi.REST, state: dict) -> None:
    positions = get_positions(api)
    if not positions:
        log("No live positions to manage.")
        return

    et_now = now_et()

    if should_force_flat(et_now):
        log("Force-flat time reached. Closing all positions.")
        for position in positions:
            close_position_market(api, position, "force_flat_eod")
        return

    order_map = get_order_map_by_symbol(api)

    managed = 0
    for position in positions:
        symbol = position.symbol
        qty_signed = safe_float(position.qty)
        qty_abs = abs(qty_signed)
        if qty_abs <= 0:
            continue

        strategy_side = "long" if qty_signed > 0 else "short"
        current_price = get_latest_trade_price(api, symbol)
        if current_price <= 0:
            continue

        meta = state.get("entry_meta", {}).get(symbol, {})
        entry_price = safe_float(meta.get("entry_price"), safe_float(getattr(position, "avg_entry_price", 0)))
        initial_stop = safe_float(meta.get("initial_stop"), 0)
        initial_target = safe_float(meta.get("target_price"), 0)
        stop_distance = safe_float(meta.get("stop_distance"), 0)

        if entry_price <= 0 or initial_stop <= 0 or stop_distance <= 0:
            continue

        current_r = (
            (current_price - entry_price) / stop_distance
            if strategy_side == "long"
            else (entry_price - current_price) / stop_distance
        )

        symbol_orders = order_map.get(symbol, [])
        stop_orders = []
        tp_orders = []

        for order in symbol_orders:
            order_type = str(getattr(order, "type", "")).lower()
            order_side = str(getattr(order, "side", "")).lower()
            if strategy_side == "long":
                if order_side == "sell":
                    if order_type in {"stop", "stop_limit"}:
                        stop_orders.append(order)
                    elif order_type == "limit":
                        tp_orders.append(order)
            else:
                if order_side == "buy":
                    if order_type in {"stop", "stop_limit"}:
                        stop_orders.append(order)
                    elif order_type == "limit":
                        tp_orders.append(order)

        current_stop_price = None
        if stop_orders:
            try:
                current_stop_price = max(
                    safe_float(getattr(o, "stop_price", 0)) if strategy_side == "short"
                    else safe_float(getattr(o, "stop_price", 0))
                    for o in stop_orders
                )
            except Exception:
                current_stop_price = None

        # Active deterioration logic
        # If position has gone materially negative after midday, flatten it.
        if et_now.hour >= 12:
            if current_r <= -0.75:
                close_position_market(api, position, "midday_deterioration")
                managed += 1
                continue

        # Trail once position reaches >= 1R
        if current_r >= TRAIL_TRIGGER_R and stop_orders:
            if strategy_side == "long":
                desired_stop = max(entry_price + (stop_distance * TRAIL_STOP_R_MULT), current_price - (stop_distance * 0.8))
                desired_stop = round(desired_stop, 2)
                improve = current_stop_price is None or desired_stop > current_stop_price + 0.01
            else:
                desired_stop = min(entry_price - (stop_distance * TRAIL_STOP_R_MULT), current_price + (stop_distance * 0.8))
                desired_stop = round(desired_stop, 2)
                improve = current_stop_price is None or desired_stop < current_stop_price - 0.01

            if improve:
                for order in stop_orders:
                    cancel_order_by_id(api, order.id)

                close_side = "sell" if strategy_side == "long" else "buy"
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=safe_round_qty(qty_abs),
                        side=close_side,
                        type="stop",
                        stop_price=desired_stop,
                        time_in_force="day",
                    )
                    log(
                        f"Trailed stop for {symbol} | side={strategy_side} "
                        f"old_stop={current_stop_price} new_stop={desired_stop} current_r={current_r:.2f}"
                    )
                    state["entry_meta"].setdefault(symbol, {})
                    state["entry_meta"][symbol]["trailed_stop"] = desired_stop
                except Exception as exc:
                    log(f"Failed trailing stop update for {symbol}: {exc}")

        managed += 1

    log(f"Managing {managed} live position(s).")


def get_bars_df(api: tradeapi.REST, symbols: list[str], timeframe: str = "1Min", limit: int = MAX_BARS) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    try:
        bars = api.get_bars(symbols, timeframe, limit=limit, adjustment="raw").df
        if bars is None or bars.empty:
            return pd.DataFrame()
        bars = bars.reset_index()
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars
    except Exception as exc:
        log(f"get_bars failed: {exc}")
        return pd.DataFrame()


def filter_to_current_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    dt_et_now = now_et()
    session_open_et = get_session_open_et(dt_et_now)
    session_open_utc = session_open_et.astimezone(UTC)

    out = df[df["timestamp"] >= pd.Timestamp(session_open_utc)].copy()
    if out.empty:
        return out

    out["timestamp_et"] = out["timestamp"].dt.tz_convert(ET)
    return out


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = filter_to_current_session(df)
    if df.empty:
        return df

    chunks = []
    for symbol, group in df.groupby("symbol"):
        g = group.sort_values("timestamp").copy()

        if g.empty:
            continue

        g["ema9"] = g["close"].ewm(span=9, adjust=False).mean()
        g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()
        g["ema50"] = g["close"].ewm(span=50, adjust=False).mean()

        # Correct session-anchored VWAP
        typical_price = (g["high"] + g["low"] + g["close"]) / 3.0
        cumulative_vol = g["volume"].cumsum()
        cumulative_pv = (typical_price * g["volume"]).cumsum()
        g["vwap"] = cumulative_pv / np.where(cumulative_vol == 0, np.nan, cumulative_vol)

        g["prev_close"] = g["close"].shift(1)
        g["tr1"] = g["high"] - g["low"]
        g["tr2"] = (g["high"] - g["prev_close"]).abs()
        g["tr3"] = (g["low"] - g["prev_close"]).abs()
        g["tr"] = g[["tr1", "tr2", "tr3"]].max(axis=1)
        g["atr14"] = g["tr"].rolling(14).mean()

        g["ret_5"] = g["close"].pct_change(5)
        g["ret_15"] = g["close"].pct_change(15)
        g["ret_30"] = g["close"].pct_change(30)

        g["avg_vol_20"] = g["volume"].rolling(20).mean()
        g["rvol"] = g["volume"] / np.where(g["avg_vol_20"] == 0, np.nan, g["avg_vol_20"])
        g["dollar_vol_1m"] = g["close"] * g["volume"]

        g["rolling_high_20"] = g["high"].rolling(20).max().shift(1)
        g["rolling_low_20"] = g["low"].rolling(20).min().shift(1)

        chunks.append(g)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def score_latest_row(row: pd.Series, symbol: str) -> tuple[str | None, float]:
    price = safe_float(row["close"])
    vwap = safe_float(row["vwap"])
    ema9 = safe_float(row["ema9"])
    ema20 = safe_float(row["ema20"])
    ema50 = safe_float(row["ema50"])
    ret_5 = safe_float(row["ret_5"])
    ret_15 = safe_float(row["ret_15"])
    ret_30 = safe_float(row["ret_30"])
    rvol = safe_float(row["rvol"])
    prior_breakout_high = safe_float(row["rolling_high_20"])
    prior_breakout_low = safe_float(row["rolling_low_20"])

    long_score = 0.0
    short_score = 0.0

    # Long side
    if symbol in LONG_UNIVERSE:
        if price > vwap:
            long_score += 0.8
        if ema9 > ema20:
            long_score += 0.9
        if ema20 > ema50:
            long_score += 0.8
        if ret_5 > 0.0010:
            long_score += min(ret_5 * 120, 1.2)
        if ret_15 > 0.0020:
            long_score += min(ret_15 * 100, 1.2)
        if ret_30 > 0.0030:
            long_score += min(ret_30 * 80, 1.0)
        if prior_breakout_high > 0 and price > prior_breakout_high:
            long_score += 1.0
        long_score += min(max(rvol - 1.0, 0.0), 1.0)

    # Short side
    if symbol in SHORT_UNIVERSE:
        if price < vwap:
            short_score += 0.8
        if ema9 < ema20:
            short_score += 0.9
        if ema20 < ema50:
            short_score += 0.8
        if ret_5 < -0.0010:
            short_score += min(abs(ret_5) * 120, 1.2)
        if ret_15 < -0.0020:
            short_score += min(abs(ret_15) * 100, 1.2)
        if ret_30 < -0.0030:
            short_score += min(abs(ret_30) * 80, 1.0)
        if prior_breakout_low > 0 and price < prior_breakout_low:
            short_score += 1.0
        short_score += min(max(rvol - 1.0, 0.0), 1.0)

    if long_score >= SIGNAL_THRESHOLD and long_score > short_score:
        return "long", round(long_score, 4)
    if short_score >= SIGNAL_THRESHOLD and short_score > long_score:
        return "short", round(short_score, 4)
    return None, 0.0


def build_candidates(api: tradeapi.REST, symbols: list[str]) -> pd.DataFrame:
    bars = get_bars_df(api, symbols, timeframe="1Min", limit=MAX_BARS)
    if bars.empty:
        return pd.DataFrame()

    bars = enrich_indicators(bars)
    if bars.empty:
        return pd.DataFrame()

    rows = []
    for symbol, group in bars.groupby("symbol"):
        g = group.sort_values("timestamp").copy()
        if len(g) < 55:
            continue

        row = g.iloc[-1]
        price = safe_float(row["close"])
        atr14 = safe_float(row["atr14"])
        rvol = safe_float(row["rvol"])
        dollar_vol_1m = safe_float(row["dollar_vol_1m"])

        if price < MIN_PRICE or price > MAX_PRICE:
            continue
        if atr14 <= 0 or np.isnan(atr14):
            continue
        if np.isnan(rvol) or rvol < MIN_RVOL:
            continue
        if dollar_vol_1m < MIN_DOLLAR_VOL_1M:
            continue

        signal_side, score = score_latest_row(row, symbol)
        if signal_side is None:
            continue

        rows.append(
            {
                "symbol": symbol,
                "side": signal_side,
                "score": score,
                "price": round(price, 4),
                "atr14": round(atr14, 4),
                "rvol": round(rvol, 4),
                "vwap": round(safe_float(row["vwap"]), 4),
                "ema9": round(safe_float(row["ema9"]), 4),
                "ema20": round(safe_float(row["ema20"]), 4),
                "ema50": round(safe_float(row["ema50"]), 4),
                "ret_5": round(safe_float(row["ret_5"]), 6),
                "ret_15": round(safe_float(row["ret_15"]), 6),
                "ret_30": round(safe_float(row["ret_30"]), 6),
                "dollar_vol_1m": round(dollar_vol_1m, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["score", "rvol", "dollar_vol_1m"], ascending=False).reset_index(drop=True)


def compute_stop_target(price: float, atr14: float, side: str) -> tuple[float, float, float]:
    stop_distance = max(price * 0.004, atr14 * 1.15)

    if side == "long":
        stop_price = price - stop_distance
        target_price = price + stop_distance * PROFIT_TARGET_R
    else:
        stop_price = price + stop_distance
        target_price = price - stop_distance * PROFIT_TARGET_R

    return round(stop_price, 2), round(target_price, 2), round(stop_distance, 4)


def compute_qty(equity: float, buying_power: float, price: float, stop_distance: float) -> float:
    if price <= 0 or stop_distance <= 0:
        return 0.0

    risk_budget = max(1.0, equity * RISK_PER_TRADE)
    qty_by_risk = risk_budget / stop_distance

    notional_cap = min(buying_power, equity * MAX_NOTIONAL_PER_POSITION_PCT)
    qty_by_notional = notional_cap / price

    qty = max(0.0, min(qty_by_risk, qty_by_notional))

    if qty * price < MIN_ORDER_NOTIONAL:
        return 0.0

    # Keep to 4 decimal places to support fractionals if broker/account permits
    qty = math.floor(qty * 10000) / 10000.0
    return qty


def estimate_sec_fee_for_exit(notional: float) -> float:
    return max(0.0, notional * SEC_FEE_RATE)


def place_bracket_order(
    api: tradeapi.REST,
    symbol: str,
    side: str,
    qty: float,
    entry_est: float,
    stop_price: float,
    target_price: float,
    state: dict,
) -> bool:
    if qty <= 0:
        return False

    order_side = "buy" if side == "long" else "sell"

    try:
        api.submit_order(
            symbol=symbol,
            qty=safe_round_qty(qty),
            side=order_side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": round(target_price, 2)},
            stop_loss={"stop_price": round(stop_price, 2)},
        )

        state["traded_today"].append(symbol)
        state["entry_meta"][symbol] = {
            "entry_price": round(entry_est, 6),
            "initial_stop": round(stop_price, 6),
            "target_price": round(target_price, 6),
            "stop_distance": round(abs(entry_est - stop_price), 6),
            "side": side,
            "entry_time_et": now_et().isoformat(),
        }

        append_csv(
            TRADE_LOG_FILE,
            {
                "utc_time": now_utc().isoformat(),
                "et_time": now_et().isoformat(),
                "pt_time": now_pt().isoformat(),
                "symbol": symbol,
                "action": "submit_bracket",
                "strategy_side": side,
                "order_side": order_side,
                "qty": qty,
                "entry_est": round(entry_est, 4),
                "stop_price": round(stop_price, 4),
                "target_price": round(target_price, 4),
                "sec_fee_est_exit": round(estimate_sec_fee_for_exit(qty * entry_est), 6),
            },
        )

        log(
            f"Submitted {side.upper()} bracket | "
            f"{symbol} qty={qty} entry≈{entry_est:.2f} stop={stop_price:.2f} target={target_price:.2f}"
        )
        return True
    except Exception as exc:
        log(f"Order failed for {symbol}: {exc}")
        return False


def refresh_state_for_closed_positions(api: tradeapi.REST, state: dict) -> None:
    live_symbols = {p.symbol for p in get_positions(api)}
    to_remove = []

    for symbol in list(state.get("entry_meta", {}).keys()):
        if symbol not in live_symbols:
            to_remove.append(symbol)

    for symbol in to_remove:
        state["entry_meta"].pop(symbol, None)


def run_bot() -> None:
    state = reset_state_if_new_day()

    summary = {
        "bot": BOT_NAME,
        "utc_time": now_utc().isoformat(),
        "et_time": now_et().isoformat(),
        "pt_time": now_pt().isoformat(),
        "status": "started",
        "market_open": None,
        "entry_window": None,
        "equity": None,
        "buying_power": None,
        "scanner_candidates": 0,
        "new_orders_submitted": 0,
        "traded_today_count": len(state.get("traded_today", [])),
    }

    log(f"=== {BOT_NAME} start ===")
    api = get_api()

    equity = get_equity(api)
    buying_power = get_buying_power(api)
    summary["equity"] = equity
    summary["buying_power"] = buying_power

    log(f"Account snapshot | equity={equity:.2f} buying_power={buying_power:.2f}")

    is_open = market_is_open(api)
    summary["market_open"] = is_open

    et_now = now_et()
    summary["entry_window"] = in_entry_window(et_now)

    log(f"Current ET time: {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')} | PT {now_pt().strftime('%H:%M:%S %Z')}")

    if not is_open:
        log("Market is closed. Exiting cleanly.")
        summary["status"] = "market_closed"
        save_state(state)
        write_summary(summary)
        return

    refresh_state_for_closed_positions(api, state)
    manage_open_positions(api, state)

    if not in_entry_window(et_now):
        log("Outside entry window for new entries.")
        summary["status"] = "outside_entry_window"
        save_state(state)
        write_summary(summary)
        return

    open_positions = get_positions(api)
    if len(open_positions) >= MAX_TOTAL_POSITIONS:
        log(f"Max position count reached ({len(open_positions)}/{MAX_TOTAL_POSITIONS}).")
        summary["status"] = "max_positions_reached"
        save_state(state)
        write_summary(summary)
        return

    candidates = build_candidates(api, BASE_UNIVERSE)
    if candidates.empty:
        log("Scanner found 0 candidates.")
        summary["status"] = "no_candidates"
        save_state(state)
        write_summary(summary)
        return

    blocked = symbols_with_exposure(api)
    traded_today = set(state.get("traded_today", []))
    candidates = candidates[
        (~candidates["symbol"].isin(blocked)) &
        (~candidates["symbol"].isin(traded_today))
    ].copy()

    if candidates.empty:
        log("All candidates blocked by open exposure or same-day re-entry guard.")
        summary["status"] = "all_candidates_blocked"
        save_state(state)
        write_summary(summary)
        return

    summary["scanner_candidates"] = int(len(candidates))
    preview = candidates.head(8)[["symbol", "side", "score", "price", "rvol", "atr14"]].to_dict("records")
    log(f"Scanner found {len(candidates)} candidates.")
    log(f"Top candidates: {preview}")

    available_slots = max(0, MAX_TOTAL_POSITIONS - len(open_positions))
    max_orders = min(MAX_NEW_ORDERS_PER_RUN, available_slots)

    submitted = 0
    for _, row in candidates.iterrows():
        if submitted >= max_orders:
            break

        symbol = row["symbol"]
        side = row["side"]
        price = safe_float(row["price"])
        atr14 = safe_float(row["atr14"])

        stop_price, target_price, stop_distance = compute_stop_target(price, atr14, side)
        qty = compute_qty(equity, buying_power, price, stop_distance)

        if qty <= 0:
            log(f"Skipped {symbol} {side.upper()} | qty=0 price={price:.2f} stop_distance={stop_distance:.4f}")
            continue

        ok = place_bracket_order(
            api=api,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_est=price,
            stop_price=stop_price,
            target_price=target_price,
            state=state,
        )
        if ok:
            submitted += 1

    summary["new_orders_submitted"] = submitted
    summary["traded_today_count"] = len(state.get("traded_today", []))
    summary["status"] = "completed"

    save_state(state)
    write_summary(summary)
    log(f"Run complete | new_orders_submitted={submitted}")


if __name__ == "__main__":
    try:
        run_bot()
    except Exception as exc:
        error_text = f"FATAL ERROR: {exc}\n{traceback.format_exc()}"
        log(error_text)
        write_summary(
            {
                "bot": BOT_NAME,
                "utc_time": now_utc().isoformat(),
                "et_time": now_et().isoformat(),
                "pt_time": now_pt().isoformat(),
                "status": "fatal_error",
                "error": error_text,
            }
        )
        raise
