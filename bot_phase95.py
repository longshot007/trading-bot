import os
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import alpaca_trade_api as tradeapi


# ============================================================
# bot_phase95.py
# Phase 9.5
# - Opening-range signal based on first 5-minute candle
# - Softer SPY filter
# - Actual fill reconciliation
# - Lightweight profitability learning from logged pattern tags
# - Entries allowed through 11:00 ET
# - Forced flat at 11:00 ET
# - Turnkey IEX-feed revision to avoid recent SIP subscription errors
# ============================================================


# -----------------------------
# Environment / API
# -----------------------------
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise RuntimeError("Missing Alpaca credentials in environment variables.")

api = tradeapi.REST(
    APCA_API_KEY_ID,
    APCA_API_SECRET_KEY,
    APCA_API_BASE_URL,
    api_version="v2"
)


# -----------------------------
# Time zones
# -----------------------------
EASTERN = pytz.timezone("America/New_York")
UTC = pytz.UTC
PACIFIC = pytz.timezone("America/Los_Angeles")


# -----------------------------
# Files
# -----------------------------
DATA_DIR = "data"
LOG_DIR = "logs"

STATE_FILE = os.path.join(DATA_DIR, "phase95_state.json")
TRADES_LOG_FILE = os.path.join(LOG_DIR, "phase95_trades_log.csv")
SIGNALS_LOG_FILE = os.path.join(LOG_DIR, "phase95_signals_log.csv")
PATTERN_STATS_FILE = os.path.join(DATA_DIR, "phase95_pattern_stats.json")


# -----------------------------
# Strategy parameters
# -----------------------------
MAX_TRADES_PER_DAY = 10
MAX_OPEN_POSITIONS = 5
RISK_PER_TRADE_PCT = 0.01
MAX_DAILY_NET_LOSS_PCT = 0.02
MAX_DAILY_LOSING_TRADES = 3

MIN_PRICE = 5.0
MAX_PRICE = 200.0

MIN_DOLLAR_VOLUME = 1_500_000
MIN_RELATIVE_VOLUME = 1.05

MIN_5M_RANGE_PCT = 0.0020
MAX_5M_RANGE_PCT = 0.08
MAX_RISK_PER_SHARE_PCT = 0.04

ENTRY_BUFFER_PCT = 0.0003
MIN_ENTRY_BUFFER_DOLLARS = 0.01
NEAR_BREAKOUT_PCT = 0.0015

TARGET_R_MULTIPLE = 2.0

ENTRY_START_HOUR = 9
ENTRY_START_MINUTE = 35
ENTRY_END_HOUR = 11
ENTRY_END_MINUTE = 0

FORCE_EXIT_HOUR = 11
FORCE_EXIT_MINUTE = 0

SPY_SYMBOL = "SPY"
SCANNER_LIMIT = 200

SEC_FEE_RATE = 0.0000206

# IEX feed revision for free Alpaca market data access
DATA_FEED_INTRADAY = "iex"
DATA_FEED_DAILY = "iex"

# Lenient signal settings
MIN_BODY_FRACTION = 0.25
MAX_OPPOSITE_WICK_FRACTION = 0.55
MIN_CLOSE_NEAR_EXTREME_LONG = 0.55
MAX_CLOSE_NEAR_EXTREME_SHORT = 0.45

# Learning filter kept mild so it does not block too many trades
MIN_ADAPTIVE_SCORE = -0.50

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL", "NFLX", "INTC",
    "MU", "PLTR", "CRM", "ADBE", "AVGO", "QCOM", "SHOP", "UBER", "COIN", "SMCI",
    "ARM", "SOFI", "PYPL", "SNOW", "PANW", "CRWD", "ANET", "MRVL", "F", "GM",
    "BAC", "JPM", "C", "WFC", "XOM", "CVX", "OXY", "SLB", "LLY", "UNH",
    "JNJ", "PFE", "NKE", "DIS", "BA", "CAT", "DE", "RIOT", "MARA", "HOOD",
    "RIVN", "LCID", "QQQ", "IWM", "DIA", "ARKK", "TQQQ", "SQQQ", "XLF", "XLK"
]


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def now_et() -> datetime:
    return datetime.now(EASTERN)


def now_pacific() -> datetime:
    return datetime.now(PACIFIC)


def log(msg: str) -> None:
    et = now_et().strftime("%Y-%m-%d %H:%M:%S")
    pt = now_pacific().strftime("%H:%M:%S")
    print(f"[ET {et} | PT {pt}] {msg}")


def safe_float(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def append_csv(path: str, row: Dict, columns: List[str]) -> None:
    ensure_dirs()
    exists = os.path.exists(path)
    df = pd.DataFrame([row]).reindex(columns=columns)
    if exists:
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# -----------------------------
# Persistent state
# -----------------------------
def default_state() -> Dict:
    return {
        "date": "",
        "trades_today": 0,
        "losing_trades_today": 0,
        "realized_net_pnl_today": 0.0,
        "daily_start_equity": 0.0,
        "symbols_traded_today": [],
        "positions": {},
        "last_scan_candidates": [],
        "kill_switch": False,
        "kill_switch_reason": ""
    }


def load_state() -> Dict:
    ensure_dirs()
    if not os.path.exists(STATE_FILE):
        return default_state()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict) -> None:
    ensure_dirs()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reset_daily_state_if_needed(state: Dict) -> Dict:
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        equity = get_account_equity()
        state = default_state()
        state["date"] = today
        state["daily_start_equity"] = equity
        save_state(state)
        log(f"Daily state reset. Start equity={equity:.2f}")
    return state


# -----------------------------
# Log columns
# -----------------------------
TRADE_COLUMNS = [
    "timestamp_et", "symbol", "direction", "event", "qty",
    "entry_order_id", "entry_price_est", "entry_price_fill",
    "stop_price", "target_price",
    "exit_order_id", "exit_price_est", "exit_price_fill",
    "reason", "gross_pnl", "sec_fee", "net_pnl",
    "score", "relative_volume", "range_pct",
    "spy_regime", "pattern_tags"
]

SIGNAL_COLUMNS = [
    "timestamp_et", "symbol", "direction", "latest_price",
    "opening_open", "opening_high", "opening_low", "opening_close",
    "opening_volume", "relative_volume", "range_pct",
    "body_fraction", "close_position", "adaptive_score",
    "base_score", "score", "spy_regime", "pattern_tags", "action"
]


# -----------------------------
# Market / account
# -----------------------------
def market_is_open() -> bool:
    return bool(api.get_clock().is_open)


def get_account_equity() -> float:
    return safe_float(api.get_account().equity)


def get_buying_power() -> float:
    return safe_float(api.get_account().buying_power)


def get_current_positions() -> Dict[str, Dict]:
    positions = {}
    try:
        for p in api.list_positions():
            qty_signed = int(float(p.qty))
            positions[p.symbol] = {
                "symbol": p.symbol,
                "signed_qty": qty_signed,
                "qty": abs(qty_signed),
                "side": "long" if qty_signed > 0 else "short",
                "avg_entry_price": safe_float(p.avg_entry_price),
                "current_price": safe_float(getattr(p, "current_price", None)),
                "market_value": safe_float(getattr(p, "market_value", None)),
                "unrealized_pl": safe_float(getattr(p, "unrealized_pl", None))
            }
    except Exception as e:
        log(f"Could not list positions: {e}")
    return positions


# Define entry window (NY time)
entry_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
entry_end   = now_et.replace(hour=13, minute=0, second=0, microsecond=0)  # 10:00 PT

if not (entry_start <= now_et <= entry_end):
    log("Outside entry window for new entries.")
    return


def is_force_exit_time(current_et: datetime) -> bool:
   def is_entry_window(current_et: datetime) -> bool:
    entry_start = current_et.replace(
        hour=9,
        minute=30,
        second=0,
        microsecond=0
    )
    entry_end = current_et.replace(
        hour=13,
        minute=0,
        second=0,
        microsecond=0
    )
    return entry_start <= current_et <= entry_end
    cutoff = current_et.replace(
        hour=FORCE_EXIT_HOUR,
        minute=FORCE_EXIT_MINUTE,
        second=0,
        microsecond=0
    )
    return current_et >= cutoff


# -----------------------------
# Data retrieval
# -----------------------------
def _extract_symbol_bars(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if bars.empty:
        return bars

    if isinstance(bars.index, pd.MultiIndex):
        try:
            bars = bars.xs(symbol, level=0)
        except Exception:
            try:
                bars = bars.xs(symbol, level="symbol")
            except Exception:
                pass

    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(EASTERN)
    return bars.sort_index()


def get_minute_bars(symbol: str, minutes_back: int = 120) -> pd.DataFrame:
    end_utc = datetime.now(UTC)
    start_utc = end_utc - timedelta(minutes=minutes_back + 30)

    try:
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            adjustment="raw",
            feed=DATA_FEED_INTRADAY
        ).df
    except Exception as e:
        log(f"Minute bars error for {symbol}: {e}")
        return pd.DataFrame()

    return _extract_symbol_bars(bars, symbol)


def get_daily_bars(symbol: str, days: int = 20) -> pd.DataFrame:
    end_utc = datetime.now(UTC)
    start_utc = end_utc - timedelta(days=days + 10)

    try:
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Day,
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            adjustment="raw",
            feed=DATA_FEED_DAILY
        ).df
    except Exception as e:
        log(f"Daily bars error for {symbol}: {e}")
        return pd.DataFrame()

    return _extract_symbol_bars(bars, symbol)


def get_latest_trade_price(symbol: str) -> Optional[float]:
    minute_bars = get_minute_bars(symbol, 3)
    if minute_bars.empty:
        return None

    for col in ["close", "vwap", "open"]:
        if col in minute_bars.columns:
            price = safe_float(minute_bars.iloc[-1][col], None)
            if price is not None and price > 0:
                return price
    return None


# -----------------------------
# Session filtering
# -----------------------------
def filter_today_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    today = now_et().date()
    df = df[df.index.date == today]
    return df[
        (df.index.time >= datetime.strptime("09:30", "%H:%M").time()) &
        (df.index.time <= datetime.strptime("16:00", "%H:%M").time())
    ]


# -----------------------------
# Opening signal
# -----------------------------
def build_opening_5m_candle(minute_bars: pd.DataFrame) -> Optional[Dict]:
    df = filter_today_regular_session(minute_bars)
    if df.empty:
        return None

    first_five = df[
        (df.index.time >= datetime.strptime("09:30", "%H:%M").time()) &
        (df.index.time < datetime.strptime("09:35", "%H:%M").time())
    ]

    if len(first_five) < 5:
        return None

    open_ = float(first_five.iloc[0]["open"])
    high_ = float(first_five["high"].max())
    low_ = float(first_five["low"].min())
    close_ = float(first_five.iloc[-1]["close"])
    volume_ = float(first_five["volume"].sum())

    candle_range = max(high_ - low_, 0.0001)
    body = abs(close_ - open_)
    body_fraction = body / candle_range
    upper_wick = high_ - max(open_, close_)
    lower_wick = min(open_, close_) - low_
    close_position = (close_ - low_) / candle_range

    direction = "long" if close_ >= open_ else "short"
    opposite_wick_fraction = (
        lower_wick / candle_range if direction == "long" else upper_wick / candle_range
    )

    return {
        "open": open_,
        "high": high_,
        "low": low_,
        "close": close_,
        "volume": volume_,
        "range": candle_range,
        "range_pct": candle_range / max(open_, 0.0001),
        "body_fraction": body_fraction,
        "close_position": close_position,
        "upper_wick_fraction": upper_wick / candle_range,
        "lower_wick_fraction": lower_wick / candle_range,
        "opposite_wick_fraction": opposite_wick_fraction,
        "direction": direction
    }


def opening_candle_is_valid(opening: Dict) -> bool:
    if opening["range_pct"] < MIN_5M_RANGE_PCT or opening["range_pct"] > MAX_5M_RANGE_PCT:
        return False
    if opening["body_fraction"] < MIN_BODY_FRACTION:
        return False
    if opening["opposite_wick_fraction"] > MAX_OPPOSITE_WICK_FRACTION:
        return False

    if opening["direction"] == "long":
        if opening["close_position"] < MIN_CLOSE_NEAR_EXTREME_LONG:
            return False
    else:
        if opening["close_position"] > MAX_CLOSE_NEAR_EXTREME_SHORT:
            return False

    return True


def build_trade_levels(opening: Dict, latest_price: float) -> Dict:
    buffer_amt = max(latest_price * ENTRY_BUFFER_PCT, MIN_ENTRY_BUFFER_DOLLARS)

    if opening["direction"] == "long":
        stop_price = opening["low"] - buffer_amt
        entry_price = opening["high"] + buffer_amt
        risk_per_share = entry_price - stop_price
        target_price = entry_price + TARGET_R_MULTIPLE * risk_per_share
    else:
        stop_price = opening["high"] + buffer_amt
        entry_price = opening["low"] - buffer_amt
        risk_per_share = stop_price - entry_price
        target_price = entry_price - TARGET_R_MULTIPLE * risk_per_share

    return {
        "direction": opening["direction"],
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "risk_per_share": round(risk_per_share, 4),
        "risk_per_share_pct": round(risk_per_share / max(entry_price, 0.0001), 4)
    }


def breakout_is_confirmed(opening: Dict, latest_price: float, levels: Dict) -> bool:
    trigger = levels["entry_price"]
    if opening["direction"] == "long":
        return latest_price >= trigger or latest_price >= trigger * (1 - NEAR_BREAKOUT_PCT)
    return latest_price <= trigger or latest_price <= trigger * (1 + NEAR_BREAKOUT_PCT)


# -----------------------------
# SPY regime
# -----------------------------
def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def get_spy_regime() -> Dict:
    try:
        spy_bars = get_minute_bars(SPY_SYMBOL, 120)
        spy_bars = filter_today_regular_session(spy_bars)
        if spy_bars.empty or len(spy_bars) < 10:
            return {"regime": "neutral", "last": None, "vwap": None}

        spy_bars = spy_bars.copy()
        spy_bars["vwap"] = compute_intraday_vwap(spy_bars)

        opening = build_opening_5m_candle(spy_bars)
        if not opening:
            return {"regime": "neutral", "last": None, "vwap": None}

        last_close = float(spy_bars.iloc[-1]["close"])
        last_vwap = float(spy_bars.iloc[-1]["vwap"])
        opening_mid = (opening["high"] + opening["low"]) / 2.0

        if last_close > last_vwap and last_close > opening_mid:
            regime = "bullish"
        elif last_close < last_vwap and last_close < opening_mid:
            regime = "bearish"
        else:
            regime = "neutral"

        return {"regime": regime, "last": round(last_close, 4), "vwap": round(last_vwap, 4)}
    except Exception as e:
        log(f"SPY regime error: {e}")
        return {"regime": "neutral", "last": None, "vwap": None}


def spy_score_adjustment(spy_regime: str, direction: str) -> float:
    if spy_regime == "neutral":
        return 0.0
    if direction == "long":
        return 0.75 if spy_regime == "bullish" else -0.75
    return 0.75 if spy_regime == "bearish" else -0.75


# -----------------------------
# Relative volume
# -----------------------------
def compute_relative_volume(symbol: str, opening_volume: float) -> float:
    daily = get_daily_bars(symbol, 15)
    if daily.empty or len(daily) < 6:
        return 0.0

    avg_daily_volume = float(daily["volume"].tail(10).mean())
    avg_opening_5m_proxy = avg_daily_volume / 78.0 if avg_daily_volume > 0 else 0.0
    if avg_opening_5m_proxy <= 0:
        return 0.0

    return opening_volume / avg_opening_5m_proxy


# -----------------------------
# Pattern learning
# -----------------------------
def read_trade_log() -> pd.DataFrame:
    if not os.path.exists(TRADES_LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(TRADES_LOG_FILE)
    except Exception:
        return pd.DataFrame()


def build_pattern_stats() -> Dict[str, Dict]:
    df = read_trade_log()
    if df.empty:
        return {}

    exits = df[df["event"] == "exit_submitted"].copy()
    if exits.empty or "pattern_tags" not in exits.columns:
        return {}

    stats: Dict[str, Dict] = {}

    for _, row in exits.iterrows():
        tags_str = str(row.get("pattern_tags", "")).strip()
        if not tags_str or tags_str == "nan":
            continue

        net_pnl = safe_float(row.get("net_pnl"), 0.0)
        win = 1 if net_pnl > 0 else 0
        tags = [t for t in tags_str.split("|") if t]

        for tag in tags:
            if tag not in stats:
                stats[tag] = {"count": 0, "wins": 0, "net_pnl_sum": 0.0}
            stats[tag]["count"] += 1
            stats[tag]["wins"] += win
            stats[tag]["net_pnl_sum"] += net_pnl

    return stats


def save_pattern_stats(stats: Dict[str, Dict]) -> None:
    ensure_dirs()
    with open(PATTERN_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_pattern_stats() -> Dict[str, Dict]:
    if os.path.exists(PATTERN_STATS_FILE):
        try:
            with open(PATTERN_STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    stats = build_pattern_stats()
    save_pattern_stats(stats)
    return stats


def refresh_pattern_stats() -> Dict[str, Dict]:
    stats = build_pattern_stats()
    save_pattern_stats(stats)
    return stats


def adaptive_pattern_score(tags: List[str], stats: Dict[str, Dict]) -> float:
    if not tags:
        return 0.0

    scores = []
    for tag in tags:
        item = stats.get(tag)
        if not item:
            continue

        count = int(item.get("count", 0))
        wins = int(item.get("wins", 0))
        net_pnl_sum = safe_float(item.get("net_pnl_sum", 0.0))

        smoothed_win_rate = (wins + 2.0) / (count + 4.0)
        pnl_per_trade = net_pnl_sum / max(count, 1)

        tag_score = (smoothed_win_rate - 0.5) * 1.5
        tag_score += clamp(pnl_per_trade / 50.0, -0.20, 0.20)

        if count < 5:
            tag_score *= 0.5

        scores.append(tag_score)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def candidate_pattern_tags(
    opening: Dict,
    rel_vol: float,
    spy_regime: str,
    risk_per_share_pct: float
) -> List[str]:
    tags = [f"dir_{opening['direction']}", f"spy_{spy_regime}"]

    if rel_vol >= 2.0:
        tags.append("rvol_high")
    elif rel_vol >= 1.3:
        tags.append("rvol_good")
    else:
        tags.append("rvol_ok")

    if opening["body_fraction"] >= 0.60:
        tags.append("body_strong")
    elif opening["body_fraction"] >= 0.40:
        tags.append("body_ok")
    else:
        tags.append("body_light")

    if opening["direction"] == "long":
        if opening["close_position"] >= 0.85:
            tags.append("close_at_high")
    else:
        if opening["close_position"] <= 0.15:
            tags.append("close_at_low")

    if opening["range_pct"] >= 0.01:
        tags.append("range_expanded")
    else:
        tags.append("range_normal")

    if risk_per_share_pct <= 0.015:
        tags.append("tight_risk")
    elif risk_per_share_pct <= 0.025:
        tags.append("normal_risk")
    else:
        tags.append("wide_risk")

    return tags


# -----------------------------
# Candidate scoring
# -----------------------------
def base_candidate_score(opening: Dict, rel_vol: float, latest_price: float) -> float:
    score = 0.0
    score += rel_vol * 3.5
    score += opening["range_pct"] * 90.0
    score += opening["body_fraction"] * 1.5

    if opening["direction"] == "long":
        score += opening["close_position"] * 1.0
    else:
        score += (1.0 - opening["close_position"]) * 1.0

    score += latest_price / 1200.0
    return score


# -----------------------------
# Risk / sizing
# -----------------------------
def calculate_qty(entry_price: float, stop_price: float, equity: float, buying_power: float) -> int:
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0

    risk_budget = equity * RISK_PER_TRADE_PCT
    qty_by_risk = int(risk_budget // risk_per_share)
    qty_by_bp = int((buying_power * 0.95) // entry_price)
    return max(0, min(qty_by_risk, qty_by_bp))


# -----------------------------
# Orders / fill reconciliation
# -----------------------------
def wait_for_fill_price(order_id: str, retries: int = 6, sleep_seconds: int = 2) -> Tuple[Optional[float], Optional[str]]:
    for _ in range(retries):
        try:
            order = api.get_order(order_id)
            status = str(order.status).lower()
            filled_avg_price = safe_float(getattr(order, "filled_avg_price", None), None)

            if status in {"filled", "partially_filled"} and filled_avg_price is not None:
                return filled_avg_price, status

            if status in {"canceled", "expired", "rejected"}:
                return None, status
        except Exception:
            pass

        time.sleep(sleep_seconds)

    return None, "timeout"


def submit_entry_order(symbol: str, qty: int, direction: str) -> Optional[Dict]:
    side = "buy" if direction == "long" else "sell"

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        log(f"ENTRY submitted: {symbol} {direction} qty={qty} order_id={order.id}")

        fill_price, fill_status = wait_for_fill_price(order.id)
        return {
            "order_id": order.id,
            "fill_price": fill_price,
            "fill_status": fill_status
        }
    except Exception as e:
        log(f"Entry order failed for {symbol}: {e}")
        return None


def submit_exit_order(symbol: str, qty: int, position_side: str) -> Optional[Dict]:
    exit_side = "sell" if position_side == "long" else "buy"

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=exit_side,
            type="market",
            time_in_force="day"
        )
        log(f"EXIT submitted: {symbol} {position_side} qty={qty} order_id={order.id}")

        fill_price, fill_status = wait_for_fill_price(order.id)
        return {
            "order_id": order.id,
            "fill_price": fill_price,
            "fill_status": fill_status
        }
    except Exception as e:
        log(f"Exit order failed for {symbol}: {e}")
        return None


# -----------------------------
# PnL / fees
# -----------------------------
def estimate_sec_fee(exit_side: str, exit_price: float, qty: int) -> float:
    if exit_side.lower() != "sell":
        return 0.0
    return round(exit_price * qty * SEC_FEE_RATE, 6)


def compute_trade_pnl(direction: str, entry_fill: float, exit_fill: float, qty: int) -> Dict[str, float]:
    gross = (exit_fill - entry_fill) * qty if direction == "long" else (entry_fill - exit_fill) * qty
    exit_side = "sell" if direction == "long" else "buy"
    sec_fee = estimate_sec_fee(exit_side, exit_fill, qty)
    net = gross - sec_fee
    return {
        "gross_pnl": round(gross, 4),
        "sec_fee": round(sec_fee, 6),
        "net_pnl": round(net, 4)
    }


# -----------------------------
# Kill switch
# -----------------------------
def update_kill_switch(state: Dict) -> Dict:
    daily_start_equity = safe_float(state.get("daily_start_equity"), 0.0)
    realized_net_pnl_today = safe_float(state.get("realized_net_pnl_today"), 0.0)
    losing_trades_today = int(state.get("losing_trades_today", 0))

    if daily_start_equity > 0:
        daily_loss_pct = max(0.0, -realized_net_pnl_today / daily_start_equity)
        if daily_loss_pct >= MAX_DAILY_NET_LOSS_PCT:
            state["kill_switch"] = True
            state["kill_switch_reason"] = f"daily_net_loss_limit_{MAX_DAILY_NET_LOSS_PCT:.2%}"
            return state

    if losing_trades_today >= MAX_DAILY_LOSING_TRADES:
        state["kill_switch"] = True
        state["kill_switch_reason"] = f"max_losing_trades_{MAX_DAILY_LOSING_TRADES}"
        return state

    return state


# -----------------------------
# Scanner
# -----------------------------
def scan_candidates(pattern_stats: Dict[str, Dict]) -> List[Dict]:
    spy = get_spy_regime()
    spy_regime = spy["regime"]

    equity = get_account_equity()
    buying_power = get_buying_power()

    candidates = []

    for symbol in DEFAULT_UNIVERSE[:SCANNER_LIMIT]:
        if symbol == SPY_SYMBOL:
            continue

        try:
            minute_bars = get_minute_bars(symbol, 120)
            if minute_bars.empty:
                continue

            opening = build_opening_5m_candle(minute_bars)
            if not opening:
                continue

            if not opening_candle_is_valid(opening):
                continue

            latest_price = get_latest_trade_price(symbol)
            if latest_price is None:
                continue

            if latest_price < MIN_PRICE or latest_price > MAX_PRICE:
                continue

            rel_vol = compute_relative_volume(symbol, opening["volume"])
            if rel_vol < MIN_RELATIVE_VOLUME:
                continue

            dollar_volume = latest_price * opening["volume"]
            if dollar_volume < MIN_DOLLAR_VOLUME:
                continue

            levels = build_trade_levels(opening, latest_price)
            if levels["risk_per_share_pct"] > MAX_RISK_PER_SHARE_PCT:
                continue

            if not breakout_is_confirmed(opening, latest_price, levels):
                continue

            qty = calculate_qty(levels["entry_price"], levels["stop_price"], equity, buying_power)
            if qty < 1:
                continue

            pattern_tags = candidate_pattern_tags(
                opening=opening,
                rel_vol=rel_vol,
                spy_regime=spy_regime,
                risk_per_share_pct=levels["risk_per_share_pct"]
            )

            adaptive_score = adaptive_pattern_score(pattern_tags, pattern_stats)
            if adaptive_score < MIN_ADAPTIVE_SCORE:
                continue

            base_score = base_candidate_score(opening, rel_vol, latest_price)
            total_score = base_score + adaptive_score + spy_score_adjustment(spy_regime, opening["direction"])

            candidates.append({
                "symbol": symbol,
                "latest_price": latest_price,
                "opening": opening,
                "direction": opening["direction"],
                "entry_price": levels["entry_price"],
                "stop_price": levels["stop_price"],
                "target_price": levels["target_price"],
                "risk_per_share": levels["risk_per_share"],
                "risk_per_share_pct": levels["risk_per_share_pct"],
                "qty": qty,
                "relative_volume": rel_vol,
                "dollar_volume": dollar_volume,
                "spy_regime": spy_regime,
                "pattern_tags": pattern_tags,
                "adaptive_score": adaptive_score,
                "base_score": base_score,
                "score": total_score
            })
        except Exception as e:
            log(f"Scanner skip {symbol}: {e}")

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates


# -----------------------------
# Entry management
# -----------------------------
def maybe_enter_new_positions(state: Dict, pattern_stats: Dict[str, Dict]) -> Dict:
    current = now_et()

    if not is_entry_window(current):
        log("Outside entry window for new trades.")
        return state

    if state.get("kill_switch", False):
        log(f"Kill switch active: {state.get('kill_switch_reason', '')}")
        return state

    if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
        log("Max trades per day reached.")
        return state

    broker_positions = get_current_positions()
    if len(broker_positions) >= MAX_OPEN_POSITIONS:
        log("Max open positions reached.")
        return state

    candidates = scan_candidates(pattern_stats)
    state["last_scan_candidates"] = [
        {
            "symbol": c["symbol"],
            "score": round(c["score"], 4),
            "direction": c["direction"],
            "latest_price": round(c["latest_price"], 4),
            "entry_price": c["entry_price"],
            "stop_price": c["stop_price"],
            "target_price": c["target_price"]
        }
        for c in candidates[:10]
    ]

    if not candidates:
        log("No valid candidates found.")
        save_state(state)
        return state

    symbols_traded_today = set(state.get("symbols_traded_today", []))
    tracked_positions = state.get("positions", {})

    for c in candidates:
        if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
            break
        if len(get_current_positions()) >= MAX_OPEN_POSITIONS:
            break

        symbol = c["symbol"]
        if symbol in symbols_traded_today:
            continue
        if symbol in tracked_positions and tracked_positions[symbol].get("status") in {"open", "entry_submitted"}:
            continue

        result = submit_entry_order(symbol, c["qty"], c["direction"])
        action = "entry_submitted" if result else "entry_failed"

        append_csv(SIGNALS_LOG_FILE, {
            "timestamp_et": current.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "direction": c["direction"],
            "latest_price": round(c["latest_price"], 4),
            "opening_open": round(c["opening"]["open"], 4),
            "opening_high": round(c["opening"]["high"], 4),
            "opening_low": round(c["opening"]["low"], 4),
            "opening_close": round(c["opening"]["close"], 4),
            "opening_volume": round(c["opening"]["volume"], 2),
            "relative_volume": round(c["relative_volume"], 4),
            "range_pct": round(c["opening"]["range_pct"], 4),
            "body_fraction": round(c["opening"]["body_fraction"], 4),
            "close_position": round(c["opening"]["close_position"], 4),
            "adaptive_score": round(c["adaptive_score"], 4),
            "base_score": round(c["base_score"], 4),
            "score": round(c["score"], 4),
            "spy_regime": c["spy_regime"],
            "pattern_tags": "|".join(c["pattern_tags"]),
            "action": action
        }, SIGNAL_COLUMNS)

        if not result:
            continue

        fill_price = result["fill_price"]
        if fill_price is None:
            fill_price = c["latest_price"]

        tracked_positions[symbol] = {
            "symbol": symbol,
            "status": "open" if result.get("fill_status") in {"filled", "partially_filled", "timeout", None} else "entry_submitted",
            "direction": c["direction"],
            "qty": int(c["qty"]),
            "entry_order_id": result["order_id"],
            "entry_price_est": round(c["entry_price"], 4),
            "entry_price_fill": round(fill_price, 4),
            "stop_price": round(c["stop_price"], 4),
            "target_price": round(c["target_price"], 4),
            "score": round(c["score"], 4),
            "relative_volume": round(c["relative_volume"], 4),
            "range_pct": round(c["opening"]["range_pct"], 4),
            "spy_regime": c["spy_regime"],
            "pattern_tags": "|".join(c["pattern_tags"]),
            "opened_at_et": current.strftime("%Y-%m-%d %H:%M:%S")
        }

        append_csv(TRADES_LOG_FILE, {
            "timestamp_et": current.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "direction": c["direction"],
            "event": "entry_submitted",
            "qty": int(c["qty"]),
            "entry_order_id": result["order_id"],
            "entry_price_est": round(c["entry_price"], 4),
            "entry_price_fill": round(fill_price, 4),
            "stop_price": round(c["stop_price"], 4),
            "target_price": round(c["target_price"], 4),
            "exit_order_id": "",
            "exit_price_est": "",
            "exit_price_fill": "",
            "reason": "",
            "gross_pnl": "",
            "sec_fee": "",
            "net_pnl": "",
            "score": round(c["score"], 4),
            "relative_volume": round(c["relative_volume"], 4),
            "range_pct": round(c["opening"]["range_pct"], 4),
            "spy_regime": c["spy_regime"],
            "pattern_tags": "|".join(c["pattern_tags"])
        }, TRADE_COLUMNS)

        state["positions"] = tracked_positions
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        symbols_traded_today.add(symbol)
        state["symbols_traded_today"] = sorted(symbols_traded_today)
        save_state(state)

    return state


# -----------------------------
# Exit management
# -----------------------------
def maybe_manage_and_exit_positions(state: Dict) -> Dict:
    current = now_et()
    broker_positions = get_current_positions()
    tracked_positions = state.get("positions", {})

    if not broker_positions:
        log("No live positions to manage.")
        return state

    for symbol, broker_pos in broker_positions.items():
        tracked = tracked_positions.get(symbol)
        if not tracked:
            tracked_positions[symbol] = {
                "symbol": symbol,
                "status": "open",
                "direction": broker_pos["side"],
                "qty": int(broker_pos["qty"]),
                "entry_order_id": "",
                "entry_price_est": round(broker_pos["avg_entry_price"], 4),
                "entry_price_fill": round(broker_pos["avg_entry_price"], 4),
                "stop_price": round(broker_pos["avg_entry_price"], 4),
                "target_price": round(broker_pos["avg_entry_price"], 4),
                "score": 0.0,
                "relative_volume": 0.0,
                "range_pct": 0.0,
                "spy_regime": "",
                "pattern_tags": "",
                "opened_at_et": current.strftime("%Y-%m-%d %H:%M:%S")
            }
            tracked = tracked_positions[symbol]

        direction = tracked.get("direction", broker_pos["side"])
        qty = int(tracked.get("qty", broker_pos["qty"]))
        entry_fill = safe_float(tracked.get("entry_price_fill"), broker_pos["avg_entry_price"])
        stop_price = safe_float(tracked.get("stop_price"), entry_fill)
        target_price = safe_float(tracked.get("target_price"), entry_fill)

        current_price = get_latest_trade_price(symbol)
        if current_price is None:
            current_price = safe_float(broker_pos.get("current_price"), entry_fill)

        reason = None

        if direction == "long":
            if current_price <= stop_price:
                reason = "stop_hit"
            elif current_price >= target_price:
                reason = "target_hit"
        else:
            if current_price >= stop_price:
                reason = "stop_hit"
            elif current_price <= target_price:
                reason = "target_hit"

        if is_force_exit_time(current):
            reason = "time_exit_1100"

        if not reason:
            continue

        exit_result = submit_exit_order(symbol, qty, direction)
        if not exit_result:
            tracked["status"] = "exit_rejected"
            tracked_positions[symbol] = tracked
            state["positions"] = tracked_positions
            save_state(state)
            continue

        exit_fill = exit_result["fill_price"]
        if exit_fill is None:
            exit_fill = current_price

        pnl = compute_trade_pnl(direction, entry_fill, exit_fill, qty)

        append_csv(TRADES_LOG_FILE, {
            "timestamp_et": current.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "direction": direction,
            "event": "exit_submitted",
            "qty": qty,
            "entry_order_id": tracked.get("entry_order_id", ""),
            "entry_price_est": tracked.get("entry_price_est", ""),
            "entry_price_fill": round(entry_fill, 4),
            "stop_price": round(stop_price, 4),
            "target_price": round(target_price, 4),
            "exit_order_id": exit_result["order_id"],
            "exit_price_est": round(current_price, 4),
            "exit_price_fill": round(exit_fill, 4),
            "reason": reason,
            "gross_pnl": pnl["gross_pnl"],
            "sec_fee": pnl["sec_fee"],
            "net_pnl": pnl["net_pnl"],
            "score": safe_float(tracked.get("score"), 0.0),
            "relative_volume": safe_float(tracked.get("relative_volume"), 0.0),
            "range_pct": safe_float(tracked.get("range_pct"), 0.0),
            "spy_regime": tracked.get("spy_regime", ""),
            "pattern_tags": tracked.get("pattern_tags", "")
        }, TRADE_COLUMNS)

        state["realized_net_pnl_today"] = round(
            safe_float(state.get("realized_net_pnl_today"), 0.0) + pnl["net_pnl"],
            4
        )
        if pnl["net_pnl"] < 0:
            state["losing_trades_today"] = int(state.get("losing_trades_today", 0)) + 1

        tracked["status"] = "closed"
        tracked["exit_order_id"] = exit_result["order_id"]
        tracked["exit_price_fill"] = round(exit_fill, 4)
        tracked["exit_reason"] = reason
        tracked["closed_at_et"] = current.strftime("%Y-%m-%d %H:%M:%S")
        tracked["gross_pnl"] = pnl["gross_pnl"]
        tracked["sec_fee"] = pnl["sec_fee"]
        tracked["net_pnl"] = pnl["net_pnl"]
        tracked_positions[symbol] = tracked

        state["positions"] = tracked_positions
        state = update_kill_switch(state)
        save_state(state)

    return state


# -----------------------------
# Main
# -----------------------------
def run_bot() -> None:
    log("=== bot_phase95.py start ===")
    ensure_dirs()

    state = load_state()
    state = reset_daily_state_if_needed(state)
    pattern_stats = load_pattern_stats()

    try:
        if not market_is_open():
            log("Market is closed. Exiting cleanly.")
            return

        current = now_et()
        log(
            f"Current ET time: {current.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"| PT {now_pacific().strftime('%H:%M:%S %Z')}"
        )

        state = maybe_manage_and_exit_positions(state)

        pattern_stats = refresh_pattern_stats()

        if is_entry_window(current):
            state = maybe_enter_new_positions(state, pattern_stats)
        else:
            log("Outside entry window for new entries.")

        save_state(state)
        log(
            f"Done. trades_today={state.get('trades_today', 0)} "
            f"losing_trades_today={state.get('losing_trades_today', 0)} "
            f"realized_net_pnl_today={state.get('realized_net_pnl_today', 0.0)} "
            f"kill_switch={state.get('kill_switch', False)}"
        )
        log("=== bot_phase95.py end ===")

    except Exception as e:
        log(f"Fatal error: {e}")
        traceback.print_exc()
        save_state(state)
        raise


if __name__ == "__main__":
    run_bot()
