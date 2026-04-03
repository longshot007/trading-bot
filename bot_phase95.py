import os
import json
import time
import traceback
from datetime import datetime, timedelta, time as dtime
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
# - Uses IEX by default to avoid SIP subscription failures
# - Adds scanner reject logging so 0-candidate days are diagnosable
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
PACIFIC = pytz.timezone("America/Los_Angeles")
UTC = pytz.UTC


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

MIN_PRICE = 3.0
MAX_PRICE = 250.0
MIN_DOLLAR_VOLUME = 400_000
MIN_RELATIVE_VOLUME = 0.35

MIN_5M_RANGE_PCT = 0.0012
MAX_5M_RANGE_PCT = 0.12
MAX_RISK_PER_SHARE_PCT = 0.05

ENTRY_BUFFER_PCT = 0.0003
MIN_ENTRY_BUFFER_DOLLARS = 0.01
NEAR_BREAKOUT_PCT = 0.0025

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

DATA_FEED_INTRADAY = "iex"
DATA_FEED_DAILY = "iex"

# Leniency increased so the scanner is less likely to filter everything out.
MIN_BODY_FRACTION = 0.18
MAX_OPPOSITE_WICK_FRACTION = 0.72
MIN_CLOSE_NEAR_EXTREME_LONG = 0.52
MAX_CLOSE_NEAR_EXTREME_SHORT = 0.48
MIN_ADAPTIVE_SCORE = -0.75

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL", "NFLX", "INTC",
    "MU", "PLTR", "CRM", "ADBE", "AVGO", "QCOM", "SHOP", "UBER", "COIN", "SMCI",
    "ARM", "SOFI", "PYPL", "SNOW", "PANW", "CRWD", "ANET", "MRVL", "F", "GM",
    "BAC", "JPM", "C", "WFC", "XOM", "CVX", "OXY", "SLB", "LLY", "UNH",
    "JNJ", "PFE", "NKE", "DIS", "BA", "CAT", "DE", "RIOT", "MARA", "HOOD",
    "RIVN", "LCID", "QQQ", "IWM", "DIA", "ARKK", "TQQQ", "SQQQ", "XLF", "XLK",
    "XLE", "XBI", "SMH", "SOXX", "HIMS", "DKNG", "AFRM", "UPST", "TEM", "NIO"
]


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
    "base_score", "score", "spy_regime", "pattern_tags", "action", "detail"
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


def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def append_csv(path: str, row: Dict, columns: List[str]) -> None:
    ensure_dirs()
    df = pd.DataFrame([row]).reindex(columns=columns)
    exists = os.path.exists(path)
    df.to_csv(path, mode="a" if exists else "w", index=False, header=not exists)


def load_json(path: str, fallback):
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback


def save_json(path: str, payload) -> None:
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        "last_scan_summary": {},
        "kill_switch": False,
        "kill_switch_reason": ""
    }


def load_state() -> Dict:
    return load_json(STATE_FILE, default_state())


def save_state(state: Dict) -> None:
    save_json(STATE_FILE, state)


def load_pattern_stats() -> Dict[str, Dict]:
    return load_json(PATTERN_STATS_FILE, {})


def save_pattern_stats(stats: Dict[str, Dict]) -> None:
    save_json(PATTERN_STATS_FILE, stats)


# -----------------------------
# Market / account
# -----------------------------
def get_account_equity() -> float:
    return safe_float(api.get_account().equity, 0.0)


def get_buying_power() -> float:
    return safe_float(api.get_account().buying_power, 0.0)


def reset_daily_state_if_needed(state: Dict) -> Dict:
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = default_state()
        state["date"] = today
        state["daily_start_equity"] = get_account_equity()
        save_state(state)
        log(f"Daily state reset. Start equity={state['daily_start_equity']:.2f}")
    return state


def market_is_open() -> Tuple[bool, datetime, str]:
    current = now_et()

    if current.weekday() >= 5:
        return False, current, "weekend"

    try:
        clock = api.get_clock()
        return bool(clock.is_open), current, f"alpaca_clock_{'open' if clock.is_open else 'closed'}"
    except Exception as e:
        # Fallback so clock failures do not incorrectly shut the bot down.
        is_open = dtime(9, 30) <= current.time() < dtime(16, 0)
        return is_open, current, f"fallback_time_window_due_to_clock_error:{e}"


def is_entry_window(current_et: datetime) -> bool:
    start = current_et.replace(hour=ENTRY_START_HOUR, minute=ENTRY_START_MINUTE, second=0, microsecond=0)
    end = current_et.replace(hour=ENTRY_END_HOUR, minute=ENTRY_END_MINUTE, second=0, microsecond=0)
    return start <= current_et <= end


def is_force_exit_time(current_et: datetime) -> bool:
    cutoff = current_et.replace(hour=FORCE_EXIT_HOUR, minute=FORCE_EXIT_MINUTE, second=0, microsecond=0)
    return current_et >= cutoff


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
                "avg_entry_price": safe_float(p.avg_entry_price, 0.0),
                "current_price": safe_float(getattr(p, "current_price", None), 0.0),
            }
    except Exception as e:
        log(f"Could not list positions: {e}")
    return positions


# -----------------------------
# Data retrieval
# -----------------------------
def _normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        if "symbol" in out.index.names:
            out = out.xs(symbol, level="symbol")
        else:
            out = out.xs(symbol, level=0)

    idx = pd.to_datetime(out.index, utc=True)
    out.index = idx.tz_convert(EASTERN)
    return out.sort_index()


def get_bars_safe(symbol: str, timeframe, start_iso: str, end_iso: str, feed: str) -> pd.DataFrame:
    try:
        bars = api.get_bars(
            symbol,
            timeframe,
            start=start_iso,
            end=end_iso,
            adjustment="raw",
            feed=feed
        ).df
        return _normalize_bars(bars, symbol)
    except Exception as e:
        log(f"Bars error for {symbol} feed={feed}: {e}")
        return pd.DataFrame()


def get_minute_bars(symbol: str, minutes_back: int = 120) -> pd.DataFrame:
    end_utc = datetime.now(UTC)
    start_utc = end_utc - timedelta(minutes=minutes_back + 30)

    bars = get_bars_safe(
        symbol,
        tradeapi.TimeFrame.Minute,
        start_utc.isoformat(),
        end_utc.isoformat(),
        DATA_FEED_INTRADAY
    )
    if not bars.empty:
        return bars

    return get_bars_safe(
        symbol,
        tradeapi.TimeFrame.Minute,
        start_utc.isoformat(),
        end_utc.isoformat(),
        "sip"
    )


def get_daily_bars(symbol: str, days: int = 20) -> pd.DataFrame:
    end_utc = datetime.now(UTC)
    start_utc = end_utc - timedelta(days=days + 15)

    bars = get_bars_safe(
        symbol,
        tradeapi.TimeFrame.Day,
        start_utc.isoformat(),
        end_utc.isoformat(),
        DATA_FEED_DAILY
    )
    if not bars.empty:
        return bars

    return get_bars_safe(
        symbol,
        tradeapi.TimeFrame.Day,
        start_utc.isoformat(),
        end_utc.isoformat(),
        "sip"
    )


def get_latest_trade_price(symbol: str) -> Optional[float]:
    minute_bars = get_minute_bars(symbol, 5)
    if minute_bars.empty:
        return None

    for col in ("close", "vwap", "open"):
        if col in minute_bars.columns:
            px = safe_float(minute_bars.iloc[-1][col], None)
            if px is not None and px > 0:
                return px
    return None


# -----------------------------
# Session filtering
# -----------------------------
def filter_today_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    today = now_et().date()
    out = df[df.index.date == today].copy()
    return out[(out.index.time >= dtime(9, 30)) & (out.index.time <= dtime(16, 0))]


# -----------------------------
# Opening candle logic
# -----------------------------
def build_opening_5m_candle(minute_bars: pd.DataFrame) -> Optional[Dict]:
    session = filter_today_regular_session(minute_bars)
    if session.empty:
        return None

    first = session[(session.index.time >= dtime(9, 30)) & (session.index.time < dtime(9, 35))]
    if len(first) < 5:
        return None

    open_ = float(first.iloc[0]["open"])
    high_ = float(first["high"].max())
    low_ = float(first["low"].min())
    close_ = float(first.iloc[-1]["close"])
    volume_ = float(first["volume"].sum())

    range_ = max(high_ - low_, 0.0001)
    body_fraction = abs(close_ - open_) / range_
    upper_wick = high_ - max(open_, close_)
    lower_wick = min(open_, close_) - low_
    close_position = (close_ - low_) / range_
    direction = "long" if close_ >= open_ else "short"
    opposite_wick_fraction = (lower_wick / range_) if direction == "long" else (upper_wick / range_)

    return {
        "open": open_,
        "high": high_,
        "low": low_,
        "close": close_,
        "volume": volume_,
        "range": range_,
        "range_pct": range_ / max(open_, 0.0001),
        "body_fraction": body_fraction,
        "close_position": close_position,
        "direction": direction,
        "opposite_wick_fraction": opposite_wick_fraction
    }


def opening_candle_is_valid(opening: Dict) -> bool:
    if opening["range_pct"] < MIN_5M_RANGE_PCT or opening["range_pct"] > MAX_5M_RANGE_PCT:
        return False
    if opening["body_fraction"] < MIN_BODY_FRACTION:
        return False
    if opening["opposite_wick_fraction"] > MAX_OPPOSITE_WICK_FRACTION:
        return False

    if opening["direction"] == "long":
        return opening["close_position"] >= MIN_CLOSE_NEAR_EXTREME_LONG
    return opening["close_position"] <= MAX_CLOSE_NEAR_EXTREME_SHORT


def build_trade_levels(opening: Dict, latest_price: float) -> Dict:
    buffer_amt = max(latest_price * ENTRY_BUFFER_PCT, MIN_ENTRY_BUFFER_DOLLARS)

    if opening["direction"] == "long":
        entry_price = opening["high"] + buffer_amt
        stop_price = opening["low"] - buffer_amt
        risk_per_share = entry_price - stop_price
        target_price = entry_price + TARGET_R_MULTIPLE * risk_per_share
    else:
        entry_price = opening["low"] - buffer_amt
        stop_price = opening["high"] + buffer_amt
        risk_per_share = stop_price - entry_price
        target_price = entry_price - TARGET_R_MULTIPLE * risk_per_share

    return {
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "risk_per_share": round(risk_per_share, 4),
        "risk_per_share_pct": round(risk_per_share / max(abs(entry_price), 0.0001), 4)
    }


def breakout_is_confirmed(opening: Dict, latest_price: float, levels: Dict) -> bool:
    trigger = levels["entry_price"]
    if opening["direction"] == "long":
        return latest_price >= trigger * (1 - NEAR_BREAKOUT_PCT)
    return latest_price <= trigger * (1 + NEAR_BREAKOUT_PCT)


# -----------------------------
# SPY regime
# -----------------------------
def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].replace(0, np.nan).cumsum()
    return cum_pv / cum_vol


def get_spy_regime() -> Dict:
    try:
        bars = filter_today_regular_session(get_minute_bars(SPY_SYMBOL, 180))
        if len(bars) < 10:
            return {"regime": "neutral"}

        bars = bars.copy()
        bars["vwap"] = compute_intraday_vwap(bars)
        opening = build_opening_5m_candle(bars)
        if not opening:
            return {"regime": "neutral"}

        last_close = float(bars.iloc[-1]["close"])
        vwap = float(bars.iloc[-1]["vwap"])
        midpoint = (opening["high"] + opening["low"]) / 2.0

        if last_close > vwap and last_close > midpoint:
            regime = "bullish"
        elif last_close < vwap and last_close < midpoint:
            regime = "bearish"
        else:
            regime = "neutral"

        return {"regime": regime, "last": round(last_close, 4), "vwap": round(vwap, 4)}
    except Exception as e:
        log(f"SPY regime error: {e}")
        return {"regime": "neutral"}


def spy_score_adjustment(spy_regime: str, direction: str) -> float:
    if spy_regime == "neutral":
        return 0.0
    if direction == "long":
        return 0.60 if spy_regime == "bullish" else -0.35
    return 0.60 if spy_regime == "bearish" else -0.35


# -----------------------------
# Volume / learning / scoring
# -----------------------------
def compute_relative_volume(symbol: str, opening_volume: float) -> float:
    daily = get_daily_bars(symbol, 20)
    if daily.empty or len(daily) < 5:
        return 1.0

    avg_daily_vol = float(daily["volume"].tail(10).mean())
    if avg_daily_vol <= 0:
        return 1.0

    opening_proxy = avg_daily_vol / 78.0
    if opening_proxy <= 0:
        return 1.0

    return opening_volume / opening_proxy


def read_trade_log() -> pd.DataFrame:
    if not os.path.exists(TRADES_LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(TRADES_LOG_FILE)
    except Exception:
        return pd.DataFrame()


def build_pattern_stats() -> Dict[str, Dict]:
    df = read_trade_log()
    if df.empty or "event" not in df.columns:
        return {}

    exits = df[df["event"] == "exit_submitted"].copy()
    stats: Dict[str, Dict] = {}

    for _, row in exits.iterrows():
        tags_str = str(row.get("pattern_tags", "")).strip()
        if not tags_str or tags_str == "nan":
            continue

        pnl = safe_float(row.get("net_pnl"), 0.0)
        win = 1 if pnl > 0 else 0

        for tag in [t for t in tags_str.split("|") if t]:
            stats.setdefault(tag, {"count": 0, "wins": 0, "net_pnl_sum": 0.0})
            stats[tag]["count"] += 1
            stats[tag]["wins"] += win
            stats[tag]["net_pnl_sum"] += pnl

    return stats


def refresh_pattern_stats() -> Dict[str, Dict]:
    stats = build_pattern_stats()
    save_pattern_stats(stats)
    return stats


def adaptive_pattern_score(tags: List[str], stats: Dict[str, Dict]) -> float:
    if not tags:
        return 0.0

    vals = []
    for tag in tags:
        item = stats.get(tag)
        if not item:
            continue

        count = int(item.get("count", 0))
        wins = int(item.get("wins", 0))
        net_sum = safe_float(item.get("net_pnl_sum", 0.0), 0.0)

        win_rate = (wins + 2.0) / (count + 4.0)
        pnl_per_trade = net_sum / max(count, 1)
        score = (win_rate - 0.5) * 1.4 + max(-0.25, min(0.25, pnl_per_trade / 60.0))
        if count < 5:
            score *= 0.5
        vals.append(score)

    return float(np.mean(vals)) if vals else 0.0


def candidate_pattern_tags(opening: Dict, rel_vol: float, spy_regime: str, risk_per_share_pct: float) -> List[str]:
    tags = [f"dir_{opening['direction']}", f"spy_{spy_regime}"]
    tags.append("rvol_high" if rel_vol >= 2.0 else "rvol_good" if rel_vol >= 1.0 else "rvol_ok")
    tags.append("body_strong" if opening["body_fraction"] >= 0.55 else "body_ok")

    if opening["direction"] == "long" and opening["close_position"] >= 0.85:
        tags.append("close_at_high")
    if opening["direction"] == "short" and opening["close_position"] <= 0.15:
        tags.append("close_at_low")

    tags.append("range_expanded" if opening["range_pct"] >= 0.01 else "range_normal")
    tags.append("tight_risk" if risk_per_share_pct <= 0.015 else "normal_risk" if risk_per_share_pct <= 0.03 else "wide_risk")
    return tags


def base_candidate_score(opening: Dict, rel_vol: float, latest_price: float) -> float:
    score = rel_vol * 3.0
    score += opening["range_pct"] * 80.0
    score += opening["body_fraction"] * 1.2
    score += opening["close_position"] if opening["direction"] == "long" else (1.0 - opening["close_position"])
    score += latest_price / 1500.0
    return score


# -----------------------------
# Sizing / orders / pnl
# -----------------------------
def calculate_qty(entry_price: float, stop_price: float, equity: float, buying_power: float) -> int:
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0

    risk_budget = equity * RISK_PER_TRADE_PCT
    qty_by_risk = int(risk_budget // risk_per_share)
    qty_by_bp = int((buying_power * 0.95) // max(entry_price, 0.01))
    return max(0, min(qty_by_risk, qty_by_bp))


def wait_for_fill_price(order_id: str, retries: int = 6, sleep_seconds: int = 2) -> Tuple[Optional[float], str]:
    for _ in range(retries):
        try:
            order = api.get_order(order_id)
            status = str(order.status).lower()
            fill = safe_float(getattr(order, "filled_avg_price", None), None)
            if status in {"filled", "partially_filled"} and fill is not None:
                return fill, status
            if status in {"canceled", "expired", "rejected"}:
                return None, status
        except Exception:
            pass
        time.sleep(sleep_seconds)
    return None, "timeout"


def submit_entry_order(symbol: str, qty: int, direction: str) -> Optional[Dict]:
    side = "buy" if direction == "long" else "sell"
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
        log(f"ENTRY submitted: {symbol} {direction} qty={qty} order_id={order.id}")
        fill_price, fill_status = wait_for_fill_price(order.id)
        return {"order_id": order.id, "fill_price": fill_price, "fill_status": fill_status}
    except Exception as e:
        log(f"Entry order failed for {symbol}: {e}")
        return None


def submit_exit_order(symbol: str, qty: int, position_side: str) -> Optional[Dict]:
    exit_side = "sell" if position_side == "long" else "buy"
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=exit_side, type="market", time_in_force="day")
        log(f"EXIT submitted: {symbol} {position_side} qty={qty} order_id={order.id}")
        fill_price, fill_status = wait_for_fill_price(order.id)
        return {"order_id": order.id, "fill_price": fill_price, "fill_status": fill_status}
    except Exception as e:
        log(f"Exit order failed for {symbol}: {e}")
        return None


def estimate_sec_fee(direction: str, exit_price: float, qty: int) -> float:
    return round(exit_price * qty * SEC_FEE_RATE, 6) if direction == "long" else 0.0


def compute_trade_pnl(direction: str, entry_fill: float, exit_fill: float, qty: int) -> Dict[str, float]:
    gross = (exit_fill - entry_fill) * qty if direction == "long" else (entry_fill - exit_fill) * qty
    sec_fee = estimate_sec_fee(direction, exit_fill, qty)
    return {
        "gross_pnl": round(gross, 4),
        "sec_fee": round(sec_fee, 6),
        "net_pnl": round(gross - sec_fee, 4)
    }


# -----------------------------
# Kill switch
# -----------------------------
def update_kill_switch(state: Dict) -> Dict:
    start_equity = safe_float(state.get("daily_start_equity"), 0.0)
    realized = safe_float(state.get("realized_net_pnl_today"), 0.0)
    losing = int(state.get("losing_trades_today", 0))

    if start_equity > 0:
        loss_pct = max(0.0, -realized / start_equity)
        if loss_pct >= MAX_DAILY_NET_LOSS_PCT:
            state["kill_switch"] = True
            state["kill_switch_reason"] = f"daily_net_loss_limit_{MAX_DAILY_NET_LOSS_PCT:.2%}"
            return state

    if losing >= MAX_DAILY_LOSING_TRADES:
        state["kill_switch"] = True
        state["kill_switch_reason"] = f"max_losing_trades_{MAX_DAILY_LOSING_TRADES}"

    return state


# -----------------------------
# Signal logging
# -----------------------------
def log_signal(
    symbol: str,
    direction: str,
    latest_price: float,
    opening: Dict,
    rel_vol: float,
    adaptive_score: float,
    base_score: float,
    score: float,
    spy_regime: str,
    pattern_tags: str,
    action: str,
    detail: str
) -> None:
    current = now_et()
    append_csv(SIGNALS_LOG_FILE, {
        "timestamp_et": current.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction,
        "latest_price": round(latest_price, 4),
        "opening_open": round(opening["open"], 4),
        "opening_high": round(opening["high"], 4),
        "opening_low": round(opening["low"], 4),
        "opening_close": round(opening["close"], 4),
        "opening_volume": int(opening["volume"]),
        "relative_volume": round(rel_vol, 4),
        "range_pct": round(opening["range_pct"], 4),
        "body_fraction": round(opening["body_fraction"], 4),
        "close_position": round(opening["close_position"], 4),
        "adaptive_score": round(adaptive_score, 4),
        "base_score": round(base_score, 4),
        "score": round(score, 4),
        "spy_regime": spy_regime,
        "pattern_tags": pattern_tags,
        "action": action,
        "detail": detail
    }, SIGNAL_COLUMNS)


# -----------------------------
# Scanner
# -----------------------------
def scan_candidates(pattern_stats: Dict[str, Dict]) -> List[Dict]:
    spy = get_spy_regime()
    spy_regime = spy.get("regime", "neutral")
    equity = get_account_equity()
    buying_power = get_buying_power()

    candidates: List[Dict] = []
    reason_counts: Dict[str, int] = {}
    checked = 0

    for symbol in DEFAULT_UNIVERSE[:SCANNER_LIMIT]:
        if symbol == SPY_SYMBOL:
            continue

        checked += 1

        try:
            minute_bars = get_minute_bars(symbol, 180)
            if minute_bars.empty:
                reason_counts["no_bars"] = reason_counts.get("no_bars", 0) + 1
                continue

            opening = build_opening_5m_candle(minute_bars)
            if not opening:
                reason_counts["no_opening_5m"] = reason_counts.get("no_opening_5m", 0) + 1
                continue

            latest_price = get_latest_trade_price(symbol)
            if latest_price is None:
                reason_counts["no_latest_price"] = reason_counts.get("no_latest_price", 0) + 1
                continue

            if latest_price < MIN_PRICE or latest_price > MAX_PRICE:
                reason_counts["price_filter"] = reason_counts.get("price_filter", 0) + 1
                continue

            rel_vol = compute_relative_volume(symbol, opening["volume"])
            dollar_volume = latest_price * opening["volume"]
            levels = build_trade_levels(opening, latest_price)
            pattern_tags = candidate_pattern_tags(opening, rel_vol, spy_regime, levels["risk_per_share_pct"])
            adaptive_score = adaptive_pattern_score(pattern_tags, pattern_stats)
            base_score = base_candidate_score(opening, rel_vol, latest_price)
            total_score = base_score + adaptive_score + spy_score_adjustment(spy_regime, opening["direction"])
            pattern_tags_str = "|".join(pattern_tags)

            if not opening_candle_is_valid(opening):
                reason_counts["opening_invalid"] = reason_counts.get("opening_invalid", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "opening_invalid")
                continue

            if rel_vol < MIN_RELATIVE_VOLUME:
                reason_counts["rvol_filter"] = reason_counts.get("rvol_filter", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "rvol_filter")
                continue

            if dollar_volume < MIN_DOLLAR_VOLUME:
                reason_counts["dollar_volume_filter"] = reason_counts.get("dollar_volume_filter", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "dollar_volume_filter")
                continue

            if levels["risk_per_share_pct"] > MAX_RISK_PER_SHARE_PCT:
                reason_counts["risk_too_wide"] = reason_counts.get("risk_too_wide", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "risk_too_wide")
                continue

            if adaptive_score < MIN_ADAPTIVE_SCORE:
                reason_counts["adaptive_filter"] = reason_counts.get("adaptive_filter", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "adaptive_filter")
                continue

            if not breakout_is_confirmed(opening, latest_price, levels):
                reason_counts["breakout_not_ready"] = reason_counts.get("breakout_not_ready", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "breakout_not_ready")
                continue

            qty = calculate_qty(levels["entry_price"], levels["stop_price"], equity, buying_power)
            if qty < 1:
                reason_counts["qty_zero"] = reason_counts.get("qty_zero", 0) + 1
                log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_reject", "qty_zero")
                continue

            candidate = {
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
            }

            log_signal(symbol, opening["direction"], latest_price, opening, rel_vol, adaptive_score, base_score, total_score, spy_regime, pattern_tags_str, "scan_accept", "candidate")
            candidates.append(candidate)

        except Exception as e:
            reason_counts["exception"] = reason_counts.get("exception", 0) + 1
            log(f"Scanner skip {symbol}: {e}")

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    log(f"Scanner checked {checked} symbols. Candidates={len(candidates)}. Rejects={reason_counts}")
    return candidates


# -----------------------------
# Entry management
# -----------------------------
def maybe_enter_new_positions(state: Dict, pattern_stats: Dict[str, Dict]) -> Dict:
    current = now_et()

    if not is_entry_window(current):
        log("Outside entry window for new entries.")
        return state

    state = update_kill_switch(state)
    if state.get("kill_switch", False):
        log(f"Kill switch active. Reason: {state.get('kill_switch_reason', '')}")
        return state

    broker_positions = get_current_positions()
    if len(broker_positions) >= MAX_OPEN_POSITIONS:
        log("Max open positions reached.")
        return state

    if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
        log("Max trades per day reached.")
        return state

    candidates = scan_candidates(pattern_stats)
    state["last_scan_candidates"] = [
        {
            "symbol": c["symbol"],
            "score": round(c["score"], 4),
            "direction": c["direction"],
            "latest_price": round(c["latest_price"], 4),
            "entry_price": round(c["entry_price"], 4),
            "stop_price": round(c["stop_price"], 4),
            "target_price": round(c["target_price"], 4),
            "relative_volume": round(c["relative_volume"], 4)
        }
        for c in candidates[:10]
    ]
    state["last_scan_summary"] = {
        "checked_at_et": current.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidates)
    }
    save_state(state)

    if not candidates:
        log("No valid candidates found.")
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

        order_result = submit_entry_order(symbol, c["qty"], c["direction"])
        pattern_tags_str = "|".join(c["pattern_tags"])

        if not order_result:
            log_signal(symbol, c["direction"], c["latest_price"], c["opening"], c["relative_volume"], c["adaptive_score"], c["base_score"], c["score"], c["spy_regime"], pattern_tags_str, "entry_failed", "order_failed")
            continue

        entry_fill = order_result["fill_price"] if order_result["fill_price"] is not None else c["latest_price"]

        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        state.setdefault("symbols_traded_today", []).append(symbol)
        state.setdefault("positions", {})[symbol] = {
            "symbol": symbol,
            "direction": c["direction"],
            "qty": c["qty"],
            "entry_order_id": order_result["order_id"],
            "entry_price_est": round(c["entry_price"], 4),
            "entry_price_fill": round(entry_fill, 4),
            "stop_price": round(c["stop_price"], 4),
            "target_price": round(c["target_price"], 4),
            "score": round(c["score"], 4),
            "relative_volume": round(c["relative_volume"], 4),
            "range_pct": round(c["opening"]["range_pct"], 4),
            "spy_regime": c["spy_regime"],
            "pattern_tags": pattern_tags_str,
            "status": "open",
            "entered_at_et": current.strftime("%Y-%m-%d %H:%M:%S")
        }

        append_csv(TRADES_LOG_FILE, {
            "timestamp_et": current.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "direction": c["direction"],
            "event": "entry_submitted",
            "qty": c["qty"],
            "entry_order_id": order_result["order_id"],
            "entry_price_est": round(c["entry_price"], 4),
            "entry_price_fill": round(entry_fill, 4),
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
            "pattern_tags": pattern_tags_str
        }, TRADE_COLUMNS)

        log_signal(symbol, c["direction"], c["latest_price"], c["opening"], c["relative_volume"], c["adaptive_score"], c["base_score"], c["score"], c["spy_regime"], pattern_tags_str, "entry_submitted", "order_submitted")
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
            continue

        direction = tracked.get("direction")
        qty = int(tracked.get("qty", broker_pos["qty"]))
        entry_fill = safe_float(tracked.get("entry_price_fill"), 0.0)
        stop_price = safe_float(tracked.get("stop_price"), 0.0)
        target_price = safe_float(tracked.get("target_price"), 0.0)

        current_price = get_latest_trade_price(symbol)
        if current_price is None or current_price <= 0:
            current_price = safe_float(broker_pos.get("current_price"), 0.0)
        if current_price <= 0:
            continue

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
            tracked["status"] = "exit_failed"
            tracked_positions[symbol] = tracked
            save_state(state)
            continue

        exit_fill = exit_result["fill_price"] if exit_result["fill_price"] is not None else current_price
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

        state["realized_net_pnl_today"] = round(safe_float(state.get("realized_net_pnl_today"), 0.0) + pnl["net_pnl"], 4)
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
        is_open, current, market_reason = market_is_open()
        log(
            f"Current ET time: {current.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"| PT {now_pacific().strftime('%H:%M:%S %Z')}"
        )
        log(f"Market check: is_open={is_open} reason={market_reason}")

        if not is_open:
            log("Market is closed. Exiting cleanly.")
            return

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
