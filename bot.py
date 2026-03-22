import os
import json
import math
import time
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PHASE 9 - ADAPTIVE LEARNING BOT
# Persistent rolling learning + capped storage + 48+ signals
# ============================================================

# -------------------------
# Environment / API
# -------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing Alpaca credentials in environment variables.")

api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version="v2")


# -------------------------
# Core config
# -------------------------
TIMEFRAME = tradeapi.TimeFrame.Minute
LOOKBACK_BARS = 260
LABEL_HORIZON = 8
LABEL_THRESHOLD = 0.0025        # 0.25% future move
MIN_TRAIN_ROWS = 400
MAX_TRAIN_ROWS = 5000
MAX_TRADE_HISTORY_ROWS = 500
MAX_CANDIDATES = 10
MAX_POSITIONS = 4
MAX_ALLOC_PER_POSITION = 0.22
CASH_RESERVE = 0.08
MIN_PRICE = 7.0
MAX_PRICE = 300.0
MIN_DOLLAR_VOLUME = 5_000_000
MIN_PROB_TO_BUY = 0.57
PROB_SELL_FLOOR = 0.45

STOP_LOSS_PCT = 0.018
TAKE_PROFIT_PCT = 0.03
TRAIL_ACTIVATE_PCT = 0.015
TRAIL_GIVEBACK_PCT = 0.008
MAX_HOLD_MINUTES = 180
MIN_ORDER_NOTIONAL = 25.0
RETRAIN_EVERY_N_NEW_ROWS = 250
RANDOM_STATE = 42

SEC_FEE_RATE = 0.0000206   # user-required fee model on sells
TRADING_FEE_BUFFER = 0.0005

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL",
    "NFLX", "AVGO", "PLTR", "SMCI", "MU", "UBER", "CRM", "ORCL",
    "INTC", "QCOM", "SHOP", "SQ", "PANW", "CRWD", "SNOW", "COIN",
    "ARM", "ADBE", "PYPL", "TTD", "MRVL", "NOW"
]

# -------------------------
# Storage
# -------------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"

for p in [DATA_DIR, MODELS_DIR, LOGS_DIR, STATE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.csv"
MODEL_FILE = MODELS_DIR / "phase9_model.pkl"
MODEL_META_FILE = MODELS_DIR / "phase9_model_meta.json"
OPEN_STATE_FILE = STATE_DIR / "open_positions.json"
RUN_LOG_FILE = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"


# -------------------------
# Utilities
# -------------------------
def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def rolling_trim_csv(path: Path, max_rows: int) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.tail(max_rows).copy()
        df.to_csv(path, index=False)


def append_df(path: Path, new_df: pd.DataFrame, dedupe_cols=None, max_rows=None) -> int:
    if new_df is None or new_df.empty:
        return 0
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df.copy()

    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows).copy()

    df.to_csv(path, index=False)
    return len(new_df)


def market_is_open() -> bool:
    try:
        return bool(api.get_clock().is_open)
    except Exception as e:
        log(f"Clock check failed: {e}")
        return False


def get_account():
    return api.get_account()


def get_cash() -> float:
    return safe_float(get_account().cash, 0.0)


def get_equity() -> float:
    return safe_float(get_account().equity, 0.0)


def get_positions_dict():
    out = {}
    try:
        for p in api.list_positions():
            out[p.symbol] = p
    except Exception as e:
        log(f"Could not load positions: {e}")
    return out


def cancel_open_orders(symbol: str = None) -> None:
    try:
        orders = api.list_orders(status="open")
        for o in orders:
            if symbol is None or o.symbol == symbol:
                try:
                    api.cancel_order(o.id)
                except Exception:
                    pass
    except Exception as e:
        log(f"Cancel open orders error: {e}")


# -------------------------
# Indicators / signal helpers
# -------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df).replace(0, np.nan)
    atr_n = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_n
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_n = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_n = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    return -100 * (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw = tp * df["volume"]
    direction = tp.diff()
    pos_flow = pd.Series(np.where(direction > 0, raw, 0.0), index=df.index)
    neg_flow = pd.Series(np.where(direction < 0, raw, 0.0), index=df.index)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).fillna(0).cumsum()


def adl(df: pd.DataFrame) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm.fillna(0) * df["volume"]
    return mfv.cumsum()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm.fillna(0) * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def vpt(df: pd.DataFrame) -> pd.Series:
    return (df["volume"] * (df["close"].pct_change().fillna(0))).cumsum()


def linear_slope(series: pd.Series, window: int) -> pd.Series:
    vals = series.values
    out = np.full(len(series), np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, len(series)):
        y = vals[i - window + 1:i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        out[i] = ((x - x_mean) * (y - y_mean)).sum() / denom
    return pd.Series(out, index=series.index)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vv = df["volume"].cumsum().replace(0, np.nan)
    return pv / vv


def detect_candles(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    prev_o = o.shift(1)
    prev_c = c.shift(1)

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    out = pd.DataFrame(index=df.index)
    out["cdl_doji"] = (body / rng < 0.1).astype(int)
    out["cdl_hammer"] = ((lower / rng > 0.5) & (upper / rng < 0.15) & (c > o)).astype(int)
    out["cdl_shooting_star"] = ((upper / rng > 0.5) & (lower / rng < 0.15) & (c < o)).astype(int)
    out["cdl_bull_engulf"] = ((prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)).astype(int)
    out["cdl_bear_engulf"] = ((prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)).astype(int)
    out["cdl_inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    out["cdl_outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)

    out["cdl_morning_star"] = (
        (c.shift(2) < o.shift(2)) &
        ((c.shift(1) - o.shift(1)).abs() < (h.shift(1) - l.shift(1)) * 0.2) &
        (c > o) &
        (c > ((o.shift(2) + c.shift(2)) / 2))
    ).astype(int)

    out["cdl_evening_star"] = (
        (c.shift(2) > o.shift(2)) &
        ((c.shift(1) - o.shift(1)).abs() < (h.shift(1) - l.shift(1)) * 0.2) &
        (c < o) &
        (c < ((o.shift(2) + c.shift(2)) / 2))
    ).astype(int)

    out["cdl_three_white_soldiers"] = (
        (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
        (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    ).astype(int)

    out["cdl_three_black_crows"] = (
        (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) &
        (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    ).astype(int)

    return out


# -------------------------
# Data fetch / normalization
# -------------------------
def normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        if "symbol" in df.index.names:
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                try:
                    df = df.reset_index()
                    df = df[df["symbol"] == symbol].copy()
                except Exception:
                    pass

    df = df.copy().reset_index()

    if "timestamp" not in df.columns:
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})

    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in expected:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' for {symbol}")

    if "trade_count" not in df.columns:
        if "n" in df.columns:
            df["trade_count"] = df["n"]
        else:
            df["trade_count"] = 0.0

    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df["symbol"] = symbol

    numeric_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def get_bars(symbol: str, limit: int = LOOKBACK_BARS) -> pd.DataFrame:
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=limit).df
        return normalize_bars(bars, symbol)
    except Exception as e:
        log(f"Data fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# -------------------------
# Feature engineering
# -------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 120:
        return pd.DataFrame()

    x = df.copy()
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_5"] = x["close"].pct_change(5)
    x["ret_10"] = x["close"].pct_change(10)
    x["ret_20"] = x["close"].pct_change(20)
    x["logret_1"] = np.log(x["close"] / x["close"].shift(1))

    x["range_pct"] = (x["high"] - x["low"]) / x["close"]
    x["body_pct"] = (x["close"] - x["open"]).abs() / x["close"]
    x["upper_wick_pct"] = (x["high"] - x[["open", "close"]].max(axis=1)) / x["close"]
    x["lower_wick_pct"] = (x[["open", "close"]].min(axis=1) - x["low"]) / x["close"]
    x["close_pos_in_bar"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    x["gap_pct"] = (x["open"] - x["close"].shift(1)) / x["close"].shift(1)

    x["tr"] = true_range(x)
    x["tr_pct"] = x["tr"] / x["close"]
    x["atr14"] = atr(x, 14)
    x["atr20"] = atr(x, 20)
    x["atr14_pct"] = x["atr14"] / x["close"]
    x["atr20_pct"] = x["atr20"] / x["close"]
    x["atr_ratio"] = x["atr14"] / x["atr20"].replace(0, np.nan)

    x["volatility_5"] = x["ret_1"].rolling(5).std()
    x["volatility_10"] = x["ret_1"].rolling(10).std()
    x["volatility_20"] = x["ret_1"].rolling(20).std()

    x["zscore_20"] = (x["close"] - x["close"].rolling(20).mean()) / x["close"].rolling(20).std().replace(0, np.nan)
    x["zscore_50"] = (x["close"] - x["close"].rolling(50).mean()) / x["close"].rolling(50).std().replace(0, np.nan)

    for n in [5, 10, 20, 50, 100]:
        x[f"sma_{n}"] = x["close"].rolling(n).mean()
        x[f"dist_sma_{n}"] = (x["close"] - x[f"sma_{n}"]) / x[f"sma_{n}"]

    for n in [8, 12, 21, 34, 55]:
        x[f"ema_{n}"] = x["close"].ewm(span=n, adjust=False).mean()
        x[f"dist_ema_{n}"] = (x["close"] - x[f"ema_{n}"]) / x[f"ema_{n}"]

    x["sma_cross_5_20"] = (x["sma_5"] - x["sma_20"]) / x["sma_20"]
    x["ema_cross_12_26"] = (x["close"].ewm(span=12, adjust=False).mean() - x["close"].ewm(span=26, adjust=False).mean()) / x["close"]
    x["ema_slope_8"] = linear_slope(x["ema_8"], 8)
    x["ema_slope_21"] = linear_slope(x["ema_21"], 10)
    x["ema_slope_55"] = linear_slope(x["ema_55"], 12)

    bb_mid = x["close"].rolling(20).mean()
    bb_std = x["close"].rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    x["bb_width"] = (bb_up - bb_dn) / bb_mid.replace(0, np.nan)
    x["bb_pos"] = (x["close"] - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)

    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    kc_up = ema20 + 2 * x["atr20"]
    kc_dn = ema20 - 2 * x["atr20"]
    x["keltner_width"] = (kc_up - kc_dn) / ema20.replace(0, np.nan)

    dc_high = x["high"].rolling(20).max()
    dc_low = x["low"].rolling(20).min()
    x["donchian_pos"] = (x["close"] - dc_low) / (dc_high - dc_low).replace(0, np.nan)
    x["breakout_20_up"] = (x["close"] > dc_high.shift(1)).astype(int)
    x["breakout_20_down"] = (x["close"] < dc_low.shift(1)).astype(int)

    x["rsi_14"] = rsi(x["close"], 14)
    x["stoch_k_14"] = stochastic_k(x, 14)
    x["stoch_d_3"] = x["stoch_k_14"].rolling(3).mean()
    x["willr_14"] = williams_r(x, 14)
    x["cci_20"] = cci(x, 20)
    x["adx_14"] = adx(x, 14)

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    x["mfi_14"] = mfi(x, 14)
    x["obv"] = obv(x)
    x["obv_slope_10"] = linear_slope(x["obv"], 10)
    x["adl"] = adl(x)
    x["adl_slope_10"] = linear_slope(x["adl"], 10)
    x["cmf_20"] = cmf(x, 20)
    x["vpt"] = vpt(x)
    x["vpt_slope_10"] = linear_slope(x["vpt"], 10)

    x["rel_volume_5"] = x["volume"] / x["volume"].rolling(5).mean().replace(0, np.nan)
    x["rel_volume_10"] = x["volume"] / x["volume"].rolling(10).mean().replace(0, np.nan)
    x["rel_volume_20"] = x["volume"] / x["volume"].rolling(20).mean().replace(0, np.nan)
    x["volume_z_20"] = (x["volume"] - x["volume"].rolling(20).mean()) / x["volume"].rolling(20).std().replace(0, np.nan)
    x["trade_count_z_20"] = (x["trade_count"] - x["trade_count"].rolling(20).mean()) / x["trade_count"].rolling(20).std().replace(0, np.nan)
    x["dollar_volume"] = x["close"] * x["volume"]
    x["dollar_volume_ma20"] = x["dollar_volume"].rolling(20).mean()

    x["vwap_dist"] = (x["close"] - x["vwap"]) / x["vwap"].replace(0, np.nan)
    x["session_vwap"] = session_vwap(x)
    x["session_vwap_dist"] = (x["close"] - x["session_vwap"]) / x["session_vwap"].replace(0, np.nan)

    x["minute_of_session"] = np.arange(len(x))
    x["minute_norm"] = x["minute_of_session"] / max(len(x), 1)

    candle_df = detect_candles(x)
    x = pd.concat([x, candle_df], axis=1)

    x["support_20"] = x["low"].rolling(20).min()
    x["resistance_20"] = x["high"].rolling(20).max()
    x["dist_support_20"] = (x["close"] - x["support_20"]) / x["close"]
    x["dist_resistance_20"] = (x["resistance_20"] - x["close"]) / x["close"]

    x["future_return"] = x["close"].shift(-LABEL_HORIZON) / x["close"] - 1
    x["target"] = (x["future_return"] > LABEL_THRESHOLD).astype(int)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "logret_1",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct", "close_pos_in_bar", "gap_pct",
    "tr_pct", "atr14_pct", "atr20_pct", "atr_ratio",
    "volatility_5", "volatility_10", "volatility_20",
    "zscore_20", "zscore_50",
    "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_sma_50", "dist_sma_100",
    "dist_ema_8", "dist_ema_12", "dist_ema_21", "dist_ema_34", "dist_ema_55",
    "sma_cross_5_20", "ema_cross_12_26", "ema_slope_8", "ema_slope_21", "ema_slope_55",
    "bb_width", "bb_pos", "keltner_width", "donchian_pos", "breakout_20_up", "breakout_20_down",
    "rsi_14", "stoch_k_14", "stoch_d_3", "willr_14", "cci_20", "adx_14",
    "macd", "macd_signal", "macd_hist",
    "mfi_14", "obv_slope_10", "adl_slope_10", "cmf_20", "vpt_slope_10",
    "rel_volume_5", "rel_volume_10", "rel_volume_20", "volume_z_20", "trade_count_z_20",
    "vwap_dist", "session_vwap_dist", "minute_norm",
    "cdl_doji", "cdl_hammer", "cdl_shooting_star", "cdl_bull_engulf", "cdl_bear_engulf",
    "cdl_inside_bar", "cdl_outside_bar", "cdl_morning_star", "cdl_evening_star",
    "cdl_three_white_soldiers", "cdl_three_black_crows",
    "dist_support_20", "dist_resistance_20"
]


# -------------------------
# Training data build / persistence
# -------------------------
def make_training_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    needed = ["timestamp", "symbol", "close", "future_return", "target"] + FEATURE_COLUMNS
    rows = feature_df[needed].copy()
    rows = rows.iloc[:-LABEL_HORIZON].copy() if len(rows) > LABEL_HORIZON else pd.DataFrame()
    if rows.empty:
        return rows

    rows = rows.dropna(subset=FEATURE_COLUMNS + ["target"])
    rows["target"] = rows["target"].astype(int)
    return rows


def rebuild_symbol_training_data(symbol: str) -> pd.DataFrame:
    bars = get_bars(symbol, LOOKBACK_BARS)
    if bars.empty:
        return pd.DataFrame()
    feats = build_features(bars)
    train_rows = make_training_rows(feats)
    return train_rows


def update_training_store(symbols) -> int:
    parts = []
    for s in symbols:
        try:
            df = rebuild_symbol_training_data(s)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            log(f"Training build failed for {s}: {e}")

    if not parts:
        return 0

    batch = pd.concat(parts, ignore_index=True)
    batch["timestamp"] = pd.to_datetime(batch["timestamp"], utc=True)
    batch["timestamp"] = batch["timestamp"].astype(str)

    n = append_df(
        TRAIN_DATA_FILE,
        batch,
        dedupe_cols=["symbol", "timestamp"],
        max_rows=MAX_TRAIN_ROWS
    )
    return n


def load_training_data() -> pd.DataFrame:
    if not TRAIN_DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRAIN_DATA_FILE)
    if df.empty:
        return df
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])
    df["target"] = df["target"].astype(int)
    return df


def load_model():
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None


def load_model_meta():
    return load_json(MODEL_META_FILE, default={
        "trained_at": None,
        "train_rows": 0,
        "new_rows_since_train": 0,
        "feature_count": len(FEATURE_COLUMNS),
        "model_type": "RandomForestClassifier"
    })


def save_model(model, meta: dict) -> None:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    save_json(MODEL_META_FILE, meta)


def train_or_load_model(force=False):
    meta = load_model_meta()
    model = load_model()
    train_df = load_training_data()

    need_train = force or model is None
    if meta.get("new_rows_since_train", 0) >= RETRAIN_EVERY_N_NEW_ROWS:
        need_train = True
    if len(train_df) < MIN_TRAIN_ROWS:
        need_train = False if model is not None else False

    if not need_train and model is not None:
        return model, meta

    if len(train_df) < MIN_TRAIN_ROWS:
        log(f"Not enough train rows yet: {len(train_df)}")
        return model, meta

    X = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["target"].astype(int)

    class_balance = y.mean()
    log(f"Training model on {len(train_df)} rows | positive rate={class_balance:.3f}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=12,
        min_samples_split=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X, y)

    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_rows": int(len(train_df)),
        "new_rows_since_train": 0,
        "feature_count": len(FEATURE_COLUMNS),
        "model_type": "RandomForestClassifier"
    }
    save_model(model, meta)
    return model, meta


# -------------------------
# Adaptive thresholds from trade history
# -------------------------
def load_trade_history() -> pd.DataFrame:
    if not TRADE_HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADE_HISTORY_FILE)
    return df


def adaptive_buy_threshold() -> float:
    df = load_trade_history()
    if df.empty or len(df) < 25 or "net_pnl_pct" not in df.columns:
        return MIN_PROB_TO_BUY

    recent = df.tail(50).copy()
    win_rate = (recent["net_pnl_pct"] > 0).mean()
    avg_pnl = recent["net_pnl_pct"].mean()

    threshold = MIN_PROB_TO_BUY
    if win_rate < 0.45 or avg_pnl < 0:
        threshold += 0.03
    elif win_rate > 0.60 and avg_pnl > 0.003:
        threshold -= 0.02

    return max(0.54, min(0.65, threshold))


# -------------------------
# Position state persistence
# -------------------------
def load_open_state() -> dict:
    return load_json(OPEN_STATE_FILE, default={})


def save_open_state(state: dict) -> None:
    save_json(OPEN_STATE_FILE, state)


def sync_open_state_with_broker():
    state = load_open_state()
    positions = get_positions_dict()
    for sym in list(state.keys()):
        if sym not in positions:
            state.pop(sym, None)

    for sym, pos in positions.items():
        if sym not in state:
            state[sym] = {
                "entry_time": datetime.utcnow().isoformat(),
                "entry_price": safe_float(pos.avg_entry_price, 0.0),
                "qty": safe_float(pos.qty, 0.0),
                "highest_price": safe_float(pos.current_price, safe_float(pos.avg_entry_price, 0.0)),
                "entry_score": None
            }

    save_open_state(state)
    return state


# -------------------------
# Candidate scoring
# -------------------------
def latest_feature_row(symbol: str) -> pd.Series | None:
    bars = get_bars(symbol, LOOKBACK_BARS)
    if bars.empty:
        return None
    feats = build_features(bars)
    if feats.empty:
        return None
    row = feats.iloc[-1].copy()

    dollar_volume = safe_float(row.get("dollar_volume_ma20", 0.0))
    close_px = safe_float(row.get("close", 0.0))
    if close_px < MIN_PRICE or close_px > MAX_PRICE:
        return None
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None
    return row


def score_candidates(model, symbols):
    rows = []
    for s in symbols:
        row = latest_feature_row(s)
        if row is None:
            continue
        X = pd.DataFrame([row[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)])
        try:
            prob = float(model.predict_proba(X)[0][1]) if model is not None else 0.5
        except Exception:
            prob = 0.5

        # Hard filters: avoid obvious low-quality setups
        if safe_float(row["adx_14"]) < 12:
            prob -= 0.015
        if safe_float(row["rel_volume_20"]) < 0.85:
            prob -= 0.015
        if safe_float(row["dist_resistance_20"]) < 0.004:
            prob -= 0.02
        if safe_float(row["atr14_pct"]) > 0.04:
            prob -= 0.02

        # Positive context boosts
        if safe_float(row["ema_slope_21"]) > 0:
            prob += 0.01
        if safe_float(row["session_vwap_dist"]) > 0:
            prob += 0.01
        if int(safe_float(row["cdl_bull_engulf"])) == 1 or int(safe_float(row["cdl_hammer"])) == 1:
            prob += 0.01
        if int(safe_float(row["breakout_20_up"])) == 1:
            prob += 0.01

        rows.append({
            "symbol": s,
            "prob": prob,
            "price": safe_float(row["close"]),
            "atr14_pct": safe_float(row["atr14_pct"]),
            "adx_14": safe_float(row["adx_14"]),
            "rel_volume_20": safe_float(row["rel_volume_20"]),
            "session_vwap_dist": safe_float(row["session_vwap_dist"]),
            "dist_resistance_20": safe_float(row["dist_resistance_20"]),
            "feature_row": row
        })

    if not rows:
        return pd.DataFrame()

    cands = pd.DataFrame(rows).sort_values("prob", ascending=False).head(MAX_CANDIDATES).reset_index(drop=True)
    return cands


# -------------------------
# Trade management
# -------------------------
def submit_market_buy(symbol: str, qty: int) -> bool:
    if qty <= 0:
        return False
    try:
        cancel_open_orders(symbol)
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        log(f"BUY submitted: {symbol} qty={qty}")
        return True
    except Exception as e:
        log(f"BUY failed {symbol}: {e}")
        return False


def submit_market_sell(symbol: str, qty: int) -> bool:
    if qty <= 0:
        return False
    try:
        cancel_open_orders(symbol)
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day"
        )
        log(f"SELL submitted: {symbol} qty={qty}")
        return True
    except Exception as e:
        log(f"SELL failed {symbol}: {e}")
        return False


def calc_sell_net_pnl_pct(entry_price: float, exit_price: float) -> float:
    gross = (exit_price / entry_price) - 1.0
    sec_fee_pct = SEC_FEE_RATE
    return gross - sec_fee_pct - TRADING_FEE_BUFFER


def update_trade_history_record(symbol: str, state_row: dict, exit_price: float, exit_reason: str) -> None:
    entry_price = safe_float(state_row.get("entry_price"), 0.0)
    qty = safe_float(state_row.get("qty"), 0.0)
    highest = safe_float(state_row.get("highest_price"), entry_price)
    entry_score = state_row.get("entry_score")
    entry_time = state_row.get("entry_time")

    if entry_price <= 0 or qty <= 0:
        return

    net_pnl_pct = calc_sell_net_pnl_pct(entry_price, exit_price)
    gross_pnl = (exit_price - entry_price) * qty
    sec_fee = exit_price * qty * SEC_FEE_RATE
    est_other_cost = exit_price * qty * TRADING_FEE_BUFFER

    hold_minutes = None
    try:
        et = pd.to_datetime(entry_time, utc=True)
        hold_minutes = (pd.Timestamp.utcnow().tz_localize("UTC") - et).total_seconds() / 60.0
    except Exception:
        pass

    row = pd.DataFrame([{
        "closed_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "highest_price": highest,
        "entry_score": entry_score,
        "gross_pnl": gross_pnl,
        "sec_fee": sec_fee,
        "est_other_cost": est_other_cost,
        "net_pnl_pct": net_pnl_pct,
        "hold_minutes": hold_minutes,
        "exit_reason": exit_reason
    }])

    append_df(TRADE_HISTORY_FILE, row, dedupe_cols=None, max_rows=MAX_TRADE_HISTORY_ROWS)


def manage_positions(model) -> None:
    state = sync_open_state_with_broker()
    positions = get_positions_dict()

    for symbol, pos in positions.items():
        current_price = safe_float(pos.current_price, safe_float(pos.avg_entry_price, 0.0))
        entry_price = safe_float(pos.avg_entry_price, 0.0)
        qty = int(float(pos.qty))

        if entry_price <= 0 or qty <= 0:
            continue

        s = state.get(symbol, {})
        s["qty"] = qty
        s["entry_price"] = entry_price
        s["highest_price"] = max(safe_float(s.get("highest_price"), entry_price), current_price)

        held_minutes = 0.0
        try:
            et = pd.to_datetime(s.get("entry_time"), utc=True)
            held_minutes = (pd.Timestamp.utcnow().tz_localize("UTC") - et).total_seconds() / 60.0
        except Exception:
            pass

        pnl_pct = current_price / entry_price - 1.0
        trail_stop_hit = False
        if s["highest_price"] >= entry_price * (1 + TRAIL_ACTIVATE_PCT):
            trail_floor = s["highest_price"] * (1 - TRAIL_GIVEBACK_PCT)
            trail_stop_hit = current_price <= trail_floor

        exit_reason = None
        if pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = "stop_loss"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = "take_profit"
        elif trail_stop_hit:
            exit_reason = "trailing_stop"
        elif held_minutes >= MAX_HOLD_MINUTES:
            exit_reason = "max_hold"

        # Score deterioration exit
        if exit_reason is None and model is not None:
            row = latest_feature_row(symbol)
            if row is not None:
                X = pd.DataFrame([row[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)])
                try:
                    prob = float(model.predict_proba(X)[0][1])
                    if prob < PROB_SELL_FLOOR:
                        exit_reason = "score_deterioration"
                except Exception:
                    pass

        state[symbol] = s

        if exit_reason:
            if submit_market_sell(symbol, qty):
                update_trade_history_record(symbol, s, current_price, exit_reason)
                state.pop(symbol, None)

    save_open_state(state)


def open_new_positions(model) -> None:
    if model is None:
        log("No model available yet; skipping entries.")
        return

    threshold = adaptive_buy_threshold()
    positions = get_positions_dict()
    current_symbols = set(positions.keys())

    open_slots = max(0, MAX_POSITIONS - len(current_symbols))
    if open_slots <= 0:
        log("No open slots available.")
        return

    cands = score_candidates(model, [s for s in UNIVERSE if s not in current_symbols])
    if cands.empty:
        log("No viable candidates.")
        return

    buys = cands[cands["prob"] >= threshold].head(open_slots).copy()
    if buys.empty:
        log(f"No candidates above threshold {threshold:.3f}")
        return

    equity = get_equity()
    cash = get_cash()
    usable_cash = max(0.0, cash - equity * CASH_RESERVE)
    per_position_budget = min(equity * MAX_ALLOC_PER_POSITION, usable_cash / max(len(buys), 1))

    if per_position_budget < MIN_ORDER_NOTIONAL:
        log("Not enough usable cash for new positions.")
        return

    state = load_open_state()
    for _, row in buys.iterrows():
        price = safe_float(row["price"], 0.0)
        if price <= 0:
            continue
        qty = int(per_position_budget // price)
        notional = qty * price
        if qty <= 0 or notional < MIN_ORDER_NOTIONAL:
            continue

        if submit_market_buy(row["symbol"], qty):
            state[row["symbol"]] = {
                "entry_time": datetime.utcnow().isoformat(),
                "entry_price": price,
                "qty": qty,
                "highest_price": price,
                "entry_score": float(row["prob"])
            }

    save_open_state(state)


# -------------------------
# Main
# -------------------------
def run_bot():
    log("=== Phase 9 bot start ===")

    if not market_is_open():
        log("Market is closed. Exiting cleanly.")
        return

    sync_open_state_with_broker()

    # Refresh compact rolling learning set
    new_rows = update_training_store(UNIVERSE)
    meta = load_model_meta()
    meta["new_rows_since_train"] = int(meta.get("new_rows_since_train", 0)) + int(new_rows)
    save_json(MODEL_META_FILE, meta)
    log(f"Training store updated with ~{new_rows} rows")

    model, meta = train_or_load_model(force=False)

    manage_positions(model)
    time.sleep(1)
    open_new_positions(model)

    rolling_trim_csv(TRAIN_DATA_FILE, MAX_TRAIN_ROWS)
    rolling_trim_csv(TRADE_HISTORY_FILE, MAX_TRADE_HISTORY_ROWS)

    log("=== Phase 9 bot end ===")


if __name__ == "__main__":
    run_bot()
