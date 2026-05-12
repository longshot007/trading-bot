"""
candlestick_rules.py
====================
Candlestick signal detection and trading rules encoded from:

  [1] "The Candlestick Trading Bible" - Munehisa Homma / modern interpretation
  [2] "Profitable Candlestick Trading" - Stephen Bigalow (Wiley)

This file is STANDALONE and SAFE — it imports only pandas and numpy,
defines pure functions that accept a DataFrame of OHLCV bars, and returns
signal names or boolean values. It does not connect to Alpaca, does not
place orders, and does not modify any state.

DataFrame contract (same as bot_phase97.py):
  Columns: open, high, low, close, volume
  Index:   datetime
  Typical timeframe: 1Min bars, but logic works on any timeframe

Usage:
  from candlestick_rules import detect_signals, market_structure, confluence_score

  df = <your OHLCV dataframe>
  signals = detect_signals(df)
  structure = market_structure(df)
  score = confluence_score(df)

ML usage:
  Each function returns a value that can be used as a feature vector column.
  Run detect_signals(df) on every bar window and store results alongside
  trade outcomes to build a labeled training dataset.
"""

import numpy as np
import pandas as pd


def body_size(o, c):
    return abs(c - o)

def upper_shadow(o, h, c):
    return h - max(o, c)

def lower_shadow(o, l, c):
    return min(o, c) - l

def is_bullish(o, c):
    return c > o

def is_bearish(o, c):
    return c < o

def body_range_pct(o, h, l, c):
    rng = h - l
    if rng == 0:
        return 0
    return body_size(o, c) / rng

def candle_midpoint(o, c):
    return (o + c) / 2

def stochastics(df, k_period=14, d_period=3):
    df = df.copy()
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    rng = high_max - low_min
    df["stoch_k"] = np.where(rng == 0, 50, (df["close"] - low_min) / rng * 100)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df

def is_oversold(df, threshold=20):
    if "stoch_k" not in df.columns:
        df = stochastics(df)
    return float(df["stoch_k"].iloc[-1]) < threshold

def is_overbought(df, threshold=80):
    if "stoch_k" not in df.columns:
        df = stochastics(df)
    return float(df["stoch_k"].iloc[-1]) > threshold

def stoch_turning_up(df):
    if "stoch_k" not in df.columns:
        df = stochastics(df)
    if len(df) < 2:
        return False
    return (df["stoch_k"].iloc[-2] <= df["stoch_d"].iloc[-2]) and (df["stoch_k"].iloc[-1] > df["stoch_d"].iloc[-1])

def stoch_turning_down(df):
    if "stoch_k" not in df.columns:
        df = stochastics(df)
    if len(df) < 2:
        return False
    return (df["stoch_k"].iloc[-2] >= df["stoch_d"].iloc[-2]) and (df["stoch_k"].iloc[-1] < df["stoch_d"].iloc[-1])

def add_moving_averages(df):
    df = df.copy()
    df["ema8"]   = df["close"].ewm(span=8, adjust=False).mean()
    df["sma21"]  = df["close"].rolling(21).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    return df

def above_200_sma(df):
    if "sma200" not in df.columns:
        df = add_moving_averages(df)
    sma200 = df["sma200"].iloc[-1]
    if pd.isna(sma200):
        return False
    return float(df["close"].iloc[-1]) > sma200

def near_21_sma(df, tolerance_pct=0.003):
    if "sma21" not in df.columns:
        df = add_moving_averages(df)
    sma21 = df["sma21"].iloc[-1]
    if pd.isna(sma21):
        return False
    close = float(df["close"].iloc[-1])
    return abs(close - sma21) / sma21 < tolerance_pct

def market_structure(df, lookback=20):
    if len(df) < lookback:
        return "choppy"
    window = df.tail(lookback)
    highs  = window["high"].values
    lows   = window["low"].values
    mid = lookback // 2
    first_high  = highs[:mid].mean()
    second_high = highs[mid:].mean()
    first_low   = lows[:mid].mean()
    second_low  = lows[mid:].mean()
    higher_highs = second_high > first_high
    higher_lows  = second_low  > first_low
    lower_highs  = second_high < first_high
    lower_lows   = second_low  < first_low
    price_range = highs.max() - lows.min()
    avg_atr = (window["high"] - window["low"]).mean()
    if avg_atr == 0:
        return "choppy"
    range_ratio = price_range / avg_atr
    if higher_highs and higher_lows:
        return "uptrend"
    elif lower_highs and lower_lows:
        return "downtrend"
    elif range_ratio < 3.0:
        return "choppy"
    else:
        return "ranging"

def find_support_resistance(df, lookback=50, tolerance_pct=0.005):
    if len(df) < lookback:
        return [], []
    window = df.tail(lookback)
    highs  = window["high"].values
    lows   = window["low"].values
    resistance_levels = []
    support_levels    = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            resistance_levels.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            support_levels.append(lows[i])
    return support_levels, resistance_levels

def near_support(df, tolerance_pct=0.005):
    supports, _ = find_support_resistance(df)
    if not supports:
        return False
    close = float(df["close"].iloc[-1])
    return any(abs(close - s) / s < tolerance_pct for s in supports)

def near_resistance(df, tolerance_pct=0.005):
    _, resistances = find_support_resistance(df)
    if not resistances:
        return False
    close = float(df["close"].iloc[-1])
    return any(abs(close - r) / r < tolerance_pct for r in resistances)

def fibonacci_levels(df, lookback=50):
    if len(df) < lookback:
        return {}
    window    = df.tail(lookback)
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())
    diff       = swing_high - swing_low
    if diff == 0:
        return {}
    levels = {}
    for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
        levels[ratio] = swing_high - (diff * ratio)
    return levels

def near_fibonacci(df, ratios=(0.5, 0.618), tolerance_pct=0.005):
    levels = fibonacci_levels(df)
    if not levels:
        return False
    close = float(df["close"].iloc[-1])
    for ratio in ratios:
        if ratio in levels:
            lvl = levels[ratio]
            if abs(close - lvl) / lvl < tolerance_pct:
                return True
    return False

def volume_confirms(df, multiplier=1.2):
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return False
    return float(df["volume"].iloc[-1]) > avg_vol * multiplier

def is_doji(o, h, l, c, body_threshold=0.05):
    rng = h - l
    if rng == 0:
        return False
    return body_size(o, c) / rng < body_threshold

def is_hammer(o, h, l, c, shadow_multiplier=2.0):
    body   = body_size(o, c)
    lo_shd = lower_shadow(o, l, c)
    up_shd = upper_shadow(o, h, c)
    if body == 0:
        return False
    long_lower  = lo_shd >= shadow_multiplier * body
    small_upper = up_shd <= body * 0.3
    body_at_top = min(o, c) > l + lo_shd * 0.5
    return long_lower and small_upper and body_at_top

def is_shooting_star(o, h, l, c, shadow_multiplier=2.0):
    body   = body_size(o, c)
    up_shd = upper_shadow(o, h, c)
    lo_shd = lower_shadow(o, l, c)
    if body == 0:
        return False
    long_upper  = up_shd >= shadow_multiplier * body
    small_lower = lo_shd <= body * 0.3
    return long_upper and small_lower

def is_hanging_man(o, h, l, c, shadow_multiplier=2.0):
    return is_hammer(o, h, l, c, shadow_multiplier)

def is_inverted_hammer(o, h, l, c, shadow_multiplier=2.0):
    return is_shooting_star(o, h, l, c, shadow_multiplier)

def is_marubozu_bullish(o, h, l, c, shadow_threshold=0.05):
    rng = h - l
    if rng == 0:
        return False
    no_lower = lower_shadow(o, l, c) / rng < shadow_threshold
    no_upper = upper_shadow(o, h, c) / rng < shadow_threshold
    return is_bullish(o, c) and no_lower and no_upper

def is_marubozu_bearish(o, h, l, c, shadow_threshold=0.05):
    rng = h - l
    if rng == 0:
        return False
    no_lower = lower_shadow(o, l, c) / rng < shadow_threshold
    no_upper = upper_shadow(o, h, c) / rng < shadow_threshold
    return is_bearish(o, c) and no_lower and no_upper

def is_spinning_top(o, h, l, c, body_threshold=0.3):
    rng = h - l
    if rng == 0:
        return False
    small_body  = body_size(o, c) / rng < body_threshold
    has_shadows = (upper_shadow(o, h, c) > 0) and (lower_shadow(o, l, c) > 0)
    return small_body and has_shadows

def is_long_day(o, h, l, c, body_threshold=0.6):
    rng = h - l
    if rng == 0:
        return False
    return body_size(o, c) / rng > body_threshold

def is_bullish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    first_bearish    = is_bearish(o1, c1)
    second_bullish   = is_bullish(o2, c2)
    body1            = body_size(o1, c1)
    body2            = body_size(o2, c2)
    engulfs          = o2 <= c1 and c2 >= o1
    larger           = body2 > body1
    return first_bearish and second_bullish and engulfs and larger

def is_bearish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    first_bullish  = is_bullish(o1, c1)
    second_bearish = is_bearish(o2, c2)
    body1          = body_size(o1, c1)
    body2          = body_size(o2, c2)
    engulfs        = o2 >= c1 and c2 <= o1
    larger         = body2 > body1
    return first_bullish and second_bearish and engulfs and larger

def is_bullish_harami(o1, h1, l1, c1, o2, h2, l2, c2):
    first_large_bearish = is_bearish(o1, c1) and is_long_day(o1, h1, l1, c1)
    second_small        = is_bullish(o2, c2)
    contained           = c2 < o1 and o2 > c1
    return first_large_bearish and second_small and contained

def is_bearish_harami(o1, h1, l1, c1, o2, h2, l2, c2):
    first_large_bullish = is_bullish(o1, c1) and is_long_day(o1, h1, l1, c1)
    second_small        = is_bearish(o2, c2)
    contained           = o2 < c1 and c2 > o1
    return first_large_bullish and second_small and contained

def is_piercing_pattern(o1, h1, l1, c1, o2, h2, l2, c2):
    first_bearish   = is_bearish(o1, c1)
    second_bullish  = is_bullish(o2, c2)
    opens_lower     = o2 < l1
    closes_midway   = c2 > candle_midpoint(o1, c1)
    not_full_engulf = c2 < o1
    return first_bearish and second_bullish and opens_lower and closes_midway and not_full_engulf

def is_dark_cloud_cover(o1, h1, l1, c1, o2, h2, l2, c2):
    first_bullish   = is_bullish(o1, c1)
    second_bearish  = is_bearish(o2, c2)
    opens_higher    = o2 > h1
    closes_midway   = c2 < candle_midpoint(o1, c1)
    not_full_engulf = c2 > o1
    return first_bullish and second_bearish and opens_higher and closes_midway and not_full_engulf

def is_kicker_bullish(o1, c1, o2, c2, gap_threshold=0.001):
    first_bearish  = is_bearish(o1, c1)
    second_bullish = is_bullish(o2, c2)
    gap_up         = o2 >= o1 * (1 - gap_threshold)
    strong_move    = c2 > o1
    return first_bearish and second_bullish and gap_up and strong_move

def is_kicker_bearish(o1, c1, o2, c2, gap_threshold=0.001):
    first_bullish  = is_bullish(o1, c1)
    second_bearish = is_bearish(o2, c2)
    gap_down       = o2 <= o1 * (1 + gap_threshold)
    strong_move    = c2 < o1
    return first_bullish and second_bearish and gap_down and strong_move

def is_tweezer_bottom(o1, h1, l1, c1, o2, h2, l2, c2, tolerance_pct=0.001):
    matching_lows = abs(l1 - l2) / l1 < tolerance_pct if l1 != 0 else False
    return is_bearish(o1, c1) and is_bullish(o2, c2) and matching_lows

def is_tweezer_top(o1, h1, l1, c1, o2, h2, l2, c2, tolerance_pct=0.001):
    matching_highs = abs(h1 - h2) / h1 < tolerance_pct if h1 != 0 else False
    return is_bullish(o1, c1) and is_bearish(o2, c2) and matching_highs

def is_meeting_lines_bullish(o1, c1, o2, c2, tolerance_pct=0.001):
    matching_close = abs(c1 - c2) / c1 < tolerance_pct if c1 != 0 else False
    return is_bearish(o1, c1) and is_bullish(o2, c2) and matching_close

def is_morning_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3, midpoint_threshold=0.5):
    first_bearish     = is_bearish(o1, c1) and is_long_day(o1, h1, l1, c1)
    middle_small      = body_size(o2, c2) < body_size(o1, c1) * 0.5
    third_bullish     = is_bullish(o3, c3)
    closes_midway     = c3 >= candle_midpoint(o1, c1)
    gap_after_first   = max(o2, c2) < c1
    return first_bearish and middle_small and third_bullish and closes_midway and gap_after_first

def is_evening_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3, midpoint_threshold=0.5):
    first_bullish   = is_bullish(o1, c1) and is_long_day(o1, h1, l1, c1)
    middle_small    = body_size(o2, c2) < body_size(o1, c1) * 0.5
    third_bearish   = is_bearish(o3, c3)
    closes_midway   = c3 <= candle_midpoint(o1, c1)
    gap_after_first = min(o2, c2) > c1
    return first_bullish and middle_small and third_bearish and closes_midway and gap_after_first

def is_three_white_soldiers(o1, c1, o2, c2, o3, c3):
    all_bullish     = is_bullish(o1, c1) and is_bullish(o2, c2) and is_bullish(o3, c3)
    progressive     = c1 < c2 < c3
    opens_in_body2  = o2 > o1 and o2 < c1
    opens_in_body3  = o3 > o2 and o3 < c2
    return all_bullish and progressive and opens_in_body2 and opens_in_body3

def is_three_black_crows(o1, c1, o2, c2, o3, c3):
    all_bearish     = is_bearish(o1, c1) and is_bearish(o2, c2) and is_bearish(o3, c3)
    progressive     = c1 > c2 > c3
    opens_in_body2  = o2 < o1 and o2 > c1
    opens_in_body3  = o3 < o2 and o3 > c2
    return all_bearish and progressive and opens_in_body2 and opens_in_body3

def is_three_inside_up(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
    harami   = is_bullish_harami(o1, h1, l1, c1, o2, h2, l2, c2)
    confirm  = is_bullish(o3, c3) and c3 > o1
    return harami and confirm

def is_three_inside_down(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
    harami   = is_bearish_harami(o1, h1, l1, c1, o2, h2, l2, c2)
    confirm  = is_bearish(o3, c3) and c3 < o1
    return harami and confirm

def is_inside_bar(o1, h1, l1, c1, o2, h2, l2, c2):
    return h2 < h1 and l2 > l1

def is_inside_bar_false_breakout_bullish(h1, l1, h2, l2, o3, h3, l3, c3):
    broke_below   = l3 < l2
    reversed_up   = c3 > h2
    return broke_below and reversed_up

def is_inside_bar_false_breakout_bearish(h1, l1, h2, l2, o3, h3, l3, c3):
    broke_above  = h3 > h2
    reversed_down = c3 < l2
    return broke_above and reversed_down

def gap_up(prev_h, curr_l):
    return curr_l > prev_h

def gap_down(prev_l, curr_h):
    return curr_h < prev_l

def gap_size_pct(prev_c, curr_o):
    if prev_c == 0:
        return 0
    return abs(curr_o - prev_c) / prev_c

def stop_loss_hammer(l, atr_buffer=0.0):
    return l - atr_buffer

def stop_loss_engulfing(o2, atr_buffer=0.0):
    return o2 - atr_buffer

def stop_loss_morning_star(l2, atr_buffer=0.0):
    return l2 - atr_buffer

def fifty_percent_exit_triggered(entry_o, entry_c, current_price):
    midpoint = candle_midpoint(entry_o, entry_c)
    if is_bullish(entry_o, entry_c):
        return current_price <= midpoint
    return False


def detect_signals(df, min_bars=5):
    if len(df) < min_bars:
        return {}
    signals = {}
    bars = df.tail(4)
    rows = [bars.iloc[i] for i in range(len(bars))]
    def vals(row):
        return float(row.open), float(row.high), float(row.low), float(row.close)
    o0, h0, l0, c0 = vals(rows[-1])
    if is_doji(o0, h0, l0, c0):
        signals["DOJI"] = {"direction": "neutral", "confidence": 2}
    if is_marubozu_bullish(o0, h0, l0, c0):
        signals["MARUBOZU_BULL"] = {"direction": "bull", "confidence": 2}
    if is_marubozu_bearish(o0, h0, l0, c0):
        signals["MARUBOZU_BEAR"] = {"direction": "bear", "confidence": 2}
    if len(rows) >= 2:
        o1, h1, l1, c1 = vals(rows[-2])
        if is_hammer(o0, h0, l0, c0):
            signals["HAMMER"] = {"direction": "bull", "confidence": 1}
        if is_shooting_star(o0, h0, l0, c0):
            signals["SHOOTING_STAR"] = {"direction": "bear", "confidence": 1}
        if is_bullish_engulfing(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["BULLISH_ENGULFING"] = {"direction": "bull", "confidence": 2}
        if is_bearish_engulfing(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["BEARISH_ENGULFING"] = {"direction": "bear", "confidence": 2}
        if is_bullish_harami(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["BULLISH_HARAMI"] = {"direction": "bull", "confidence": 1}
        if is_bearish_harami(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["BEARISH_HARAMI"] = {"direction": "bear", "confidence": 1}
        if is_kicker_bullish(o1, c1, o0, c0):
            signals["KICKER_BULL"] = {"direction": "bull", "confidence": 3}
        if is_kicker_bearish(o1, c1, o0, c0):
            signals["KICKER_BEAR"] = {"direction": "bear", "confidence": 3}
        if is_piercing_pattern(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["PIERCING"] = {"direction": "bull", "confidence": 2}
        if is_dark_cloud_cover(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["DARK_CLOUD"] = {"direction": "bear", "confidence": 2}
        if is_tweezer_bottom(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["TWEEZER_BOTTOM"] = {"direction": "bull", "confidence": 1}
        if is_tweezer_top(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["TWEEZER_TOP"] = {"direction": "bear", "confidence": 1}
        if is_inside_bar(o1, h1, l1, c1, o0, h0, l0, c0):
            signals["INSIDE_BAR"] = {"direction": "neutral", "confidence": 1}
    if len(rows) >= 3:
        o1, h1, l1, c1 = vals(rows[-2])
        o2, h2, l2, c2 = vals(rows[-3])
        if is_morning_star(o2, h2, l2, c2, o1, h1, l1, c1, o0, h0, l0, c0):
            signals["MORNING_STAR"] = {"direction": "bull", "confidence": 2}
        if is_evening_star(o2, h2, l2, c2, o1, h1, l1, c1, o0, h0, l0, c0):
            signals["EVENING_STAR"] = {"direction": "bear", "confidence": 2}
        if is_three_white_soldiers(o2, c2, o1, c1, o0, c0):
            signals["THREE_WHITE_SOLDIERS"] = {"direction": "bull", "confidence": 3}
        if is_three_black_crows(o2, c2, o1, c1, o0, c0):
            signals["THREE_BLACK_CROWS"] = {"direction": "bear", "confidence": 3}
        if is_three_inside_up(o2, h2, l2, c2, o1, h1, l1, c1, o0, h0, l0, c0):
            signals["THREE_INSIDE_UP"] = {"direction": "bull", "confidence": 2}
        if is_three_inside_down(o2, h2, l2, c2, o1, h1, l1, c1, o0, h0, l0, c0):
            signals["THREE_INSIDE_DOWN"] = {"direction": "bear", "confidence": 2}
    if len(rows) >= 4:
        o1, h1, l1, c1 = vals(rows[-2])
        o2, h2, l2, c2 = vals(rows[-3])
        if (is_inside_bar(o2, h2, l2, c2, o1, h1, l1, c1) and
                is_inside_bar_false_breakout_bullish(h2, l2, h1, l1, o0, h0, l0, c0)):
            signals["INSIDE_BAR_FALSE_BREAK_BULL"] = {"direction": "bull", "confidence": 3}
        if (is_inside_bar(o2, h2, l2, c2, o1, h1, l1, c1) and
                is_inside_bar_false_breakout_bearish(h2, l2, h1, l1, o0, h0, l0, c0)):
            signals["INSIDE_BAR_FALSE_BREAK_BEAR"] = {"direction": "bear", "confidence": 3}
    return signals


def confluence_score(df, direction="bull"):
    score = 0
    details = []
    df = stochastics(df)
    df = add_moving_averages(df)
    structure = market_structure(df)
    if direction == "bull" and structure == "uptrend":
        score += 1
        details.append("uptrend")
    elif direction == "bear" and structure == "downtrend":
        score += 1
        details.append("downtrend")
    elif structure == "ranging":
        score += 1
        details.append("ranging_boundary")
    elif structure == "choppy":
        return 0, ["CHOPPY_MARKET_NO_TRADE"]
    if direction == "bull" and is_oversold(df):
        score += 1
        details.append("stoch_oversold")
    elif direction == "bear" and is_overbought(df):
        score += 1
        details.append("stoch_overbought")
    if direction == "bull" and stoch_turning_up(df):
        score += 1
        details.append("stoch_turning_up")
    elif direction == "bear" and stoch_turning_down(df):
        score += 1
        details.append("stoch_turning_down")
    if direction == "bull" and near_support(df):
        score += 1
        details.append("near_support")
    elif direction == "bear" and near_resistance(df):
        score += 1
        details.append("near_resistance")
    if near_21_sma(df):
        score += 1
        details.append("near_21_sma")
    if near_fibonacci(df):
        score += 1
        details.append("near_fibonacci_50_618")
    if volume_confirms(df):
        score += 1
        details.append("volume_confirmed")
    return score, details


class CandlestickRules:
    MIN_CONFLUENCE_SCORE = 3
    MIN_SIGNAL_CONFIDENCE = 1

    @staticmethod
    def check_entry(df, signals, direction="bull"):
        structure = market_structure(df)
        if structure == "choppy":
            return False, "CHOPPY_MARKET"
        if not signals:
            return False, "NO_SIGNAL"
        directional = {k: v for k, v in signals.items()
                       if v["direction"] == direction or v["direction"] == "neutral"}
        if not directional:
            return False, "NO_DIRECTIONAL_SIGNAL"
        best_confidence = max(v["confidence"] for v in directional.values())
        if best_confidence < CandlestickRules.MIN_SIGNAL_CONFIDENCE:
            return False, "SIGNAL_CONFIDENCE_TOO_LOW"
        score, details = confluence_score(df, direction)
        if score < CandlestickRules.MIN_CONFLUENCE_SCORE:
            return False, f"CONFLUENCE_TOO_LOW_{score}"
        if "DOJI" in signals and is_overbought(df):
            return False, "DOJI_AT_TOP_OVERBOUGHT"
        df_ma = add_moving_averages(df)
        sma21 = df_ma["sma21"].iloc[-1]
        close = float(df["close"].iloc[-1])
        if not pd.isna(sma21) and direction == "bull":
            extension = (close - sma21) / sma21
            if extension > 0.05:
                return False, "PRICE_TOO_EXTENDED_FROM_21SMA"
        return True, f"VALID_ENTRY_SCORE_{score}_SIGNALS_{list(directional.keys())}"

    @staticmethod
    def check_exit(df, entry_open, entry_close, entry_low, current_price, entry_direction="bull"):
        if entry_direction == "bull" and is_overbought(df):
            o = float(df["open"].iloc[-1])
            h = float(df["high"].iloc[-1])
            l = float(df["low"].iloc[-1])
            c = float(df["close"].iloc[-1])
            if is_doji(o, h, l, c):
                return True, "DOJI_AT_TOP_EXIT"
        if fifty_percent_exit_triggered(entry_open, entry_close, current_price):
            return True, "FIFTY_PCT_PULLBACK_EXIT"
        if entry_direction == "bull" and len(df) >= 2:
            o1 = float(df["open"].iloc[-2]); h1 = float(df["high"].iloc[-2])
            l1 = float(df["low"].iloc[-2]); c1 = float(df["close"].iloc[-2])
            o2 = float(df["open"].iloc[-1]); h2 = float(df["high"].iloc[-1])
            l2 = float(df["low"].iloc[-1]); c2 = float(df["close"].iloc[-1])
            if is_bearish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
                return True, "BEARISH_ENGULFING_EXIT"
            if is_kicker_bearish(o1, c1, o2, c2):
                return True, "KICKER_REVERSAL_EXIT"
        return False, "HOLD"

    @staticmethod
    def position_size_factor(score):
        if score <= 2:
            return 0.5
        elif score == 3:
            return 0.7
        elif score == 4:
            return 0.85
        else:
            return 1.0


def extract_ml_features(df):
    if len(df) < 5:
        return {}
    df = stochastics(df)
    df = add_moving_averages(df)
    signals  = detect_signals(df)
    structure = market_structure(df)
    bull_score, _ = confluence_score(df, "bull")
    bear_score, _ = confluence_score(df, "bear")
    o = float(df["open"].iloc[-1]); h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1]); c = float(df["close"].iloc[-1])
    features = {
        "body_pct_of_range":    body_range_pct(o, h, l, c),
        "upper_shadow_pct":     upper_shadow(o, h, c) / (h - l) if (h - l) > 0 else 0,
        "lower_shadow_pct":     lower_shadow(o, l, c) / (h - l) if (h - l) > 0 else 0,
        "is_bullish_candle":    int(is_bullish(o, c)),
        "market_structure":     structure,
        "is_uptrend":           int(structure == "uptrend"),
        "is_downtrend":         int(structure == "downtrend"),
        "is_ranging":           int(structure == "ranging"),
        "is_choppy":            int(structure == "choppy"),
        "stoch_k":              float(df["stoch_k"].iloc[-1]),
        "stoch_d":              float(df["stoch_d"].iloc[-1]),
        "is_oversold":          int(is_oversold(df)),
        "is_overbought":        int(is_overbought(df)),
        "near_support":         int(near_support(df)),
        "near_resistance":      int(near_resistance(df)),
        "near_21_sma":          int(near_21_sma(df)),
        "near_fibonacci":       int(near_fibonacci(df)),
        "above_200_sma":        int(above_200_sma(df)),
        "volume_confirms":      int(volume_confirms(df)),
        "bull_confluence_score": bull_score,
        "bear_confluence_score": bear_score,
        "detected_signals":     list(signals.keys()),
        "close":   c,
        "volume":  float(df["volume"].iloc[-1]),
    }
    return features
