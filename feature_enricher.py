"""
Feature engineering for a 180-tick (≈15-minute) futures window.

- ORDERED_KEYS defines full feature set
- enrich_features() is used for both training (strict=False) and live prediction (strict=True)
"""

from statistics import mean, stdev
import numpy as np
import math
from datetime import datetime, timezone, timedelta, time as dtime

from block_feature_extractor import compute_block_features, BLOCK_FEATURE_KEYS
from helper import encode_zone_tag
from session_tracker import update_session_tracker_from_buffer, get_session_features, SESSION_FEATURE_KEYS


# ─── Microstructure helpers ───────────────────────────────
def _ltq_bias(seq):
    buys = sum(q for q in seq if q > 0)
    sells = sum(-q for q in seq if q < 0)
    tot = buys + sells or 1
    return (buys - sells) / tot

def _bounce_freq(seq):
    flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return flips / max(1, len(seq) - 1)

def _iceberg_score(depth_seq):
    suspicious = 0
    for (bb0, _), (bb1, _) in zip(depth_seq, depth_seq[1:]):
        if bb0 > 3 * max(1, bb1):
            suspicious += 1
    return suspicious / max(1, len(depth_seq) - 1)

def _is_opening_range_time(timestamp_ns: int) -> bool:
    """Check if tick is within opening range (9:15-9:20 AM IST) following existing pattern"""
    if timestamp_ns <= 0:
        return False
    
    try:
        ts_sec = timestamp_ns / 1e9
        dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        dt_ist = dt_utc + timedelta(hours=5, minutes=30)
        time_ist = dt_ist.time()
        
        return dtime(9, 15) <= time_ist <= dtime(9, 20)
    except (ValueError, OSError):
        return False

def _calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range as percentage of price"""
    if len(highs) < period + 1:
        return 0
    
    true_ranges = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    atr = mean(true_ranges[-period:])
    return (atr / max(closes[-1], 1e-6)) * 100  # As percentage

_clip = lambda x, lo, hi: max(lo, min(hi, x))

try:
    from feature_config import get_active_features
    ORDERED_KEYS = get_active_features()
    print(f"✅ Using {len(ORDERED_KEYS)} features from config")
except ImportError:
    # Final feature order for model training and inference
    ORDERED_KEYS = [
        "ltp_roc", "ltq_spike", "spread", "depth_imbalance",
        "bid_accumulating", "ask_vanish", "oi_change",
        "ltp_std", "vwap_deviation",
        "momentum_score", "roc_pct", "direction_streak",
        "ltp_slope", "oi_roc", "volume_zscore", "ltp_range",
        "agg_ratio", "range_position", "std_compression", "net_depth_trend",
        # time-aware
        "minutes_since_open", "tod_sin", "tod_cos",
        # behavioural microstructure features
        "ltq_bias_30", "bounce_freq_30", "iceberg_score_30",
        "flag_break_up30", "flag_break_dn30",
        "flag_trap_up30", "flag_trap_dn30",
        "inv_long_short", "inv_abs_ratio",
        # new regime detection features
        "atr_pct", "volatility_ratio",
        # new trap filter features
        "trend_conflict", "slope_90",
    ] + SESSION_FEATURE_KEYS + BLOCK_FEATURE_KEYS

    print(f"⚠️  Using original {len(ORDERED_KEYS)} features")


def _direction_streak(ltp_series):
    if len(ltp_series) < 2:
        return 0
    streak = 0
    sign = 1 if ltp_series[-1] > ltp_series[-2] else -1
    for i in range(2, len(ltp_series) + 1):
        if (ltp_series[-i + 1] - ltp_series[-i]) * sign > 0:
            streak += 1
        else:
            break
    return streak * sign

#tick_buffer: oldest to newest time order for sequence builder             
def enrich_features(tick_buffer: list, expected_len: int = 180, *, strict: bool = True, inv_snap=None, memory_zones=None, signed_buffer: list = None) -> dict:
    if not tick_buffer:
        raise ValueError("tick_buffer is empty")
    if len(tick_buffer) < expected_len:
        if strict:
            raise ValueError(f"tick_buffer size {len(tick_buffer)} < required {expected_len}")
        tick_buffer = (tick_buffer + [tick_buffer[-1]])[:expected_len]
    # Ensure signed_buffer matches tick_buffer length if provided
    if signed_buffer is None:
        # Create signed buffer from unsigned tick_buffer
        signed_buffer = tag_signed_ltq(tick_buffer.copy())

    if signed_buffer and len(signed_buffer) != len(tick_buffer):
        if strict:
            raise ValueError(f"signed_buffer size {len(signed_buffer)} != tick_buffer size {len(tick_buffer)}")
        # Truncate or pad signed_buffer to match
        signed_buffer = (signed_buffer + [signed_buffer[-1] if signed_buffer else {}])[:len(tick_buffer)]
        
    ltq_series_signed = [t["ltq"] for t in signed_buffer]
    first, last = tick_buffer[0], tick_buffer[-1] #last is the latest trade at prediction time, first is 15m back trade 
    ltp_series = [t["ltp"] for t in tick_buffer]
    ltq_series = [t["ltq"] for t in signed_buffer]
    current_ltp = last["ltp"]
    bid_qty0 = [t["bid_qty_0"] for t in tick_buffer]
    ask_qty0 = [t["ask_qty_0"] for t in tick_buffer]
    oi_series = [t["oi"] for t in tick_buffer]

    ltp_roc = (last["ltp"] - first["ltp"]) / max(first["ltp"], 1e-6)
    ltq_mean = mean(ltq_series)
    ltq_std = stdev(ltq_series) if len(set(ltq_series)) > 1 else 1e-6

    ltq_spike = int(last["ltq"] > ltq_mean + 2 * ltq_std)
    ltq_zscore = (last["ltq"] - ltq_mean) / ltq_std

    spread = _clip(last["ask_price_0"] - last["bid_price_0"], 0, 10)
    depth_imb = (last["bid_qty_0"] - last["ask_qty_0"]) / (last["bid_qty_0"] + last["ask_qty_0"] + 10)

    bid_accum = np.polyfit(range(len(bid_qty0)), bid_qty0, 1)[0]
    ask_vanish = -np.polyfit(range(len(ask_qty0)), ask_qty0, 1)[0]
    oi_change = last["oi"] - first["oi"]
    ltp_mean = mean(ltp_series)
    ltp_std = stdev(ltp_series) if len(set(ltp_series)) > 1 else 1e-6

    trimmed_ltqs = np.array(ltq_series)
    trimmed_ltps = np.array(ltp_series)
    if len(ltq_series) > 10:
        lower = int(0.05 * len(ltq_series))
        upper = int(0.95 * len(ltq_series))
        qty_window = np.abs(trimmed_ltqs[lower:upper])
        vwap = np.sum(trimmed_ltps[lower:upper] * qty_window) / max(np.sum(qty_window), 1)
    else:
        abs_ltqs = [abs(q) for q in ltq_series]
        vwap = sum(p * abs(q) for p, q in zip(ltp_series, ltq_series)) / max(sum(abs_ltqs), 1)

    vwap_dev = last["ltp"] - vwap
    ltp_range = max(ltp_series) - min(ltp_series)
    ltp_std_pct = ltp_std / max(ltp_mean, 1e-6)
    ltp_range_pct = ltp_range / max(ltp_std, 1e-6)
    spread_pct = spread / max(ltp_std, 1e-6)

    momentum = (last["ltp"] - ltp_mean) / (ltp_std or 1)
    roc_pct = ltp_roc * 100
    streak = _direction_streak(ltp_series)
    ltp_slope = np.polyfit(range(len(ltp_series)), ltp_series, 1)[0]
    oi_roc = _clip((last["oi"] - first["oi"]) / max(first["oi"], 1), -0.3, 0.3)
    volume_zscore = _clip((ltq_series[-1] - ltq_mean) / ltq_std, -4, 4)
    ltp_range = max(ltp_series) - min(ltp_series)

    ltq_bid = sum(abs(t["ltq"]) for t in signed_buffer if t["ltq"] > 0)
    ltq_ask = sum(abs(t["ltq"]) for t in signed_buffer if t["ltq"] < 0)
    agg_ratio = (ltq_bid - ltq_ask) / max(ltq_bid + ltq_ask, 1)

    range_pos = (last["ltp"] - min(ltp_series)) / max(ltp_range, 1e-6)

    std_now = np.std(ltp_series[-30:]) if len(set(ltp_series[-30:])) > 1 else 1e-6
    std_past = np.std(ltp_series[:30]) if len(set(ltp_series[:30])) > 1 else 1e-6
    std_compression = _clip(std_now / max(std_past, 1e-6), 0.01, 10)

    net_depth_trend = bid_accum - ask_vanish

    ts_ns = last.get("ts", 0)
    ist_ts_ns = ts_ns + (5 * 3600 + 30 * 60) * 1_000_000_000  # Add 5.5 hours
    sec_of_day = (ist_ts_ns // 1_000_000_000) % 86_400    
    minutes_since_open = ((sec_of_day - (9 * 3600 + 15 * 60)) / 60.0) if sec_of_day >= (9*3600 + 15*60) else -1
    tod_frac = sec_of_day / 86_400.0
    tod_sin  = math.sin(2 * math.pi * tod_frac)
    tod_cos  = math.cos(2 * math.pi * tod_frac)

    side_series = [np.sign(q) for q in ltq_series]
    depth_seq = [(t["bid_qty_0"], t["ask_qty_0"]) for t in tick_buffer]

    ltq_bias_30 = _ltq_bias(ltq_series_signed[-30:])
    bounce_freq_30 = _bounce_freq(side_series[-30:])
    iceberg_score_30 = _iceberg_score(depth_seq[-30:])

    # WITH THESE:
    if len(ltp_series) >= 60:
        hi30 = max(ltp_series[-60:-5])
        lo30 = min(ltp_series[-60:-5])
        
        if hi30 - lo30 >= 5:  # Minimum meaningful range
            flag_break_up30 = int(ltp_series[-1] > hi30 + 2)
            flag_break_dn30 = int(ltp_series[-1] < lo30 - 2)
        else:
            flag_break_up30 = 0
            flag_break_dn30 = 0
    else:
        hi30 = lo30 = ltp_series[-1]
        flag_break_up30 = flag_break_dn30 = 0

    # WITH THESE:
    if len(ltp_series) >= 60 and hi30 - lo30 >= 5:
        flag_trap_up30 = int(len(ltp_series) >= 3 and 
                            ltp_series[-3] > hi30 + 2 and 
                            ltp_series[-1] <= hi30 - 1)
        flag_trap_dn30 = int(len(ltp_series) >= 3 and 
                            ltp_series[-3] < lo30 - 2 and 
                            ltp_series[-1] >= lo30 + 1)
    else:
        flag_trap_up30 = flag_trap_dn30 = 0
        
    inv_long_short = getattr(inv_snap, "inv_long_short", 0)
    inv_abs = getattr(inv_snap, "inv_abs", 1)
    ltq_recent_mean = np.mean([abs(q) for q in ltq_series[-30:] if abs(q) > 0]) or 1
    inv_abs_ratio = math.log1p(inv_abs / ltq_recent_mean)

    # ─── NEW REGIME DETECTION FEATURES ───────────────────────
    # Create 1-minute OHLC bars from 5-second ticks (12 ticks = 1 minute)
    minute_bars = []
    for i in range(12, len(ltp_series) + 1, 12):  # Every 12 ticks = 1 minute
        window = ltp_series[i-12:i]
        minute_bars.append({
            'high': max(window),
            'low': min(window), 
            'close': window[-1]
        })
    
    # Calculate ATR on 1-minute bars (14-period ATR ≈ 14 minutes)
    if len(minute_bars) >= 15:
        highs = [bar['high'] for bar in minute_bars]
        lows = [bar['low'] for bar in minute_bars]
        closes = [bar['close'] for bar in minute_bars]
        atr_pct = _calculate_atr(highs, lows, closes, period=14)
    else:
        atr_pct = 0
    
    # Volatility regime: recent 5 minutes vs earlier 10 minutes
    recent_period = 60  # 5 minutes = 60 ticks
    historical_period = 120  # 10 minutes = 120 ticks
    
    if len(ltp_series) >= recent_period + historical_period:
        recent_data = ltp_series[-recent_period:]
        historical_data = ltp_series[-(recent_period + historical_period):-recent_period]
        
        recent_vol = np.std(recent_data) if len(set(recent_data)) > 1 else 1e-6
        historical_vol = np.std(historical_data) if len(set(historical_data)) > 1 else 1e-6
        
        volatility_ratio = _clip(recent_vol / historical_vol, 0.1, 5.0)
    else:
        volatility_ratio = 1.0


    # ─── NEW TRAP FILTER FEATURES ─────────────────────────────
    # Trend timeframes adjusted for 5-second ticks:
    # Short-term: 2 minutes = 24 ticks
    # Long-term: 7.5 minutes = 90 ticks (half the window)
    short_period = 24  # 2 minutes
    long_period = 90   # 7.5 minutes
    
    slope_24 = np.polyfit(range(short_period), ltp_series[-short_period:], 1)[0] if len(ltp_series) >= short_period else 0
    slope_90 = np.polyfit(range(long_period), ltp_series[-long_period:], 1)[0] if len(ltp_series) >= long_period else 0
    
    # Trend conflict: when short and long term trends disagree
    trend_conflict = 0
    if abs(slope_24) > 1e-6 and abs(slope_90) > 1e-6:
        # Check if slopes have opposite signs or significantly different magnitudes
        sign_conflict = (slope_24 > 0) != (slope_90 > 0)
        magnitude_conflict = abs(slope_24 / slope_90) > 2.5 if abs(slope_90) > 1e-6 else False
        trend_conflict = int(sign_conflict or magnitude_conflict)

    features = {
        "ltp_roc": ltp_roc,
        "ltq_spike": ltq_spike,
        "spread": spread_pct,
        "depth_imbalance": depth_imb,
        "bid_accumulating": bid_accum,
        "ask_vanish": ask_vanish,
        "oi_change": oi_change,
        "ltp_std": ltp_std_pct,
        #"vwap": vwap,
        "vwap_deviation": vwap_dev,
        "momentum_score": momentum,
        "roc_pct": roc_pct,
        "direction_streak": streak,
        "ltp_slope": ltp_slope,
        "oi_roc": oi_roc,
        "volume_zscore": volume_zscore,
        "ltp_range": ltp_range_pct,
        "agg_ratio": agg_ratio,
        "range_position": range_pos,
        "std_compression": std_compression,
        "net_depth_trend": net_depth_trend,
        "minutes_since_open": minutes_since_open,
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "ltq_bias_30": ltq_bias_30,
        "bounce_freq_30": bounce_freq_30,
        "iceberg_score_30": iceberg_score_30,
        "flag_break_up30": flag_break_up30,
        "flag_break_dn30": flag_break_dn30,
        "flag_trap_up30": flag_trap_up30,
        "flag_trap_dn30": flag_trap_dn30,
        "inv_long_short": inv_long_short,
        "inv_abs_ratio": inv_abs_ratio,
        "atr_pct": atr_pct,
        "volatility_ratio": volatility_ratio,
        "trend_conflict": trend_conflict,
        "slope_90": slope_90
    }
    # ✅ SESSION FEATURES - Following same pattern as block features
    try:
        # Calculate opening range from ticks in 9:15-9:20 AM IST window
        opening_range = None
        if len(tick_buffer) >= 10:
            or_ticks = []
            for tick in tick_buffer:
                tick_ts = tick.get("ts", 0)
                if tick_ts > 0 and _is_opening_range_time(tick_ts):
                    ltp = tick.get("ltp", 0)
                    if ltp > 0:
                        or_ticks.append(ltp)
            
            # Calculate OR if we have meaningful opening range data
            if len(or_ticks) >= 5:  # Minimum viable OR sample
                opening_range = {
                    'or_high': max(or_ticks),
                    'or_low': min(or_ticks),
                    'or_range': max(or_ticks) - min(or_ticks),
                    'or_completed': True
                }
        
        # Update session tracker and get features
        update_session_tracker_from_buffer(tick_buffer, opening_range=opening_range)
        session_features = get_session_features()
        features.update(session_features)
        
    except Exception as e:
        # Graceful degradation - add zero session features if calculation fails
        for key in SESSION_FEATURE_KEYS:
            features[key] = 0.0
        if not strict:  # Only log in non-strict mode to avoid spam
            print(f"⚠️ Session features failed: {e}")

    # Add zone dummies to features
    block_features = compute_block_features(tick_buffer, signed_buffer=signed_buffer)
    features.update(block_features)

    return {k: features[k] for k in ORDERED_KEYS}

