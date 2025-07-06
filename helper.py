# helper.py - Enhanced with market intelligence display only

from datetime import datetime, time as dtime, timedelta, timezone
from utils import log_text
import asyncio

def is_within_capture_window():
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    current_time = now_ist.time()
    return dtime(9, 15) <= current_time <= dtime(15, 30)

def is_valid_180_window(window):
    """Check if 180 synthetic ticks span reasonable time (≤20 minutes)"""
    if len(window) < 180:
        return False
    
    timestamps = [tick.get("ts_batch", tick.get("ts", 0)) for tick in window]
    valid_timestamps = [ts for ts in timestamps if ts > 0]
    if len(valid_timestamps) < 180:
        return False
    
    window_span = (valid_timestamps[-1] - valid_timestamps[0]) / 1e9
    return window_span <= 1200  # 20 minutes max

async def wait_until_capture_start():
    now_utc = datetime.now(timezone.utc)
    market_open_utc = now_utc.replace(hour=3, minute=45, second=0, microsecond=0)

    # Handle if market_open is on the next day (unlikely, but safe)
    if now_utc >= market_open_utc:
        return

    wait_seconds = (market_open_utc - now_utc).total_seconds()
    log_text(f"Sleeping until market open ({wait_seconds:.2f} seconds)...")
    await asyncio.sleep(wait_seconds)

# Trade state tracking (simple global state)
class TradeState:
    def __init__(self):
        self.last_signal_time = 0
        self.daily_count = 0
        self.last_reset_date = None
    
    def reset_if_new_day(self, current_timestamp):
        """Reset daily counter at market open (9:15 AM IST)"""
        current_dt = datetime.fromtimestamp(current_timestamp / 1e9, tz=timezone.utc) + timedelta(hours=5, minutes=30)
        current_date = current_dt.date()
        current_time = current_dt.time()
        
        # Reset at market open or new day
        if (self.last_reset_date != current_date and 
            current_time >= dtime(9, 15)):
            self.daily_count = 0
            self.last_reset_date = current_date
            log_text(f"🔄 Daily trade counter reset: {current_date}")

# Global trade state
_trade_state = TradeState()

def display_market_microstructure(features: dict, prediction: str, confidence: float, timestamp: int):
    """Clean, readable market intelligence display"""
    
    current_dt = datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc) + timedelta(hours=5, minutes=30)
    ist_time_str = current_dt.strftime('%H:%M:%S')
    
    try:
        # === Extract key metrics ===
        # Smart Money (recent blocks 4,5)
        smart_4 = features.get("smart_money_ratio_4", 0.0)
        smart_5 = features.get("smart_money_ratio_5", 0.0)
        dumb_4 = features.get("dumb_money_ratio_4", 0.0) 
        dumb_5 = features.get("dumb_money_ratio_5", 0.0)
        net_smart = ((smart_4 + smart_5) - (dumb_4 + dumb_5)) / 2
        
        # Market Makers
        mm_bias = (features.get("mm_directional_bias_4", 0.0) + features.get("mm_directional_bias_5", 0.0)) / 2
        mm_pressure = (features.get("mm_inventory_pressure_4", 0.0) + features.get("mm_inventory_pressure_5", 0.0)) / 2
        
        # Order Book
        book_imbal = (features.get("book_imbalance_4", 0.0) + features.get("book_imbalance_5", 0.0)) / 2
        
        # Price Action
        slope = features.get("ltp_slope", 0.0)
        momentum = features.get("momentum_score", 0.0)
        agg_ratio = features.get("agg_ratio", 0.0)
        streak = features.get("direction_streak", 0)
        
        # Session
        vwap_dev = features.get("session_vwap_deviation_pct", 0.0)
        range_pos = features.get("session_range_position", 0.0)
        
        # === Compact Intelligence Report ===
        log_text(f"🧠 Smart Money: {_format_flow(net_smart)} | 🏪 MM: {_format_mm(mm_pressure, mm_bias)} | 📖 Book: {_format_book(book_imbal)}")
        log_text(f"📈 Price: Slope{slope:+.2f} Mom{momentum:+.2f} Agg{agg_ratio:+.2f} | Session: VWAP{vwap_dev:+.2f} Pos{range_pos:.2f}")
        
        # === Block Detail (optional, only if significant differences) ===
        if abs(smart_4 - smart_5) > 0.05 or abs(mm_bias - features.get("mm_directional_bias_4", 0.0)) > 0.05:
            log_text(f"📋 Blocks: B4[S{smart_4:.2f} D{dumb_4:.2f} MM{features.get('mm_directional_bias_4', 0.0):+.2f}] B5[S{smart_5:.2f} D{dumb_5:.2f} MM{features.get('mm_directional_bias_5', 0.0):+.2f}]")
            
    except Exception as e:
        log_text(f"⚠️ Market display error: {e}")

def _format_flow(net_smart: float) -> str:
    """Format smart money flow"""
    if net_smart > 0.15:
        return "STRONG BUY"
    elif net_smart > 0.05:
        return "BUYING"
    elif net_smart < -0.15:
        return "STRONG SELL"
    elif net_smart < -0.05:
        return "SELLING"
    return "NEUTRAL"

def _format_mm(pressure: float, bias: float) -> str:
    """Format market maker activity"""
    activity = "HIGH" if pressure > 0.4 else "MED" if pressure > 0.2 else "LOW"
    lean = "BULL" if bias > 0.1 else "BEAR" if bias < -0.1 else "NEU"
    return f"{activity}/{lean}"

def _format_book(imbal: float) -> str:
    """Format order book pressure"""
    if imbal > 0.1:
        return "BUY PRESS"
    elif imbal < -0.1:
        return "SELL PRESS"
    return "BALANCED"

def is_trade_allowed(prediction: str, prediction_30: str, confidence: float, 
                    quality: float, timestamp: int, features: dict) -> tuple[bool, str]:
    """
    Enhanced trade filter with market microstructure physics and time controls.
    
    Args:
        prediction: "UP" or "DOWN" (15-min prediction)
        prediction_30: 30-min prediction (not used for now)
        confidence: Model confidence 0-1 (saved for later use)
        quality: Trade quality score 0-1 (saved for later use)
        timestamp: Current timestamp in nanoseconds
        features: Feature dictionary with market data
        
    Returns:
        (should_trade: bool, strategy_type: str)
    """
    
    # Reset daily counter if new day
    _trade_state.reset_if_new_day(timestamp)
    
    # === DISPLAY MARKET INTELLIGENCE (No validation) ===
    display_market_microstructure(features, prediction, confidence, timestamp)
    
    # === TIME-BASED FILTERS ===
    
    # 1. Only directional signals
    if prediction not in ["UP", "DOWN"]:
        return False, ""
    
    # 2. Lunch period block (12:00 - 12:45 PM IST)
    current_dt = datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc) + timedelta(hours=5, minutes=30)
    current_time = current_dt.time()
    
    #if dtime(12, 0) <= current_time <= dtime(12, 45):
    #    return False, ""  # Block during lunch
    
    # 3. Cooldown period (5 minutes)
    cooldown_ns = 5 * 60 * 1_000_000_000  # 5 minutes in nanoseconds
    #if timestamp - _trade_state.last_signal_time < cooldown_ns:
    #    return False, ""
    
    # 4. Daily trade limit (< 10 trades per day)
    #if _trade_state.daily_count >= 10:
    #    return False, ""
    
    # === MARKET MICROSTRUCTURE PHYSICS ===
    
    # Core directional features
    ltp_slope = features.get("ltp_slope", 0)
    momentum_score = features.get("momentum_score", 0)
    agg_ratio = features.get("agg_ratio", 0)
    slope_90 = features.get("slope_90", 0)
    
    # Volume confirmation
    ltq_bias_recent = features.get("ltq_bias_4", 0)
    volume_trend = features.get("volume_trend_5", 0)
    
    # Momentum confluence
    direction_streak = features.get("direction_streak", 0)
    
    if prediction == "UP":
        # Basic physics: Price should be rising for UP prediction
        basic_rules = [
            ("ltp_slope > 0", ltp_slope > 0),                    # Price trending up
            ("momentum_score > 0", momentum_score > 0),          # Positive momentum
            ("agg_ratio > 0", agg_ratio > 0),                   # More buyers than sellers
        ]
        
        # Supporting evidence (not mandatory but preferred)
        supporting_rules = [
            ("slope_90 > 0", slope_90 > 0),                     # Long-term uptrend
            ("ltq_bias_recent > 0", ltq_bias_recent > 0),       # Recent buying
            ("volume_trend > 0", volume_trend > 0),             # Rising volume
            ("direction_streak > 0", direction_streak > 0),     # Consecutive up moves
        ]
        
    elif prediction == "DOWN":
        # Basic physics: Price should be falling for DOWN prediction
        basic_rules = [
            ("ltp_slope < 0", ltp_slope < 0),                   # Price trending down
            ("momentum_score < 0", momentum_score < 0),         # Negative momentum  
            ("agg_ratio < 0", agg_ratio < 0),                  # More sellers than buyers
        ]
        
        # Supporting evidence
        supporting_rules = [
            ("slope_90 < 0", slope_90 < 0),                    # Long-term downtrend
            ("ltq_bias_recent < 0", ltq_bias_recent < 0),      # Recent selling
            ("volume_trend > 0", volume_trend > 0),            # Rising volume (confirms move)
            ("direction_streak < 0", direction_streak < 0),    # Consecutive down moves
        ]
    
    # Check basic rules (these should almost always pass)
    failed_basic = []
    for rule_name, condition in basic_rules:
        if not condition:
            failed_basic.append(rule_name)
    
    # Check supporting rules (nice to have)
    failed_supporting = []
    for rule_name, condition in supporting_rules:
        if not condition:
            failed_supporting.append(rule_name)
    
    # Calculate alignment score
    basic_alignment = (len(basic_rules) - len(failed_basic)) / len(basic_rules)
    supporting_alignment = (len(supporting_rules) - len(failed_supporting)) / len(supporting_rules)
    
    # Decision logic - require at least 2/3 basic rules to pass
    #if basic_alignment < 0.67:  # At least 2/3 basic rules
    #    return False, ""
    
    # Overall logic score threshold
    overall_logic_score = (basic_alignment * 0.7 + supporting_alignment * 0.3)
    if overall_logic_score < 0:
        return False, ""
    
    # === SIGNAL APPROVAL ===
    
    # Record this signal for cooldown/daily limit tracking
    _trade_state.last_signal_time = timestamp
    _trade_state.daily_count += 1
    
    # Always return "intraday" strategy for now
    strategy_type = "intraday"
    
    # Log successful signal
    ist_time = current_dt.strftime('%H:%M:%S')
    log_text(f"✅ Trade allowed: {prediction} at {ist_time} "
             f"(basic: {basic_alignment:.2f}, overall: {overall_logic_score:.2f}, "
             f"daily: {_trade_state.daily_count}/10)")
    
    return True, strategy_type

def encode_zone_tag(zone_tag: str) -> dict:
    """One-hot encode ltp_zone_tag consistently for model input."""
    return {
        "zone_SUPPORT": int(zone_tag == "SUPPORT"),
        "zone_RESIST": int(zone_tag == "RESIST"),
        "zone_SUPPORT_RESIST": int(zone_tag == "SUPPORT_RESIST")
    }