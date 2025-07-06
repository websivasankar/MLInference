#utils.py
from datetime import datetime
import os
from datetime import datetime, timezone, timedelta

def ts_to_ist_str(ts_ns: int) -> str:
    # Convert nanoseconds to seconds
    ts_sec = ts_ns / 1e9
    # UTC-aware datetime
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    # Convert to IST
    dt_ist = dt_utc + timedelta(hours=5, minutes=30)
    return dt_ist.strftime('%Y-%m-%d %H:%M:%S')

def log_text(alert_text, log_file_path="logs/general/logs.txt"):
    """
    Logs alert text to a specified file and auto-creates nested folders.
    """
    single_line_alert = alert_text.replace('\n', ' ').strip()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} | {single_line_alert}"

    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'a', encoding="utf-8") as log_file:
            log_file.write(log_entry + '\n')
        return True
    except Exception as e:
        print(f"Error logging alert: {e}")
        return False

def tag_signed_ltq(records):
    """
    🔥 FIXED: Assigns correct trade direction to LTQ using proper market microstructure logic:
    
    CORRECT FINANCIAL LOGIC:
    - If LTP >= Ask Price → Aggressive BUY (buyer hitting ask) → +LTQ
    - If LTP <= Bid Price → Aggressive SELL (seller hitting bid) → -LTQ  
    - If Bid < LTP < Ask → Use tick-by-tick rule or mid-point rule
    
    This is the STANDARD way to classify trade direction in all professional trading systems!
    """
    for row in records:
        bid_price = row.get("bid_price_0", 0)
        ask_price = row.get("ask_price_0", 0)
        ltp = row.get("ltp", 0)
        ltq_raw = abs(row.get("ltq", 0))
        
        # Ensure valid order book data
        if bid_price <= 0 or ask_price <= 0 or ltp <= 0:
            # Fallback: keep original unsigned LTQ if no valid order book
            row["ltq"] = ltq_raw
            continue
        
        # Validate bid <= ask (sanity check)
        if bid_price >= ask_price:
            # Invalid order book - keep unsigned
            row["ltq"] = ltq_raw
            continue
            
        # ✅ CORRECT TRADE CLASSIFICATION LOGIC
        if ltp >= ask_price:
            # Trade at or above ask = Aggressive BUY (market taker hitting ask)
            direction = 1
        elif ltp <= bid_price:
            # Trade at or below bid = Aggressive SELL (market taker hitting bid)  
            direction = -1
        else:
            # Trade inside spread: Bid < LTP < Ask
            # Use mid-point rule as fallback
            mid_price = (bid_price + ask_price) / 2
            direction = 1 if ltp >= mid_price else -1
            
            # Alternative: Could use tick-by-tick rule (compare with previous LTP)
            # But mid-point rule is simpler and widely used
        
        # Apply direction to LTQ
        row["ltq"] = direction * ltq_raw
        
        # Optional: Add debug info for verification
        # print(f"LTP:{ltp} Bid:{bid_price} Ask:{ask_price} → Dir:{direction}")
    
    return records

def detect_regime_from_blocks(agg: dict) -> str:
    """
    Determine current market regime (UP, DOWN, FLAT) based on block features.
    Uses the 6 blocks of 30-tick features from block_feature_extractor.
    
    Parameters:
    - agg: Dictionary of features from enrich_features()

    Returns:
    - "UP", "DOWN", or "FLAT" based on trend slope and momentum
    """
    try:
        # Use the correct feature names from your block extractor
        # Block 0 is oldest (first 30 ticks), Block 5 is newest (last 30 ticks)
        
        # Primary trend signal: Compare slopes across blocks
        early_slopes = [agg.get(f"ltp_slope_{i}", 0) for i in range(2)]  # Blocks 0-1
        late_slopes = [agg.get(f"ltp_slope_{i}", 0) for i in range(4, 6)]  # Blocks 4-5
        
        early_slope_avg = sum(early_slopes) / len(early_slopes) if early_slopes else 0
        late_slope_avg = sum(late_slopes) / len(late_slopes) if late_slopes else 0
        
        # Secondary confirmation: Recent momentum and volatility
        recent_momentum = agg.get("ltp_momentum_5", 0)  # Most recent block momentum
        volatility_ratio = agg.get("volatility_ratio", 1.0)  # From main features
        atr_pct = agg.get("atr_pct", 0)
        
        # Trend strength indicators
        trend_conflict = agg.get("trend_conflict", 0)
        slope_90 = agg.get("slope_90", 0)  # Long-term slope from main features
        
        # Volume confirmation
        recent_volume_trend = agg.get("volume_trend_5", 0)
        volume_intensity = agg.get("volume_intensity_5", 0)
        
        # Calculate regime score
        trend_score = 0
        
        # 1. Slope progression (early vs late blocks)
        slope_change = late_slope_avg - early_slope_avg
        if slope_change > 0.05:
            trend_score += 2
        elif slope_change < -0.05:
            trend_score -= 2
            
        # 2. Recent momentum confirmation
        if recent_momentum > 0.01:
            trend_score += 1
        elif recent_momentum < -0.01:
            trend_score -= 1
            
        # 3. Long-term slope alignment
        if slope_90 > 0.02:
            trend_score += 1
        elif slope_90 < -0.02:
            trend_score -= 1
            
        # 4. Volume confirmation
        if recent_volume_trend > 0 and volume_intensity > 100:
            trend_score += 1 if trend_score > 0 else -1
            
        # 5. Regime stability check
        if trend_conflict == 1:
            trend_score = int(trend_score * 0.5)  # Reduce confidence
            
        # 6. Volatility filter
        if volatility_ratio > 2.0 or atr_pct > 0.5:
            # High volatility - be more conservative
            threshold = 3
        else:
            threshold = 2
            
        # Final regime decision
        if trend_score >= threshold:
            return "UP"
        elif trend_score <= -threshold:
            return "DOWN"
        else:
            return "FLAT"
            
    except (KeyError, TypeError, ZeroDivisionError) as e:
        print(f"Regime detection error: {e}")
        return "FLAT"  # Safe fallback


def get_regime_confidence(agg: dict) -> float:
    """
    Calculate confidence score (0-1) for the regime detection.
    Higher values indicate stronger conviction in the regime signal.
    """
    try:
        # Factors that increase confidence
        confidence = 0.5  # Base confidence
        
        # Trend consistency across blocks
        slopes = [agg.get(f"ltp_slope_{i}", 0) for i in range(6)]
        slope_signs = [1 if s > 0 else -1 if s < 0 else 0 for s in slopes]
        consistency = abs(sum(slope_signs)) / len(slope_signs)
        confidence += consistency * 0.3
        
        # Volume support
        volume_trends = [agg.get(f"volume_trend_{i}", 0) for i in range(6)]
        avg_volume_trend = sum(volume_trends) / len(volume_trends)
        if abs(avg_volume_trend) > 10:
            confidence += 0.1
            
        # Low trend conflict
        if agg.get("trend_conflict", 1) == 0:
            confidence += 0.1
            
        # Stable volatility
        volatility_ratio = agg.get("volatility_ratio", 1.0)
        if 0.8 <= volatility_ratio <= 1.5:
            confidence += 0.1
            
        return min(1.0, max(0.0, confidence))
        
    except Exception:
        return 0.5

