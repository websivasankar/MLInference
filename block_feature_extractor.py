import numpy as np
from statistics import mean, stdev
import warnings
from smart_money_detector import analyze_smart_money_flow, SMART_MONEY_KEYS
from market_maker_detector import detect_mm_behavior, MARKET_MAKER_KEYS


# List of feature suffixes per block
BLOCK_KEYS = [
    "ltp_slope", "ltp_volatility", "ltp_momentum", "ltp_acceleration",
    "volume_zscore", "ltq_bias", "volume_intensity", "volume_trend",
    "oi_change", "oi_trend",
    "bid_trend", "ask_trend", "net_depth_trend", "avg_spread",
    "spread_volatility", "book_imbalance",
    "trap_up", "trap_dn", "breakout_strength",
    "price_efficiency", "pv_correlation"
] + SMART_MONEY_KEYS + MARKET_MAKER_KEYS

BLOCK_FEATURE_KEYS = [f"{k}_{i}" for i in range(6) for k in BLOCK_KEYS]

# Provide for feature_enricher.py to import
__all__ = ["compute_block_features", "BLOCK_FEATURE_KEYS"]

def compute_block_features(tick_buffer, block_size=30, total_blocks=6, signed_buffer=None):
    """
    Compute block-wise microstructure features over a 180-tick window.
    Each block represents 30 seconds of data, extracting 6 blocks total.
    
    Features extracted per block:
    - Price dynamics: slope, volatility, momentum
    - Volume patterns: z-score, bias, intensity
    - Order book: depth trends, spread dynamics
    - Trap detection: false breakout signals
    - Market quality: efficiency metrics
    
    Args:
        tick_buffer: List of tick dictionaries with keys like 'ltp', 'ltq', 'oi', etc.
        block_size: Number of ticks per block (default: 30)
        total_blocks: Number of blocks to extract (default: 6)
    
    Returns:
        Dictionary of features with keys like 'ltp_slope_0', 'volume_zscore_1', etc.
    """
    features = {}
    n_ticks = len(tick_buffer)
    
    if n_ticks < block_size * total_blocks:
        warnings.warn(f"Insufficient ticks: {n_ticks} < {block_size * total_blocks}")
        # Pad with zeros for missing features
        for i in range(total_blocks):
            features.update(_get_empty_block_features(i))
        return features

    for i in range(total_blocks):
        start = i * block_size
        end = start + block_size
        block = tick_buffer[start:end]
        # Get corresponding signed block if available
        signed_block = None
        if signed_buffer and len(signed_buffer) >= end:
            signed_block = signed_buffer[start:end]        
        # Extract block features
        block_features = _extract_block_features(block,i, signed_block=signed_block)
        features.update(block_features)

    return features

def _extract_block_features(block, block_idx, signed_block=None):
    """Extract all features for a single block"""
    features = {}
    
    # Extract raw data arrays
    ltp = np.array([t.get("ltp", 0) for t in block])
    ltq = np.array([t.get("ltq", 0) for t in block])
    oi = np.array([t.get("oi", 0) for t in block])
    
    # Order book data
    bid_qty = np.array([t.get("bid_qty_0", 0) for t in block])
    ask_qty = np.array([t.get("ask_qty_0", 0) for t in block])
    bid_price = np.array([t.get("bid_price_0", 0) for t in block])
    ask_price = np.array([t.get("ask_price_0", 0) for t in block])
    
    # Remove zero prices for more accurate calculations
    valid_ltp = ltp[ltp > 0]
    valid_ltq = ltq[ltq != 0]
    
    # === PRICE DYNAMICS FEATURES ===
    features.update(_compute_price_features(valid_ltp, block_idx))
    
    # === VOLUME FEATURES ===
    features.update(_compute_volume_features(valid_ltq, block_idx))
    
    # === OPEN INTEREST FEATURES ===
    features.update(_compute_oi_features(oi, block_idx))
    
    # === ORDER BOOK FEATURES ===
    features.update(_compute_orderbook_features(
        bid_qty, ask_qty, bid_price, ask_price, block_idx
    ))
    
    # === TRAP DETECTION FEATURES ===
    features.update(_compute_trap_features(valid_ltp, block_idx))
    
    # === MARKET QUALITY FEATURES ===
    features.update(_compute_market_quality_features(
        valid_ltp, valid_ltq, bid_price, ask_price, block_idx
    ))

   # Smart Money and Market Maker features - use signed block if available
    analysis_block = signed_block if signed_block else block
    features.update(analyze_smart_money_flow(analysis_block, block_idx))
    features.update(detect_mm_behavior(analysis_block, block_idx))
            
    return features

def _compute_price_features(ltp, block_idx):
    """Compute price-related features"""
    features = {}
    
    if len(ltp) >= 2:
        # Linear trend
        x = np.arange(len(ltp))
        slope, intercept = np.polyfit(x, ltp, 1)
        
        # Price volatility
        returns = np.diff(ltp) / ltp[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        
        # Price momentum (recent vs earlier)
        mid_point = len(ltp) // 2
        early_mean = np.mean(ltp[:mid_point]) if mid_point > 0 else ltp[0]
        late_mean = np.mean(ltp[mid_point:])
        momentum = (late_mean - early_mean) / early_mean if early_mean != 0 else 0.0
        
        # Price acceleration (second derivative)
        if len(ltp) >= 3:
            acceleration = np.polyfit(x, ltp, 2)[0]  # Quadratic coefficient
        else:
            acceleration = 0.0
            
    else:
        slope = volatility = momentum = acceleration = 0.0
    
    features.update({
        f"ltp_slope_{block_idx}": slope,
        f"ltp_volatility_{block_idx}": volatility,
        f"ltp_momentum_{block_idx}": momentum,
        f"ltp_acceleration_{block_idx}": acceleration,
    })
    
    return features

def _compute_volume_features(ltq, block_idx):
    """Compute volume-related features"""
    features = {}
    
    if len(ltq) > 0:
        ltq_mean = np.mean(ltq)
        ltq_std = np.std(ltq) if len(ltq) > 1 else 1e-6
        
        # Volume z-score (current vs historical)
        volume_zscore = (ltq[-1] - ltq_mean) / ltq_std if ltq_std > 0 else 0.0
        
        # Buy/sell bias
        buys = np.sum(ltq[ltq > 0])
        sells = np.sum(np.abs(ltq[ltq < 0]))
        total_volume = buys + sells
        ltq_bias = (buys - sells) / total_volume if total_volume > 0 else 0.0
        
        # Volume intensity (total volume in block)
        volume_intensity = np.log1p(total_volume)
        
        # Volume trend
        if len(ltq) >= 2:
            volume_trend = np.polyfit(np.arange(len(ltq)), np.abs(ltq), 1)[0]
        else:
            volume_trend = 0.0
            
    else:
        volume_zscore = ltq_bias = volume_intensity = volume_trend = 0.0
    
    features.update({
        f"volume_zscore_{block_idx}": volume_zscore,
        f"ltq_bias_{block_idx}": ltq_bias,
        f"volume_intensity_{block_idx}": volume_intensity,
        f"volume_trend_{block_idx}": volume_trend,
    })
    
    return features

def _compute_oi_features(oi, block_idx):
    """Compute open interest features"""
    features = {}
    
    if len(oi) >= 2:
        oi_change = (oi[-1] - oi[0]) / max(oi[0], 1.0)  # Percentage change
        oi_trend_raw = np.polyfit(np.arange(len(oi)), oi, 1)[0] if len(oi) > 2 else 0.0
        oi_trend = oi_trend_raw / max(np.mean(oi), 1.0)  # Normalize by block mean
    else:
        oi_change = oi_trend = 0.0
    
    features.update({
        f"oi_change_{block_idx}": oi_change,
        f"oi_trend_{block_idx}": oi_trend,
    })
    
    return features

def _compute_orderbook_features(bid_qty, ask_qty, bid_price, ask_price, block_idx):
    """Compute order book depth and spread features"""
    features = {}
    
    # Depth trends
    if len(bid_qty) > 1:
        bid_trend = np.polyfit(np.arange(len(bid_qty)), bid_qty, 1)[0]
    else:
        bid_trend = 0.0
        
    if len(ask_qty) > 1:
        ask_trend = np.polyfit(np.arange(len(ask_qty)), ask_qty, 1)[0]
    else:
        ask_trend = 0.0
    
    net_depth_trend = bid_trend - ask_trend
    
    # Spread dynamics
    valid_spreads = []
    for b_p, a_p in zip(bid_price, ask_price):
        if b_p > 0 and a_p > 0 and a_p > b_p:
            valid_spreads.append(a_p - b_p)
    
    if valid_spreads:
        avg_spread = np.mean(valid_spreads)
        spread_volatility = np.std(valid_spreads) if len(valid_spreads) > 1 else 0.0
    else:
        avg_spread = spread_volatility = 0.0
    
    # Order book imbalance
    total_bid = np.sum(bid_qty)
    total_ask = np.sum(ask_qty)
    book_imbalance = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0.0
    
    features.update({
        f"bid_trend_{block_idx}": bid_trend,
        f"ask_trend_{block_idx}": ask_trend,
        f"net_depth_trend_{block_idx}": net_depth_trend,
        f"avg_spread_{block_idx}": avg_spread,
        f"spread_volatility_{block_idx}": spread_volatility,
        f"book_imbalance_{block_idx}": book_imbalance,
    })
    
    return features

def _compute_trap_features(ltp, block_idx):
    """Compute trap/false breakout detection features"""
    features = {}
    
    # Add explicit length check with higher minimum requirement
    if len(ltp) < 10:  # Increased from 3 to 10 for more robust detection
        trap_up = trap_dn = breakout_strength = 0
    else:
        # Define recent range (excluding last 5 ticks instead of 2)
        range_data = ltp[:-5]  # Changed from ltp[:-2]
        
        if len(range_data) >= 5:  # Ensure sufficient range data
            hi = np.max(range_data)
            lo = np.min(range_data)
            
            # Only proceed if we have a meaningful price range
            if hi - lo >= 2.0:  # Minimum 2-point range for Nifty futures
                # Now safe to check trap patterns with lookback of 4 ticks
                # Trap up: price broke above range but came back down
                trap_up = int(len(ltp) >= 8 and 
                             ltp[-4] > hi + 1 and  # Broke above with buffer
                             ltp[-1] <= hi - 1)    # Came back down with buffer
                
                # Trap down: price broke below range but came back up  
                trap_dn = int(len(ltp) >= 8 and 
                             ltp[-4] < lo - 1 and  # Broke below with buffer
                             ltp[-1] >= lo + 1)    # Came back up with buffer
                
                # Range breakout strength - only if meaningful range
                breakout_strength = max(
                    (ltp[-1] - hi) / (hi - lo) if ltp[-1] > hi else 0,
                    (lo - ltp[-1]) / (hi - lo) if ltp[-1] < lo else 0
                )
            else:
                # Range too small - no meaningful breakouts possible
                trap_up = trap_dn = breakout_strength = 0.0
        else:
            # Insufficient range data
            trap_up = trap_dn = breakout_strength = 0
              
    features.update({
        f"trap_up_{block_idx}": trap_up,
        f"trap_dn_{block_idx}": trap_dn,
        f"breakout_strength_{block_idx}": breakout_strength,
    })
    
    return features

def _compute_market_quality_features(ltp, ltq, bid_price, ask_price, block_idx):
    """Compute market efficiency and quality metrics"""
    features = {}
    
    # Price efficiency (how much price moves per unit volume)
    if len(ltp) >= 2 and len(ltq) > 0:
        price_change = abs(ltp[-1] - ltp[0])
        total_volume = np.sum(np.abs(ltq))
        price_efficiency = price_change / total_volume if total_volume > 0 else 0.0
    else:
        price_efficiency = 0.0
        
    # Price-volume correlation
    if len(ltp) >= 3 and len(ltq) >= 3:
        min_len = min(len(ltp), len(ltq))
        try:
            ltp_sub = ltp[:min_len]
            ltq_sub = np.abs(ltq[:min_len])
            std_ltp = np.std(ltp_sub)
            std_ltq = np.std(ltq_sub)
            if std_ltp == 0 or std_ltq == 0:
                pv_correlation = 0.0
            else:
                corr_coef = np.corrcoef(ltp_sub, ltq_sub)[0, 1]
                pv_correlation = corr_coef if not np.isnan(corr_coef) else 0.0
        except Exception:
            pv_correlation = 0.0
    else:
        pv_correlation = 0.0
    
    features.update({
        f"price_efficiency_{block_idx}": price_efficiency,
        f"pv_correlation_{block_idx}": pv_correlation,
    })
    
    return features

def _get_empty_block_features(block_idx):
    """Return zero-filled features for missing blocks"""
    return {
        f"ltp_slope_{block_idx}": 0.0,
        f"ltp_volatility_{block_idx}": 0.0,
        f"ltp_momentum_{block_idx}": 0.0,
        f"ltp_acceleration_{block_idx}": 0.0,
        f"volume_zscore_{block_idx}": 0.0,
        f"ltq_bias_{block_idx}": 0.0,
        f"volume_intensity_{block_idx}": 0.0,
        f"volume_trend_{block_idx}": 0.0,
        f"oi_change_{block_idx}": 0.0,
        f"oi_trend_{block_idx}": 0.0,
        f"bid_trend_{block_idx}": 0.0,
        f"ask_trend_{block_idx}": 0.0,
        f"net_depth_trend_{block_idx}": 0.0,
        f"avg_spread_{block_idx}": 0.0,
        f"spread_volatility_{block_idx}": 0.0,
        f"book_imbalance_{block_idx}": 0.0,
        f"trap_up_{block_idx}": 0,
        f"trap_dn_{block_idx}": 0,
        f"breakout_strength_{block_idx}": 0.0,
        f"price_efficiency_{block_idx}": 0.0,
        f"pv_correlation_{block_idx}": 0.0,
    }
