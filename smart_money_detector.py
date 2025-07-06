# smart_money_detector.py - BULLETPROOFED VERSION
"""
Smart Money Flow Detection Module - PRODUCTION SAFE

Analyzes order flow patterns to distinguish between smart money (institutional/informed)
and dumb money (retail/reactive) trading behavior.

SAFETY FEATURES:
- Zero-division protection on all calculations
- Empty array validation
- Graceful degradation with sensible defaults
- Input sanitization and bounds checking
"""

import numpy as np
from typing import List, Dict, Any
import warnings

# Configuration constants - Calibrated for Nifty Futures
SMART_VOLUME_MULTIPLIER = 2.5  # 2.5x median volume (more realistic threshold)
PRICE_IMPACT_THRESHOLD = 0.00005  # 0.005% per lot (5 points on 25000 level)
MIN_VOLUME_THRESHOLD = 5  # Minimum lot size to consider
PERSISTENCE_WINDOW = 5  # Ticks to measure flow persistence

# ================================
# SAFETY UTILITIES
# ================================

def safe_mean(arr):
    """Safe mean calculation with empty array protection"""
    if len(arr) == 0:
        return 0.0
    try:
        return float(np.mean(arr))
    except (ValueError, TypeError):
        return 0.0

def safe_median(arr):
    """Safe median calculation with empty array protection"""
    if len(arr) == 0:
        return 0.0
    try:
        return float(np.median(arr))
    except (ValueError, TypeError):
        return 0.0

def safe_std(arr):
    """Safe standard deviation with empty array protection"""
    if len(arr) <= 1:
        return 0.0
    try:
        return float(np.std(arr))
    except (ValueError, TypeError):
        return 0.0

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with zero-denominator protection"""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    try:
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return float(result)
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def safe_clip(value, min_val, max_val):
    """Safe clipping with NaN protection"""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return max(min_val, min(max_val, value))

def validate_and_clean_arrays(*arrays):
    """Validate and clean input arrays, removing NaN/inf values"""
    cleaned = []
    for arr in arrays:
        if len(arr) == 0:
            cleaned.append(np.array([]))
            continue
        
        # Convert to numpy array and remove NaN/inf
        arr_np = np.array(arr)
        mask = np.isfinite(arr_np)
        cleaned_arr = arr_np[mask]
        cleaned.append(cleaned_arr)
    
    return cleaned

# ================================
# MAIN ANALYSIS FUNCTION
# ================================

def analyze_smart_money_flow(tick_buffer: List[Dict], block_idx: int) -> Dict[str, float]:
    """
    BULLETPROOF smart money analysis with comprehensive error handling.
    
    Returns 4 features per block:
    - smart_money_ratio: % of volume from smart orders
    - dumb_money_ratio: % of volume from reactive orders  
    - smart_persistence: Smart money flow consistency
    - dumb_reactivity: Dumb money reaction speed to price moves
    """
    # Early validation
    if not tick_buffer or len(tick_buffer) < 10:
        return _empty_features(block_idx)
    
    try:
        # Extract and validate core data vectors
        volumes = np.array([abs(t.get("ltq", 0)) for t in tick_buffer if t.get("ltq") is not None])
        prices = np.array([t.get("ltp", 0) for t in tick_buffer if t.get("ltp") is not None])
        signed_volumes = np.array([t.get("ltq", 0) for t in tick_buffer if t.get("ltq") is not None])
        
        # Clean and validate arrays
        volumes, prices, signed_volumes = validate_and_clean_arrays(volumes, prices, signed_volumes)
        
        # Ensure all arrays have same length
        min_len = min(len(volumes), len(prices), len(signed_volumes))
        if min_len < 5:  # Need minimum viable data
            return _empty_features(block_idx)
        
        volumes = volumes[:min_len]
        prices = prices[:min_len]
        signed_volumes = signed_volumes[:min_len]
        
        # Filter out zero/invalid data
        valid_mask = (volumes > 0) & (prices > 0)
        if np.sum(valid_mask) < 3:
            return _empty_features(block_idx)
        
        volumes_valid = volumes[valid_mask]
        prices_valid = prices[valid_mask]
        signed_volumes_valid = signed_volumes[valid_mask]
        
        # Core calculations with safety
        smart_ratio = _calculate_smart_ratio_safe(volumes_valid, prices_valid, signed_volumes_valid)
        dumb_ratio = _calculate_dumb_ratio_safe(volumes_valid, prices_valid, signed_volumes_valid)
        smart_persist = _calculate_persistence_safe(signed_volumes_valid, prices_valid, is_smart=True)
        dumb_react = _calculate_reactivity_safe(signed_volumes_valid, prices_valid)
        
        return {
            f"smart_money_ratio_{block_idx}": safe_clip(smart_ratio, 0.0, 1.0),
            f"dumb_money_ratio_{block_idx}": safe_clip(dumb_ratio, 0.0, 1.0),
            f"smart_persistence_{block_idx}": safe_clip(smart_persist, 0.0, 1.0),
            f"dumb_reactivity_{block_idx}": safe_clip(dumb_react, 0.0, 1.0),
        }
        
    except Exception as e:
        warnings.warn(f"Smart money analysis failed for block {block_idx}: {e}")
        return _empty_features(block_idx)

# ================================
# SAFE CALCULATION FUNCTIONS
# ================================

def _calculate_smart_ratio_safe(volumes: np.ndarray, prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """SAFE: Smart money detection with comprehensive error handling"""
    if len(volumes) < 3:
        return 0.0
    
    try:
        # Filter meaningful volumes with safety
        meaningful_volumes = volumes[volumes >= MIN_VOLUME_THRESHOLD]
        if len(meaningful_volumes) == 0:
            return 0.0
        
        median_vol = safe_median(meaningful_volumes)
        if median_vol <= 0:
            return 0.0
        
        smart_threshold = max(median_vol * SMART_VOLUME_MULTIPLIER, MIN_VOLUME_THRESHOLD * 3)
        total_volume = np.sum(volumes)
        
        if total_volume <= 0:
            return 0.0
        
        smart_volume = 0.0
        
        for i in range(1, len(volumes)):
            vol = volumes[i]
            if vol >= smart_threshold and i < len(prices):
                # Safe price impact calculation
                if prices[i] > 0 and prices[i-1] > 0:
                    price_change = abs(prices[i] - prices[i-1])
                    price_change_pct = safe_divide(price_change, prices[i-1], 0.0)
                    volume_normalized = safe_divide(vol, median_vol, 1.0)
                    
                    if volume_normalized > 0:
                        impact_efficiency = safe_divide(price_change_pct, volume_normalized, float('inf'))
                        
                        # Smart money: high volume relative to impact
                        if impact_efficiency <= PRICE_IMPACT_THRESHOLD:
                            smart_volume += vol
        
        return safe_divide(smart_volume, total_volume, 0.0)
        
    except Exception:
        return 0.0

def _calculate_dumb_ratio_safe(volumes: np.ndarray, prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """SAFE: Dumb money detection with comprehensive error handling"""
    if len(volumes) < 3:
        return 0.0
    
    try:
        total_volume = np.sum(volumes)
        if total_volume <= 0:
            return 0.0
        
        dumb_volume = 0.0
        
        for i in range(2, len(volumes)):
            if (volumes[i] >= MIN_VOLUME_THRESHOLD and 
                i-1 < len(prices) and i-2 < len(prices) and
                prices[i-1] > 0 and prices[i-2] > 0):
                
                # Safe momentum calculation
                recent_move = prices[i-1] - prices[i-2]
                current_side = np.sign(signed_volumes[i]) if i < len(signed_volumes) else 0
                
                # Price move threshold with safety
                price_move_threshold = prices[i-2] * 0.0001 if prices[i-2] > 0 else 0.01
                
                if abs(recent_move) > price_move_threshold and current_side != 0:
                    move_direction = np.sign(recent_move)
                    
                    # Dumb money: chasing momentum
                    if move_direction == current_side:
                        dumb_volume += volumes[i]
                        
                        # Extra penalty for persistent chasing
                        if i >= 3 and i-3 < len(prices) and prices[i-3] > 0:
                            prev_move = prices[i-2] - prices[i-3]
                            if (abs(prev_move) > 0 and 
                                np.sign(prev_move) == np.sign(recent_move) == current_side):
                                dumb_volume += volumes[i] * 0.5
        
        return safe_divide(dumb_volume, total_volume, 0.0)
        
    except Exception:
        return 0.0

def _calculate_persistence_safe(signed_volumes: np.ndarray, prices: np.ndarray, is_smart: bool) -> float:
    """SAFE: Flow persistence measurement with comprehensive error handling"""
    if len(signed_volumes) < PERSISTENCE_WINDOW:
        return 0.0
    
    try:
        # Filter meaningful flows with safety
        meaningful_flows = signed_volumes[np.abs(signed_volumes) >= MIN_VOLUME_THRESHOLD]
        if len(meaningful_flows) < 3:
            return 0.0
        
        # Calculate directional consistency
        flow_directions = np.sign(meaningful_flows)
        flow_directions = flow_directions[flow_directions != 0]  # Remove zeros
        
        if len(flow_directions) < 3:
            return 0.0
        
        # Rolling window persistence calculation
        persistence_scores = []
        window_size = min(PERSISTENCE_WINDOW, len(flow_directions))
        
        for i in range(window_size, len(flow_directions)):
            window = flow_directions[i-window_size:i]
            
            if len(window) > 0:
                positive_flows = np.sum(window > 0)
                negative_flows = np.sum(window < 0)
                total_flows = len(window)
                
                if total_flows > 0:
                    directional_bias = safe_divide(max(positive_flows, negative_flows), total_flows, 0.0)
                    persistence_scores.append(directional_bias)
        
        return safe_mean(persistence_scores)
        
    except Exception:
        return 0.0

def _calculate_reactivity_safe(signed_volumes: np.ndarray, prices: np.ndarray) -> float:
    """SAFE: Reactivity measurement with comprehensive error handling"""
    if len(signed_volumes) < 3:
        return 0.0
    
    try:
        reaction_scores = []
        
        # Safe mean calculation for normalization
        nonzero_vols = np.abs(signed_volumes[signed_volumes != 0])
        mean_vol = safe_mean(nonzero_vols)
        if mean_vol <= 0:
            mean_vol = 1.0  # Fallback
        
        for i in range(2, len(signed_volumes)):
            if (abs(signed_volumes[i]) >= MIN_VOLUME_THRESHOLD and 
                i-1 < len(prices) and i-2 < len(prices) and
                prices[i-1] > 0 and prices[i-2] > 0):
                
                # Safe reaction calculation
                immediate_price_move = prices[i-1] - prices[i-2]
                order_direction = np.sign(signed_volumes[i])
                
                price_threshold = prices[i-2] * 0.0001 if prices[i-2] > 0 else 0.01
                
                if abs(immediate_price_move) > price_threshold and order_direction != 0:
                    price_direction = np.sign(immediate_price_move)
                    
                    # Reaction score calculation
                    if price_direction == order_direction:
                        volume_weight = safe_divide(abs(signed_volumes[i]), mean_vol, 1.0)
                        price_weight = safe_divide(abs(immediate_price_move), prices[i-2], 0.0)
                        
                        reaction_score = volume_weight * price_weight
                        if reaction_score > 0 and not np.isinf(reaction_score):
                            reaction_scores.append(reaction_score)
        
        return safe_mean(reaction_scores)
        
    except Exception:
        return 0.0

def _empty_features(block_idx: int) -> Dict[str, float]:
    """Return zero-filled features for invalid blocks"""
    return {
        f"smart_money_ratio_{block_idx}": 0.0,
        f"dumb_money_ratio_{block_idx}": 0.0,
        f"smart_persistence_{block_idx}": 0.0,
        f"dumb_reactivity_{block_idx}": 0.0,
    }

# Export feature keys for integration
SMART_MONEY_KEYS = [
    "smart_money_ratio", "dumb_money_ratio", 
    "smart_persistence", "dumb_reactivity"
]