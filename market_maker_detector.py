# market_maker_detector.py - BULLETPROOFED VERSION COMPLETE
"""
Market Maker Behavior Detection Module - PRODUCTION SAFE

Identifies market maker manipulation patterns and inventory management signals.

SAFETY FEATURES:
- Zero-division protection on all calculations
- Empty array validation
- Graceful degradation with sensible defaults
- Input sanitization and bounds checking
- NaN/infinity value handling
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import warnings

# Configuration constants - Calibrated for Nifty Futures
SPREAD_MANIPULATION_THRESHOLD = 2.0  # 2x normal spread (more conservative)
FAKE_LIQUIDITY_RATIO = 0.2  # 20% execution rate threshold
MIN_QUOTE_SIZE = 10  # Minimum meaningful quote size
INVENTORY_PRESSURE_WINDOW = 8  # Ticks for pressure calculation
ABNORMAL_SPREAD_POINTS = 3  # Absolute points threshold for Nifty

# ================================
# SAFETY UTILITIES 
# ================================

def safe_mean(arr):
    """Safe mean calculation with empty array protection"""
    if not hasattr(arr, '__len__') or len(arr) == 0:
        return 0.0
    try:
        result = float(np.mean(arr))
        return result if np.isfinite(result) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0

def safe_median(arr):
    """Safe median calculation with empty array protection"""
    if not hasattr(arr, '__len__') or len(arr) == 0:
        return 0.0
    try:
        result = float(np.median(arr))
        return result if np.isfinite(result) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0

def safe_std(arr):
    """Safe standard deviation with empty array protection"""
    if not hasattr(arr, '__len__') or len(arr) <= 1:
        return 0.0
    try:
        result = float(np.std(arr))
        return result if np.isfinite(result) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0

def safe_sum(arr):
    """Safe summation with empty array protection"""
    if not hasattr(arr, '__len__') or len(arr) == 0:
        return 0.0
    try:
        result = float(np.sum(arr))
        return result if np.isfinite(result) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with zero-denominator protection"""
    if (denominator == 0 or np.isnan(denominator) or np.isinf(denominator) or
        np.isnan(numerator) or np.isinf(numerator)):
        return default
    try:
        result = numerator / denominator
        return result if np.isfinite(result) else default
    except (ZeroDivisionError, TypeError, ValueError, OverflowError):
        return default

def safe_clip(value, min_val, max_val):
    """Safe clipping with NaN protection"""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    try:
        return max(min_val, min(max_val, float(value)))
    except (TypeError, ValueError):
        return 0.0

def validate_and_clean_arrays(*arrays):
    """Validate and clean multiple input arrays, removing NaN/inf values"""
    cleaned = []
    for arr in arrays:
        if not hasattr(arr, '__len__') or len(arr) == 0:
            cleaned.append(np.array([]))
            continue
        
        try:
            # Convert to numpy array and remove NaN/inf
            arr_np = np.array(arr, dtype=float)
            mask = np.isfinite(arr_np)
            cleaned_arr = arr_np[mask]
            cleaned.append(cleaned_arr)
        except (ValueError, TypeError):
            cleaned.append(np.array([]))
    
    return tuple(cleaned)

def safe_filter_positive(arr):
    """Safely filter positive values from array"""
    if not hasattr(arr, '__len__') or len(arr) == 0:
        return np.array([])
    try:
        arr_np = np.array(arr)
        return arr_np[arr_np > 0]
    except (TypeError, ValueError):
        return np.array([])

def safe_filter_range(arr, min_val, max_val):
    """Safely filter array within range"""
    if not hasattr(arr, '__len__') or len(arr) == 0:
        return np.array([])
    try:
        arr_np = np.array(arr)
        return arr_np[(arr_np >= min_val) & (arr_np <= max_val)]
    except (TypeError, ValueError):
        return np.array([])

# ================================
# INPUT VALIDATION FUNCTIONS
# ================================

def _validate_mm_inputs(tick_buffer: List[Dict]) -> bool:
    """Validate inputs for MM detection"""
    if not tick_buffer or len(tick_buffer) < 5:
        return False
    
    # Check if we have essential fields
    required_fields = ["bid_price_0", "ask_price_0", "bid_qty_0", "ask_qty_0", "ltq"]
    
    valid_ticks = 0
    for tick in tick_buffer:
        if all(field in tick and tick[field] is not None for field in required_fields):
            valid_ticks += 1
    
    # Need at least 50% valid ticks
    return valid_ticks >= len(tick_buffer) * 0.5

def _extract_mm_data_safely(tick_buffer: List[Dict]) -> Tuple[np.ndarray, ...]:
    """Safely extract and validate MM detection data"""
    try:
        # Pre-validate buffer
        if not _validate_mm_inputs(tick_buffer):
            return tuple(np.array([]) for _ in range(6))
        
        # Extract data with None checking
        bid_prices = []
        ask_prices = []
        bid_qtys = []
        ask_qtys = []
        volumes = []
        signed_volumes = []
        
        for tick in tick_buffer:
            # Only include ticks with valid order book data
            bid_price = tick.get("bid_price_0")
            ask_price = tick.get("ask_price_0")
            bid_qty = tick.get("bid_qty_0")
            ask_qty = tick.get("ask_qty_0")
            ltq = tick.get("ltq")
            
            if all(x is not None for x in [bid_price, ask_price, bid_qty, ask_qty, ltq]):
                try:
                    bid_p = float(bid_price)
                    ask_p = float(ask_price)
                    bid_q = max(0, int(bid_qty))
                    ask_q = max(0, int(ask_qty))
                    ltq_val = int(ltq)
                    
                    if bid_p > 0 and ask_p > 0 and ask_p > bid_p:
                        bid_prices.append(bid_p)
                        ask_prices.append(ask_p)
                        bid_qtys.append(bid_q)
                        ask_qtys.append(ask_q)
                        volumes.append(abs(ltq_val))
                        signed_volumes.append(ltq_val)
                except (ValueError, TypeError):
                    continue  # Skip invalid data
        
        return (np.array(bid_prices), np.array(ask_prices), np.array(bid_qtys),
                np.array(ask_qtys), np.array(volumes), np.array(signed_volumes))
        
    except Exception:
        return tuple(np.array([]) for _ in range(6))

# ================================
# MAIN DETECTION FUNCTION
# ================================

def detect_mm_behavior(tick_buffer: List[Dict], block_idx: int) -> Dict[str, float]:
    """
    BULLETPROOF market maker behavior detection with comprehensive error handling.
    
    Returns 4 features per block:
    - mm_spread_manipulation: Artificial spread widening score
    - mm_fake_liquidity_ratio: Quote stuffing vs real liquidity
    - mm_directional_bias: MM inventory-driven directional preference  
    - mm_inventory_pressure: Pressure relief trading intensity
    """
    # Early validation
    if not tick_buffer or len(tick_buffer) < 10:
        return _empty_features(block_idx)
    
    try:
        # Use safe data extraction
        (bid_prices, ask_prices, bid_qtys, 
         ask_qtys, volumes, signed_volumes) = _extract_mm_data_safely(tick_buffer)
        
        # Validate extracted data
        if (len(bid_prices) < 5 or len(ask_prices) < 5 or 
            len(bid_qtys) < 5 or len(ask_qtys) < 5):
            return _empty_features(block_idx)
        
        # Ensure all arrays have same length (they should from extraction but double-check)
        min_len = min(len(bid_prices), len(ask_prices), len(bid_qtys), 
                     len(ask_qtys), len(volumes), len(signed_volumes))
        
        if min_len < 3:
            return _empty_features(block_idx)
        
        # Truncate to same length for safety
        bid_prices = bid_prices[:min_len]
        ask_prices = ask_prices[:min_len]
        bid_qtys = bid_qtys[:min_len]
        ask_qtys = ask_qtys[:min_len]
        volumes = volumes[:min_len]
        signed_volumes = signed_volumes[:min_len]
        
        # Core MM detection calculations with additional safety layers
        spread_manip = _detect_spread_manipulation_safe(bid_prices, ask_prices)
        fake_liquidity = _calculate_fake_liquidity_ratio_safe(bid_qtys, ask_qtys, volumes)
        directional_bias = _calculate_directional_bias_safe(bid_qtys, ask_qtys, signed_volumes)
        inventory_pressure = _calculate_inventory_pressure_safe(signed_volumes, volumes)
        
        # Final validation and clipping
        return {
            f"mm_spread_manipulation_{block_idx}": safe_clip(spread_manip, 0.0, 1.0),
            f"mm_fake_liquidity_ratio_{block_idx}": safe_clip(fake_liquidity, 0.0, 1.0),
            f"mm_directional_bias_{block_idx}": safe_clip(directional_bias, -1.0, 1.0),
            f"mm_inventory_pressure_{block_idx}": safe_clip(inventory_pressure, 0.0, 1.0),
        }
        
    except Exception as e:
        warnings.warn(f"MM behavior detection failed for block {block_idx}: {e}")
        return _empty_features(block_idx)

# ================================
# SAFE CALCULATION FUNCTIONS
# ================================

def _detect_spread_manipulation_safe(bid_prices: np.ndarray, ask_prices: np.ndarray) -> float:
    """SAFE: Detect artificial spread widening by market makers"""
    if len(bid_prices) < 5 or len(ask_prices) < 5:
        return 0.0
    
    try:
        spreads = ask_prices - bid_prices
        
        # Filter out unrealistic spreads with safety
        valid_spreads = safe_filter_range(spreads, 0.1, 50.0)  # 0.1 to 50 points for Nifty
        if len(valid_spreads) < 3:
            return 0.0
        
        median_spread = safe_median(valid_spreads)
        if median_spread <= 0:
            return 0.0
        
        normal_threshold = max(median_spread * SPREAD_MANIPULATION_THRESHOLD, ABNORMAL_SPREAD_POINTS)
        
        manipulation_events = 0
        total_events = 0
        
        for i in range(1, len(spreads) - 1):
            if spreads[i] > 0:  # Valid spread
                total_events += 1
                
                # Check for spread manipulation pattern
                if spreads[i] > normal_threshold:
                    # Look for quick normalization (MM signature)
                    end_idx = min(i + 4, len(spreads))
                    post_manipulation_spreads = spreads[i+1:end_idx]
                    
                    if len(post_manipulation_spreads) > 0:
                        post_spreads_positive = safe_filter_positive(post_manipulation_spreads)
                        if len(post_spreads_positive) > 0:
                            avg_post_spread = safe_mean(post_spreads_positive)
                            
                            # If spread quickly returns to normal, likely manipulation
                            if avg_post_spread <= median_spread * 1.2:
                                manipulation_events += 1
        
        return safe_divide(manipulation_events, total_events, 0.0)
        
    except Exception:
        return 0.0

def _calculate_fake_liquidity_ratio_safe(bid_qtys: np.ndarray, ask_qtys: np.ndarray, volumes: np.ndarray) -> float:
    """SAFE: Calculate ratio of fake liquidity with comprehensive error handling"""
    if len(bid_qtys) < 3 or len(ask_qtys) < 3:
        return 0.0
    
    try:
        # Filter meaningful quote sizes with safety
        meaningful_bids = bid_qtys[bid_qtys >= MIN_QUOTE_SIZE]
        meaningful_asks = ask_qtys[ask_qtys >= MIN_QUOTE_SIZE]
        meaningful_volumes = volumes[volumes > 0] if len(volumes) > 0 else np.array([])
        
        if len(meaningful_bids) == 0 and len(meaningful_asks) == 0:
            return 0.0
        
        # Calculate quoted vs executed liquidity with safety
        avg_quoted_bid = safe_mean(meaningful_bids)
        avg_quoted_ask = safe_mean(meaningful_asks)
        total_avg_quoted = avg_quoted_bid + avg_quoted_ask
        
        avg_executed = safe_mean(meaningful_volumes)
        
        if total_avg_quoted <= 0:
            return 0.0
        
        # Execution ratio - how much actually trades vs what's quoted
        execution_ratio = safe_divide(avg_executed, total_avg_quoted, 0.0)
        
        # Higher fake liquidity when execution ratio is low
        fake_ratio = max(0, 1 - safe_divide(execution_ratio, FAKE_LIQUIDITY_RATIO, 0.0))
        
        return safe_clip(fake_ratio, 0.0, 1.0)
        
    except Exception:
        return 0.0

def _calculate_directional_bias_safe(bid_qtys: np.ndarray, ask_qtys: np.ndarray, signed_volumes: np.ndarray) -> float:
    """SAFE: Detect MM directional bias with comprehensive error handling"""
    if len(bid_qtys) < 5 or len(ask_qtys) < 5:
        return 0.0
    
    try:
        # Calculate meaningful imbalances with safety
        imbalances = []
        for i in range(min(len(bid_qtys), len(ask_qtys))):
            bid_qty = bid_qtys[i]
            ask_qty = ask_qtys[i]
            
            if bid_qty >= MIN_QUOTE_SIZE or ask_qty >= MIN_QUOTE_SIZE:
                total_depth = bid_qty + ask_qty
                if total_depth > 0:
                    imbalance = safe_divide(bid_qty - ask_qty, total_depth, 0.0)
                    if np.isfinite(imbalance):
                        imbalances.append(imbalance)
        
        if len(imbalances) < 3:
            return 0.0
        
        # Look for persistent imbalance patterns
        avg_imbalance = safe_mean(imbalances)
        imbalance_consistency = safe_std(imbalances)
        
        # Calculate recent order flow direction with safety
        min_flow_len = min(8, len(signed_volumes))
        recent_flows = signed_volumes[-min_flow_len:] if len(signed_volumes) > 0 else np.array([])
        meaningful_flows = recent_flows[np.abs(recent_flows) >= 5] if len(recent_flows) > 0 else np.array([])
        
        if len(meaningful_flows) > 0:
            net_flow = safe_sum(meaningful_flows)
            flow_direction = np.sign(net_flow)
            
            # MM bias: Often quote against recent flow (inventory management)
            bias_against_flow = -avg_imbalance * flow_direction
            
            # Weight by consistency (lower std = more consistent bias)
            consistency_factor = max(0, 1 - safe_divide(imbalance_consistency, 1.0, 1.0))
            final_bias = bias_against_flow * consistency_factor
            
            return safe_clip(final_bias, -1.0, 1.0)
        
        return 0.0
        
    except Exception:
        return 0.0

def _calculate_inventory_pressure_safe(signed_volumes: np.ndarray, volumes: np.ndarray) -> float:
    """SAFE: Detect MM inventory pressure relief patterns with comprehensive error handling"""
    window_size = min(INVENTORY_PRESSURE_WINDOW, len(signed_volumes))
    if len(signed_volumes) < window_size + 2:
        return 0.0
    
    try:
        pressure_events = []
        
        # Ensure volumes array is at least as long as signed_volumes
        if len(volumes) < len(signed_volumes):
            # Pad volumes array with last value if shorter
            last_vol = volumes[-1] if len(volumes) > 0 else 0
            padding = np.full(len(signed_volumes) - len(volumes), last_vol)
            volumes_padded = np.concatenate([volumes, padding])
        else:
            volumes_padded = volumes[:len(signed_volumes)]
        
        for i in range(window_size, len(signed_volumes)):
            # Calculate cumulative inventory in rolling window with safety
            start_idx = max(0, i - window_size)
            window_inventory = safe_sum(signed_volumes[start_idx:i])
            
            current_volume = volumes_padded[i] if i < len(volumes_padded) else 0
            current_flow = signed_volumes[i]
            
            # Only consider meaningful volumes
            if current_volume < 5:  # Min 5 lots
                continue
            
            # Calculate volume-weighted average for the window with safety
            window_volumes = volumes_padded[start_idx:i]
            window_volumes_positive = safe_filter_positive(window_volumes)
            avg_window_volume = safe_mean(window_volumes_positive)
            
            if avg_window_volume <= 0:
                continue
            
            # Check for significant inventory accumulation
            inventory_threshold = avg_window_volume * window_size * 0.3  # 30% of window
            
            if abs(window_inventory) > inventory_threshold:
                inventory_direction = np.sign(window_inventory)
                current_direction = np.sign(current_flow)
                
                # Pressure relief: large trade in opposite direction to inventory
                if current_direction == -inventory_direction and current_volume > avg_window_volume * 1.5:
                    # Calculate pressure intensity with safety
                    inventory_magnitude = safe_divide(abs(window_inventory), inventory_threshold, 0.0)
                    volume_magnitude = safe_divide(current_volume, avg_window_volume, 0.0)
                    
                    pressure_score = min(inventory_magnitude * volume_magnitude, 5.0)
                    if pressure_score > 0 and np.isfinite(pressure_score):
                        pressure_events.append(pressure_score)
        
        # Return average pressure relief intensity
        avg_pressure = safe_mean(pressure_events)
        return safe_clip(avg_pressure, 0.0, 1.0)
        
    except Exception:
        return 0.0

def _empty_features(block_idx: int) -> Dict[str, float]:
    """Return zero-filled features for invalid blocks"""
    return {
        f"mm_spread_manipulation_{block_idx}": 0.0,
        f"mm_fake_liquidity_ratio_{block_idx}": 0.0,
        f"mm_directional_bias_{block_idx}": 0.0,
        f"mm_inventory_pressure_{block_idx}": 0.0,
    }

# Export feature keys for integration
MARKET_MAKER_KEYS = [
    "mm_spread_manipulation", "mm_fake_liquidity_ratio",
    "mm_directional_bias", "mm_inventory_pressure"
]