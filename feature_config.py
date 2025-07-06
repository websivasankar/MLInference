# feature_config.py - FIXED VERSION WITH BLOCK IMPORT
"""
Centralized feature configuration without circular imports.
All feature sets defined here to ensure consistency.
"""

# Import block features to avoid duplication
from block_feature_extractor import BLOCK_FEATURE_KEYS
from session_tracker import SESSION_FEATURE_KEYS
# ════════════════════════════════════════════════════════════════
# TOGGLE LINE - CHANGE ONLY THIS
# ════════════════════════════════════════════════════════════════
USE_SHAP_OPTIMIZED = False  # Set to True to use SHAP-optimized 35 features

# Core feature set (without block features)
CORE_FEATURE_SET = [
    # Core microstructure features
    "ltp_roc", "ltq_spike", "spread", "depth_imbalance",
    "bid_accumulating", "ask_vanish", "oi_change",
    "ltp_std", "vwap_deviation",
    "momentum_score", "roc_pct", "direction_streak",
    "ltp_slope", "oi_roc", "volume_zscore", "ltp_range",
    "agg_ratio", "range_position", "std_compression", "net_depth_trend",
    
    # Time-aware features
    "minutes_since_open", "tod_sin", "tod_cos",
    
    # Behavioral microstructure features
    "ltq_bias_30", "bounce_freq_30", "iceberg_score_30",
    "flag_break_up30", "flag_break_dn30",
    "flag_trap_up30", "flag_trap_dn30",
    "inv_long_short", "inv_abs_ratio",
    
    # Regime detection features
    "atr_pct", "volatility_ratio",
    "trend_conflict", "slope_90"
]

# Full feature set = Core + Block features
FULL_FEATURE_SET = CORE_FEATURE_SET + SESSION_FEATURE_KEYS + BLOCK_FEATURE_KEYS

# Your SHAP-optimized features (35 features)
SHAP_SELECTED_FEATURES = ['atr_pct', 'avg_spread_0', 'avg_spread_3', 'bid_accumulating', 'bid_trend_0', 'book_imbalance_1', 'dumb_money_ratio_0', 'dumb_money_ratio_4', 'dumb_reactivity_4', 'ltp_std', 'ltp_volatility_0', 'ltp_volatility_1', 'ltp_volatility_3', 'ltp_volatility_4', 'ltp_volatility_5', 'ltp_vs_tick_poc_pct', 'ltp_vs_volume_poc_pct', 'ltq_bias_2', 'ltq_bias_4', 'minutes_since_open', 'mm_directional_bias_2', 'oi_change_0', 'oi_trend_2', 'session_or_breakout_strength', 'session_or_position', 'session_or_range', 'session_range_position', 'session_tick_poc_strength', 'session_value_area_width_pct', 'session_volume_poc_strength', 'session_vwap_deviation_pct', 'slope_90', 'smart_money_ratio_5', 'smart_persistence_2', 'spread_volatility_0', 'spread_volatility_1', 'spread_volatility_2', 'spread_volatility_4', 'spread_volatility_5', 'tod_cos', 'tod_sin', 'volume_intensity_1', 'volume_intensity_4']

def get_active_features():
    """Return currently active feature set."""
    return SHAP_SELECTED_FEATURES if USE_SHAP_OPTIMIZED else FULL_FEATURE_SET

def get_feature_set_name():
    """Return human-readable name."""
    return f"SHAP_{len(SHAP_SELECTED_FEATURES)}" if USE_SHAP_OPTIMIZED else f"FULL_{len(FULL_FEATURE_SET)}"

def get_feature_count():
    """Return count of active features."""
    return len(get_active_features())

def get_core_feature_count():
    """Return count of core (non-block) features."""
    return len(CORE_FEATURE_SET)

def get_block_feature_count():
    """Return count of block features."""
    return len(BLOCK_FEATURE_KEYS)

# Debug info
def print_feature_breakdown():
    """Print detailed feature breakdown for debugging."""
    print(f"📊 Feature Breakdown:")
    print(f"   Core features: {len(CORE_FEATURE_SET)}")
    print(f"   Block features: {len(BLOCK_FEATURE_KEYS)}")
    print(f"   Total full features: {len(FULL_FEATURE_SET)}")
    print(f"   SHAP features: {len(SHAP_SELECTED_FEATURES)}")
    print(f"   Currently active: {get_feature_set_name()} ({get_feature_count()} features)")

# Test function
if __name__ == "__main__":
    print(f"✅ Feature configuration loaded")
    print_feature_breakdown()
    print(f"🔧 USE_SHAP_OPTIMIZED = {USE_SHAP_OPTIMIZED}")
    
    # Verify no duplicates
    if len(FULL_FEATURE_SET) != len(set(FULL_FEATURE_SET)):
        print("⚠️ WARNING: Duplicate features detected in FULL_FEATURE_SET")
    else:
        print("✅ No duplicate features detected")