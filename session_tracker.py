# session_tracker.py - FINAL OPTIMIZED SESSION-WIDE VERSION
"""
Session feature tracker with smart caching and normalized features.

Key Optimizations:
- Session-wide accumulation for true macro context
- Conditional POC/Value Area recalculation only when significant changes
- JSON persistence on major state changes only
- Incremental updates for cheap operations (VWAP, ranges)
- All features normalized using session range

Performance:
- 95% of ticks: Light updates only (microseconds)
- 5% of ticks: Heavy recalculation when POC shifts (milliseconds)
- JSON writes: ~10-20 per session (not 4,000+)
"""

import json
import os
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

class SessionStreamingVWAP:
    """Session-wide VWAP calculation with incremental updates"""
    
    def __init__(self):
        self.sum_pv = 0.0  # Session sum of price * volume
        self.sum_v = 0.0   # Session sum of volume
        self.vwap = 0.0
    
    def update(self, price: float, volume: float):
        """Update session VWAP with new tick"""
        if volume > 0:
            self.sum_pv += price * volume
            self.sum_v += volume
            self.vwap = self.sum_pv / self.sum_v if self.sum_v > 0 else price
    
    def get_vwap(self) -> float:
        return self.vwap
    
    def to_dict(self) -> dict:
        """For JSON persistence"""
        return {
            'sum_pv': self.sum_pv,
            'sum_v': self.sum_v,
            'vwap': self.vwap
        }
    
    def from_dict(self, data: dict):
        """From JSON persistence"""
        self.sum_pv = data.get('sum_pv', 0.0)
        self.sum_v = data.get('sum_v', 0.0)
        self.vwap = data.get('vwap', 0.0)

class SessionVolumeHistogram:
    """Session-wide volume histogram with smart caching"""
    
    def __init__(self, tick_size: float = 0.25):
        self.tick_size = tick_size
        
        # Session-wide accumulation
        self.volume_profile = defaultdict(float)
        self.tick_count_profile = defaultdict(int)
        self.total_volume = 0.0
        
        # Cached expensive calculations
        self.cached_volume_poc_price = 0.0
        self.cached_volume_poc_strength = 0.0
        self.cached_tick_poc_price = 0.0
        self.cached_tick_poc_strength = 0.0
        self.cached_value_area_high = 0.0
        self.cached_value_area_low = 0.0
        
        # Change detection
        self.last_max_volume_bucket = None
        self.last_max_tick_bucket = None
        self.last_recalc_volume = 0.0
        
        # Performance tracking
        self.recalc_count = 0
        self.tick_count = 0
    
    def _price_to_bucket(self, price: float) -> int:
        """Convert price to histogram bucket"""
        return int(price / self.tick_size)
    
    def update_incremental(self, price: float, volume: float):
        """✅ LIGHT: Incremental update - called every tick"""
        if price <= 0:
            return False
            
        self.tick_count += 1
        bucket = self._price_to_bucket(price)
        
        # Update session accumulation
        self.volume_profile[bucket] += volume
        self.total_volume += volume
        self.tick_count_profile[bucket] += 1
        
        # Check if expensive recalculation needed
        return self._should_recalculate()
    
    def _should_recalculate(self) -> bool:
        """Detect if POC has shifted significantly"""
        if not self.volume_profile:
            return True
            
        # Find current max volume bucket
        current_max_bucket = max(self.volume_profile.items(), key=lambda x: x[1])[0]
        
        # POC price level changed
        if self.last_max_volume_bucket != current_max_bucket:
            return True
            
        # Force recalculation if volume increased significantly since last calc
        volume_change_pct = (self.total_volume - self.last_recalc_volume) / max(self.last_recalc_volume, 1.0)
        if volume_change_pct > 0.1:  # 10% volume increase
            return True
            
        # Force recalculation periodically
        if self.tick_count % 300 == 0:  # Every 300 ticks (~25 minutes)
            return True
            
        return False
    
    def recalculate_expensive_features(self):
        """✅ HEAVY: Expensive recalculation - called only when POC shifts"""
        if not self.volume_profile:
            return
            
        self.recalc_count += 1
        self.last_recalc_volume = self.total_volume
        
        # Volume POC calculation
        max_volume_item = max(self.volume_profile.items(), key=lambda x: x[1])
        self.last_max_volume_bucket = max_volume_item[0]
        self.cached_volume_poc_price = max_volume_item[0] * self.tick_size
        self.cached_volume_poc_strength = max_volume_item[1] / self.total_volume if self.total_volume > 0 else 0.0
        
        # Tick POC calculation
        if self.tick_count_profile:
            total_ticks = sum(self.tick_count_profile.values())
            max_tick_item = max(self.tick_count_profile.items(), key=lambda x: x[1])
            self.last_max_tick_bucket = max_tick_item[0]
            self.cached_tick_poc_price = max_tick_item[0] * self.tick_size
            self.cached_tick_poc_strength = max_tick_item[1] / total_ticks if total_ticks > 0 else 0.0
        
        # Value Area calculation (70% of volume)
        self._calculate_value_area()
    
    def _calculate_value_area(self, percentage: float = 0.7):
        """Calculate Value Area from session volume distribution"""
        if not self.volume_profile or self.total_volume <= 0:
            self.cached_value_area_high = 0.0
            self.cached_value_area_low = 0.0
            return
        
        # Sort buckets by volume (descending)
        sorted_buckets = sorted(self.volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        target_volume = self.total_volume * percentage
        cumulative_volume = 0.0
        value_buckets = []
        
        for bucket, volume in sorted_buckets:
            value_buckets.append(bucket)
            cumulative_volume += volume
            if cumulative_volume >= target_volume:
                break
        
        if len(value_buckets) >= 2:
            prices = [bucket * self.tick_size for bucket in value_buckets]
            self.cached_value_area_low = min(prices)
            self.cached_value_area_high = max(prices)
            
            if self.cached_value_area_high <= self.cached_value_area_low:
                self.cached_value_area_high = self.cached_value_area_low + self.tick_size
        elif len(value_buckets) == 1:
            price = value_buckets[0] * self.tick_size
            self.cached_value_area_low = price
            self.cached_value_area_high = price + self.tick_size
        else:
            self.cached_value_area_low = 0.0
            self.cached_value_area_high = 0.0
    
    def get_cached_features(self) -> Tuple[float, float, float, float, float, float]:
        """Get cached POC and Value Area features (fast)"""
        return (
            self.cached_volume_poc_price,
            self.cached_volume_poc_strength,
            self.cached_tick_poc_price, 
            self.cached_tick_poc_strength,
            self.cached_value_area_low,
            self.cached_value_area_high
        )
    
    def to_dict(self) -> dict:
        """For JSON persistence"""
        return {
            'volume_profile': dict(self.volume_profile),
            'tick_count_profile': dict(self.tick_count_profile),
            'total_volume': self.total_volume,
            'cached_volume_poc_price': self.cached_volume_poc_price,
            'cached_volume_poc_strength': self.cached_volume_poc_strength,
            'cached_tick_poc_price': self.cached_tick_poc_price,
            'cached_tick_poc_strength': self.cached_tick_poc_strength,
            'cached_value_area_high': self.cached_value_area_high,
            'cached_value_area_low': self.cached_value_area_low,
            'last_max_volume_bucket': self.last_max_volume_bucket,
            'last_max_tick_bucket': self.last_max_tick_bucket,
            'last_recalc_volume': self.last_recalc_volume,
            'recalc_count': self.recalc_count,
            'tick_count': self.tick_count
        }
    
    def from_dict(self, data: dict):
        """From JSON persistence"""
        self.volume_profile = defaultdict(float, data.get('volume_profile', {}))
        self.tick_count_profile = defaultdict(int, data.get('tick_count_profile', {}))
        self.total_volume = data.get('total_volume', 0.0)
        self.cached_volume_poc_price = data.get('cached_volume_poc_price', 0.0)
        self.cached_volume_poc_strength = data.get('cached_volume_poc_strength', 0.0)
        self.cached_tick_poc_price = data.get('cached_tick_poc_price', 0.0)
        self.cached_tick_poc_strength = data.get('cached_tick_poc_strength', 0.0)
        self.cached_value_area_high = data.get('cached_value_area_high', 0.0)
        self.cached_value_area_low = data.get('cached_value_area_low', 0.0)
        self.last_max_volume_bucket = data.get('last_max_volume_bucket')
        self.last_max_tick_bucket = data.get('last_max_tick_bucket')
        self.last_recalc_volume = data.get('last_recalc_volume', 0.0)
        self.recalc_count = data.get('recalc_count', 0)
        self.tick_count = data.get('tick_count', 0)

class OptimizedSessionTracker:
    """
    Performance-optimized session tracker with smart caching and normalized features.
    
    Design:
    - Session-wide accumulation for true macro context
    - Light updates every tick (VWAP, ranges)  
    - Heavy recalculation only when POC shifts
    - JSON persistence on significant changes only
    - All features normalized using session range
    """
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        
        # Session identification
        self.session_date = None
        self.json_file_path = None
        
        # Session state tracking
        self.session_started = False
        self.session_start_ts = None
        self.session_open = 0.0
        
        # Opening Range (calculated once, cached forever)
        self.or_high = 0.0
        self.or_low = float('inf')
        self.or_range = 0.0
        self.or_completed = False
        
        # Session extremes (updated every tick - cheap)
        self.session_high = 0.0
        self.session_low = float('inf')
        
        # Session VWAP (updated every tick - cheap)
        self.session_vwap = SessionStreamingVWAP()
        
        # Volume histogram (smart caching)
        self.volume_histogram = SessionVolumeHistogram()
        
        # Current session features cache
        self.current_features = {}
        
        # Performance tracking
        self.total_ticks_processed = 0
        self.heavy_recalc_count = 0
        self.json_write_count = 0
    
    def _get_session_date(self, timestamp_ns: int) -> str:
        """Get session date for file naming"""
        if self.test_mode:
            return "test_session"
            
        ts_sec = timestamp_ns / 1e9
        dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        dt_ist = dt_utc + timedelta(hours=5, minutes=30)
        return dt_ist.strftime('%Y%m%d')
    
    def _get_json_file_path(self, session_date: str) -> str:
        """Get JSON file path for session state"""
        return f"session_state_{session_date}.json"
    
    def _should_write_json(self, poc_shifted: bool) -> bool:
        """Decide if JSON file should be updated - PERFORMANCE OPTIMIZED"""
        
        # Only write on major milestones, not every POC shift
        if self.total_ticks_processed % 1200 == 0:  # Every 100 minutes
            return True
        
        # Write at session end (market close detection)
        current_hour = datetime.now().hour
        if current_hour >= 15:  # After 3 PM, assume session ending
            if self.total_ticks_processed % 300 == 0:  # Every 25 minutes
                return True
        
        # Emergency backup - very infrequent
        if self.json_write_count == 0 and self.total_ticks_processed > 600:
            return True  # First backup after 50 minutes
        
        return False  # DEFAULT: Don't write JSON
    
    def _write_session_state(self):
        """Write session state to JSON file"""
        if self.test_mode or not self.json_file_path:
            return
            
        try:
            state = {
                'session_date': self.session_date,
                'session_started': self.session_started,
                'session_start_ts': self.session_start_ts,
                'session_open': self.session_open,
                'or_high': self.or_high,
                'or_low': self.or_low,
                'or_range': self.or_range,
                'or_completed': self.or_completed,
                'session_high': self.session_high,
                'session_low': self.session_low,
                'session_vwap': self.session_vwap.to_dict(),
                'volume_histogram': self.volume_histogram.to_dict(),
                'total_ticks_processed': self.total_ticks_processed,
                'heavy_recalc_count': self.heavy_recalc_count,
                'json_write_count': self.json_write_count
            }
            
            with open(self.json_file_path, 'w') as f:
                json.dump(state, f)
                
            self.json_write_count += 1
            
        except Exception as e:
            print(f"⚠️ Failed to write session state: {e}")
    
    def _load_session_state(self) -> bool:
        """Load session state from JSON file"""
        if self.test_mode or not self.json_file_path or not os.path.exists(self.json_file_path):
            return False
            
        try:
            with open(self.json_file_path, 'r') as f:
                state = json.load(f)
            
            self.session_date = state.get('session_date')
            self.session_started = state.get('session_started', False)
            self.session_start_ts = state.get('session_start_ts')
            self.session_open = state.get('session_open', 0.0)
            self.or_high = state.get('or_high', 0.0)
            self.or_low = state.get('or_low', float('inf'))
            self.or_range = state.get('or_range', 0.0)
            self.or_completed = state.get('or_completed', False)
            self.session_high = state.get('session_high', 0.0)
            self.session_low = state.get('session_low', float('inf'))
            self.total_ticks_processed = state.get('total_ticks_processed', 0)
            self.heavy_recalc_count = state.get('heavy_recalc_count', 0)
            self.json_write_count = state.get('json_write_count', 0)
            
            # Restore VWAP state
            if 'session_vwap' in state:
                self.session_vwap.from_dict(state['session_vwap'])
            
            # Restore histogram state
            if 'volume_histogram' in state:
                self.volume_histogram.from_dict(state['volume_histogram'])
            
            print(f"✅ Loaded session state: {self.total_ticks_processed} ticks, {self.heavy_recalc_count} recalcs")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load session state: {e}")
            return False
    
    def _cleanup_session_file(self):
        """Delete session file at session end"""
        if self.test_mode or not self.json_file_path:
            return
            
        try:
            if os.path.exists(self.json_file_path):
                os.remove(self.json_file_path)
                print(f"🗑️ Cleaned up session file: {self.json_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup session file: {e}")
    
    def is_session_time(self, timestamp_ns: int) -> bool:
        """Check if timestamp is within session hours"""
        if self.test_mode:
            return True
            
        try:
            ts_sec = timestamp_ns / 1e9
            dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            dt_ist = dt_utc + timedelta(hours=5, minutes=30)
            time_ist = dt_ist.time()
            
            from datetime import time as dtime
            return dtime(9, 10) <= time_ist <= dtime(15, 35)
        except:
            return False
    
    def update_from_buffer(self, tick_buffer: list, synthetic_tick_index: Optional[int] = None, 
                          opening_range: Optional[dict] = None):
        """
        ✅ MAIN METHOD: Update session state from latest tick in buffer
        
        Performance Strategy:
        - Process only the LATEST tick for session accumulation  
        - Use 180-tick buffer for buffer-based features only
        - Smart caching for expensive POC calculations
        """
        if not tick_buffer:
            return
        
        # Get latest tick for session updates
        latest_tick = tick_buffer[-1]
        timestamp_ns = latest_tick.get("ts", 0)
        
        # Initialize session if needed
        if not self.session_started:
            self._initialize_session(timestamp_ns)
        
        # ✅ LIGHT: Update session state with latest tick only
        poc_shifted = self._update_session_incremental(latest_tick)
        
        # ✅ HEAVY: Expensive recalculation only when needed
        if poc_shifted:
            self.volume_histogram.recalculate_expensive_features()
            self.heavy_recalc_count += 1
        
        # Set opening range if provided
        if opening_range and not self.or_completed:
            self.or_high = opening_range.get('or_high', 0.0)
            self.or_low = opening_range.get('or_low', 0.0)
            self.or_range = opening_range.get('or_range', 0.0)
            self.or_completed = opening_range.get('or_completed', True)
        
        # Calculate final features (mix of session-wide and buffer-based)
        self._calculate_final_features(tick_buffer, latest_tick)
        
        # ✅ CONDITIONAL: Write JSON only on significant changes
        if self._should_write_json(poc_shifted):
            self._write_session_state()
        
        self.total_ticks_processed += 1
    
    def _initialize_session(self, timestamp_ns: int):
        """Initialize session tracking with smart state recovery"""
        self.session_date = self._get_session_date(timestamp_ns)
        self.json_file_path = self._get_json_file_path(self.session_date)
        
        # Try to load existing state (for mid-day restart)
        if not self._load_session_state():
            # Fresh session start
            self.session_started = True
            self.session_start_ts = timestamp_ns
            print(f"🚀 Started new session: {self.session_date}")
        else:
            print(f"🔄 Resumed session: {self.session_date}")
    
    def _update_session_incremental(self, tick_data: dict) -> bool:
        """✅ LIGHT: Incremental session updates - called every tick"""
        ltp = tick_data.get("ltp", 0)
        volume = abs(tick_data.get("ltq", 0))
        
        if ltp <= 0:
            return False
        
        # Update session open (first valid tick)
        if self.session_open <= 0:
            self.session_open = ltp
        
        # Update session extremes (cheap)
        self.session_high = max(self.session_high, ltp)
        self.session_low = min(self.session_low, ltp)
        
        # Update session VWAP (cheap)
        if volume > 0:
            self.session_vwap.update(ltp, volume)
        
        # Update volume histogram and check if expensive recalc needed
        return self.volume_histogram.update_incremental(ltp, volume)
    
    def _calculate_final_features(self, tick_buffer: list, latest_tick: dict):
        """Calculate final feature set (normalized using session range)"""
        features = {}
        current_ltp = latest_tick.get("ltp", 0)
        current_ltq = abs(latest_tick.get("ltq", 0))
        
        # Session range for normalization (avoid division by zero)
        session_range = max(self.session_high - self.session_low, 1.0)
        
        # ✅ OPENING RANGE FEATURES (keep existing good normalizations)
        features["session_or_range"] = max(self.or_range, 0.0)
        
        if self.or_range > 0:
            features["session_or_position"] = (current_ltp - self.or_low) / self.or_range
        else:
            features["session_or_position"] = 0.5
        
        if current_ltp > self.or_high and self.or_range > 0:
            features["session_or_breakout_strength"] = (current_ltp - self.or_high) / self.or_range
        elif current_ltp < self.or_low and self.or_range > 0:
            features["session_or_breakout_strength"] = (self.or_low - current_ltp) / self.or_range
        else:
            features["session_or_breakout_strength"] = 0.0
        
        # ✅ SESSION VWAP FEATURES (normalized by session range)
        session_vwap = self.session_vwap.get_vwap()
        if session_vwap > 0:
            features["session_vwap_deviation_pct"] = (current_ltp - session_vwap) / session_range
        else:
            features["session_vwap_deviation_pct"] = 0.0
        
        # ✅ SESSION RANGE FEATURES (keep existing)
        features["session_range_position"] = (current_ltp - self.session_low) / session_range
        
        # ✅ POC FEATURES (normalized - remove absolute prices)
        (volume_poc_price, volume_poc_strength, 
         tick_poc_price, tick_poc_strength,
         value_area_low, value_area_high) = self.volume_histogram.get_cached_features()
        
        # POC strength features (already 0-1)
        features["session_volume_poc_strength"] = volume_poc_strength
        features["session_tick_poc_strength"] = tick_poc_strength
        
        # POC deviations (normalized by session range)
        if volume_poc_price > 0:
            features["ltp_vs_volume_poc_pct"] = (current_ltp - volume_poc_price) / session_range
        else:
            features["ltp_vs_volume_poc_pct"] = 0.0
        
        if tick_poc_price > 0:
            features["ltp_vs_tick_poc_pct"] = (current_ltp - tick_poc_price) / session_range
        else:
            features["ltp_vs_tick_poc_pct"] = 0.0
        
        # POC divergence (normalized by session range)
        if volume_poc_price > 0 and tick_poc_price > 0:
            features["session_poc_divergence_pct"] = abs(volume_poc_price - tick_poc_price) / session_range
        else:
            features["session_poc_divergence_pct"] = 0.0
        
        # ✅ VALUE AREA FEATURES (categorical position + normalized width)
        if value_area_high > 0 and value_area_low > 0 and value_area_high > value_area_low:
            if current_ltp > value_area_high:
                features["ltp_vs_value_area"] = 1.0  # Above value area
            elif current_ltp < value_area_low:
                features["ltp_vs_value_area"] = -1.0  # Below value area
            else:
                features["ltp_vs_value_area"] = 0.0  # Inside value area
                
            # Value area width as % of session range
            features["session_value_area_width_pct"] = (value_area_high - value_area_low) / session_range
        else:
            features["ltp_vs_value_area"] = 0.0
            features["session_value_area_width_pct"] = 0.0
        
        # ✅ BUFFER-BASED FEATURES (from 180-tick window)
        
        # Buffer volume stats (normalized)
        volumes = [abs(tick.get("ltq", 0)) for tick in tick_buffer if abs(tick.get("ltq", 0)) > 0]
        if len(volumes) > 1:
            volume_mean = np.mean(volumes)
            volume_std = max(np.std(volumes), 1.0)
            features["session_volume_zscore"] = (current_ltq - volume_mean) / volume_std
        else:
            features["session_volume_zscore"] = 0.0
        
        # Buffer momentum (normalized by session range)
        prices = [tick.get("ltp", 0) for tick in tick_buffer if tick.get("ltp", 0) > 0]
        if len(prices) >= 2:
            buffer_momentum = prices[-1] - prices[0]
            features["session_price_momentum_pct"] = buffer_momentum / session_range
        else:
            features["session_price_momentum_pct"] = 0.0
        
        self.current_features = features
    
    def get_session_features(self) -> Dict[str, float]:
        """Get current session features for ML model"""
        return self.current_features.copy()
    
    def reset_session(self):
        """Reset tracker for new trading session"""
        # Clean up old session file
        self._cleanup_session_file()
        
        # Reset state
        test_mode = self.test_mode
        self.__init__(test_mode=test_mode)
    
    def get_performance_stats(self) -> Dict[str, int]:
        """Get performance statistics"""
        return {
            'total_ticks_processed': self.total_ticks_processed,
            'heavy_recalc_count': self.heavy_recalc_count,
            'json_write_count': self.json_write_count,
            'histogram_recalc_count': self.volume_histogram.recalc_count
        }

# ============================================================================
# INTEGRATION FUNCTIONS (Same Interface)
# ============================================================================

_session_tracker = None

def get_session_tracker(test_mode: bool = False) -> OptimizedSessionTracker:
    """Get or create session tracker instance"""
    global _session_tracker
    if _session_tracker is None:
        _session_tracker = OptimizedSessionTracker(test_mode=test_mode)
    return _session_tracker

def update_session_tracker_from_buffer(tick_buffer: list, synthetic_tick_index: Optional[int] = None,
                                     opening_range: Optional[dict] = None):
    """
    ✅ MAIN INTEGRATION: Same interface, optimized implementation
    """
    tracker = get_session_tracker()
    tracker.update_from_buffer(tick_buffer, synthetic_tick_index, opening_range)
    
def get_session_features() -> Dict[str, float]:
    """Get session features for ML model"""
    tracker = get_session_tracker()
    return tracker.get_session_features()

def reset_session_tracker_for_testing():
    """Reset global tracker for testing"""
    global _session_tracker
    if _session_tracker:
        _session_tracker.reset_session()
    _session_tracker = None

def get_session_performance_stats() -> Dict[str, int]:
    """Get performance statistics"""
    tracker = get_session_tracker()
    return tracker.get_performance_stats()

# Session feature keys (normalized - final version)
SESSION_FEATURE_KEYS = [
    # Opening Range features
    "session_or_range", "session_or_position", "session_or_breakout_strength",
    
    # Session VWAP features  
    "session_vwap_deviation_pct", "session_range_position",
    
    # Volume stats
    "session_volume_zscore", "session_price_momentum_pct",
    
    # POC features (normalized)
    "session_volume_poc_strength", "ltp_vs_volume_poc_pct",
    "session_tick_poc_strength", "ltp_vs_tick_poc_pct", 
    "session_poc_divergence_pct",
    
    # Value Area features (categorical + normalized width)
    "ltp_vs_value_area", "session_value_area_width_pct"
]

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🚀 Testing optimized session tracker performance...")
    
    tracker = OptimizedSessionTracker(test_mode=True)
    
    # Simulate session ticks
    import time
    start_time = time.time()
    
    for i in range(1000):
        tick = {"ltp": 25000 + i * 0.25, "ltq": 50 + i % 100, "ts": i * 1000000000}
        tick_buffer = [tick]  # Simplified for test
        tracker.update_from_buffer(tick_buffer)
    
    end_time = time.time()
    stats = tracker.get_performance_stats()
    features = tracker.get_session_features()
    
    print(f"✅ Performance Test Results:")
    print(f"   Processed: {stats['total_ticks_processed']} ticks")
    print(f"   Heavy recalculations: {stats['heavy_recalc_count']}")
    print(f"   JSON writes: {stats['json_write_count']}")
    print(f"   Time taken: {(end_time - start_time)*1000:.2f}ms")
    print(f"   Avg per tick: {(end_time - start_time)*1000000/1000:.2f}μs")
    print(f"   Features extracted: {len(features)}")
    print(f"   Feature sample: {list(features.items())[:5]}")