# unified_trade_executor.py - Updated to use is_trade_allowed from helper.py
"""
Simplified Unified Trade Executor using market microstructure physics
- Replaced complex TradeFilter with simple is_trade_allowed logic
- KISS approach: basic market physics + time controls
"""

import os
import asyncio
from utils import log_text
from helper import is_trade_allowed  # Import the enhanced function

# Simple AlgoTest API wrapper (unchanged)
class SimpleAlgoTestAPI:
    def __init__(self, enabled=False):
        self.enabled = enabled
        
    async def send_signal(self, direction, strategy_type):
        """Send signal - simulation or live"""
        if not self.enabled:
            print(f"🎭 SIMULATION: {direction} {strategy_type} signal sent")
            return True
        
        # Import here to avoid dependency issues
        try:
            from algotest_api import AlgoTestAPI
            api = AlgoTestAPI(enabled=True)
            return await api.send_signal(direction, strategy_type)
        except ImportError:
            print(f"🎭 SIMULATION (AlgoTest not available): {direction} {strategy_type}")
            return True
        except Exception as e:
            print(f"❌ AlgoTest error: {e}")
            return False

class UnifiedTradeExecutor:
    """Simplified trade executor using market physics from helper.py"""
    
    # Class-level instances (simple singleton pattern)
    algotest_api = None
    
    @classmethod
    def _initialize(cls):
        """Lazy initialization"""
        if cls.algotest_api is None:
            # Control live trading with environment variable
            live_enabled = os.getenv("ALGOTEST_ENABLED", "false").lower() == "true"
            cls.algotest_api = SimpleAlgoTestAPI(enabled=live_enabled)
    
    @classmethod
    async def handle_prediction(cls, tick_data, prediction, confidence, quality, prediction_30=""):
        """
        Simplified entry point using market microstructure physics
        
        Args:
            tick_data: Dict with 'ts', 'ltp', etc.
            prediction: "UP" or "DOWN" 
            confidence: Float 0-1
            quality: Float 0-1
            prediction_30: Optional 30min prediction
        """
        try:
            cls._initialize()
            
            # === BASIC QUALITY FILTER ===
            # Minimum quality gates (very basic)
            if confidence < 0.75:  # Minimum confidence
                return
            
            #if quality < 0.65:  # Minimum quality
            #    return
            
            # === ENHANCED TRADE FILTER ===
            # Use market microstructure physics from helper.py
            try:
                # Extract features from tick_data (in real implementation, features come from feature_enricher)
                # For now, we'll use empty dict - in live_predictor.py, pass the actual features
                features = tick_data.get('features', {})
                
                should_trade, strategy_type = is_trade_allowed(
                    prediction=prediction,
                    prediction_30=prediction_30 or prediction,
                    confidence=confidence,
                    quality=quality,
                    timestamp=tick_data["ts"],
                    features=features
                )
            except Exception as e:
                log_text(f"⚠️ Trade filter error: {e}")
                return
            
            if not should_trade:
                return
                
            # === EXECUTE TRADE ===
            direction = "Long" if prediction == "UP" else "Short"
            
            try:
                success = await cls.algotest_api.send_signal(direction, strategy_type)
                
                if success:
                    mode = "LIVE" if cls.algotest_api.enabled else "SIM"
                    log_text(f"✅ {mode}: {prediction} @ ₹{tick_data['ltp']:.1f} | "
                            f"Conf: {confidence:.3f} | Qual: {quality:.3f}")
                    
            except Exception as e:
                log_text(f"❌ API execution error: {e}")
                
        except Exception as e:
            log_text(f"❌ Trade execution system error: {e}")

# Convenience function for external use (unchanged interface)
async def execute_if_qualified(tick_data, prediction, confidence, quality, prediction_30=""):
    """
    Convenience wrapper - can be imported directly
    
    NOTE: In live_predictor.py, you should pass features in tick_data:
    tick_data['features'] = features  # Add this line before calling
    await execute_if_qualified(tick_data, prediction, confidence, quality, prediction_30)
    """
    await UnifiedTradeExecutor.handle_prediction(tick_data, prediction, confidence, quality, prediction_30)

# For testing
if __name__ == "__main__":
    async def test():
        # Test with sample features
        test_tick = {
            "ts": 1234567890000000000,  # nanoseconds
            "ltp": 25000.0,
            "features": {
                "ltp_slope": 0.5,
                "momentum_score": 0.3,
                "agg_ratio": 0.2,
                "slope_90": 0.1,
                "ltq_bias_4": 0.1,
                "volume_trend_5": 0.1,
                "direction_streak": 2
            }
        }
        await execute_if_qualified(test_tick, "UP", 0.85, 0.80, "UP")
    
    asyncio.run(test())