# ml_tick_trainer/flatten_tick.py

import json
def flatten_tick(ts_ns, tick):
    row = {
        "ts": ts_ns,
        "ltp": float(tick.get("LTP", 0) or 0),
        "ltq": int(tick.get("LTQ", 0) or 0),
        "oi": int(tick.get("OI", 0) or 0),
        "volume": int(tick.get("volume", 0) or 0)
    }

    # Ensure all depth levels 0, 1, 2 are always present
    for i in range(3):
        row[f"bid_price_{i}"] = 0.0
        row[f"bid_qty_{i}"] = 0
        row[f"ask_price_{i}"] = 0.0
        row[f"ask_qty_{i}"] = 0

    depth = tick.get("depth", [])
    for i in range(min(3, len(depth))):
        lvl = depth[i]
        row[f"bid_price_{i}"] = float(lvl.get("bid_price", 0) or 0)
        row[f"bid_qty_{i}"] = int(lvl.get("bid_quantity", 0) or 0)
        row[f"ask_price_{i}"] = float(lvl.get("ask_price", 0) or 0)
        row[f"ask_qty_{i}"] = int(lvl.get("ask_quantity", 0) or 0)

    return row

# Add at end of flatten_tick.py

def build_synthetic_tick(ts_ns: int, ticks: list) -> dict:
    """
    Create a synthetic tick from a batch of raw ticks.
    Uses last tick's LTP/depth, sums LTQ/volume, sets ts = ts_ns.
    """
    if not ticks:
        return {}

    latest = ticks[-1].copy()
    latest["ts"] = ts_ns
    latest["ltq"] = sum(t.get("ltq", 0) for t in ticks)
    latest["volume"] = sum(t.get("volume", 0) for t in ticks)
    return latest

def build_synthetic_tick_signed(ts_ns: int, signed_ticks: list) -> dict:
    """
    Create a synthetic tick from signed raw ticks, preserving directional LTQ.
    Uses last tick's LTP/depth, sums signed LTQ/volume, sets ts = ts_ns.
    
    Args:
        ts_ns: Target timestamp in nanoseconds
        signed_ticks: List of ticks with signed LTQ values
    
    Returns:
        Synthetic tick with aggregated signed LTQ
    """
    if not signed_ticks:
        return {}

    latest = signed_ticks[-1].copy()
    latest["ts"] = ts_ns
    latest["ltq"] = sum(t.get("ltq", 0) for t in signed_ticks)  # Sum preserves direction
    latest["volume"] = sum(t.get("volume", 0) for t in signed_ticks)
    return latest