from dataclasses import dataclass
from collections import deque

@dataclass
class InventorySnapshot:
    ts: int
    inv_long_short: int
    inv_abs: int

class InventoryTracker:
    def __init__(self, maxlen: int = 3600):
        self._ltqs = deque(maxlen=maxlen)  # Store signed LTQs only

    def update(self, row: dict):
        qty = int(row.get("ltq", 0))
        if qty != 0:
            self._ltqs.append(qty)
        elif row.get("ltp") and row.get("bid_price_0") and row.get("ask_price_0"):
            # add dummy signed LTQ based on execution side
            if row["ltp"] >= row["ask_price_0"]:
                self._ltqs.append(1)
            elif row["ltp"] <= row["bid_price_0"]:
                self._ltqs.append(-1)
                
    def snapshot(self, ts_ns: int) -> InventorySnapshot:
        buys = sum(q for q in self._ltqs if q > 0)
        sells = sum(-q for q in self._ltqs if q < 0)
        return InventorySnapshot(
            ts=ts_ns,
            inv_long_short=buys - sells,
            inv_abs=buys + sells,
        )
