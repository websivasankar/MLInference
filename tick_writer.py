# tick_writer.py - Fixed version addressing the 0 rows issue

import threading
import time
import queue
import pandas as pd
from sqlalchemy import create_engine
from utils import log_text
import copy

class TickWriter:
    def __init__(self, pg_url: str, table="nifty_fut_ticks"):
        self.engine = create_engine(pg_url)
        self.table = table
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def write_tick(self, row: dict):
        self.q.put(("single", row))

    def write_5s_batch(self, ts_ns: int, tick_rows: list):
        #log_text(f"[DEBUG] write_5s_batch() rows={len(tick_rows)} at ts_batch={ts_ns}")
        if not tick_rows:
            return
        self.q.put(("batch", ts_ns, copy.deepcopy(tick_rows)))  # ✅ fixes mutation

    def _run_loop(self):
        while True:
            item = self.q.get()
            if not item:
                continue

            if item[0] == "single":
                _, row = item
                self._insert_to_db([row])

            elif item[0] == "batch":
                _, _, rows = item
                self._insert_to_db(rows)

    # Replace your current _insert_to_db method with this exact code:

    def _insert_to_db(self, rows: list, max_retries=3):
        if not rows:
            log_text("ℹ️ _insert_to_db early return: empty input rows")
            return

        expected_keys = [
            "ts", "ltp", "ltq", "oi", "volume",
            "bid_price_0", "bid_qty_0",
            "bid_price_1", "bid_qty_1",
            "bid_price_2", "bid_qty_2",
            "ask_price_0", "ask_qty_0",
            "ask_price_1", "ask_qty_1",
            "ask_price_2", "ask_qty_2",
            "ts_batch"
        ]

        normalized = []
        for i, r in enumerate(rows):
            if "ts" not in r or r["ts"] is None or r["ts"] == 0:
                log_text(f"❌ Row {i} has invalid/missing 'ts': {r}")
                continue

            try:
                ts_val = int(r["ts"])
            except (ValueError, TypeError):
                log_text(f"❌ Row {i}: Invalid timestamp format: {r['ts']}")
                continue

            # No rejection for missing keys — fallback fill is enough
            norm_row = {k: r.get(k, 0) for k in expected_keys}
            norm_row["ts"] = ts_val
            normalized.append(norm_row)

        #log_text(f"🔍 DEBUG: Normalized {len(normalized)} rows from {len(rows)} input rows")

        if not normalized:
            log_text("⚠️ _insert_to_db: all input rows skipped after normalization")
            return

        df = pd.DataFrame(normalized)
        try:
            df = df.astype({
                "ts": "int64", "ltp": "float64", "ltq": "int32",
                "oi": "int64", "volume": "int64",
                "bid_price_0": "float64", "bid_qty_0": "int32",
                "bid_price_1": "float64", "bid_qty_1": "int32",
                "bid_price_2": "float64", "bid_qty_2": "int32",
                "ask_price_0": "float64", "ask_qty_0": "int32",
                "ask_price_1": "float64", "ask_qty_1": "int32",
                "ask_price_2": "float64", "ask_qty_2": "int32",
                "ts_batch": "int64"
            })
        except Exception as e:
            log_text(f"❌ _insert_to_db: dtype coercion failed — {e}")
            return

        if df.isnull().any().any():
            log_text("⚠️ _insert_to_db: null values detected, filling with 0")
            df = df.fillna(0)

        try:
            df.to_sql(
                self.table,
                con=self.engine,
                if_exists="append",
                index=False,
                method="multi"
            )
            #log_text(f"✅ Inserted {len(df)} ticks into {self.table}")
        except Exception as e:
            log_text(f"❌ _insert_to_db: insert failed — {e}")
            if len(df) > 0:
                log_text(f"🔍 Sample row: {df.iloc[0].to_dict()}")

    def shutdown(self):
        """Graceful shutdown"""
        self.q.put(None)  # Signal thread to stop
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            log_text("⚠️ TickWriter thread did not shut down gracefully")
