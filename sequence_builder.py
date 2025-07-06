# sequence_builder.py
"""
Build a dataset from one *or many* label CSV files.

Vector order:
  1. 180 RAW_COLUMNS  (oldest → newest tick)
  2. ORDERED_KEYS     (aggregated features from the full window)
  3. "label", "label_30"
"""

from dotenv import load_dotenv
load_dotenv()

import os, argparse, psycopg2, pandas as pd
from urllib.parse import urlparse
from feature_enricher import enrich_features, ORDERED_KEYS
import time
from inventory_tracker import InventoryTracker
from utils import tag_signed_ltq
from flatten_tick import build_synthetic_tick_signed
import psycopg2.extras
# ──────── config ────────
PG_URL = os.getenv("PG_URL")

# ──────── helpers ────────

def pg_conn_dict(url):
    u = urlparse(url)
    return dict(dbname=u.path.lstrip("/"), user=u.username, password=u.password,
                host=u.hostname, port=u.port)

def fetch_synthetic_window(cur, ts_batch_anchor, num_ticks=180):
    cur.execute(
        f"""
        WITH latest_per_batch AS (
            SELECT DISTINCT ON (ts_batch)
                ts_batch, ts, ltp,
                bid_price_0, bid_qty_0,
                bid_price_1, bid_qty_1,
                bid_price_2, bid_qty_2,
                ask_price_0, ask_qty_0,
                ask_price_1, ask_qty_1,
                ask_price_2, ask_qty_2,
                oi
            FROM {TICK_TABLE}
            WHERE ts_batch IS NOT NULL AND ts_batch < %s
            ORDER BY ts_batch DESC, ts DESC
            LIMIT %s
        ),
        ltq_volume_sums AS (
            SELECT ts_batch,
                   SUM(ltq) AS ltq,
                   SUM(volume) AS volume
            FROM {TICK_TABLE}
            WHERE ts_batch IS NOT NULL AND ts_batch < %s
            GROUP BY ts_batch
        )
        SELECT l.ts_batch, l.ts, l.ltp,
               l.bid_price_0, l.bid_qty_0,
               l.bid_price_1, l.bid_qty_1,
               l.bid_price_2, l.bid_qty_2,
               l.ask_price_0, l.ask_qty_0,
               l.ask_price_1, l.ask_qty_1,
               l.ask_price_2, l.ask_qty_2,
               l.oi,
               s.ltq, s.volume
        FROM latest_per_batch l
        JOIN ltq_volume_sums s ON l.ts_batch = s.ts_batch
        ORDER BY l.ts_batch ASC
        """,
        (ts_batch_anchor, num_ticks, ts_batch_anchor)
    )

    rows = cur.fetchall()
    if len(rows) < num_ticks:
        return None

    columns = [
        "ts_batch", "ts", "ltp",
        "bid_price_0", "bid_qty_0",
        "bid_price_1", "bid_qty_1",
        "bid_price_2", "bid_qty_2",
        "ask_price_0", "ask_qty_0",
        "ask_price_1", "ask_qty_1",
        "ask_price_2", "ask_qty_2",
        "oi", "ltq", "volume"  # volume is just for reference
    ]
    return pd.DataFrame(rows, columns=columns)

def fetch_raw_inventory_ticks(cur, start_ts, end_ts):
    cur.execute(
        """
        SELECT *
        FROM nifty_fut_ticks
        WHERE ts > %s AND ts <= %s
        ORDER BY ts ASC
        """,
        (start_ts, end_ts)
    )
    return [dict(row) for row in cur.fetchall()]

# ──────── main build ────────

def build_dataset(label_paths, output_csv, num_ticks=180):
    conn = psycopg2.connect(**pg_conn_dict(PG_URL))
    cur  = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    samples = []
    for path in label_paths:
        inv = InventoryTracker(maxlen=3600)
        print("reading", path)
        global TICK_TABLE, PRED_TABLE
        basename = os.path.basename(path)
        datepart = basename.split("_")[-1].split(".")[0]
        TICK_TABLE = f"nifty_fut_ticks_{datepart}"
        #PRED_TABLE = f"ml_predictions_{datepart}"        
        
        for ts_batch, label, label30 in pd.read_csv(path)[["ts", "label", "label_30"]].itertuples(index=False):
            # Fetch raw ticks for this ts_batch (5-second span)

            raw_window = fetch_raw_inventory_ticks(cur, ts_batch - 900_000_000_000, ts_batch)
            tagged_window = tag_signed_ltq(raw_window)
            for row in tagged_window:
                inv.update(row)

            win_df = fetch_synthetic_window(cur, ts_batch, num_ticks) #latest to oldest time order

            if win_df is None:
                continue
            records = win_df.to_dict("records")

            #Group tagged_window by ts_batch and create signed synthetic records
            from collections import defaultdict
            signed_raw_by_batch = defaultdict(list)
            
            # Reuse already signed tagged_window - no need to fetch again!
            for raw_tick in tagged_window:
                batch_ts = raw_tick.get("ts_batch")
                if batch_ts:
                    signed_raw_by_batch[batch_ts].append(raw_tick)

            # Create signed synthetic records from grouped signed raw ticks
            signed_records = []
            for record in records:
                record_ts_batch = record.get("ts_batch")
                # Build signed synthetic tick from grouped raw signed ticks
                signed_synthetic = build_synthetic_tick_signed(
                    record_ts_batch, 
                    signed_raw_by_batch[record_ts_batch]
                )
                signed_records.append(signed_synthetic)

            # ✅ STEP 5: Feature extraction with both buffers
            agg = enrich_features(
                records,  # unsigned synthetic records
                strict=True, 
                inv_snap=inv.snapshot(ts_batch),
                signed_buffer=signed_records  # signed synthetic records from raw
            )

            ordered_values = [agg[k] for k in ORDERED_KEYS]
            vec = ordered_values + [label, label30]

            samples.append(vec)

    cur.close(); conn.close()
    cols = ORDERED_KEYS + ["label", "label_30"]

    df_out = pd.DataFrame(samples, columns=cols)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"dataset saved → {output_csv}  (rows={len(df_out)})")

# ──────── CLI ────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", required=True,
                   help="One or more label CSV files")
    p.add_argument("--output", required=True,
                   help="Output dataset CSV path")
    p.add_argument("--num_ticks", type=int, default=180)
    args = p.parse_args()

    build_dataset(args.labels, args.output, args.num_ticks)
