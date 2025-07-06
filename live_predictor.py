# live_predictor.py (Dual-model version)

from dotenv import load_dotenv
import os
load_dotenv()

import asyncio
import time
import joblib
import pandas as pd
from collections import deque
from datetime import datetime
from dhan_api_connector import connect
from async_market_feed import AsyncMarketFeed
from flatten_tick import flatten_tick, build_synthetic_tick, build_synthetic_tick_signed
from feature_enricher import enrich_features, ORDERED_KEYS
from tick_writer import TickWriter
from prediction_writer import PredictionWriter
from inventory_tracker import InventoryTracker
import numpy as np
from utils                import log_text, tag_signed_ltq
from helper import wait_until_capture_start
from unified_trade_executor import execute_if_qualified
import traceback
# ────────── CONFIG ──────────
raw_pg_url = os.getenv("PG_URL", "").strip()
pg_url = raw_pg_url if raw_pg_url else "postgresql+psycopg2://siva:securepass@localhost:5432/bsengine"

# Dynamic model paths based on feature configuration
try:
    from feature_config import USE_SHAP_OPTIMIZED, get_feature_set_name
    suffix = "_shap" if USE_SHAP_OPTIMIZED else ""
    print(f"🎯 Live predictor using {get_feature_set_name()} models")
except ImportError:
    suffix = ""
    print("🎯 Live predictor using legacy models")

MODEL_PATH_15 = f"models/lgbm_label15{suffix}.pkl"
MODEL_PATH_30 = f"models/lgbm_label30{suffix}.pkl"
MODEL_PATH_UP = f"models/up{suffix}.pkl"
MODEL_PATH_DOWN = f"models/down{suffix}.pkl"
WRITE_INTERVAL = 5_000_000_000   # 5 s (ns)
BUFFER_SIZE    = 180             # 180×5s = 15m window

LABEL_MAP_15 = {0: "DOWN", 1: "FLAT", 2: "UP", 3: "TRAP_UP", 4: "TRAP_DOWN"}
LABEL_MAP_30 = {0: "DOWN", 1: "FLAT", 2: "UP"}

# ────────── MAIN ──────────
async def live_predict():

    try:
        model_15 = joblib.load(MODEL_PATH_15)
        model_30 = joblib.load(MODEL_PATH_30)
        model_up = joblib.load(MODEL_PATH_UP)
        model_down = joblib.load(MODEL_PATH_DOWN)
    except Exception as e:
        log_text(f"🔥 Model load failed: {e}\n{traceback.format_exc()}")
        return
 
    dhan_ctx, _ = connect()
    tokens = [53216] #NiftyFuture ID AUG:64103
    instruments = [(AsyncMarketFeed.NSE_FNO, str(t), AsyncMarketFeed.Full) for t in tokens]
    mf = AsyncMarketFeed(dhan_ctx, instruments, version="v2")
    mf.start()

    raw_writer = TickWriter(pg_url)
    pred_writer = PredictionWriter(pg_url)

    window = deque(maxlen=BUFFER_SIZE)
    window_signed = deque(maxlen=BUFFER_SIZE)
    inv = InventoryTracker(maxlen=3600)

    tick_buffer_raw = []
    last_write_ts = time.time_ns() #handle 5 s interval
    last_tick_ts = None #handle tick gaps

    try:
        while True:
            ts_ns = time.time_ns()                 
            try:
                tick = await mf.get_data_async(timeout=1.0)
            except asyncio.TimeoutError:
                # ← Add timeout gap detection HERE before continue
                log_text(f"⏰ Dhan timeout at {datetime.now()}")
                if last_tick_ts and ts_ns - last_tick_ts > 15_000_000_000:
                    notick = datetime.fromtimestamp(ts_ns / 1e9)
                    log_text(f"🔴 TIMEOUT GAP: No tick for {(ts_ns - last_tick_ts)/1e9:.1f}s at {notick}")
                continue
            except Exception as e:
                log_text(f"🔥 Market feed fetch failed: {e}\n{traceback.format_exc()}")
                continue
            
            if tick is None:
                # ← Add null tick gap detection HERE too
                if last_tick_ts and ts_ns - last_tick_ts > 15_000_000_000:
                    log_text(f"🔴 NULL TICK GAP: No data for {(ts_ns - last_tick_ts)/1e9:.1f}s")
                continue

            # log if synthetic tick is not received for more than 15 seconds
            if last_tick_ts and ts_ns - last_tick_ts > 15_000_000_000:
                notick = datetime.fromtimestamp(ts_ns / 1e9)
                log_text(f"No tick received for >15s at {notick}")
            # log if no tick received for more than 60 seconds    
            if last_tick_ts and ts_ns - last_tick_ts > 60_000_000_000:
                window.clear()
                window_signed.clear()
                tick_buffer_raw.clear()   
                log_text(f"Cleared buffer after long gap at {datetime.fromtimestamp(ts_ns / 1e9)}")
            
            last_tick_ts = ts_ns 
            row = flatten_tick(ts_ns, tick)
            if row["ltp"] == 0 or row["volume"] == 0 or row["ltq"] == 0:
                log_text(f"🔍 Dropped stale tick: LTP={row['ltp']}, VOL={row['volume']}, LTQ={row['ltq']}")
                continue

            row_for_signing = row.copy()  # ← ADD THIS
            row_signed = tag_signed_ltq([row_for_signing])[0]
            inv.update(row_signed)            

            # Store raw tick (unsigned)
            tick_buffer_raw.append(row)            

            if ts_ns - last_write_ts >= WRITE_INTERVAL:
                ts_batch = ts_ns                        
                # Create unsigned synthetic tick (existing)
                synthetic_row = build_synthetic_tick(ts_ns, tick_buffer_raw)
                synthetic_row["ts"] = ts_ns
                synthetic_row["ts_batch"] = ts_batch

                # Create signed synthetic tick (NEW)
                signed_raw_buffer = tag_signed_ltq(tick_buffer_raw.copy())
                synthetic_signed = build_synthetic_tick_signed(ts_ns, signed_raw_buffer)
                synthetic_signed["ts"] = ts_ns
                synthetic_signed["ts_batch"] = ts_batch

                if synthetic_row and synthetic_signed:
                    # Both created successfully - add to buffers atomically
                    window.append(synthetic_row)
                    window_signed.append(synthetic_signed)
                    
                    # Verify buffers stay in sync
                    if len(window) != len(window_signed):
                        log_text(f"🔥 CRITICAL: Buffer desync detected! window:{len(window)} signed:{len(window_signed)}")
                        # Recovery: clear both buffers and restart
                        window.clear()
                        window_signed.clear()
                        continue
                        
                elif synthetic_row or synthetic_signed:
                    # Only one created - log error and skip this cycle
                    log_text(f"⚠️ Synthetic tick creation failed: row={bool(synthetic_row)} signed={bool(synthetic_signed)}")
                    # Don't add anything to buffers - keep them in sync
                    
                else:
                    # Both failed - log and continue
                    log_text("⚠️ Both synthetic ticks failed to create")

                inv_snap = inv.snapshot(ts_ns)
                
                if len(window) == BUFFER_SIZE:
                    features = enrich_features(list(window), strict=True, inv_snap=inv_snap, signed_buffer=list(window_signed))
                    # Always remove "ltp_zone_tag" from features and add one-hot encoding in fixed order
                    ordered_values = [features[k] for k in ORDERED_KEYS]
                    feats = ordered_values
                    df_feat = pd.DataFrame([feats])           
                    probs_15 = model_15.predict_proba(df_feat)[0]
                    probs_30 = model_30.predict_proba(df_feat)[0]
                    idx_15 = int(np.argmax(probs_15))
                    idx_30 = int(np.argmax(probs_30))

                    label_15 = LABEL_MAP_15[idx_15]
                    label_30 = LABEL_MAP_30[idx_30]
                    prob_15 = float(probs_15[idx_15])
                    prob_30 = float(probs_30[idx_30])

                    quality = 0                  
                    if label_15 == "UP":
                        prob_up = model_up.predict_proba(df_feat)[0][1]
                        quality = prob_up                   
                    elif label_15 == "DOWN":
                        prob_down = model_down.predict_proba(df_feat)[0][1]
                        quality = prob_down
       
                    # Save both predictions                    
                    try:
                        # Write to prediction table
                        pred_writer.write(synthetic_row, label_15, prob_15, label_30, prob_30, quality)
                    except Exception as e:
                        log_text(f"⚠️ Prediction write failed: {e}\n{traceback.format_exc()}")                                 
                    # SAFE: Trade execution won't crash prediction system
                    try:
                        synthetic_row['features'] = features
                        await execute_if_qualified(synthetic_row, label_15, prob_15, quality, label_30)
                    except Exception as e:
                        log_text(f"⚠️ Trade execution  failed: {e}\n{traceback.format_exc()}")

                batch_to_write = tick_buffer_raw.copy()
                for r in batch_to_write:
                    r["ts_batch"] = ts_batch

                try:
                    raw_writer.write_5s_batch(ts_batch, batch_to_write)
                except Exception as e:
                    log_text(f"⚠️ Raw tick writing: {e}\n{traceback.format_exc()}")
                tick_buffer_raw.clear()
                last_write_ts = ts_ns 

                await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        log_text("Cancelled Error.")
        raise  
    except Exception as e:
        log_text(f"🔥 Market Feed : {e}\n{traceback.format_exc()}")
        raise  
    finally:
        await mf.stop()
        raw_writer.shutdown()
        pred_writer.shutdown()    

if __name__ == "__main__":
    try:
        asyncio.run(wait_until_capture_start())
        asyncio.run(live_predict())
    except KeyboardInterrupt:
        log_text("Graceful shutdown.")
    except Exception as e:
        log_text(f"Startup failed: {e}")
