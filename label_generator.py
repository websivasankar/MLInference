from dotenv import load_dotenv
load_dotenv()

import os, argparse, psycopg2, pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from sqlalchemy import create_engine

# ─── config ──────────────────────────────────────────────
PG_URL = os.getenv("PG_URL")
SEQ_WINDOW_TICKS = 180

# ─── helpers ─────────────────────────────────────────────
def parse_date(s):
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Date must be in YYYYMMDD or YYYY-MM-DD format")

def load_day(table: str) -> pd.DataFrame:
    sql = f"""
    SELECT ts, ts_batch, ltp, ltq, oi, volume
    FROM {table}
    ORDER BY ts ASC
    """
    engine = create_engine(PG_URL)
    return pd.read_sql(sql, engine)

def filter_pure_consensus_labels(df_merged):
    """
    Keep only logically consistent, high-quality label combinations.
    
    KEEPS:
    - Perfect Agreements: UP/UP, DOWN/DOWN
    - True FLAT: FLAT/FLAT with max movement ≤20 points in both timeframes
    - Trap-to-Reversal Cases: TRAP_UP/DOWN, TRAP_DOWN/UP
    
    REMOVES:
    - Mixed signals: UP/FLAT, DOWN/FLAT, FLAT/UP, FLAT/DOWN
    - Direct conflicts: UP/DOWN, DOWN/UP
    - Unresolved traps: TRAP_UP/FLAT, TRAP_DOWN/FLAT, TRAP_UP/UP, TRAP_DOWN/DOWN
    - High-volatility FLAT: FLAT/FLAT with movement >20 points in either timeframe
    """
    # Perfect Agreements (Both timeframes see same opportunity)
    perfect_consensus = (
        ((df_merged['label'] == 'UP') & (df_merged['label_30'] == 'UP')) |
        ((df_merged['label'] == 'DOWN') & (df_merged['label_30'] == 'DOWN')) |
        # FLAT/FLAT with movement constraint (both timeframes must stay within 20 points)
        ((df_merged['label'] == 'FLAT') & (df_merged['label_30'] == 'FLAT') &
         (df_merged['max_up'].abs() <= 20) & (df_merged['max_down'].abs() <= 20) &
         (df_merged['max_up_30'].abs() <= 35) & (df_merged['max_down_30'].abs() <= 35))
    )
    
    # Trap-to-Reversal Cases (15min trap resolves opposite in 30min)
    trap_cases = (
        # Bullish trap (15min) that resolves bearish (30min)
        ((df_merged['label'] == 'TRAP_UP') & (df_merged['label_30'] == 'DOWN')) |
        # Bearish trap (15min) that resolves bullish (30min)  
        ((df_merged['label'] == 'TRAP_DOWN') & (df_merged['label_30'] == 'UP'))
    )
    
    # Keep only pure signals
    pure_mask = perfect_consensus | trap_cases
    filtered_df = df_merged[pure_mask].copy()
    
    # Calculate removed categories for analysis
    excluded_df = df_merged[~pure_mask]
    excluded_count = len(excluded_df)
    
    print(f"📊 Pure Consensus Filtering:")
    print(f"   Before: {len(df_merged):4d} samples")
    print(f"   After:  {len(filtered_df):4d} samples")
    print(f"   Removed: {excluded_count:3d} mixed signals ({100*excluded_count/len(df_merged):.1f}%)")
    
    # Show kept combinations (clean signals)
    if len(filtered_df) > 0:
        kept_combinations = filtered_df.groupby(['label', 'label_30']).size().sort_values(ascending=False)
        print(f"   ✅ KEPT (Pure Signals):")
        for (label_15, label_30), count in kept_combinations.items():
            print(f"     {label_15:9} / {label_30:8} : {count:4d} samples")
    
    # Show removed combinations (mixed signals) 
    if excluded_count > 0:
        removed_combinations = excluded_df.groupby(['label', 'label_30']).size().sort_values(ascending=False)
        print(f"   ❌ REMOVED (Mixed/Junk Signals):")
        for (label_15, label_30), count in removed_combinations.items():
            print(f"     {label_15:9} / {label_30:8} : {count:4d} samples")
    
    # Quality assessment
    perfect_agreements_in_filtered = (
        ((filtered_df['label'] == 'UP') & (filtered_df['label_30'] == 'UP')) |
        ((filtered_df['label'] == 'DOWN') & (filtered_df['label_30'] == 'DOWN')) |
        # FLAT/FLAT with movement constraint
        ((filtered_df['label'] == 'FLAT') & (filtered_df['label_30'] == 'FLAT') &
         (filtered_df['max_up'].abs() <= 20) & (filtered_df['max_down'].abs() <= 20) &
         (filtered_df['max_up_30'].abs() <= 35) & (filtered_df['max_down_30'].abs() <= 35))
    )
    perfect_count = len(filtered_df[perfect_agreements_in_filtered])
    trap_reversal_count = len(filtered_df) - perfect_count
    
    print(f"   📊 Quality Breakdown:")
    print(f"     Perfect Agreements:  {perfect_count:4d} ({100*perfect_count/len(filtered_df):.1f}%)")
    print(f"     Trap-to-Reversals:   {trap_reversal_count:4d} ({100*trap_reversal_count/len(filtered_df):.1f}%)")
    
    return filtered_df

def time_aware_smart_sampling(df, max_consecutive=3, time_window_minutes=5):
    """
    Remove redundant identical samples within time windows.
    
    KISS Logic:
    - Same pattern within time window: keep only first N occurrences  
    - Same pattern after time window: always keep (new context)
    """
    if df.empty:
        return df
    
    # Simple setup
    feature_cols = ['label', 'max_up', 'max_down', 'label_30', 'max_up_30', 'max_down_30']
    time_window_ns = time_window_minutes * 60 * 1_000_000_000
    
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    keep_indices = [0]  # Always keep first row
    
    # Track state for current identical sequence
    consecutive_count = 1
    sequence_start_ts = df_sorted.iloc[0]['ts']
    
    # Single pass through data
    for i in range(1, len(df_sorted)):
        current_row = df_sorted.iloc[i]
        previous_row = df_sorted.iloc[i-1]
        
        # Check if features are identical
        is_identical = (current_row[feature_cols] == previous_row[feature_cols]).all()
        
        if is_identical:
            time_since_sequence_start = current_row['ts'] - sequence_start_ts
            
            if time_since_sequence_start <= time_window_ns:
                # Within time window - apply consecutive limit
                consecutive_count += 1
                if consecutive_count <= max_consecutive:
                    keep_indices.append(i)
            else:
                # Beyond time window - start new sequence
                keep_indices.append(i)
                consecutive_count = 1
                sequence_start_ts = current_row['ts']
        else:
            # Features changed - reset and keep
            keep_indices.append(i)
            consecutive_count = 1
            sequence_start_ts = current_row['ts']
    
    return df_sorted.iloc[keep_indices].copy()

def sampling_stats(df_before, df_after, time_window_minutes=5):
    """Calculate and display sampling statistics."""
    total_removed = len(df_before) - len(df_after)
    reduction_pct = 100 * total_removed / len(df_before) if len(df_before) > 0 else 0
    
    print(f"📊 Smart Sampling Results:")
    print(f"   Before: {len(df_before):4d} samples")
    print(f"   After:  {len(df_after):4d} samples")
    print(f"   Removed: {total_removed:3d} redundant ({reduction_pct:.1f}%)")
    
    if len(df_after) > 1:
        time_span_hours = (df_after['ts'].max() - df_after['ts'].min()) / 1e9 / 3600
        avg_interval = (df_after['ts'].max() - df_after['ts'].min()) / len(df_after) / 1e9
        print(f"   Time span: {time_span_hours:.1f}h, Avg interval: {avg_interval:.1f}s")

def apply_smart_sampling(df, max_consecutive=3, time_window_minutes=5, show_stats=True):
    """Apply sampling and optionally show stats."""
    df_sampled = time_aware_smart_sampling(df, max_consecutive, time_window_minutes)
    
    if show_stats:
        sampling_stats(df, df_sampled, time_window_minutes)
    
    return df_sampled

def synthesize_ticks(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"ts", "ltp", "ltq", "volume", "oi", "ts_batch"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    df = df.sort_values("ts")  # Ensure time order
    return (
        df.groupby("ts_batch")
          .agg({
              "ltp": "last",       # Final LTP in batch
              "ts": "last",        # Final tick's timestamp
              "ltq": lambda x: x.abs().sum(),        # Total traded qty in 5s
              "volume": "sum",     # Total volume
              "oi": "last"         # Last OI snapshot
          })
          .reset_index()
          .rename(columns={"ts": "ts_anchor"})
    )

def rolling_label_improved(df: pd.DataFrame, synth_df: pd.DataFrame, forward_ns: int, abs_thr: float) -> pd.DataFrame:
    """
    Improved labeling with consistent synthetic tick windows:
    1. Both backward and forward use 180 synthetic ticks within 20-minute boundaries
    2. Accounts for real market gaps in synthetic tick generation
    3. Partial trap detection (60% progress toward TP, then SL)
    4. Path quality check (adverse movement < 50% of SL)
    5. Clean directional moves only for UP/DOWN labels
    """
    ts_arr = synth_df["ts_anchor"].values
    ltp_arr = synth_df["ltp"].values
    labels = []
    valid_ts = []
    max_ups = []
    max_downs = []
    
    # Define stop levels based on timeframe
    if forward_ns == 900_000_000_000:  # 15 min
        sl_pts = 20
        tp_pts = abs_thr  # e.g., 25
    else:  # 30 min
        sl_pts = 20  # Larger stop for longer timeframe
        tp_pts = abs_thr  # e.g., 50
    
    for i in range(len(synth_df)):
        if i < SEQ_WINDOW_TICKS:
            continue
            
        t0 = ts_arr[i]
        ltp0 = ltp_arr[i]
        
        # Backward quality check (synthetic ticks)
        back_ts = ts_arr[i - SEQ_WINDOW_TICKS:i]
        if len(back_ts) < SEQ_WINDOW_TICKS:
            continue
            
        # Time boundary check: 180 backward synthetic ticks should not span more than 20 minutes
        back_time_span = (back_ts[-1] - back_ts[0]) / 1e9  # Convert to seconds
        max_allowed_span = 1200  # 20 minutes in seconds
        
        if back_time_span > max_allowed_span:
            continue  # Backward window spans too long (>20 minutes)
            
        back_diff = np.diff(back_ts) / 1e9
        # Relaxed timing for synthetic ticks (5±2 second gaps expected)
        if np.median(back_diff) > 8 or np.std(back_diff) > 4:
            continue
            
        # Forward window extraction - using SYNTHETIC ticks for consistency
        forward_synth_candidates = synth_df[synth_df["ts_anchor"] > t0]
        if len(forward_synth_candidates) < SEQ_WINDOW_TICKS:
            continue  # Not enough forward synthetic ticks
            
        forward_synth = forward_synth_candidates.head(SEQ_WINDOW_TICKS)
        forward_ltp = forward_synth["ltp"].values       # Synthetic LTP values
        ts_forward = forward_synth["ts_anchor"].values  # Synthetic timestamps
        
        # Time boundary check: 180 synthetic ticks should not span more than 20 minutes
        forward_time_span = (ts_forward[-1] - ts_forward[0]) / 1e9  # Convert to seconds
        max_allowed_span = 1200  # 20 minutes in seconds
        
        if forward_time_span > max_allowed_span:
            continue  # Forward window spans too long (>20 minutes)
            
        # Forward quality check (synthetic tick timing)
        fut_diff = np.diff(ts_forward) / 1e9
        # Relaxed timing for synthetic ticks (5±2 second gaps expected)
        if np.median(fut_diff) > 8 or np.std(fut_diff) > 4:
            continue

        # Calculate opportunity levels
        long_tp = ltp0 + tp_pts      # e.g., 25000 + 25 = 25025
        long_sl = ltp0 - sl_pts      # e.g., 25000 - 20 = 24980
        short_tp = ltp0 - tp_pts     # e.g., 25000 - 25 = 24975  
        short_sl = ltp0 + sl_pts     # e.g., 25000 + 20 = 25020

        # Track movement and path quality
        first_target_hit = None
        sl_hit_after_tp = False
        max_up = 0.0
        max_down = 0.0
        
        # Thresholds for quality checks (relaxed adverse movement)
        partial_trap_threshold = tp_pts * 0.5  # 60% progress toward TP
        adverse_threshold = sl_pts * 0.5       # 50% of SL movement (relaxed from 30%)

        # Path analysis using synthetic tick LTP values
        for ltp in forward_ltp:
            move = ltp - ltp0
            max_up = max(max_up, move)
            max_down = min(max_down, move)
            
            # Check for FIRST target hit
            if first_target_hit is None:
                if ltp >= long_tp:
                    first_target_hit = "UP"
                elif ltp <= short_tp:
                    first_target_hit = "DOWN"
                # NEW: Check for partial trap (significant progress toward TP, then SL)
                elif ltp <= long_sl and max_up >= partial_trap_threshold:
                    first_target_hit = "TRAP_UP"  # Went 60% toward TP, then hit SL
                    break
                elif ltp >= short_sl and abs(max_down) >= partial_trap_threshold:
                    first_target_hit = "TRAP_DOWN"  # Went 60% toward TP, then hit SL
                    break
            
            # After first target, check for reversal to SL
            elif first_target_hit == "UP":
                if ltp <= long_sl:
                    sl_hit_after_tp = True  # UP target hit first, now SL hit = TRAP_UP
            elif first_target_hit == "DOWN":
                if ltp >= short_sl:
                    sl_hit_after_tp = True  # DOWN target hit first, now SL hit = TRAP_DOWN

        # Final labeling based on first move, reversal, and path quality
        if forward_ns == 900_000_000_000:  # 15 min - include TRAP labels
            if first_target_hit and sl_hit_after_tp:
                label = "TRAP_UP" if first_target_hit == "UP" else "TRAP_DOWN"
            elif first_target_hit == "UP":
                # Check path quality - downward excursion shouldn't exceed threshold
                if abs(max_down) <= adverse_threshold:
                    label = "UP"      # Clean upward path
                else:
                    label = "FLAT"    # TP hit but path was too messy
            elif first_target_hit == "DOWN":
                # Check path quality - upward excursion shouldn't exceed threshold  
                if max_up <= adverse_threshold:
                    label = "DOWN"    # Clean downward path
                else:
                    label = "FLAT"    # TP hit but path was too messy
            elif first_target_hit == "TRAP_UP":
                label = "TRAP_UP"     # Partial move up, then SL
            elif first_target_hit == "TRAP_DOWN":
                label = "TRAP_DOWN"   # Partial move down, then SL
            else:
                label = "FLAT"        # No clear opportunity in either direction
        else:  # 30 min - simplified, no TRAP labels but keep path quality
            if first_target_hit == "UP":
                # Check path quality for 30min too
                if abs(max_down) <= adverse_threshold:
                    label = "UP"      # Clean upward path
                else:
                    label = "FLAT"    # TP hit but path was messy
            elif first_target_hit == "DOWN":
                # Check path quality for 30min too
                if max_up <= adverse_threshold:
                    label = "DOWN"    # Clean downward path  
                else:
                    label = "FLAT"    # TP hit but path was messy
            else:
                label = "FLAT"        # No clear opportunity

        labels.append(label)
        valid_ts.append(t0)
        max_ups.append(round(max_up))
        max_downs.append(round(max_down))
    
    return pd.DataFrame({
        "ts": valid_ts,
        "label": labels,
        "max_up": max_ups,
        "max_down": max_downs
    })

# ─── main ────────────────────────────────────────────────
def process_day(for_date, abs_thr_15, abs_thr_30, output_path):
    suffix = for_date.strftime("%Y%m%d")
    table  = f"nifty_fut_ticks_{suffix}"
    df     = load_day(table)

    if df.empty:
        print(f" no ticks for {suffix}")
        return
    synth_df = synthesize_ticks(df)

    # Use improved labeling function
    label_15_df = rolling_label_improved(df, synth_df, forward_ns=900_000_000_000, abs_thr=abs_thr_15)
    label_30_df = rolling_label_improved(df, synth_df, forward_ns=1_800_000_000_000, abs_thr=abs_thr_30)

    df_merged = pd.merge(label_15_df, label_30_df, on="ts", how="inner", suffixes=("", "_30"))

    if df_merged.empty:
        print("No valid labels found")
        return

    # Apply consensus filtering
    df_consensus = filter_pure_consensus_labels(df_merged)

    if df_consensus.empty:
        print("❌ No pure consensus labels found after filtering")
        return

    # NEW: Apply time-aware smart sampling to reduce redundancy
    df_final = apply_smart_sampling(
    df_consensus, 
    time_window_minutes=5  # Add this to use 5 minutes
    )

    if df_final.empty:
        print("❌ No samples remaining after smart sampling")
        return

    # Save final results
    out = output_path or f"labels/labels_{suffix}.csv"
    dirn = os.path.dirname(out)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    df_final.to_csv(out, index=False)
    print(f"💾 Final labels saved → {out}  (rows={len(df_final)})")
    
    # Summary of actionable signals
    actionable_count = len(df_final)  # All remaining are actionable since we filtered out mixed signals
    print(f"📈 Total actionable signals: {actionable_count}")
    print("-" * 60)

# ─── CLI ─────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date",
                   type=parse_date,
                   default=datetime.today(),
                   help="trading date (YYYY-MM-DD)")
    p.add_argument("--abs15", type=float, required=True,
                   help="absolute TP threshold in ₹ for 15-min label")
    p.add_argument("--abs30", type=float, required=True,
                   help="absolute TP threshold in ₹ for 30-min label")
    p.add_argument("--output",
                   dest="output_path",
                   help="path to write the labels CSV")
    args = p.parse_args()

    process_day(
        args.date,
        abs_thr_15=args.abs15,
        abs_thr_30=args.abs30,
        output_path=args.output_path
    )