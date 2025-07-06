-- create_table.sql (extended for dual-model predictions)

-- Step 1: Drop old table (if exists)
-- Drop old table
DROP TABLE IF EXISTS nifty_fut_ticks CASCADE;

-- Updated tick schema (removed: open_price, close_price, high_price, low_price, avg_price)
CREATE TABLE nifty_fut_ticks (
    ts  BIGINT PRIMARY KEY NOT NULL,
    ltp FLOAT,
    ltq INT,
    oi BIGINT,
    volume BIGINT,
    
    bid_price_0 FLOAT,
    bid_qty_0 INT,
    bid_price_1 FLOAT,
    bid_qty_1 INT,
    bid_price_2 FLOAT,
    bid_qty_2 INT,
    
    ask_price_0 FLOAT,
    ask_qty_0 INT,
    ask_price_1 FLOAT,
    ask_qty_1 INT,
    ask_price_2 FLOAT,
    ask_qty_2 INT
) PARTITION BY RANGE (ts);

-- Drop and recreate prediction table
DROP TABLE IF EXISTS ml_predictions CASCADE;

CREATE TABLE ml_predictions (
    ts BIGINT NOT NULL,
    prediction TEXT NOT NULL,       -- 15-minute prediction
    confidence FLOAT,
    prediction_30 TEXT,             -- 30-minute prediction
    confidence_30 FLOAT,
    ltp FLOAT,
    ltq INT,
    oi BIGINT,
    quality FLOAT, 
    signal_id TEXT,
    fired BOOLEAN DEFAULT FALSE,
    fired_by_executor BOOLEAN DEFAULT FALSE,     -- ✅ NEW
    trade_ts BIGINT,
    outcome TEXT,
    ltp_post_15s FLOAT,
    execution_comment TEXT                        -- ✅ NEW
) PARTITION BY RANGE (ts);
