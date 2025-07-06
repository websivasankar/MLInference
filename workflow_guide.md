# 🔄 Complete Workflow Guide

This guide walks you through the entire process from setup to live trading.

## 🚀 Phase 1: Initial Setup

### 1.1 Environment Setup
```bash
# Clone and setup
git clone https://github.com/websivasankar/MLInference
cd MLInference

# Automated setup (recommended)
python setup.py

# Manual setup (alternative)
pip install -r requirements.txt
createdb bsengine
psql -d bsengine -f create_table.sql
```

### 1.2 Configuration
```bash
# Configure database connection
echo 'PG_URL="postgresql://user:pass@localhost:5432/bsengine"' > .env

# Configure Dhan API credentials  
cat > ws.json << EOF
{
    "client_id": "your_dhan_client_id",
    "access_token": "your_dhan_access_token"
}
EOF
```

### 1.3 Verification
```bash
# Test setup
python -c "
import pandas as pd
import lightgbm as lgb
from feature_enricher import enrich_features
print('✅ Setup successful!')
"
```

## 📊 Phase 2: Data Collection & Labeling

### 2.1 Historical Data Requirements
You need historical tick data stored in PostgreSQL tables:
```sql
-- Table format: nifty_fut_ticks_YYYYMMDD
-- Required columns: ts, ltp, ltq, oi, volume, bid_price_0, ask_price_0, etc.
```

### 2.2 Generate Training Labels
```bash
# Single day labeling
python label_generator.py \
    --date 2024-01-15 \
    --abs15 25 \
    --abs30 50 \
    --output labels/labels_20240115.csv

# Multiple days (batch processing)
for date in 2024-01-15 2024-01-16 2024-01-17; do
    python label_generator.py \
        --date $date \
        --abs15 25 \
        --abs30 50 \
        --output labels/labels_${date//-/}.csv
done
```

**Label Parameters:**
- `--abs15 25`: 15-minute profit target = 25 points
- `--abs30 50`: 30-minute profit target = 50 points
- Adjust based on market volatility and trading strategy

### 2.3 Label Quality Check
```bash
# Review generated labels
python -c "
import pandas as pd
df = pd.read_csv('labels/labels_20240115.csv')
print('Label distribution:')
print(df['label'].value_counts())
print(df['label_30'].value_counts())
print(f'Total samples: {len(df)}')
"
```

## 🧠 Phase 3: Dataset Construction & Training

### 3.1 Build Training Dataset
```bash
# Single file
python sequence_builder.py \
    --labels labels/labels_20240115.csv \
    --output datasets/dataset_20240115.csv

# Multiple files (recommended)
python sequence_builder.py \
    --labels labels/labels_*.csv \
    --output datasets/dataset_combined.csv \
    --num_ticks 180
```

### 3.2 Dataset Validation
```bash
# Check dataset quality
python -c "
import pandas as pd
df = pd.read_csv('datasets/dataset_combined.csv')
print(f'Dataset shape: {df.shape}')
print(f'Features: {df.shape[1] - 2}')  # -2 for label columns
print(f'Null values: {df.isnull().sum().sum()}')
print('\\nLabel distribution:')
print(df[['label', 'label_30']].describe())
"
```

### 3.3 Model Training
```bash
# Train dual models
python train_lightgbm.py \
    --dataset datasets/dataset_combined.csv \
    --model_dir models/

# Expected outputs:
# models/lgbm_label15.pkl - 15-minute model (5 classes)
# models/lgbm_label30.pkl - 30-minute model (3 classes)
```

### 3.4 Model Validation
```bash
# Check model performance
python -c "
import joblib
model_15 = joblib.load('models/lgbm_label15.pkl')
model_30 = joblib.load('models/lgbm_label30.pkl')
print(f'15-min model classes: {model_15.classes_}')
print(f'30-min model classes: {model_30.classes_}')
print('✅ Models loaded successfully')
"
```

## 🎯 Phase 4: Live Trading

### 4.1 Pre-flight Checks
```bash
# Verify all components
./run_tests.sh

# Check models exist
ls -la models/lgbm_label*.pkl

# Test database connection
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('PG_URL'))
with engine.connect() as conn:
    result = conn.execute('SELECT version()')
    print('✅ Database connected')
"
```

### 4.2 Live Trading Modes

#### Simulation Mode (Safe)
```bash
# Set simulation mode
export ALGOTEST_ENABLED=false

# Start live predictor
python live_predictor.py

# Expected output:
# 🎭 SIMULATION: UP intraday signal sent
# ✅ SIM: UP @ ₹25025.0 | Conf: 0.856 | Qual: 0.723
```

#### Live Trading Mode (Real Money)
```bash
# Configure live trading
export ALGOTEST_ENABLED=true
export ALGOTEST_API_KEY="your_api_key"
export ALGOTEST_API_SECRET="your_api_secret"

# Start live trading (WARNING: Real money!)
python live_predictor.py

# Expected output:
# ✅ LIVE: UP @ ₹25025.0 | Conf: 0.856 | Qual: 0.723
```

### 4.3 Monitoring Live System

#### Real-time Logs
```bash
# Main system logs
tail -f logs/general/logs.txt

# Key patterns to monitor:
# "✅ Trade allowed" - Successful signal
# "🔴 WebSocket" - Connection issues  
# "⚠️ Trade execution failed" - Execution errors
```

#### Performance Monitoring
```bash
# System resources
top -p $(pgrep -f live_predictor.py)

# Database performance
psql -d bsengine -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd 
FROM pg_stat_user_tables 
WHERE tablename LIKE 'nifty_fut_ticks%'
ORDER BY n_tup_ins DESC;
"

# Prediction statistics
psql -d bsengine -c "
SELECT prediction, COUNT(*), AVG(confidence) 
FROM ml_predictions 
WHERE ts > EXTRACT(epoch FROM NOW() - INTERVAL '1 day') * 1000000000
GROUP BY prediction;
"
```

## 🔧 Phase 5: Maintenance & Optimization

### 5.1 Daily Maintenance
```bash
# Clean old logs (keep last 7 days)
find logs/ -name "*.txt" -mtime +7 -delete

# Archive old session state files
mv session_state_*.json archive/ 2>/dev/null || true

# Database maintenance
psql -d bsengine -c "VACUUM ANALYZE;"
```

### 5.2 Model Retraining
```bash
# Weekly retraining workflow
# 1. Generate new labels
python label_generator.py --date $(date +%Y-%m-%d) --abs15 25 --abs30 50

# 2. Update dataset
python sequence_builder.py \
    --labels labels/labels_*.csv \
    --output datasets/dataset_updated.csv

# 3. Retrain models
python train_lightgbm.py \
    --dataset datasets/dataset_updated.csv \
    --model_dir models_new/

# 4. A/B test performance before deploying
```

### 5.3 Performance Optimization
```bash
# Profile feature extraction
python -m cProfile -o profile.prof live_predictor.py
python -c "
import pstats
p = pstats.Stats('profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Database query optimization
psql -d bsengine -c "
EXPLAIN ANALYZE 
SELECT * FROM nifty_fut_ticks_$(date +%Y%m%d) 
WHERE ts > EXTRACT(epoch FROM NOW() - INTERVAL '15 minutes') * 1000000000;
"
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Database Connection Errors
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Fix permissions
sudo -u postgres psql -c "ALTER USER $(whoami) CREATEDB;"

# Test connection
psql -d bsengine -c "SELECT NOW();"
```

#### 2. Model Loading Errors
```bash
# Check model files
ls -la models/
file models/lgbm_label15.pkl

# Test model loading
python -c "
import joblib
try:
    model = joblib.load('models/lgbm_label15.pkl')
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

#### 3. WebSocket Connection Issues
```bash
# Verify credentials
cat ws.json

# Test API connection
python -c "
from dhan_api_connector import connect
try:
    dhan_ctx, dhan = connect()
    print('✅ Dhan API connected')
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

#### 4. Memory Issues
```bash
# Monitor memory usage
free -h
ps aux | grep python

# Reduce buffer size in live_predictor.py
# Change: BUFFER_SIZE = 180 
# To: BUFFER_SIZE = 120
```

#### 5. Performance Issues
```bash
# Check system load
uptime
iostat -x 1 5

# Database performance
psql -d bsengine -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
"
```

## 📈 Advanced Workflows

### A/B Testing New Features
```bash
# 1. Create feature branch
git checkout -b feature/new-indicators

# 2. Modify feature_enricher.py with new features

# 3. Retrain models
python train_lightgbm.py --dataset datasets/ --model_dir models_test/

# 4. Run parallel testing
python live_predictor.py --model_dir models_test/ --simulation_only

# 5. Compare performance metrics
```

### Multi-Instrument Trading
```bash
# Modify live_predictor.py tokens list
tokens = [53216, 64103, 75394]  # Multiple futures contracts

# Update database schema for multiple instruments
psql -d bsengine -f create_multi_instrument_tables.sql
```

### Risk-Adjusted Position Sizing
```bash
# Implement in unified_trade_executor.py
def calculate_position_size(prediction, confidence, quality, volatility):
    base_size = 1
    confidence_multiplier = confidence
    volatility_adjustment = 1 / max(volatility, 0.5)
    return int(base_size * confidence_multiplier * volatility_adjustment)
```

## 📊 Performance Benchmarks

### Expected Performance Metrics
```
Latency Targets:
- Tick processing: <50ms
- Feature extraction: <100ms  
- Model inference: <20ms
- Trade execution: <30ms
- End-to-end: <200ms

Throughput Targets:
- Tick ingestion: 10+ ticks/sec
- Database writes: 100+ ops/sec
- Predictions: 720+ per day (every 30 seconds during market hours)

Accuracy Targets:
- Directional accuracy: 65-75%
- Precision (UP/DOWN): >70%
- Recall (signal detection): >60%
```

### Monitoring Commands
```bash
# Real-time performance dashboard
watch -n 5 "
echo '=== System Performance ==='
ps aux | grep live_predictor.py | grep -v grep
echo ''
echo '=== Database Stats ==='
psql -d bsengine -t -c \"SELECT COUNT(*) as predictions_today FROM ml_predictions WHERE ts > EXTRACT(epoch FROM current_date) * 1000000000;\"
echo ''
echo '=== Recent Predictions ==='
psql -d bsengine -t -c \"SELECT prediction, confidence, ts FROM ml_predictions ORDER BY ts DESC LIMIT 5;\"
"
```

This workflow guide provides a complete path from zero to production trading system. Follow each phase sequentially for best results.
