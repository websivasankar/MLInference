#!/usr/bin/env python3
"""
Real-Time ML Trading Pipeline Setup Script
Automated setup for development and production environments
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directories():
    """Create necessary project directories"""
    directories = [
        "logs/general",
        "models",
        "datasets", 
        "labels",
        "session_state",
        "tests",
        "docs"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def create_env_file():
    """Create .env template file"""
    env_template = """# Real-Time ML Trading Pipeline Configuration

# Database Configuration
PG_URL=postgresql://username:password@localhost:5432/bsengine

# Trading API Configuration (Optional)
ALGOTEST_ENABLED=false
ALGOTEST_API_KEY=your_api_key_here
ALGOTEST_API_SECRET=your_api_secret_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/general/logs.txt

# Feature Engineering Configuration
USE_SHAP_OPTIMIZED=true
FEATURE_SET=shap_optimized

# Performance Tuning
BUFFER_SIZE=180
WRITE_INTERVAL_NS=5000000000
MAX_THREADS=4

# Development Flags
DEBUG_MODE=false
SIMULATION_MODE=true
"""
    
    print("⚙️ Creating environment configuration...")
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_template)
        print("   ✅ Created .env file")
    else:
        print("   ⚠️ .env file already exists, skipping")

def create_ws_config():
    """Create WebSocket configuration template"""
    ws_template = {
        "client_id": "your_dhan_client_id",
        "access_token": "your_dhan_access_token",
        "api_base_url": "https://api.dhan.co",
        "websocket_url": "wss://api.dhan.co"
    }
    
    print("🔌 Creating WebSocket configuration...")
    if not os.path.exists("ws.json"):
        with open("ws.json", "w") as f:
            json.dump(ws_template, f, indent=2)
        print("   ✅ Created ws.json template")
    else:
        print("   ⚠️ ws.json already exists, skipping")

def setup_database():
    """Setup PostgreSQL database"""
    print("🗄️ Setting up PostgreSQL database...")
    
    # Check if PostgreSQL is installed
    if not run_command("which psql", "Checking PostgreSQL installation"):
        print("❌ PostgreSQL not found. Please install PostgreSQL first:")
        print("   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
        print("   MacOS: brew install postgresql")
        print("   Windows: Download from https://www.postgresql.org/download/")
        return False
    
    # Create database (will fail if already exists, which is OK)
    print("📚 Creating database...")
    subprocess.run("createdb bsengine", shell=True, capture_output=True)
    
    # Run schema creation
    if os.path.exists("create_table.sql"):
        if run_command("psql -d bsengine -f create_table.sql", "Creating database schema"):
            print("   ✅ Database schema created successfully")
        else:
            print("   ⚠️ Schema creation failed (may already exist)")
    else:
        print("   ⚠️ create_table.sql not found, skipping schema creation")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("🐍 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    else:
        print("   ⚠️ requirements.txt not found")
        return False

def create_test_files():
    """Create basic test structure"""
    print("🧪 Creating test structure...")
    
    test_init = """# Real-Time ML Trading Pipeline Tests
"""
    
    test_example = """import pytest
import pandas as pd
import numpy as np

def test_basic_imports():
    \"\"\"Test that core modules can be imported\"\"\"
    try:
        from feature_enricher import enrich_features
        from smart_money_detector import analyze_smart_money_flow
        from market_maker_detector import detect_mm_behavior
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_feature_engineering():
    \"\"\"Test basic feature engineering functionality\"\"\"
    # Mock tick data
    mock_ticks = []
    for i in range(180):
        tick = {
            'ts': i * 1000000000,
            'ltp': 25000 + i * 0.25,
            'ltq': 50 + (i % 10),
            'oi': 1000000,
            'bid_price_0': 25000 + i * 0.25 - 0.25,
            'ask_price_0': 25000 + i * 0.25 + 0.25,
            'bid_qty_0': 100,
            'ask_qty_0': 100
        }
        mock_ticks.append(tick)
    
    try:
        from feature_enricher import enrich_features
        features = enrich_features(mock_ticks, strict=False)
        assert isinstance(features, dict)
        assert len(features) > 50  # Should have many features
        print(f"✅ Generated {len(features)} features")
    except Exception as e:
        pytest.fail(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    test_basic_imports()
    test_feature_engineering()
    print("🎉 All basic tests passed!")
"""
    
    # Create test files
    with open("tests/__init__.py", "w") as f:
        f.write(test_init)
    
    with open("tests/test_basic.py", "w") as f:
        f.write(test_example)
    
    print("   ✅ Created test structure")

def create_run_scripts():
    """Create convenient run scripts"""
    print("🚀 Creating run scripts...")
    
    # Live trading script
    live_script = """#!/bin/bash
# Real-Time ML Trading Pipeline - Live Trading

echo "🚀 Starting Live Trading Pipeline..."
echo "⚠️  Make sure you have:"
echo "   1. Configured .env file"
echo "   2. Set up ws.json with valid Dhan credentials"  
echo "   3. PostgreSQL running and accessible"
echo ""

# Check if models exist
if [ ! -f "models/lgbm_label15.pkl" ]; then
    echo "❌ Models not found! Please train models first:"
    echo "   python train_lightgbm.py --dataset datasets/ --model_dir models/"
    exit 1
fi

# Start live prediction
echo "🎯 Starting live predictor..."
python live_predictor.py
"""

    # Training script
    train_script = """#!/bin/bash
# Real-Time ML Trading Pipeline - Model Training

echo "🧠 Starting Model Training Pipeline..."

# Check if datasets exist
if [ ! -d "datasets" ] || [ -z "$(ls -A datasets/)" ]; then
    echo "⚠️ No datasets found. Generate training data first:"
    echo "   1. python label_generator.py --date 2024-01-15 --abs15 25 --abs30 50"
    echo "   2. python sequence_builder.py --labels labels/*.csv --output datasets/combined.csv"
    exit 1
fi

# Start training
echo "🎯 Training models..."
python train_lightgbm.py --dataset datasets/ --model_dir models/

echo "✅ Training completed!"
echo "📁 Models saved in: models/"
echo "🚀 Ready for live trading!"
"""

    # Test script
    test_script = """#!/bin/bash
# Real-Time ML Trading Pipeline - Testing

echo "🧪 Running test suite..."

# Basic tests
echo "🔍 Running basic functionality tests..."
python -m pytest tests/test_basic.py -v

# Performance tests (if exist)
if [ -f "tests/test_performance.py" ]; then
    echo "⚡ Running performance tests..."
    python -m pytest tests/test_performance.py -v
fi

echo "✅ All tests completed!"
"""

    scripts = [
        ("run_live.sh", live_script),
        ("run_training.sh", train_script), 
        ("run_tests.sh", test_script)
    ]

    for filename, content in scripts:
        with open(filename, "w") as f:
            f.write(content)
        os.chmod(filename, 0o755)  # Make executable
        print(f"   ✅ Created: {filename}")

def verify_setup():
    """Verify setup by running basic tests"""
    print("🔍 Verifying setup...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        print("   ✅ Core dependencies imported successfully")
        
        # Test database connection (if configured)
        try:
            from sqlalchemy import create_engine
            import os
            pg_url = os.getenv("PG_URL", "postgresql://username:password@localhost:5432/bsengine")
            if "username:password" not in pg_url:
                engine = create_engine(pg_url)
                with engine.connect() as conn:
                    result = conn.execute("SELECT version()")
                    print("   ✅ Database connection successful")
        except Exception as e:
            print(f"   ⚠️ Database connection failed: {e}")
            print("      Please configure PG_URL in .env file")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import test failed: {e}")
        print("      Please install dependencies: pip install -r requirements.txt")
        return False

def main():
    """Main setup function"""
    print("🚀 Real-Time ML Trading Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version} detected")
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Creating configuration files", create_env_file),
        ("Creating WebSocket config", create_ws_config),
        ("Installing dependencies", install_dependencies),
        ("Setting up database", setup_database),
        ("Creating test structure", create_test_files),
        ("Creating run scripts", create_run_scripts),
        ("Verifying setup", verify_setup)
    ]
    
    failed_steps = []
    
    for description, func in steps:
        print(f"\n🔧 {description}...")
        try:
            if func():
                print(f"✅ {description} completed")
            else:
                print(f"⚠️ {description} completed with warnings")
                failed_steps.append(description)
        except Exception as e:
            print(f"❌ {description} failed: {e}")
            failed_steps.append(description)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Setup Summary")
    print("=" * 50)
    
    if not failed_steps:
        print("✅ All setup steps completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Configure .env with your database URL")
        print("   2. Configure ws.json with your Dhan API credentials")
        print("   3. Generate training data: ./run_training.sh")
        print("   4. Start live trading: ./run_live.sh")
    else:
        print(f"⚠️ Setup completed with {len(failed_steps)} warnings:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n📋 Please address the warnings before proceeding")
    
    print("\n📚 Documentation: README.md")
    print("🐛 Issues: https://github.com/websivasankar/MLInference/issues")
    print("💬 Discussions: https://github.com/websivasankar/MLInference/discussions")

if __name__ == "__main__":
    main()
