#!/bin/bash
# Quick Setup Script for Real-Time ML Trading Pipeline
# Run this after cloning the repository

set -e  # Exit on any error

echo "🚀 Real-Time ML Trading Pipeline - Quick Setup"
echo "=============================================="

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ⚠️ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "   ✅ Requirements installed"
else
    echo "   ⚠️ requirements.txt not found"
fi

# Create necessary directories
echo "📁 Creating directories..."
directories=("logs/general" "models" "datasets" "labels" "tests" "scripts" "docs" "examples")
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "   ✅ Created: $dir"
done

# Copy environment template
echo "⚙️ Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✅ Created .env from template"
        echo "   ⚠️ Please edit .env with your actual configuration"
    else
        echo "   ⚠️ .env.example not found"
    fi
else
    echo "   ✅ .env already exists"
fi

# Create WebSocket config template
echo "🔌 Setting up WebSocket configuration..."
if [ ! -f "ws.json" ]; then
    cat > ws.json << 'EOF'
{
    "client_id": "your_dhan_client_id",
    "access_token": "your_dhan_access_token"
}
EOF
    echo "   ✅ Created ws.json template"
    echo "   ⚠️ Please edit ws.json with your Dhan API credentials"
else
    echo "   ⚠️ ws.json already exists"
fi

# Check PostgreSQL
echo "🗄️ Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "   ✅ PostgreSQL found"
    
    # Try to create database (will fail if exists, which is OK)
    echo "📚 Setting up database..."
    createdb bsengine 2>/dev/null || echo "   ⚠️ Database 'bsengine' may already exist"
    
    # Run schema if available
    if [ -f "create_table.sql" ]; then
        psql -d bsengine -f create_table.sql 2>/dev/null && echo "   ✅ Database schema created" || echo "   ⚠️ Schema may already exist"
    else
        echo "   ⚠️ create_table.sql not found"
    fi
else
    echo "   ❌ PostgreSQL not found. Please install PostgreSQL:"
    echo "      Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "      MacOS: brew install postgresql"
    echo "      Windows: https://www.postgresql.org/download/"
fi

# Make scripts executable
echo "🔧 Setting up scripts..."
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh 2>/dev/null || true
    echo "   ✅ Scripts made executable"
fi

# Test basic imports
echo "🧪 Testing basic functionality..."
python3 -c "
try:
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    print('   ✅ Core dependencies working')
except ImportError as e:
    print(f'   ❌ Import error: {e}')
    exit(1)
" || exit 1

# Create basic test
echo "📝 Creating basic test..."
if [ ! -f "tests/test_basic.py" ]; then
    cat > tests/test_basic.py << 'EOF'
#!/usr/bin/env python3
"""Basic functionality tests"""

def test_imports():
    """Test that core modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("🎉 Basic tests passed!")
    else:
        print("❌ Basic tests failed!")
        exit(1)
EOF
    echo "   ✅ Created basic test"
fi

# Run basic test
echo "🔍 Running basic tests..."
python3 tests/test_basic.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "   1. Edit .env with your database configuration"
echo "   2. Edit ws.json with your Dhan API credentials"
echo "   3. Review README.md for detailed instructions"
echo ""
echo "🚀 Quick Commands:"
echo "   Start live trading:  python3 live_predictor.py"
echo "   Train models:        python3 train_lightgbm.py --dataset datasets/ --model_dir models/"
echo "   Run tests:           python3 -m pytest tests/ -v"
echo ""
echo "📚 Documentation:"
echo "   README.md     - Complete project overview"
echo "   WORKFLOW.md   - Step-by-step usage guide"
echo ""
echo "⚠️ Important:"
echo "   - This is educational software for research purposes"
echo "   - Trading involves substantial risk of loss"
echo "   - Always test in simulation mode first"
echo ""

# Deactivate virtual environment message
echo "💡 To activate the virtual environment later:"
echo "   source venv/bin/activate"