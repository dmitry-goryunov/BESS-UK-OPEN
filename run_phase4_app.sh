#!/bin/bash
# Quick-start script for BESS Phase 4 Streamlit App

# This script helps you quickly launch the Phase 4 Duration Sweep dashboard

set -e

echo "🔋 BESS Phase 4 Streamlit App Quick-Start"
echo "==========================================="
echo ""

# Check if requirements are installed
echo "✓ Checking dependencies..."
python -c "import streamlit, pandas, plotly" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
}

echo ""
echo "📊 Launching Phase 4 Duration Sweep Dashboard..."
echo ""
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run streamlit_phase4_sweep.py

