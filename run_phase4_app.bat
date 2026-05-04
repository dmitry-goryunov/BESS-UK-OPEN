@echo off
REM Quick-start script for BESS Phase 4 Streamlit App (Windows)

echo.
echo 🔋 BESS Phase 4 Streamlit App Quick-Start (Windows)
echo ===================================================
echo.

echo ✓ Checking Python installation...
python --version >nul 2>&1 || (
    echo ❌ Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo ✓ Checking dependencies...
python -c "import streamlit, pandas, plotly" >nul 2>&1 || (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo 📊 Launching Phase 4 Duration Sweep Dashboard...
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run streamlit_phase4_sweep.py
