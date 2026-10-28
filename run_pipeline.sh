#!/usr/bin/env bash
# ==============================================================================
# Windows / WSL Execution Script (Git Bash / MSYS2 / WSL Ubuntu/Debian)
# CSMAR Misstatement Detection: Real Data Fetch -> 102 Features -> Split Pipeline
# Reference Paper: Bertomeu, Cheynel, Floyd, and Pan (2020), RAST
# ==============================================================================

set -e

echo "=================================================================="
echo "[Step 1/4] Detecting Python environment..."
echo "=================================================================="

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v py &>/dev/null; then
    PYTHON_CMD="py -3"
else
    echo "ERROR: Python is not installed or not in PATH."
    exit 1
fi

echo "Using Python interpreter: $($PYTHON_CMD --version)"

echo "=================================================================="
echo "[Step 2/4] Setting up virtual environment (.venv) or fallback..."
echo "=================================================================="

if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ] && [ ! -f ".venv/Scripts/activate" ]; then
    echo "Removing incomplete .venv directory from previous failed attempt..."
    rm -rf .venv
fi

USE_VENV=0
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    USE_VENV=1
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    USE_VENV=1
else
    if $PYTHON_CMD -m venv .venv 2>/dev/null; then
        if [ -f ".venv/Scripts/activate" ]; then
            source .venv/Scripts/activate
            USE_VENV=1
        elif [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
            USE_VENV=1
        fi
    else
        echo "Notice: python3-venv (ensurepip) is not installed in WSL/Ubuntu."
        rm -rf .venv
        echo "Falling back to user-level installation without requiring venv..."
    fi
fi

echo "=================================================================="
echo "[Step 3/4] Fixing websocket namespace conflict & installing deps..."
echo "=================================================================="

if [ "$USE_VENV" -eq 1 ]; then
    # Uninstall conflicting obsolete `websocket` (0.2.1) package if installed, then cleanly install `websocket-client`
    python -m pip uninstall -y websocket websocket-client 2>/dev/null || true
    python -m pip install --upgrade pip
    python -m pip install -r requirement.txt
    # Sync patched csmarapi into .venv site-packages
    SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
    cp -r csmarapi "$SITE_PKG/" 2>/dev/null || true
    RUN_PY="python"
else
    if $PYTHON_CMD -m pip --version &>/dev/null; then
        $PYTHON_CMD -m pip uninstall -y websocket websocket-client 2>/dev/null || true
        $PYTHON_CMD -m pip install --user --break-system-packages -r requirement.txt 2>/dev/null || \
        $PYTHON_CMD -m pip install --user -r requirement.txt
    else
        echo "WARNING: pip is not installed. Run: sudo apt update && sudo apt install -y python3-pip python3-venv"
        exit 1
    fi
    RUN_PY="$PYTHON_CMD"
fi

echo "=================================================================="
echo "[Step 4/4] Running Real CSMAR Fetch -> 102 Features -> Train/Val/Test Split..."
echo "=================================================================="

$RUN_PY -m src.run_pipeline
