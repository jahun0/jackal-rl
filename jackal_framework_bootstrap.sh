#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

VENV_DIR=".venv_marl"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating local virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating venv and installing minimal MARL dependencies..."
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q pettingzoo "ray[rllib]" numpy

echo "[*] Rerunning ingest check inside venv..."
python3 jackal_framework_ingest_check.py
