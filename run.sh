#!/usr/bin/env bash
set -euo pipefail
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
 	python3 -m virtualenv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade -r requirements.txt -q
python bench_tui.py
