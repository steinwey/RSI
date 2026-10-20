#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3.12}
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,12), "Harbor requires Python >=3.12"'
"$PYTHON" -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -e vendor/harbor -e vendor/SWE-bench
python -m rsi.cli doctor
python -m unittest discover -s tests -v
