#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
source .venv/bin/activate

python main.py --paper-watch --asset domestic-stock --allow-paper-order --interval-sec 300 >> logs/domestic_paper_watch.log 2>&1
