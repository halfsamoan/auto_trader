#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/intraday_collect.log"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

echo "intraday collector start $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

while true; do
  {
    echo "----- collect $(date '+%Y-%m-%d %H:%M:%S') -----"
    .venv/bin/python scripts/collect_intraday_data.py --universe ai_train --period 60d --interval 5m
  } >> "$LOG_FILE" 2>&1
  sleep 300
done
