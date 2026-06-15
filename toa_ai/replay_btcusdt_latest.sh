#!/usr/bin/env bash
set -euo pipefail

cd /home/k/auto_trader

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_1m.sqlite3 replay \
  --model-id latest \
  --symbols BTCUSDT \
  --timeframe 1m \
  --sequence-length 64 \
  --max-episode-bars 20000
