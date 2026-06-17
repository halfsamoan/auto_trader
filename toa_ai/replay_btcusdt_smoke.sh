#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_smoke.sqlite3 replay \
  --model-id latest \
  --symbols BTCUSDT \
  --timeframe 1m \
  --sequence-length 64 \
  --max-episode-bars 10000
