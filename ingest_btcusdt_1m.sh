#!/usr/bin/env bash
set -euo pipefail

cd /home/k/auto_trader

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_1m.sqlite3 init-db

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_1m.sqlite3 ingest-external \
  --path data/external/binance/BTCUSDT_1m \
  --symbol BTCUSDT \
  --timeframe 1m \
  --market BINANCE \
  --asset-class crypto \
  --format binance
