#!/usr/bin/env bash
set -euo pipefail

cd /home/k/auto_trader

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_smoke.sqlite3 init-db

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_smoke.sqlite3 ingest-external \
  --path data/external/binance/BTCUSDT_1m/BTCUSDT-1m-2026-05.zip \
  --symbol BTCUSDT \
  --timeframe 1m \
  --market BINANCE \
  --asset-class crypto \
  --format binance

.venv/bin/python -m toa_ai.cli --db-path data/toa_ai/btc_smoke.sqlite3 train \
  --timeframe 1m \
  --sequence-length 64 \
  --horizon-bars 30 \
  --epochs 1 \
  --batch-size 256 \
  --symbols BTCUSDT \
  --model-dir data/toa_ai/btc_smoke_models
