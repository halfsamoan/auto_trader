#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/external/binance/BTCUSDT_1m"

mkdir -p "$OUT"
cd "$OUT"

echo "Downloading BTCUSDT 1m monthly files: 2024-06 through 2026-05"
for i in $(seq 0 23); do
  ym=$(date -u -d "2024-06-01 +$i month" +%Y-%m)
  url="https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-$ym.zip"
  echo "$url"
  curl -fL --retry 3 --retry-delay 2 -O "$url"
done

echo "Downloading BTCUSDT 1m daily files: 2026-06-01 through 2026-06-14"
for d in $(seq -w 01 14); do
  url="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-06-$d.zip"
  echo "$url"
  curl -fL --retry 3 --retry-delay 2 -O "$url" || true
done

echo
echo "Downloaded zip count:"
find "$OUT" -maxdepth 1 -type f -name "*.zip" | wc -l
echo
ls -lh "$OUT"
