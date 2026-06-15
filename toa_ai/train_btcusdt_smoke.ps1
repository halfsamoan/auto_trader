$ErrorActionPreference = "Stop"

$Root = "C:\auto_trader"
if (-not (Test-Path $Root)) {
  $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"

& $Python -m toa_ai.cli --db-path data\toa_ai\btc_smoke.sqlite3 init-db
& $Python -m toa_ai.cli --db-path data\toa_ai\btc_smoke.sqlite3 ingest-external `
  --path data\external\binance\BTCUSDT_1m\BTCUSDT-1m-2026-05.zip `
  --symbol BTCUSDT `
  --timeframe 1m `
  --market BINANCE `
  --asset-class crypto `
  --format binance
& $Python -m toa_ai.cli --db-path data\toa_ai\btc_smoke.sqlite3 train `
  --timeframe 1m `
  --sequence-length 64 `
  --horizon-bars 30 `
  --epochs 1 `
  --batch-size 256 `
  --symbols BTCUSDT `
  --model-dir data\toa_ai\btc_smoke_models
