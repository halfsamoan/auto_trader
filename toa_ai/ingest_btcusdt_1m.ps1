$ErrorActionPreference = "Stop"

$Root = "C:\auto_trader"
if (-not (Test-Path $Root)) {
  $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"

& $Python -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 init-db
& $Python -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 ingest-external `
  --path data\external\binance\BTCUSDT_1m `
  --symbol BTCUSDT `
  --timeframe 1m `
  --market BINANCE `
  --asset-class crypto `
  --format binance
