$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"

& $Python -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 train `
  --timeframe 1m `
  --sequence-length 64 `
  --horizon-bars 30 `
  --epochs 6 `
  --batch-size 256 `
  --symbols BTCUSDT `
  --model-dir data\toa_ai\btc_models
