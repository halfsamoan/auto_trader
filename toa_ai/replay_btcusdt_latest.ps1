$ErrorActionPreference = "Stop"

$Root = "C:\auto_trader"
if (-not (Test-Path $Root)) {
  $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"

& $Python -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 replay `
  --model-id latest `
  --symbols BTCUSDT `
  --timeframe 1m `
  --sequence-length 64 `
  --max-episode-bars 20000
