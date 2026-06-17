$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Out = Join-Path $Root "data\external\binance\BTCUSDT_1m"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Out

Write-Host "Downloading BTCUSDT 1m monthly files: 2024-06 through 2026-05"
$Start = Get-Date "2024-06-01"
foreach ($i in 0..23) {
  $ym = $Start.AddMonths($i).ToString("yyyy-MM")
  $url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-$ym.zip"
  $file = "BTCUSDT-1m-$ym.zip"
  Write-Host $url
  Invoke-WebRequest -Uri $url -OutFile $file
}

Write-Host "Downloading BTCUSDT 1m daily files: 2026-06-01 through 2026-06-14"
foreach ($d in 1..14) {
  $day = "{0:D2}" -f $d
  $url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-06-$day.zip"
  $file = "BTCUSDT-1m-2026-06-$day.zip"
  Write-Host $url
  try {
    Invoke-WebRequest -Uri $url -OutFile $file
  } catch {
    Write-Warning "Skipped $url"
  }
}

Write-Host ""
Write-Host "Downloaded zip count:"
(Get-ChildItem -Path $Out -Filter "*.zip" | Measure-Object).Count
Get-ChildItem -Path $Out
