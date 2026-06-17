$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Python = ".\.venv\Scripts\python.exe"

& $Python -c @"
import sqlite3
db = r"data\toa_ai\btc_1m.sqlite3"
conn = sqlite3.connect(db)
rows = conn.execute("select symbol, timeframe, count(*), min(timestamp), max(timestamp) from bars group by symbol, timeframe").fetchall()
for symbol, timeframe, count, first_ts, last_ts in rows:
    print(f"{symbol} {timeframe} rows={count} first={first_ts} last={last_ts}")
"@
