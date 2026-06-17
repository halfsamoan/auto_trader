"""Local data ingestion adapters for TOA Memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toa_ai.storage import TOAMemory


def ingest_cache_directory(
    memory: TOAMemory,
    cache_dir: str | Path,
    timeframe: str = "5m",
    symbols: list[str] | None = None,
    limit_symbols: int | None = None,
) -> dict[str, Any]:
    root = Path(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"cache_dir not found: {root}")
    wanted = {symbol.strip() for symbol in symbols or [] if symbol.strip()}
    files = sorted(root.glob(f"*_{timeframe}.csv"))
    if wanted:
        files = [path for path in files if _symbol_from_path(path, timeframe) in wanted]
    if limit_symbols is not None:
        files = files[: int(limit_symbols)]

    results = []
    for path in files:
        symbol = _symbol_from_path(path, timeframe)
        try:
            frame = pd.read_csv(path)
            rows = memory.upsert_bars(symbol, frame, timeframe=timeframe, source=f"cache:{path.name}")
            results.append({"symbol": symbol, "path": str(path), "rows_written": int(rows), "status": "ok"})
        except Exception as exc:
            results.append({"symbol": symbol, "path": str(path), "rows_written": 0, "status": "error", "error": str(exc)})
    return {
        "cache_dir": str(root),
        "timeframe": timeframe,
        "file_count": len(files),
        "success_count": sum(1 for row in results if row["status"] == "ok"),
        "rows_written": sum(int(row["rows_written"]) for row in results),
        "results": results,
    }


def _symbol_from_path(path: Path, timeframe: str) -> str:
    suffix = f"_{timeframe}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem
