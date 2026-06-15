"""External CSV ingestion for TOA AI.

Supports common Kaggle OHLCV CSVs and Binance kline CSV/ZIP files without
coupling TOA to a specific download source.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from toa_ai.storage import TOAMemory


BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


TIMESTAMP_CANDIDATES = [
    "timestamp",
    "date",
    "datetime",
    "time",
    "open_time",
    "open time",
    "opentime",
    "unix",
    "unix_timestamp",
]
OHLCV_ALIASES = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "last", "price"],
    "volume": ["volume", "vol", "base_volume", "volume_(btc)", "volume btc"],
}


def ingest_external_csv(
    memory: TOAMemory,
    path: str | Path,
    symbol: str,
    timeframe: str = "1m",
    market: str = "CRYPTO",
    asset_class: str = "crypto",
    source_format: str = "auto",
) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        return ingest_external_directory(memory, source, symbol, timeframe, market, asset_class, source_format)
    frame = load_external_ohlcv(source, source_format)
    rows = memory.upsert_bars(symbol, frame, timeframe=timeframe, market=market, asset_class=asset_class, source=f"{source_format}:{source.name}")
    return {"path": str(source), "symbol": symbol, "timeframe": timeframe, "rows_written": int(rows), "status": "ok"}


def ingest_external_directory(
    memory: TOAMemory,
    directory: str | Path,
    symbol: str,
    timeframe: str = "1m",
    market: str = "CRYPTO",
    asset_class: str = "crypto",
    source_format: str = "auto",
) -> dict[str, Any]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(str(root))
    files = sorted([*root.glob("*.csv"), *root.glob("*.zip")])
    results = []
    for path in files:
        try:
            results.append(ingest_external_csv(memory, path, symbol, timeframe, market, asset_class, source_format))
        except Exception as exc:
            results.append({"path": str(path), "symbol": symbol, "rows_written": 0, "status": "error", "error": str(exc)})
    return {
        "directory": str(root),
        "symbol": symbol,
        "timeframe": timeframe,
        "file_count": len(files),
        "success_count": sum(1 for row in results if row.get("status") == "ok"),
        "rows_written": sum(int(row.get("rows_written") or 0) for row in results),
        "results": results,
    }


def load_external_ohlcv(path: str | Path, source_format: str = "auto") -> pd.DataFrame:
    source = Path(path)
    if source_format == "binance" or _looks_like_binance_kline(source):
        return _load_binance_kline(source)
    return _load_generic_ohlcv(source)


def _load_binance_kline(path: Path) -> pd.DataFrame:
    frame = _read_csv_or_zip(path, header=None)
    if len(frame.columns) >= len(BINANCE_KLINE_COLUMNS):
        frame = frame.iloc[:, : len(BINANCE_KLINE_COLUMNS)]
        frame.columns = BINANCE_KLINE_COLUMNS
    else:
        frame = _read_csv_or_zip(path)
        frame.columns = [_clean_col(col) for col in frame.columns]
    out = pd.DataFrame()
    out["timestamp"] = _parse_timestamp(frame["open_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    return out.dropna(subset=["timestamp", "open", "high", "low", "close"])


def _load_generic_ohlcv(path: Path) -> pd.DataFrame:
    frame = _read_csv_or_zip(path)
    original_columns = list(frame.columns)
    frame = frame.rename(columns={col: _clean_col(col) for col in frame.columns})
    timestamp_col = _find_column(frame.columns, TIMESTAMP_CANDIDATES)
    if timestamp_col is None:
        raise ValueError(f"timestamp column not found in {path}; columns={original_columns}")
    out = pd.DataFrame()
    out["timestamp"] = _parse_timestamp(frame[timestamp_col])
    for target, aliases in OHLCV_ALIASES.items():
        column = _find_column(frame.columns, aliases)
        if column is None and target == "volume":
            out[target] = 0.0
            continue
        if column is None:
            raise ValueError(f"{target} column not found in {path}; columns={original_columns}")
        out[target] = pd.to_numeric(frame[column], errors="coerce")
    return out.dropna(subset=["timestamp", "open", "high", "low", "close"])


def _read_csv_or_zip(path: Path, header: int | None = "infer") -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"zip has no CSV: {path}")
            with archive.open(csv_names[0]) as handle:
                return pd.read_csv(handle, header=header)
    return pd.read_csv(path, header=header)


def _parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        unit = _timestamp_unit(values)
        return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.notna().any():
        return parsed
    values = pd.to_numeric(series, errors="coerce")
    unit = _timestamp_unit(values)
    return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")


def _timestamp_unit(values: pd.Series) -> str:
    clean = values.dropna()
    if clean.empty:
        return "s"
    median = float(clean.median())
    if median > 10_000_000_000_000:
        return "us"
    if median > 10_000_000_000:
        return "ms"
    return "s"


def _find_column(columns: Any, candidates: list[str]) -> str | None:
    available = {_clean_col(col): col for col in columns}
    for candidate in candidates:
        cleaned = _clean_col(candidate)
        if cleaned in available:
            return available[cleaned]
    return None


def _clean_col(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def _looks_like_binance_kline(path: Path) -> bool:
    name = path.name.lower()
    return "klines" in str(path).lower() or ("btcusdt" in name and ("1m" in name or "5m" in name))
