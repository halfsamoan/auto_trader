#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: yfinance, pandas, ai/dataset.py, futures_contracts.py
# Description: Yfinance-only MNQ/MES 5m futures cache backfill and data audit.
# ================================================================================

"""Backfill MNQ/MES 5-minute futures bars through yfinance only.

This script deliberately avoids KIS clients, .env loading, and order paths.
It stores normalized UTC bars in the shared intraday cache directory with
source='yfinance_futures'.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.dataset import SplitConfig, calendar_time_split
from config import AI_PRED_HORIZON_BARS, AI_SEQUENCE_LENGTH, INTRADAY_CACHE_DIR
from futures_contracts import FUTURES_CONTRACTS, get_contract


REPORT_PATH = ROOT / "ai" / "model_store" / "futures_intraday_data_report.json"
CACHE_DIR = ROOT / INTRADAY_CACHE_DIR


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{interval}.csv"


def _normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) for col in frame.columns]
    rename = {name.lower(): name for name in ["Open", "High", "Low", "Close", "Volume"]}
    lower_map = {str(col).lower(): col for col in frame.columns}
    out = pd.DataFrame(index=frame.index)
    for target in ["Open", "High", "Low", "Close", "Volume"]:
        source = lower_map.get(target.lower())
        if source is None:
            return pd.DataFrame()
        out[target] = pd.to_numeric(frame[source], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    if out.empty:
        return out
    if not isinstance(out.index, pd.DatetimeIndex):
        return pd.DataFrame()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def fetch_yfinance_futures(symbol: str, period: str, interval: str) -> pd.DataFrame:
    contract = get_contract(symbol)
    ticker = str(contract["yfinance_symbol"])
    frame = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, prepost=True, threads=False)
    return _normalize_yfinance_frame(frame)


def save_cache(symbol: str, frame: pd.DataFrame, interval: str) -> int:
    if frame.empty:
        return 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_cache(symbol, interval)
    merged = pd.concat([existing, frame]).sort_index() if not existing.empty else frame.copy()
    merged = merged[~merged.index.duplicated(keep="last")]
    out = merged.reset_index().rename(columns={merged.index.name or "index": "timestamp"})
    out = out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    out["code"] = symbol.upper()
    out["source"] = "yfinance_futures"
    out["collected_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    out = out[["timestamp", "open", "high", "low", "close", "volume", "code", "source", "collected_at"]]
    out.to_csv(_cache_path(symbol, interval), index=False)
    return int(len(merged))


def load_cache(symbol: str, interval: str) -> pd.DataFrame:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in raw.columns:
        return pd.DataFrame()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["timestamp"])
    if raw.empty:
        return pd.DataFrame()
    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    frame = raw.rename(columns=rename).set_index("timestamp")
    keep = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame.columns]
    frame = frame[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    return frame.sort_index()


def gap_report(frame: pd.DataFrame, interval_minutes: int) -> dict[str, Any]:
    if frame.empty or len(frame.index) < 2:
        return {"gap_count": 0, "large_gap_count": 0, "top_gaps": []}
    diffs = frame.index.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=interval_minutes)
    gaps = diffs[diffs > expected * 1.5]
    large = diffs[diffs >= pd.Timedelta(hours=12)]
    return {
        "gap_count": int(len(gaps)),
        "large_gap_count": int(len(large)),
        "top_gaps": [
            {
                "end_timestamp_utc": ts.isoformat(),
                "gap_minutes": float(delta.total_seconds() / 60.0),
            }
            for ts, delta in gaps.sort_values(ascending=False).head(10).items()
        ],
    }


def rollover_candidates(frame: pd.DataFrame, jump_threshold: float) -> list[dict[str, Any]]:
    if frame.empty or len(frame) < 2:
        return []
    prev_close = frame["Close"].shift(1)
    jump = frame["Open"] / prev_close - 1.0
    mask = jump.abs() >= jump_threshold
    rows = []
    for ts, value in jump[mask].dropna().items():
        rows.append(
            {
                "timestamp_utc": ts.isoformat(),
                "open_vs_prev_close_return": float(value),
                "note": "candidate only; yfinance continuous futures are not back-adjusted here",
            }
        )
    return rows[:20]


def split_sample_report(frame: pd.DataFrame, symbol: str, purge_gap_bars: int, interval_minutes: int) -> dict[str, Any]:
    if frame.empty:
        return {"symbol": symbol, "purge_gap_bars": purge_gap_bars, "status": "missing_data"}
    timestamps = frame.index[AI_SEQUENCE_LENGTH - 1 : max(len(frame) - AI_PRED_HORIZON_BARS, AI_SEQUENCE_LENGTH - 1)]
    if len(timestamps) == 0:
        return {"symbol": symbol, "purge_gap_bars": purge_gap_bars, "status": "insufficient_sequence_rows"}
    dummy = {
        "X": np.zeros((len(timestamps), 1, 1), dtype=np.float32),
        "timestamp": np.asarray(timestamps, dtype=object),
        "symbol": np.asarray([symbol] * len(timestamps), dtype=object),
    }
    _, report = calendar_time_split(
        dummy,
        SplitConfig(
            sequence_length=AI_SEQUENCE_LENGTH,
            horizon_bars=AI_PRED_HORIZON_BARS,
            purge_gap_bars=purge_gap_bars,
            interval_minutes=interval_minutes,
        ),
    )
    sizes = dict(report.get("sizes") or {})
    return {
        "symbol": symbol,
        "purge_gap_bars": purge_gap_bars,
        "purge_gap_hours": float(purge_gap_bars * interval_minutes / 60.0),
        "status": "ok",
        "sample_count_after_sequence_horizon": int(len(timestamps)),
        "split_report": report,
        "valid_samples": int(sizes.get("valid") or 0),
        "test_samples": int(sizes.get("test") or 0),
    }


def frame_quality(symbol: str, frame: pd.DataFrame, interval_minutes: int, min_valid: int, min_test: int) -> dict[str, Any]:
    contract = get_contract(symbol)
    if frame.empty:
        return {"symbol": symbol, "status": "missing_data", "contract": contract}
    first = frame.index.min()
    last = frame.index.max()
    split_96 = split_sample_report(frame, symbol, 96, interval_minutes)
    split_288 = split_sample_report(frame, symbol, 288, interval_minutes)
    status = "FUTURES_DATA_READY"
    for split in [split_288]:
        if int(split.get("valid_samples") or 0) < min_valid or int(split.get("test_samples") or 0) < min_test:
            status = "INSUFFICIENT_FUTURES_DATA"
    return {
        "symbol": symbol,
        "status": status,
        "contract": contract,
        "row_count": int(len(frame)),
        "first_timestamp_utc": first.isoformat(),
        "last_timestamp_utc": last.isoformat(),
        "first_timestamp_kst": first.tz_convert("Asia/Seoul").isoformat(),
        "last_timestamp_kst": last.tz_convert("Asia/Seoul").isoformat(),
        "first_timestamp_cme": first.tz_convert("America/Chicago").isoformat(),
        "last_timestamp_cme": last.tz_convert("America/Chicago").isoformat(),
        "unique_utc_dates": int(frame.index.normalize().nunique()),
        "gap_report": gap_report(frame, interval_minutes),
        "rollover_boundary_candidates": rollover_candidates(frame, jump_threshold=0.02),
        "rollover_handling": "not back-adjusted; large open-vs-prev-close jumps are flagged as candidates only",
        "split_samples_purge_96": split_96,
        "split_samples_purge_288_default_candidate": split_288,
        "min_valid_samples": min_valid,
        "min_test_samples": min_test,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    fetched: dict[str, Any] = {}
    qualities: dict[str, Any] = {}
    for symbol in symbols:
        try:
            frame = fetch_yfinance_futures(symbol, args.period, args.interval)
            saved_rows = save_cache(symbol, frame, args.interval)
            cached = load_cache(symbol, args.interval)
            fetched[symbol] = {
                "status": "ok" if not frame.empty else "empty",
                "fetched_rows": int(len(frame)),
                "cache_rows_after_merge": saved_rows,
                "cache_path": str(_cache_path(symbol, args.interval)),
            }
            qualities[symbol] = frame_quality(symbol, cached, args.interval_minutes, args.min_valid_samples, args.min_test_samples)
        except Exception as exc:
            fetched[symbol] = {"status": "error", "error": str(exc)}
            qualities[symbol] = {"symbol": symbol, "status": "INSUFFICIENT_FUTURES_DATA", "error": str(exc)}

    overall_status = "FUTURES_DATA_READY"
    if any(row.get("status") != "FUTURES_DATA_READY" for row in qualities.values()):
        overall_status = "INSUFFICIENT_FUTURES_DATA"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": overall_status,
        "order_api_called": False,
        "env_loaded": False,
        "data_source": "yfinance_only",
        "kis_client_import_required": False,
        "symbols": symbols,
        "interval": args.interval,
        "period": args.period,
        "default_futures_embargo_candidate_bars": 288,
        "comparison_purge_bars": [96, 288],
        "contracts": {symbol: FUTURES_CONTRACTS.get(symbol, {}) for symbol in symbols},
        "fetch_results": fetched,
        "quality_by_symbol": qualities,
        "stage_gate_note": "Stage 1 only checks data availability/quality; no model edge is implied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="MNQ,MES")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
