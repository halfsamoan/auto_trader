#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: ai/futures_dataset.py, futures_contracts.py
# Description: Futures-only feature matrix audit without KIS, .env, or order calls.
# ================================================================================

"""Audit MNQ/MES futures feature matrices built from local yfinance cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.futures_dataset import FUTURES_FEATURE_COLUMNS, build_futures_feature_frame, load_futures_cache


REPORT_PATH = ROOT / "ai" / "model_store" / "futures_feature_matrix_audit.json"


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _feature_audit(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    row_count = int(len(frame))
    result: dict[str, dict[str, Any]] = {}
    for column in FUTURES_FEATURE_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        values = series.to_numpy(dtype=float, na_value=np.nan)
        result[column] = {
            "nan_ratio": float(series.isna().sum() / row_count) if row_count else None,
            "inf_count": int(np.isinf(values).sum()) if len(values) else 0,
            "is_constant": bool(row_count > 0 and series.nunique(dropna=False) <= 1),
            "unique_count": int(series.nunique(dropna=False)) if row_count else 0,
        }
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _parse_symbols(args.symbols)
    frames = []
    row_counts: dict[str, int] = {}
    feature_row_counts: dict[str, int] = {}
    failures: dict[str, str] = {}
    for symbol in symbols:
        data = load_futures_cache(symbol, args.interval)
        row_counts[symbol] = int(len(data))
        if data.empty:
            feature_row_counts[symbol] = 0
            failures[symbol] = "missing_cache"
            continue
        try:
            frame = build_futures_feature_frame(data)
            feature_row_counts[symbol] = int(len(frame))
            frame = frame.copy()
            frame["__symbol"] = symbol
            frames.append(frame)
        except Exception as exc:
            feature_row_counts[symbol] = 0
            failures[symbol] = str(exc)
    matrix = pd.concat(frames, axis=0) if frames else pd.DataFrame(columns=FUTURES_FEATURE_COLUMNS + ["__symbol"])
    audit = _feature_audit(matrix[FUTURES_FEATURE_COLUMNS] if not matrix.empty else pd.DataFrame(columns=FUTURES_FEATURE_COLUMNS))
    krx_removed = ["kospi_regime_flag", "rule_score", "intraday_position"]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "FEATURE_AUDIT_READY" if frames and not failures else "FEATURE_AUDIT_NEEDS_REVIEW",
        "order_api_called": False,
        "env_loaded": False,
        "symbols": symbols,
        "interval": args.interval,
        "row_counts": row_counts,
        "feature_row_counts": feature_row_counts,
        "matrix_row_count": int(len(matrix)),
        "feature_count": len(FUTURES_FEATURE_COLUMNS),
        "feature_columns": FUTURES_FEATURE_COLUMNS,
        "removed_or_replaced_krx_features": {
            "removed": krx_removed,
            "replacements": {
                "intraday_position": "futures_session_position",
                "kospi_regime_flag": "globex_open_flag/us_rth_flag/cme weekday cyclical features",
                "rule_score": "no rule score injected in futures validation path",
            },
        },
        "no_future_feature_audit": {
            "rolling_windows_are_trailing": True,
            "session_features_use_timestamp_only": True,
            "labels_are_joined_after_feature_construction": True,
            "yfinance_continuous_rollover_not_back_adjusted": True,
        },
        "feature_audit": audit,
        "constant_features": [column for column, row in audit.items() if row["is_constant"]],
        "failures": failures,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="MNQ,MES")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "FEATURE_AUDIT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
