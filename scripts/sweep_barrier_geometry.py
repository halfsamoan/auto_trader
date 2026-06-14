#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.6)
# Dependency: ai/*, config.py, core/fetcher_intraday.py
# Description: Offline triple-barrier geometry sweep without order calls.
# ================================================================================

"""Sweep target/stop/horizon geometry on local intraday cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.action_label_builder import ActionLabelConfig
from ai.dataset import (
    SplitConfig,
    build_policy_symbol_samples,
    calendar_time_split,
    concat_policy_samples,
)
from ai.train_baseline_lgbm import (
    _binary_metrics,
    _flatten_last_step,
    _metric_backend,
    _model_backend,
    _select_threshold,
    _threshold_curve,
)
from ai.universe_builder import load_ai_universe_records
from config import (
    AI_LABEL_MODE,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    COMMISSION,
    EXCLUDED_CODES,
    LEVERAGE_ETN_WATCHLIST,
    TAX,
    WATCHLIST,
)
from core.fetcher_intraday import load_intraday_cache


REPORT_PATH = ROOT / "ai" / "model_store" / "barrier_geometry_sweep_report.json"
CSV_PATH = ROOT / "ai" / "model_store" / "barrier_geometry_sweep_rows.csv"


@dataclass(frozen=True)
class Geometry:
    target_return: float
    stop_return: float
    horizon_bars: int


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def excluded_codes() -> set[str]:
    leveraged = {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}
    configured = {str(code).zfill(6) for code in EXCLUDED_CODES}
    return leveraged | configured


def resolve_codes(universe: str, watchlist: str | None) -> list[str]:
    if watchlist:
        raw = [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    elif universe == "ai_train":
        raw = [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    else:
        raw = [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    excluded = excluded_codes()
    return [code for code in dict.fromkeys(raw) if code not in excluded]


def geometry_grid(args: argparse.Namespace) -> list[Geometry]:
    targets = parse_float_list(args.targets)
    stops = [-abs(value) for value in parse_float_list(args.stops)]
    horizons = parse_int_list(args.horizons)
    return [Geometry(target, stop, horizon) for target, stop, horizon in product(targets, stops, horizons)]


def label_geometry_for_frame(data, geometry: Geometry) -> dict[str, Any]:
    target_count = stop_count = neutral_count = same_bar_count = invalid_count = 0
    touch_lags: list[int] = []
    outcome_returns: list[float] = []
    close = data["Close"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    n = len(close)
    for idx in range(n):
        entry = float(close[idx])
        if entry <= 0:
            invalid_count += 1
            neutral_count += 1
            outcome_returns.append(0.0)
            continue
        target_price = entry * (1.0 + geometry.target_return)
        stop_price = entry * (1.0 + geometry.stop_return)
        outcome = "neutral"
        outcome_return = 0.0
        end = min(idx + 1 + geometry.horizon_bars, n)
        for j in range(idx + 1, end):
            hit_stop = low[j] <= stop_price
            hit_target = high[j] >= target_price
            if hit_stop and hit_target:
                same_bar_count += 1
                outcome = "stop"
                outcome_return = geometry.stop_return
                touch_lags.append(j - idx)
                break
            if hit_target:
                outcome = "target"
                outcome_return = geometry.target_return
                touch_lags.append(j - idx)
                break
            if hit_stop:
                outcome = "stop"
                outcome_return = geometry.stop_return
                touch_lags.append(j - idx)
                break
        if outcome == "neutral":
            end_idx = min(idx + geometry.horizon_bars, n - 1)
            outcome_return = float(close[end_idx] / entry - 1.0)
            neutral_count += 1
        elif outcome == "target":
            target_count += 1
        else:
            stop_count += 1
        outcome_returns.append(outcome_return)
    total = max(target_count + stop_count + neutral_count, 1)
    non_neutral = max(target_count + stop_count, 1)
    return {
        "bar_count": int(len(data)),
        "target_count": int(target_count),
        "stop_count": int(stop_count),
        "neutral_count": int(neutral_count),
        "same_bar_target_stop_count": int(same_bar_count),
        "same_bar_target_stop_ratio": float(same_bar_count / total),
        "invalid_entry_count": int(invalid_count),
        "target_ratio": float(target_count / total),
        "stop_ratio": float(stop_count / total),
        "neutral_ratio": float(neutral_count / total),
        "target_ratio_without_neutral": float(target_count / non_neutral),
        "avg_touch_lag_bars": float(np.mean(touch_lags)) if touch_lags else None,
        "avg_outcome_return": float(np.mean(outcome_returns)) if outcome_returns else None,
    }


def aggregate_label_only(codes: list[str], geometry: Geometry, interval: str, include_symbol_rows: bool = False) -> dict[str, Any]:
    totals = {
        "bar_count": 0,
        "target_count": 0,
        "stop_count": 0,
        "neutral_count": 0,
        "same_bar_target_stop_count": 0,
        "invalid_entry_count": 0,
    }
    touch_lags = []
    symbol_rows: dict[str, Any] = {}
    for code in codes:
        data = load_intraday_cache(code, interval)
        if data.empty:
            if include_symbol_rows:
                symbol_rows[code] = {"bar_count": 0, "status": "missing_cache"}
            continue
        row = label_geometry_for_frame(data, geometry)
        if include_symbol_rows:
            symbol_rows[code] = row
        for key in totals:
            totals[key] += int(row.get(key) or 0)
        if row.get("avg_touch_lag_bars") is not None:
            touch_lags.append(float(row["avg_touch_lag_bars"]))
    total = max(totals["target_count"] + totals["stop_count"] + totals["neutral_count"], 1)
    non_neutral = max(totals["target_count"] + totals["stop_count"], 1)
    cost_return = 2.0 * float(COMMISSION) + float(TAX)
    breakeven = (cost_return - geometry.stop_return) / max(geometry.target_return - geometry.stop_return, 1e-12)
    return {
        "target_return": geometry.target_return,
        "stop_return": geometry.stop_return,
        "horizon_bars": geometry.horizon_bars,
        "symbol_count": len(codes),
        "bar_count": int(totals["bar_count"]),
        "target_count": int(totals["target_count"]),
        "stop_count": int(totals["stop_count"]),
        "neutral_count": int(totals["neutral_count"]),
        "same_bar_target_stop_count": int(totals["same_bar_target_stop_count"]),
        "same_bar_target_stop_ratio": float(totals["same_bar_target_stop_count"] / total),
        "target_ratio": float(totals["target_count"] / total),
        "stop_ratio": float(totals["stop_count"] / total),
        "neutral_ratio": float(totals["neutral_count"] / total),
        "target_ratio_without_neutral": float(totals["target_count"] / non_neutral),
        "avg_touch_lag_bars_across_symbols": float(np.mean(touch_lags)) if touch_lags else None,
        "domestic_cost_return": cost_return,
        "breakeven_precision": float(breakeven),
        "precision_gap_to_breakeven_label_only": float(totals["target_count"] / non_neutral - breakeven),
        "symbol_rows": symbol_rows if include_symbol_rows else "omitted; pass --include-symbol-rows for per-symbol details",
    }


def train_geometry(codes: list[str], geometry: Geometry, args: argparse.Namespace) -> dict[str, Any]:
    backend_name, model_cls, dependency_error = _model_backend()
    metric_fns, metric_error = _metric_backend()
    report: dict[str, Any] = {
        "target_return": geometry.target_return,
        "stop_return": geometry.stop_return,
        "horizon_bars": geometry.horizon_bars,
        "backend": backend_name,
        "model_dependency_error": dependency_error,
        "metric_dependency_error": metric_error,
        "status": "not_started",
    }
    if model_cls is None:
        report["status"] = "MODEL_DEPENDENCY_MISSING"
        return report

    train_codes = codes[: args.max_train_symbols] if args.max_train_symbols and args.max_train_symbols > 0 else codes
    label_config = ActionLabelConfig(geometry.horizon_bars, geometry.target_return, geometry.stop_return, label_mode=AI_LABEL_MODE)
    samples = []
    failures = {}
    for code in train_codes:
        data = load_intraday_cache(code, args.interval)
        if data.empty:
            failures[code] = "missing_cache"
            continue
        try:
            sample = build_policy_symbol_samples(data, code, label_config, AI_SEQUENCE_LENGTH)
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            failures[code] = str(exc)
    report["failed_symbols"] = failures
    report["successful_symbol_count"] = len(samples)
    report["train_symbol_count"] = len(train_codes)
    report["train_symbol_limit_note"] = (
        "train-each limited by --max-train-symbols; label-only rows still use the full resolved universe"
        if len(train_codes) != len(codes)
        else None
    )
    if not samples:
        report["status"] = "DATA_NOT_READY"
        return report

    data = concat_policy_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, geometry.horizon_bars, AI_PURGE_GAP_BARS))
    report["split_report"] = split_report
    if int(split_report["sizes"]["valid"]) < args.min_valid_samples or int(split_report["sizes"]["test"]) < args.min_test_samples:
        report["status"] = "DATA_NOT_READY"
        return report

    x_train = _flatten_last_step(splits["train"]["X"])
    x_valid = _flatten_last_step(splits["valid"]["X"])
    x_test = _flatten_last_step(splits["test"]["X"])
    y_train = (splits["train"]["action_class"] == 1).astype(int)
    y_valid = (splits["valid"]["action_class"] == 1).astype(int)
    y_test = (splits["test"]["action_class"] == 1).astype(int)
    target_count = int(np.sum(y_train == 1))
    stop_count = int(np.sum(y_train == 0))
    scale_pos_weight = stop_count / max(target_count, 1)
    model_kwargs = {"random_state": 42}
    if backend_name == "lightgbm":
        model_kwargs.update(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": scale_pos_weight,
                "verbose": -1,
            }
        )
    else:
        model_kwargs.update(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": 4,
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
            }
        )
    model = model_cls(**model_kwargs)
    model.fit(x_train, y_train)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    thresholds = [round(float(value), 2) for value in np.arange(args.threshold_min, args.threshold_max + 0.001, args.threshold_step)]
    cost_return = 2.0 * float(COMMISSION) + float(TAX)
    valid_curve = _threshold_curve(y_valid, valid_prob, splits["valid"]["expected_return_target"], cost_return, thresholds)
    selected = _select_threshold(valid_curve, args.min_selected_trades)
    threshold = float(selected.get("threshold", args.threshold_min))
    valid_metrics = _binary_metrics(y_valid, valid_prob, threshold, metric_fns, cost_return, splits["valid"]["expected_return_target"])
    test_metrics = _binary_metrics(y_test, test_prob, threshold, metric_fns, cost_return, splits["test"]["expected_return_target"])
    report.update(
        {
            "status": "ok",
            "model_params": model_kwargs,
            "selected_threshold": selected,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "edge_status": "GEOMETRY_CANDIDATE_DOMESTIC"
            if (test_metrics.get("cost_adjusted_expected_return_on_selected") or -999.0) > 0
            and int(test_metrics.get("predicted_trade_count") or 0) >= args.min_selected_trades
            else "NO_GEOMETRY_EDGE",
        }
    )
    return report


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "target_return",
        "stop_return",
        "horizon_bars",
        "bar_count",
        "target_ratio_without_neutral",
        "neutral_ratio",
        "same_bar_target_stop_ratio",
        "breakeven_precision",
        "precision_gap_to_breakeven_label_only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--targets", default="0.0035,0.0045,0.006")
    parser.add_argument("--stops", default="0.0025,0.0035")
    parser.add_argument("--horizons", default="6,9,12")
    parser.add_argument("--mode", default="label-only", choices=["label-only", "train-each", "both"])
    parser.add_argument("--max-combinations", type=int, default=20)
    parser.add_argument("--max-train-symbols", type=int, default=None)
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--min-selected-trades", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.82)
    parser.add_argument("--threshold-step", type=float, default=0.04)
    parser.add_argument("--output", default=str(REPORT_PATH))
    parser.add_argument("--csv-output", default=str(CSV_PATH))
    parser.add_argument("--include-symbol-rows", action="store_true")
    args = parser.parse_args()

    codes = resolve_codes(args.universe, args.watchlist)
    grid = geometry_grid(args)
    label_rows = [aggregate_label_only(codes, geometry, args.interval, args.include_symbol_rows) for geometry in grid]
    train_rows = []
    if args.mode in {"train-each", "both"}:
        ranked = sorted(
            zip(grid, label_rows),
            key=lambda item: float(item[1].get("precision_gap_to_breakeven_label_only") or -999.0),
            reverse=True,
        )
        train_geometries = [geometry for geometry, _ in ranked[: max(args.max_combinations, 0)]]
        for geometry in train_geometries:
            train_rows.append(train_geometry(codes, geometry, args))
    positive_train_rows = [
        row for row in train_rows if row.get("edge_status") == "GEOMETRY_CANDIDATE_DOMESTIC"
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if label_rows else "DATA_NOT_READY",
        "edge_status": "GEOMETRY_CANDIDATE_DOMESTIC" if positive_train_rows else "NO_GEOMETRY_EDGE",
        "order_api_called": False,
        "env_loaded": False,
        "universe": args.universe,
        "symbol_count": len(codes),
        "excluded_symbols": sorted(excluded_codes()),
        "purge_gap_bars": AI_PURGE_GAP_BARS,
        "mode": args.mode,
        "geometry_count": len(grid),
        "train_each_selection": "top_label_only_precision_gap",
        "train_each_max_combinations": args.max_combinations if args.mode in {"train-each", "both"} else 0,
        "label_only_rows": label_rows,
        "train_each_rows": train_rows,
        "top_label_only_by_precision_gap": sorted(
            label_rows,
            key=lambda row: float(row.get("precision_gap_to_breakeven_label_only") or -999.0),
            reverse=True,
        )[:10],
        "multiple_testing_note": "Geometry sweep is exploratory. Do not enable orders from this report alone.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(label_rows, Path(args.csv_output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
