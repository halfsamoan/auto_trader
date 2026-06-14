#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: ai/futures_dataset.py, ai/train_baseline_lgbm.py, futures_contracts.py
# Description: Futures-only baseline and barrier sweep without KIS, .env, or orders.
# ================================================================================

"""Train LightGBM/XGBoost futures baseline and run barrier geometry sweep."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.action_label_builder import ActionLabelConfig
from ai.dataset import SplitConfig, calendar_time_split
from ai.futures_dataset import (
    FUTURES_FEATURE_COLUMNS,
    aggregate_futures_label_distribution,
    build_futures_symbol_samples,
    concat_futures_samples,
    load_futures_cache,
)
from ai.train_baseline_lgbm import _flatten_last_step, _metric_backend, _model_backend
from config import AI_LABEL_MODE, AI_PRED_HORIZON_BARS, AI_SEQUENCE_LENGTH, AI_STOP_RETURN, AI_TARGET_RETURN
from futures_contracts import get_contract
from scripts.audit_cost_model import cost_bps_futures
from scripts.sweep_barrier_geometry import Geometry, label_geometry_for_frame


REPORT_PATH = ROOT / "ai" / "model_store" / "futures_baseline_report.json"


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def futures_cost_return(symbol: str, slippage_ticks: float, commission_per_side_usd: float) -> dict[str, Any]:
    contract = get_contract(symbol)
    data = load_futures_cache(symbol)
    if data.empty:
        return {"status": "missing_cache", "symbol": symbol, "cost_return": None}
    latest_price = float(data["Close"].iloc[-1])
    tick_value = float(contract["tick_value_usd"])
    tick_size = float(contract["tick_size"])
    multiplier = float(contract.get("multiplier") or tick_value / tick_size)
    bps = cost_bps_futures(latest_price, multiplier, tick_value, slippage_ticks, commission_per_side_usd)
    return {
        "status": "ok",
        "symbol": symbol,
        "price": latest_price,
        "price_timestamp_utc": data.index[-1].isoformat(),
        "tick_value_usd": tick_value,
        "tick_size": tick_size,
        "multiplier": multiplier,
        "slippage_ticks_per_side": float(slippage_ticks),
        "commission_per_side_usd": float(commission_per_side_usd),
        "cost_bps": bps,
        "cost_return": None if bps is None else float(bps) / 10_000.0,
    }


def _metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    metric_fns: dict[str, Any] | None,
    gross_returns: np.ndarray,
    cost_returns: np.ndarray,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"sample_count": 0}
    pred = prob >= threshold
    positive = y_true == 1
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    tn = int(np.sum(~pred & ~positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    selected_net = gross_returns[pred] - cost_returns[pred]
    accuracy = float(metric_fns["accuracy_score"](y_true, pred)) if metric_fns else float(np.mean(y_true == pred))
    roc_auc = None
    pr_auc = None
    if metric_fns and len(np.unique(y_true)) >= 2:
        try:
            roc_auc = float(metric_fns["roc_auc_score"](y_true, prob))
            pr_auc = float(metric_fns["average_precision_score"](y_true, prob))
        except Exception:
            pass
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "target_ratio": float(np.mean(positive)),
        "predicted_trade_count": int(np.sum(pred)),
        "predicted_trade_ratio": float(np.mean(pred)),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "precision_lift": precision - float(np.mean(positive)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "gross_expected_return_on_selected": float(np.mean(gross_returns[pred])) if np.any(pred) else None,
        "net_expected_return_on_selected": float(np.mean(selected_net)) if len(selected_net) else None,
    }


def _threshold_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    gross_returns: np.ndarray,
    cost_returns: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    base_ratio = float(np.mean(y_true == 1)) if len(y_true) else 0.0
    for threshold in thresholds:
        pred = prob >= threshold
        if not np.any(pred):
            rows.append(
                {
                    "threshold": float(threshold),
                    "predicted_trade_count": 0,
                    "precision": None,
                    "recall": 0.0,
                    "precision_lift": None,
                    "gross_expected_return": None,
                    "net_expected_return": None,
                }
            )
            continue
        positive = y_true == 1
        selected_net = gross_returns[pred] - cost_returns[pred]
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_trade_count": int(np.sum(pred)),
                "predicted_trade_ratio": float(np.mean(pred)),
                "precision": float(np.mean(positive[pred])),
                "recall": float(np.sum(pred & positive) / max(np.sum(positive), 1)),
                "precision_lift": float(np.mean(positive[pred])) - base_ratio,
                "gross_expected_return": float(np.mean(gross_returns[pred])),
                "net_expected_return": float(np.mean(selected_net)),
            }
        )
    return rows


def _select_threshold(curve: list[dict[str, Any]], min_trades: int) -> dict[str, Any]:
    candidates = [
        row
        for row in curve
        if int(row.get("predicted_trade_count") or 0) >= min_trades and row.get("net_expected_return") is not None
    ]
    if not candidates:
        return {"threshold": 0.50, "selection_reason": "fallback_no_valid_candidate"}
    profitable = [row for row in candidates if float(row.get("net_expected_return") or -999.0) > 0.0]
    pool = profitable or candidates
    selected = max(
        pool,
        key=lambda row: (
            float(row.get("net_expected_return") or -999.0),
            float(row.get("precision_lift") or -999.0),
            int(row.get("predicted_trade_count") or 0),
        ),
    )
    return {
        **selected,
        "selection_reason": "valid_only_positive_net_ev" if profitable else "valid_only_best_available",
    }


def _sample_with_cost(sample: dict[str, Any], cost_return: float) -> dict[str, Any]:
    out = dict(sample)
    out["cost_return"] = np.full(len(sample["X"]), float(cost_return), dtype=np.float32)
    return out


def _concat_samples_with_cost(samples: list[dict[str, Any]]) -> dict[str, Any]:
    data = concat_futures_samples(samples)
    if not any(len(s["X"]) for s in samples):
        data["cost_return"] = np.asarray([], dtype=np.float32)
        return data
    costs = np.concatenate([s["cost_return"] for s in samples if len(s["X"])], axis=0)
    order = np.argsort(np.concatenate([s["timestamp"] for s in samples if len(s["X"])], axis=0))
    data["cost_return"] = costs[order]
    return data


def train_dataset_report(name: str, samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    backend_name, model_cls, dependency_error = _model_backend()
    metric_fns, metric_error = _metric_backend()
    report: dict[str, Any] = {
        "name": name,
        "status": "not_started",
        "backend": backend_name,
        "model_dependency_error": dependency_error,
        "metric_dependency_error": metric_error,
        "sample_count": int(sum(len(s["X"]) for s in samples)),
        "feature_columns": FUTURES_FEATURE_COLUMNS,
    }
    if model_cls is None:
        report["status"] = "MODEL_DEPENDENCY_MISSING"
        return report
    if not samples or not any(len(s["X"]) for s in samples):
        report["status"] = "INSUFFICIENT_FUTURES_DATA"
        return report
    data = _concat_samples_with_cost(samples)
    splits, split_report = calendar_time_split(
        data,
        SplitConfig(
            sequence_length=args.sequence_length,
            horizon_bars=args.horizon_bars,
            purge_gap_bars=args.purge_gap_bars,
            interval_minutes=5,
        ),
    )
    report["split_report"] = split_report
    if int(split_report["sizes"]["valid"]) < args.min_valid_samples or int(split_report["sizes"]["test"]) < args.min_test_samples:
        report["status"] = "INSUFFICIENT_FUTURES_DATA"
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
    kwargs: dict[str, Any] = {"random_state": 42}
    if backend_name == "lightgbm":
        kwargs.update(
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
        kwargs.update(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": 4,
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
            }
        )
    model = model_cls(**kwargs)
    model.fit(x_train, y_train)
    train_prob = model.predict_proba(x_train)[:, 1]
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    thresholds = [round(float(value), 2) for value in np.arange(args.threshold_min, args.threshold_max + 0.001, args.threshold_step)]
    valid_curve = _threshold_curve(
        y_valid,
        valid_prob,
        splits["valid"]["expected_return_target"],
        splits["valid"]["cost_return"],
        thresholds,
    )
    selected = _select_threshold(valid_curve, args.min_selected_trades)
    threshold = float(selected.get("threshold", 0.50))
    train_metrics = _metrics(y_train, train_prob, threshold, metric_fns, splits["train"]["expected_return_target"], splits["train"]["cost_return"])
    valid_metrics = _metrics(y_valid, valid_prob, threshold, metric_fns, splits["valid"]["expected_return_target"], splits["valid"]["cost_return"])
    test_metrics = _metrics(y_test, test_prob, threshold, metric_fns, splits["test"]["expected_return_target"], splits["test"]["cost_return"])
    test_net = test_metrics.get("net_expected_return_on_selected")
    edge_status = (
        "COST_SENSITIVITY_POSITIVE_ONLY"
        if int(test_metrics.get("predicted_trade_count") or 0) >= args.min_selected_trades and test_net is not None and float(test_net) > 0.0
        else "EDGE_NOT_CONFIRMED"
    )
    report.update(
        {
            "status": "ok",
            "model_params": kwargs,
            "selected_threshold": selected,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "edge_status": edge_status,
            "test_evaluation_note": "test metrics are produced once after valid-only threshold selection",
        }
    )
    return report


def geometry_grid(args: argparse.Namespace) -> list[Geometry]:
    return [
        Geometry(target, -abs(stop), horizon)
        for target, stop, horizon in product(_parse_float_list(args.targets), _parse_float_list(args.stops), _parse_int_list(args.horizons))
    ]


def barrier_sweep_for_symbol(symbol: str, cost_return: float, args: argparse.Namespace) -> dict[str, Any]:
    data = load_futures_cache(symbol)
    if data.empty:
        return {"symbol": symbol, "status": "INSUFFICIENT_FUTURES_DATA"}
    rows = []
    for geometry in geometry_grid(args):
        row = label_geometry_for_frame(data, geometry)
        trade_count = int(row["target_count"]) + int(row["stop_count"])
        breakeven = (cost_return - geometry.stop_return) / max(geometry.target_return - geometry.stop_return, 1e-12)
        precision = float(row["target_ratio_without_neutral"])
        gross_ev_non_neutral = (
            (int(row["target_count"]) * geometry.target_return + int(row["stop_count"]) * geometry.stop_return) / max(trade_count, 1)
        )
        rows.append(
            {
                "symbol": symbol,
                "target_return": geometry.target_return,
                "stop_return": geometry.stop_return,
                "horizon_bars": geometry.horizon_bars,
                "trade_count": trade_count,
                "target_count": row["target_count"],
                "stop_count": row["stop_count"],
                "neutral_count": row["neutral_count"],
                "same_bar_target_stop_ratio": row["same_bar_target_stop_ratio"],
                "precision": precision,
                "gross_ev_non_neutral": float(gross_ev_non_neutral),
                "cost_return": float(cost_return),
                "net_ev_non_neutral": float(gross_ev_non_neutral - cost_return),
                "breakeven_precision": float(breakeven),
                "precision_margin_vs_breakeven": float(precision - breakeven),
            }
        )
    top = sorted(rows, key=lambda row: (row["net_ev_non_neutral"], row["precision_margin_vs_breakeven"]), reverse=True)[:10]
    status = "COST_SENSITIVITY_POSITIVE_ONLY" if top and float(top[0]["net_ev_non_neutral"]) > 0 else "NO_GEOMETRY_EDGE"
    return {"symbol": symbol, "status": status, "rows": rows, "top_rows": top}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _parse_symbols(args.symbols)
    label_config = ActionLabelConfig(
        horizon_bars=args.horizon_bars,
        target_return=args.target_return,
        stop_return=-abs(args.stop_return),
        label_mode=AI_LABEL_MODE,
    )
    cost_by_symbol = {
        symbol: futures_cost_return(symbol, args.slippage_ticks, args.commission_per_side_usd)
        for symbol in symbols
    }
    samples_by_symbol: dict[str, list[dict[str, Any]]] = {}
    all_samples = []
    failures: dict[str, str] = {}
    for symbol in symbols:
        data = load_futures_cache(symbol, args.interval)
        if data.empty:
            failures[symbol] = "missing_cache"
            samples_by_symbol[symbol] = []
            continue
        try:
            sample = build_futures_symbol_samples(data, symbol, label_config, args.sequence_length)
            cost_return = cost_by_symbol[symbol].get("cost_return")
            if cost_return is None:
                failures[symbol] = "missing_cost"
                samples_by_symbol[symbol] = []
                continue
            sample = _sample_with_cost(sample, float(cost_return))
            samples_by_symbol[symbol] = [sample]
            all_samples.append(sample)
        except Exception as exc:
            failures[symbol] = str(exc)
            samples_by_symbol[symbol] = []
    symbol_reports = {
        symbol: train_dataset_report(symbol, samples_by_symbol.get(symbol, []), args)
        for symbol in symbols
    }
    combined_report = train_dataset_report("MNQ_MES_combined_correlated", all_samples, args)
    barrier_reports = {
        symbol: barrier_sweep_for_symbol(symbol, float(cost_by_symbol[symbol]["cost_return"]), args)
        for symbol in symbols
        if cost_by_symbol[symbol].get("cost_return") is not None
    }
    edge_values = [row.get("edge_status") for row in symbol_reports.values()]
    if any(row.get("status") == "INSUFFICIENT_FUTURES_DATA" for row in symbol_reports.values()):
        edge_status = "INSUFFICIENT_FUTURES_DATA"
    elif all(value == "COST_SENSITIVITY_POSITIVE_ONLY" for value in edge_values):
        edge_status = "COST_SENSITIVITY_POSITIVE_ONLY"
    else:
        edge_status = "EDGE_NOT_CONFIRMED"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not failures else "partial",
        "edge_status": edge_status,
        "order_api_called": False,
        "env_loaded": False,
        "symbols": symbols,
        "interval": args.interval,
        "purge_gap_bars": args.purge_gap_bars,
        "cost_scenario": {
            "slippage_ticks_per_side": args.slippage_ticks,
            "commission_per_side_usd": args.commission_per_side_usd,
            "cost_by_symbol": cost_by_symbol,
        },
        "label_config": {
            "target_return": args.target_return,
            "stop_return": -abs(args.stop_return),
            "horizon_bars": args.horizon_bars,
            "expected_return_target_definition": "target/stop use nominal barrier return; neutral timeout uses realized close-to-close return and is dropped in baseline labels",
        },
        "action_label_distribution": aggregate_futures_label_distribution(all_samples),
        "symbol_reports": symbol_reports,
        "combined_correlated_report": combined_report,
        "barrier_sweep": barrier_reports,
        "correlation_warning": "MNQ and MES are highly related US index futures; combined rows are not independent evidence.",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="MNQ,MES")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--sequence-length", type=int, default=AI_SEQUENCE_LENGTH)
    parser.add_argument("--horizon-bars", type=int, default=AI_PRED_HORIZON_BARS)
    parser.add_argument("--target-return", type=float, default=AI_TARGET_RETURN)
    parser.add_argument("--stop-return", type=float, default=abs(AI_STOP_RETURN))
    parser.add_argument("--purge-gap-bars", type=int, default=288)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    parser.add_argument("--commission-per-side-usd", type=float, default=0.62)
    parser.add_argument("--targets", default="0.0008,0.0012,0.0016,0.0020,0.0035")
    parser.add_argument("--stops", default="0.0008,0.0012,0.0016,0.0025")
    parser.add_argument("--horizons", default="6,12,24,48")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--min-selected-trades", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--threshold-min", type=float, default=0.40)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.04)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
