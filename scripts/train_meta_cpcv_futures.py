#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: ai/futures_dataset.py, ai/meta_labeling.py, futures_contracts.py
# Description: Futures-only CPCV/meta-labeling report without KIS, .env, or orders.
# ================================================================================

"""Run futures same-asset CPCV + meta-labeling validation for MNQ/MES."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.action_label_builder import ActionLabelConfig
from ai.cpcv import synthetic_overlap_self_test
from ai.dataset import SplitConfig, calendar_time_split
from ai.futures_dataset import (
    aggregate_futures_label_distribution,
    build_futures_symbol_samples,
    concat_futures_samples,
    load_futures_cache,
)
from ai.meta_labeling import MetaLabelingConfig, evaluate_meta_cpcv, fit_meta_oof_test_once
from ai.train_baseline_lgbm import _model_backend
from config import AI_LABEL_MODE, AI_SEQUENCE_LENGTH, AI_STOP_RETURN, AI_TARGET_RETURN
from scripts.train_baseline_futures import futures_cost_return


REPORT_PATH = ROOT / "ai" / "model_store" / "futures_meta_cpcv_report.json"


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def concat_split_dicts(*splits: dict[str, Any]) -> dict[str, Any]:
    keys = list(splits[0].keys())
    out = {key: np.concatenate([split[key] for split in splits], axis=0) for key in keys}
    order = np.argsort(out["timestamp"])
    return {key: value[order] for key, value in out.items()}


def symbol_samples(symbol: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    data = load_futures_cache(symbol, args.interval)
    if data.empty:
        return None, {"status": "INSUFFICIENT_FUTURES_DATA", "reason": "missing_cache"}
    label_config = ActionLabelConfig(
        horizon_bars=args.horizon_bars,
        target_return=args.target_return,
        stop_return=-abs(args.stop_return),
        label_mode=AI_LABEL_MODE,
    )
    sample = build_futures_symbol_samples(data, symbol, label_config, args.sequence_length)
    if len(sample["X"]) == 0:
        return None, {"status": "INSUFFICIENT_FUTURES_DATA", "reason": "no_samples_after_label_filter"}
    return sample, None


def evaluate_symbol(symbol: str, args: argparse.Namespace) -> dict[str, Any]:
    backend_name, _, dependency_error = _model_backend()
    cost = futures_cost_return(symbol, args.slippage_ticks, args.commission_per_side_usd)
    sample, error = symbol_samples(symbol, args)
    base: dict[str, Any] = {
        "symbol": symbol,
        "status": "not_started",
        "edge_status": "EDGE_NOT_CONFIRMED",
        "backend": backend_name,
        "model_dependency_error": dependency_error,
        "cost": cost,
    }
    if backend_name is None:
        return {**base, "status": "MODEL_DEPENDENCY_MISSING"}
    if error or sample is None:
        return {**base, **(error or {}), "edge_status": "INSUFFICIENT_FUTURES_DATA"}
    cost_return = cost.get("cost_return")
    if cost_return is None:
        return {**base, "status": "INSUFFICIENT_FUTURES_DATA", "reason": "missing_cost", "edge_status": "INSUFFICIENT_FUTURES_DATA"}

    data = concat_futures_samples([sample])
    splits, split_report = calendar_time_split(
        data,
        SplitConfig(
            sequence_length=args.sequence_length,
            horizon_bars=args.horizon_bars,
            purge_gap_bars=args.purge_gap_bars,
            interval_minutes=5,
        ),
    )
    base["split_report"] = split_report
    if int(split_report["sizes"]["valid"]) < args.min_valid_samples or int(split_report["sizes"]["test"]) < args.min_test_samples:
        return {
            **base,
            "status": "INSUFFICIENT_FUTURES_DATA",
            "reason": "valid/test sample shortage after purge and label filtering",
            "edge_status": "INSUFFICIENT_FUTURES_DATA",
        }

    train_valid = concat_split_dicts(splits["train"], splits["valid"])
    meta_config = MetaLabelingConfig(
        cost_assumption_return=float(cost_return),
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        purge_gap_bars=args.purge_gap_bars,
        interval_minutes=5,
        max_paths=args.max_paths,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_meta_train_samples=args.min_meta_train_samples,
        min_selected_trades=args.min_selected_trades,
    )
    cpcv_report = evaluate_meta_cpcv(train_valid, meta_config)
    if cpcv_report.get("status") != "ok":
        return {
            **base,
            "status": cpcv_report.get("status", "INSUFFICIENT_FUTURES_DATA"),
            "cpcv_report": cpcv_report,
            "edge_status": cpcv_report.get("edge_status", "EDGE_NOT_CONFIRMED"),
        }
    primary_threshold = float(cpcv_report.get("selected_primary_threshold", 0.50))
    meta_threshold = float(cpcv_report.get("selected_meta_threshold", 0.60))
    test_once = fit_meta_oof_test_once(train_valid, splits["test"], meta_config, primary_threshold, meta_threshold)
    gate = gate_report(cpcv_report, test_once, args)
    return {
        **base,
        "status": "ok" if test_once.get("status") == "ok" else str(test_once.get("status", "EDGE_NOT_CONFIRMED")),
        "edge_status": gate["edge_status"],
        "label_distribution": aggregate_futures_label_distribution([sample]),
        "selected_primary_threshold": primary_threshold,
        "selected_meta_threshold": meta_threshold,
        "cpcv_report": cpcv_report,
        "test_once_report": test_once,
        "gate": gate,
    }


def gate_report(cpcv_report: dict[str, Any], test_once: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    meta_summary = cpcv_report.get("meta_summary") or {}
    dsr = cpcv_report.get("dsr") or {}
    pbo = cpcv_report.get("pbo") or {}
    test_meta = test_once.get("meta_test_metrics") or {}
    cpcv_net = meta_summary.get("mean_path_net_expected_return")
    test_net = test_meta.get("net_expected_return")
    test_trades = int(test_meta.get("selected_trade_count") or 0)
    dsr_probability = dsr.get("dsr_probability")
    dsr_p_value = None if dsr_probability is None else float(1.0 - float(dsr_probability))
    pbo_value = pbo.get("pbo")
    checks = {
        "cpcv_meta_net_ev_positive": cpcv_net is not None and float(cpcv_net) > 0.0,
        "test_meta_net_ev_positive": test_net is not None and float(test_net) > 0.0,
        "sufficient_test_trades": test_trades >= args.min_selected_trades,
        "dsr_significant": dsr_probability is not None and float(dsr_probability) >= args.min_dsr_probability,
        "pbo_low": pbo_value is not None and float(pbo_value) <= args.max_pbo,
    }
    if all(checks.values()):
        edge_status = "FUTURES_EDGE_CANDIDATE"
    elif checks["test_meta_net_ev_positive"] or checks["cpcv_meta_net_ev_positive"]:
        edge_status = "COST_SENSITIVITY_POSITIVE_ONLY"
    else:
        edge_status = "EDGE_NOT_CONFIRMED"
    return {
        "edge_status": edge_status,
        "checks": checks,
        "cpcv_meta_mean_net_ev": cpcv_net,
        "test_meta_net_ev": test_net,
        "test_meta_trades": test_trades,
        "dsr_probability": dsr_probability,
        "dsr_p_value_proxy": dsr_p_value,
        "pbo": pbo_value,
        "thresholds": {
            "min_dsr_probability": args.min_dsr_probability,
            "max_pbo": args.max_pbo,
            "min_selected_trades": args.min_selected_trades,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _parse_symbols(args.symbols)
    symbol_reports = {symbol: evaluate_symbol(symbol, args) for symbol in symbols}
    statuses = [row.get("edge_status") for row in symbol_reports.values()]
    if any(status == "INSUFFICIENT_FUTURES_DATA" for status in statuses):
        final_status = "INSUFFICIENT_FUTURES_DATA"
    elif all(status == "FUTURES_EDGE_CANDIDATE" for status in statuses):
        final_status = "FUTURES_EDGE_CANDIDATE"
    elif any(status == "COST_SENSITIVITY_POSITIVE_ONLY" for status in statuses):
        final_status = "COST_SENSITIVITY_POSITIVE_ONLY"
    else:
        final_status = "EDGE_NOT_CONFIRMED"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "edge_status": final_status,
        "order_api_called": False,
        "env_loaded": False,
        "symbols": symbols,
        "purge_gap_bars": args.purge_gap_bars,
        "cpcv_overlap_self_test": synthetic_overlap_self_test(),
        "label_config": {
            "target_return": args.target_return,
            "stop_return": -abs(args.stop_return),
            "horizon_bars": args.horizon_bars,
            "expected_return_target_definition": "target/stop use nominal barrier return; neutral timeout uses realized close-to-close return and is dropped from BUY/NO_ACTION training",
        },
        "cost_scenario": {
            "slippage_ticks_per_side": args.slippage_ticks,
            "commission_per_side_usd": args.commission_per_side_usd,
        },
        "gate_rules": {
            "requires_both_mnq_and_mes": True,
            "requires_positive_cpcv_and_test_net_ev": True,
            "requires_dsr_probability_at_least": args.min_dsr_probability,
            "requires_pbo_at_most": args.max_pbo,
            "requires_test_trades_at_least": args.min_selected_trades,
        },
        "symbol_reports": symbol_reports,
        "overlap_warning": "MNQ and MES are correlated US equity index futures; passing both is not equivalent to two independent samples.",
        "order_policy": "No order gates are opened by V3.2.7 results.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="MNQ,MES")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--sequence-length", type=int, default=AI_SEQUENCE_LENGTH)
    parser.add_argument("--target-return", type=float, default=0.0012)
    parser.add_argument("--stop-return", type=float, default=abs(AI_STOP_RETURN))
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--purge-gap-bars", type=int, default=288)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    parser.add_argument("--commission-per-side-usd", type=float, default=0.62)
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--min-meta-train-samples", type=int, default=80)
    parser.add_argument("--min-selected-trades", type=int, default=20)
    parser.add_argument("--n-groups", type=int, default=6)
    parser.add_argument("--n-test-groups", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-dsr-probability", type=float, default=0.95)
    parser.add_argument("--max-pbo", type=float, default=0.20)
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
