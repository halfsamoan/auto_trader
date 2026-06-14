#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.6)
# Dependency: ai/*, config.py, core/fetcher_intraday.py
# Description: Offline CPCV + meta-labeling training report without order calls.
# ================================================================================

"""Run cost-aware meta-labeling with native CPCV validation."""

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
from ai.dataset import (
    SplitConfig,
    aggregate_action_label_distribution,
    build_policy_symbol_samples,
    calendar_time_split,
    concat_policy_samples,
)
from ai.meta_labeling import MetaLabelingConfig, evaluate_meta_cpcv, fit_meta_oof_test_once
from ai.train_baseline_lgbm import _model_backend
from ai.universe_builder import load_ai_universe_records
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    COMMISSION,
    EXCLUDED_CODES,
    LEVERAGE_ETN_WATCHLIST,
    TAX,
    WATCHLIST,
)
from core.fetcher_intraday import load_intraday_cache


REPORT_PATH = ROOT / "ai" / "model_store" / "meta_cpcv_report.json"


def excluded_codes() -> set[str]:
    leveraged = {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}
    configured = {str(code).zfill(6) for code in EXCLUDED_CODES}
    return leveraged | configured


def resolve_codes(universe: str, watchlist: str | None, max_symbols: int | None) -> list[str]:
    if watchlist:
        raw = [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    elif universe == "ai_train":
        raw = [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    else:
        raw = [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    excluded = excluded_codes()
    codes = [code for code in dict.fromkeys(raw) if code not in excluded]
    if max_symbols is not None and max_symbols > 0:
        return codes[:max_symbols]
    return codes


def concat_split_dicts(*splits: dict[str, Any]) -> dict[str, Any]:
    keys = list(splits[0].keys())
    out = {key: np.concatenate([split[key] for split in splits], axis=0) for key in keys}
    order = np.argsort(out["timestamp"])
    return {key: value[order] for key, value in out.items()}


def build_samples(codes: list[str], interval: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    label_config = ActionLabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, label_mode=AI_LABEL_MODE)
    samples: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for code in codes:
        try:
            data = load_intraday_cache(code, interval)
            if data.empty:
                failures[code] = "missing_cache"
                continue
            sample = build_policy_symbol_samples(data, code, label_config, AI_SEQUENCE_LENGTH)
            all_samples.append(sample)
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            failures[code] = str(exc)
    return samples, all_samples, failures


def breakeven_precision(target_return: float, stop_return: float, cost_return: float) -> float | None:
    denominator = target_return - stop_return
    if denominator <= 0:
        return None
    return float((cost_return - stop_return) / denominator)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    backend_name, _, dependency_error = _model_backend()
    cost_return = 2.0 * float(COMMISSION) + float(TAX)
    codes = resolve_codes(args.universe, args.watchlist, args.max_symbols)
    samples, all_samples, failures = build_samples(codes, args.interval)
    distribution = aggregate_action_label_distribution(all_samples)
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "not_started",
        "edge_status": "EDGE_NOT_CONFIRMED",
        "order_api_called": False,
        "env_loaded": False,
        "model": "cost_aware_meta_labeling_cpcv",
        "backend": backend_name,
        "model_dependency_status": "ok" if backend_name else "MODEL_DEPENDENCY_MISSING",
        "model_dependency_error": dependency_error,
        "universe": args.universe,
        "symbol_count": len(codes),
        "excluded_symbols": sorted(excluded_codes()),
        "failed_symbols": failures,
        "purge_gap_bars": AI_PURGE_GAP_BARS,
        "cpcv_overlap_self_test": synthetic_overlap_self_test(),
        "cost_assumption_return": cost_return,
        "breakeven_precision_at_config_barriers": breakeven_precision(AI_TARGET_RETURN, AI_STOP_RETURN, cost_return),
        "expected_return_target_definition": "mixed: target/stop use nominal barrier return, neutral timeout uses realized close-to-close return; neutral labels are dropped for meta target",
        "action_label_distribution": distribution,
        "split_report": None,
        "cpcv_report": None,
        "test_once_report": None,
        "test_used_once_after_cpcv_threshold_selection": True,
        "allowed_edge_status_note": "EDGE_CONFIRMED is intentionally not emitted in PHASE B.",
    }
    if backend_name is None:
        report["status"] = "MODEL_DEPENDENCY_MISSING"
        return report
    if not samples:
        report["status"] = "DATA_NOT_READY"
        report["reason"] = "no policy samples after exclusions and neutral filtering"
        return report

    data = concat_policy_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
    report["split_report"] = split_report
    if int(split_report["sizes"]["valid"]) < args.min_valid_samples or int(split_report["sizes"]["test"]) < args.min_test_samples:
        report["status"] = "DATA_NOT_READY"
        report["reason"] = (
            f"valid/test sample shortage: valid={split_report['sizes']['valid']}/{args.min_valid_samples}, "
            f"test={split_report['sizes']['test']}/{args.min_test_samples}"
        )
        return report

    train_valid = concat_split_dicts(splits["train"], splits["valid"])
    meta_config = MetaLabelingConfig(
        cost_assumption_return=cost_return,
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        purge_gap_bars=AI_PURGE_GAP_BARS,
        interval_minutes=5,
        max_paths=args.max_paths,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_meta_train_samples=args.min_meta_train_samples,
        min_selected_trades=args.min_selected_trades,
    )
    cpcv_report = evaluate_meta_cpcv(train_valid, meta_config)
    report["cpcv_report"] = cpcv_report
    if cpcv_report.get("status") != "ok":
        report["status"] = cpcv_report.get("status", "DATA_NOT_READY")
        report["edge_status"] = cpcv_report.get("edge_status", "EDGE_NOT_CONFIRMED")
        return report

    primary_threshold = float(cpcv_report.get("selected_primary_threshold", 0.50))
    meta_threshold = float(cpcv_report.get("selected_meta_threshold", 0.60))
    test_once = fit_meta_oof_test_once(train_valid, splits["test"], meta_config, primary_threshold, meta_threshold)
    report["test_once_report"] = test_once
    cpcv_edge = str(cpcv_report.get("edge_status") or "EDGE_NOT_CONFIRMED")
    test_edge = str(test_once.get("edge_status") or "EDGE_NOT_CONFIRMED")
    if test_edge == "INSUFFICIENT_TRADES":
        edge_status = "INSUFFICIENT_TRADES"
    elif cpcv_edge == "META_EDGE_CANDIDATE" and test_edge == "META_EDGE_CANDIDATE":
        edge_status = "META_EDGE_CANDIDATE"
    else:
        edge_status = "EDGE_NOT_CONFIRMED"
    report.update(
        {
            "status": "ok" if test_once.get("status") == "ok" else test_once.get("status", "DATA_NOT_READY"),
            "edge_status": edge_status,
            "selected_primary_threshold": primary_threshold,
            "selected_meta_threshold": meta_threshold,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--min-meta-train-samples", type=int, default=100)
    parser.add_argument("--min-selected-trades", type=int, default=20)
    parser.add_argument("--n-groups", type=int, default=6)
    parser.add_argument("--n-test-groups", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") not in {"DATA_NOT_READY", "MODEL_DEPENDENCY_MISSING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
