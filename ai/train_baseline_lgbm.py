# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-13 (V3.2.2)
# Dependency: ai/*, config.py, core/fetcher_intraday.py
# Description: LightGBM/XGBoost meta-policy baseline is trained and evaluated without order calls.
# ================================================================================

"""Train a no-order tabular baseline for triple-barrier policy labels."""

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
from ai.dataset import (
    POLICY_FEATURE_COLUMNS,
    SplitConfig,
    aggregate_action_label_distribution,
    build_policy_symbol_samples,
    calendar_time_split,
    concat_policy_samples,
)
from ai.universe_builder import load_ai_universe_records
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    COMMISSION,
    LEVERAGE_ETN_WATCHLIST,
    TAX,
    WATCHLIST,
)
from core.fetcher_intraday import fetch_intraday


REPORT_PATH = Path(__file__).resolve().parent / "model_store" / "baseline_lgbm_report.json"


def _model_backend() -> tuple[str | None, Any | None, str | None]:
    try:
        from lightgbm import LGBMClassifier

        return "lightgbm", LGBMClassifier, None
    except Exception as lightgbm_exc:
        try:
            from xgboost import XGBClassifier

            return "xgboost", XGBClassifier, None
        except Exception as xgboost_exc:
            return None, None, f"lightgbm:{lightgbm_exc}; xgboost:{xgboost_exc}"


def _metric_backend():
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        return {
            "accuracy_score": accuracy_score,
            "average_precision_score": average_precision_score,
            "f1_score": f1_score,
            "precision_recall_curve": precision_recall_curve,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "roc_auc_score": roc_auc_score,
        }, None
    except Exception as exc:
        return None, str(exc)


def _excluded_codes() -> set[str]:
    return {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}


def _resolve_codes(universe: str, watchlist: str | None) -> list[str]:
    if watchlist:
        raw = [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    elif universe == "ai_train":
        raw = [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    else:
        raw = [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    excluded = _excluded_codes()
    return [code for code in dict.fromkeys(raw) if code not in excluded]


def _flatten_last_step(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or len(x) == 0:
        return np.asarray([], dtype=np.float32).reshape(0, 0)
    return x[:, -1, :].astype(np.float32)


def _safe_auc(metrics: dict[str, Any] | None, name: str, y_true: np.ndarray, prob: np.ndarray) -> float | None:
    if metrics is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(metrics[name](y_true, prob))
    except Exception:
        return None


def _binary_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    metric_fns: dict[str, Any] | None,
    cost_assumption_return: float,
    gross_returns: np.ndarray,
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
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    selected_net = gross_returns[pred] - cost_assumption_return
    accuracy = float(metric_fns["accuracy_score"](y_true, pred)) if metric_fns else float(np.mean(y_true == pred))
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "target_ratio": float(np.mean(positive)),
        "predicted_trade_count": int(np.sum(pred)),
        "predicted_trade_ratio": float(np.mean(pred)),
        "accuracy": accuracy,
        "roc_auc": _safe_auc(metric_fns, "roc_auc_score", y_true, prob),
        "pr_auc": _safe_auc(metric_fns, "average_precision_score", y_true, prob),
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": f1,
        "precision_lift_vs_target_ratio": precision - float(np.mean(positive)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "cost_adjusted_expected_return_on_selected": float(np.mean(selected_net)) if len(selected_net) else None,
        "gross_expected_return_on_selected": float(np.mean(gross_returns[pred])) if np.any(pred) else None,
    }


def _threshold_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    gross_returns: np.ndarray,
    cost_assumption_return: float,
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
                    "predicted_trade_ratio": 0.0,
                    "precision": None,
                    "recall": 0.0,
                    "precision_lift": None,
                    "cost_adjusted_expected_return": None,
                }
            )
            continue
        positive = y_true == 1
        precision = float(np.mean(positive[pred]))
        recall = float(np.sum(pred & positive) / max(np.sum(positive), 1))
        net = gross_returns[pred] - cost_assumption_return
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_trade_count": int(np.sum(pred)),
                "predicted_trade_ratio": float(np.mean(pred)),
                "precision": precision,
                "recall": recall,
                "precision_lift": precision - base_ratio,
                "cost_adjusted_expected_return": float(np.mean(net)),
            }
        )
    return rows


def _select_threshold(curve: list[dict[str, Any]], min_trades: int) -> dict[str, Any]:
    candidates = [
        row
        for row in curve
        if row["predicted_trade_count"] >= min_trades
        and row["precision"] is not None
        and row["precision_lift"] is not None
    ]
    if not candidates:
        return {"threshold": 0.62, "selection_reason": "fallback_no_valid_candidate"}
    profitable = [row for row in candidates if (row["cost_adjusted_expected_return"] or -999) > 0 and row["precision_lift"] > 0]
    pool = profitable or candidates
    selected = max(
        pool,
        key=lambda row: (
            float(row["cost_adjusted_expected_return"] or -999),
            float(row["precision_lift"] or -999),
            int(row["predicted_trade_count"]),
        ),
    )
    return {
        **selected,
        "selection_reason": "valid_only_max_cost_adjusted_return_with_precision_lift"
        if profitable
        else "valid_only_best_available_candidate",
    }


def _overfit_report(train: dict[str, Any], valid: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    def gap(key: str, left: dict[str, Any], right: dict[str, Any]) -> float | None:
        if left.get(key) is None or right.get(key) is None:
            return None
        return float(left[key]) - float(right[key])

    return {
        "train_valid_precision_gap": gap("target_precision", train, valid),
        "valid_test_precision_gap": gap("target_precision", valid, test),
        "train_valid_roc_auc_gap": gap("roc_auc", train, valid),
        "valid_test_roc_auc_gap": gap("roc_auc", valid, test),
        "warning": (
            "large train/valid or valid/test gaps can indicate leakage, regime mismatch, or overfitting; "
            "do not enable orders from this report alone"
        ),
    }


def _build_samples(args: argparse.Namespace, codes: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    label_config = ActionLabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, label_mode=AI_LABEL_MODE)
    samples = []
    all_samples = []
    failures: dict[str, str] = {}
    for code in codes:
        try:
            df = fetch_intraday(code, period=args.period, interval="5m", source=args.source, use_cache=True)
            sample = build_policy_symbol_samples(df, code, label_config, AI_SEQUENCE_LENGTH)
            all_samples.append(sample)
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            failures[code] = str(exc)
    return samples, all_samples, failures


def train_baseline_report(args: argparse.Namespace) -> dict[str, Any]:
    backend_name, model_cls, dependency_error = _model_backend()
    metric_fns, metric_error = _metric_backend()
    codes = _resolve_codes(args.universe, args.watchlist)
    samples, all_samples, failures = _build_samples(args, codes)
    distribution = aggregate_action_label_distribution(all_samples)
    neutral_excluded = int(distribution.get("neutral_count", 0))
    target_count = int(distribution.get("target_count", 0))
    stop_count = int(distribution.get("stop_count", 0))
    scale_pos_weight = stop_count / max(target_count, 1)
    cost_assumption_return = 2 * COMMISSION + TAX

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "not_started",
        "model": "lightgbm_meta_policy_baseline",
        "backend": backend_name,
        "model_dependency_status": "ok" if backend_name else "MODEL_DEPENDENCY_MISSING",
        "model_dependency_error": dependency_error,
        "metric_dependency_status": "ok" if metric_fns else "METRIC_DEPENDENCY_MISSING",
        "metric_dependency_error": metric_error,
        "universe": args.universe,
        "symbol_count": len(codes),
        "excluded_symbols": sorted(_excluded_codes()),
        "successful_symbol_count": len(samples),
        "failed_symbols": failures,
        "feature_columns": POLICY_FEATURE_COLUMNS,
        "feature_construction_audit": {
            "features_use_t_or_earlier_only": True,
            "labels_use_future_window_only_for_training_targets": True,
            "neutral_labels_excluded_from_train_valid_test": True,
            "purged_calendar_split_used": True,
            "purge_gap_bars": AI_PURGE_GAP_BARS,
            "test_used_once_after_valid_threshold_selection": True,
        },
        "action_label_distribution": distribution,
        "neutral_excluded_count": neutral_excluded,
        "class_imbalance": {
            "target_count": target_count,
            "stop_count": stop_count,
            "target_ratio_without_neutral": target_count / max(target_count + stop_count, 1),
            "scale_pos_weight": scale_pos_weight,
        },
        "split_report": None,
        "order_api_called": False,
        "live_order_enabled": False,
        "cost_assumption_return": cost_assumption_return,
        "cost_model_note": "TODO: verify actual KIS commission/tax/tick/slippage before any execution.",
        "edge_policy": "EDGE_NOT_CONFIRMED - live/paper execution remains disabled in V3.2.2",
        "multiple_testing_note": "Deflated Sharpe/CPCV not implemented; treat any result as exploratory.",
    }

    if not samples:
        report["status"] = "DATA_NOT_READY"
        report["reason"] = "no policy samples after sequence_length and neutral label filters"
        return report

    data = concat_policy_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
    report["split_report"] = split_report
    valid_size = int(split_report["sizes"]["valid"])
    test_size = int(split_report["sizes"]["test"])
    if valid_size < args.min_valid_samples or test_size < args.min_test_samples:
        report["status"] = "DATA_NOT_READY"
        report["reason"] = (
            f"valid/test sample shortage: valid={valid_size}/{args.min_valid_samples}, "
            f"test={test_size}/{args.min_test_samples}"
        )
        return report
    if model_cls is None:
        report["status"] = "MODEL_DEPENDENCY_MISSING"
        return report

    x_train = _flatten_last_step(splits["train"]["X"])
    x_valid = _flatten_last_step(splits["valid"]["X"])
    x_test = _flatten_last_step(splits["test"]["X"])
    y_train = (splits["train"]["action_class"] == 1).astype(int)
    y_valid = (splits["valid"]["action_class"] == 1).astype(int)
    y_test = (splits["test"]["action_class"] == 1).astype(int)

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

    train_prob = model.predict_proba(x_train)[:, 1]
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    valid_curve = _threshold_curve(
        y_valid,
        valid_prob,
        splits["valid"]["expected_return_target"],
        cost_assumption_return,
        [round(x, 2) for x in np.arange(args.threshold_min, args.threshold_max + 0.001, args.threshold_step)],
    )
    selected = _select_threshold(valid_curve, args.min_selected_trades)
    threshold = float(selected.get("threshold", args.threshold))

    train_metrics = _binary_metrics(
        y_train,
        train_prob,
        threshold,
        metric_fns,
        cost_assumption_return,
        splits["train"]["expected_return_target"],
    )
    valid_metrics = _binary_metrics(
        y_valid,
        valid_prob,
        threshold,
        metric_fns,
        cost_assumption_return,
        splits["valid"]["expected_return_target"],
    )
    test_metrics = _binary_metrics(
        y_test,
        test_prob,
        threshold,
        metric_fns,
        cost_assumption_return,
        splits["test"]["expected_return_target"],
    )

    precision_lift = float(test_metrics.get("precision_lift_vs_target_ratio") or 0.0)
    net_edge = test_metrics.get("cost_adjusted_expected_return_on_selected")
    edge_confirmed = bool(
        test_metrics.get("predicted_trade_count", 0) >= args.min_selected_trades
        and net_edge is not None
        and net_edge > 0
        and precision_lift >= args.min_precision_lift
    )
    report.update(
        {
            "status": "ok",
            "model_params": model_kwargs,
            "valid_threshold_curve": valid_curve,
            "selected_threshold": selected,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "overfit_report": _overfit_report(train_metrics, valid_metrics, test_metrics),
            "edge_status": "EDGE_CONFIRMED" if edge_confirmed else "EDGE_NOT_CONFIRMED",
            "edge_policy": (
                "EDGE_CONFIRMED - V3.2.2 still does not enable live/paper/leveraged ETN orders"
                if edge_confirmed
                else "EDGE_NOT_CONFIRMED - live/paper/leveraged ETN orders remain disabled"
            ),
            "test_evaluation_note": "test metrics are produced once after valid-only threshold selection",
            "order_api_called": False,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cache", choices=["auto", "kis", "yfinance", "cache"])
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--period", default="120d")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--min-selected-trades", type=int, default=100)
    parser.add_argument("--min-precision-lift", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = train_baseline_report(args)
    if args.write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
