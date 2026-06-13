# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-13 (V3.2)
# Dependency: ai/*, config.py, core/fetcher_intraday.py
# Description: LightGBM/XGBoost tabular baseline을 shadow 리포트 전용으로 학습/평가합니다.
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


def _resolve_codes(universe: str, watchlist: str | None) -> list[str]:
    if watchlist:
        return [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    if universe == "ai_train":
        return [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    return [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]


def _flatten_last_step(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or len(x) == 0:
        return np.asarray([], dtype=np.float32).reshape(0, 0)
    return x[:, -1, :].astype(np.float32)


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
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
    hit_rate = float(np.mean(positive[pred])) if np.any(pred) else None
    return {
        "sample_count": int(len(y_true)),
        "threshold": threshold,
        "predicted_buy_count": int(np.sum(pred)),
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cache", choices=["auto", "kis", "yfinance", "cache"])
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--period", default="120d")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    backend_name, model_cls, dependency_error = _model_backend()
    codes = _resolve_codes(args.universe, args.watchlist)
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

    distribution = aggregate_action_label_distribution(all_samples)
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "not_started",
        "model": "lightgbm_or_xgboost_tabular_baseline",
        "backend": backend_name,
        "model_dependency_status": "ok" if backend_name else "MODEL_DEPENDENCY_MISSING",
        "model_dependency_error": dependency_error,
        "universe": args.universe,
        "symbol_count": len(codes),
        "successful_symbol_count": len(samples),
        "failed_symbols": failures,
        "action_label_distribution": distribution,
        "split_report": None,
        "order_api_called": False,
        "live_order_enabled": False,
        "cost_assumption_return": 2 * COMMISSION + TAX,
        "edge_policy": "EDGE_NOT_CONFIRMED - live/paper execution remains disabled in V3.2",
        "multiple_testing_note": "Deflated Sharpe/CPCV not implemented; treat any result as exploratory.",
    }

    if not samples:
        report["status"] = "DATA_NOT_READY"
        report["reason"] = "no policy samples after sequence_length and label filters"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

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
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if model_cls is None:
        report["status"] = "MODEL_DEPENDENCY_MISSING"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    x_train = _flatten_last_step(splits["train"]["X"])
    x_valid = _flatten_last_step(splits["valid"]["X"])
    x_test = _flatten_last_step(splits["test"]["X"])
    y_train = (splits["train"]["action_class"] == 1).astype(int)
    y_valid = (splits["valid"]["action_class"] == 1).astype(int)
    y_test = (splits["test"]["action_class"] == 1).astype(int)

    model_kwargs = {"random_state": 42}
    if backend_name == "lightgbm":
        model_kwargs.update({"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 31})
    else:
        model_kwargs.update({"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4, "eval_metric": "logloss"})
    model = model_cls(**model_kwargs)
    model.fit(x_train, y_train)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    test_buy = test_prob >= args.threshold
    net_returns = splits["test"]["expected_return_target"] - float(report["cost_assumption_return"])
    report.update(
        {
            "status": "ok",
            "valid_metrics": _binary_metrics(y_valid, valid_prob, args.threshold),
            "test_metrics": _binary_metrics(y_test, test_prob, args.threshold),
            "cost_adjusted_expected_return_on_predicted_buy": float(np.mean(net_returns[test_buy])) if np.any(test_buy) else None,
            "edge_policy": "EDGE_NOT_CONFIRMED - V3.2 does not enable live or leveraged ETN orders",
        }
    )
    if args.write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
