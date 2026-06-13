#!/usr/bin/env python3
"""Evaluate saved AI signal model with probability-bin and rule-threshold analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ai.calibration import calibrated_prob, load_calibrator
from ai.dataset import SplitConfig, aggregate_label_distribution, build_symbol_samples, calendar_time_split, concat_samples
from ai.feature_builder import FEATURE_COLUMNS
from ai.label_builder import LabelConfig
from ai.models.patchtst_model import PatchTSTLite, torch
from config import AI_LABEL_MODE, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS, AI_SEQUENCE_LENGTH, AI_STOP_RETURN, AI_TARGET_RETURN
from core.fetcher_intraday import fetch_intraday


def _metric_report(y, p):
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception:
        return {"auc": None, "pr_auc": None, "brier_score": None}
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    if len(y) == 0:
        return {"auc": None, "pr_auc": None, "brier_score": None}
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier_score": float(brier_score_loss(y, p)),
    }


def _precision_at(y, p, frac):
    n = max(int(len(p) * frac), 1)
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx])) if len(idx) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai/model_store/domestic_patchtst_model.pt")
    parser.add_argument("--watchlist", default="005930,000660")
    parser.add_argument("--period", default="120d")
    parser.add_argument("--source", default="auto", choices=["auto", "kis", "yfinance", "cache"])
    args = parser.parse_args()
    if torch is None or PatchTSTLite is None:
        print("PyTorch가 설치되어 있지 않아 AI 평가를 건너뜁니다. requirements-ai.txt를 참고해 직접 설치하세요.")
        return 0
    model_path = Path(args.model)
    metadata_path = model_path.with_name("domestic_patchtst_metadata.json")
    calibrator_path = model_path.with_name("domestic_patchtst_calibrator.pkl")
    if not model_path.exists() or not metadata_path.exists():
        print("저장된 AI 모델이 없습니다. 먼저 ai/train_patchtst.py를 실행하세요.")
        return 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label_mode = str(metadata.get("label_mode", AI_LABEL_MODE))
    sequence_length = int(metadata.get("sequence_length", AI_SEQUENCE_LENGTH))
    horizon_bars = int(metadata.get("horizon_bars", AI_PRED_HORIZON_BARS))
    target_return = float(metadata.get("target_return", AI_TARGET_RETURN))
    stop_return = float(metadata.get("stop_return", AI_STOP_RETURN))
    purge_gap_bars = int(metadata.get("purge_gap_bars", AI_PURGE_GAP_BARS))
    num_classes = int(metadata.get("num_classes", 1))
    model = PatchTSTLite(num_features=len(metadata.get("feature_columns", FEATURE_COLUMNS)), sequence_length=sequence_length, num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    samples = []
    label_config = LabelConfig(horizon_bars, target_return, stop_return, label_mode)
    for code in [item.strip() for item in args.watchlist.split(",") if item.strip()]:
        try:
            samples.append(build_symbol_samples(fetch_intraday(code, args.period, "5m", source=args.source, use_cache=True), code, label_config, sequence_length))
        except Exception as exc:
            print(f"{code}: 평가 데이터 생성 실패 - {exc}")
    data = concat_samples([s for s in samples if len(s["X"])])
    if len(data["X"]) == 0:
        print("평가 가능한 샘플이 없습니다.")
        return 0
    splits, split_report = calendar_time_split(data, SplitConfig(sequence_length, horizon_bars, purge_gap_bars))
    print("split_report=", json.dumps(split_report, ensure_ascii=False))
    test = splits["test"]
    if len(test["X"]) == 0:
        print("walk-forward split 이후 test 구간이 비어 있습니다. period 또는 watchlist를 늘리세요.")
        return 0
    with torch.no_grad():
        out = model(torch.tensor(test["X"], dtype=torch.float32))
        logits = out["logit"].detach().cpu().numpy()
        if num_classes == 3:
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            raw_probs = exp[:, 2] / exp.sum(axis=1)
            calibration_logits = logits[:, 2]
        else:
            raw_probs = 1 / (1 + np.exp(-logits))
            calibration_logits = logits
    probs = raw_probs
    if calibrator_path.exists():
        probs = calibrated_prob(load_calibrator(calibrator_path), calibration_logits)
    binary_y = (test["y"] == 2).astype(int) if label_mode == "three_class" else test["y"].astype(int)
    aggregate_distribution = aggregate_label_distribution(samples)
    report = _metric_report(binary_y, probs)
    report["precision@top10%"] = _precision_at(binary_y, probs, 0.10)
    report["precision@top20%"] = _precision_at(binary_y, probs, 0.20)
    report["avg_expected_return"] = float(np.mean(test["outcome_return"])) if len(test["outcome_return"]) else None
    report["stop_hit_rate"] = float(np.mean(test["raw_label"] == "stop")) if len(test["raw_label"]) else None
    report["neutral_ratio"] = aggregate_distribution["neutral_ratio"]
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    report["prob_bins"] = {}
    for low, high in bins:
        mask = (probs >= low) & (probs < high)
        report["prob_bins"][f"{low:.2f}~{high if high < 1 else '1.00'}"] = {
            "count": int(mask.sum()),
            "target_hit_rate": float(np.mean(binary_y[mask])) if mask.any() else None,
        }
    report["rule_thresholds"] = {}
    for threshold in [60, 65, 70, 75, 80]:
        mask = test["rule_score"] >= threshold
        report["rule_thresholds"][str(threshold)] = {
            "candidate_count": int(mask.sum()),
            "win_rate": float(np.mean(binary_y[mask])) if mask.any() else None,
            "avg_return": float(np.mean(test["outcome_return"][mask])) if mask.any() else None,
            "target_hit_rate": float(np.mean(test["raw_label"][mask] == "target")) if mask.any() else None,
            "stop_hit_rate": float(np.mean(test["raw_label"][mask] == "stop")) if mask.any() else None,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
