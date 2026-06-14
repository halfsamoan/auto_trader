#!/usr/bin/env python3
"""Evaluate the shadow PatchTST policy model without touching order paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ai.action_label_builder import ActionLabelConfig
from ai.dataset import (
    POLICY_FEATURE_COLUMNS,
    SplitConfig,
    aggregate_action_label_distribution,
    build_policy_symbol_samples,
    calendar_time_split,
    concat_policy_samples,
)
from ai.models.patchtst_policy_model import ACTION_CLASSES, PatchTSTPolicyModel, torch
from behavior_tree.tree_runner import run_domestic_stock_bt_shadow
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    AI_TRAIN_UNIVERSE,
    EXCLUDED_CODES,
    LEVERAGE_ETN_WATCHLIST,
)
from core.fetcher_intraday import fetch_intraday


DEFAULT_MODEL = Path(__file__).resolve().parent / "model_store" / "domestic_patchtst_policy.pt"
DEFAULT_METADATA = Path(__file__).resolve().parent / "model_store" / "domestic_patchtst_policy_metadata.json"


def _metrics(y_true, y_pred):
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception:
        return {"action_accuracy": None, "macro_f1": None, "weighted_f1": None, "buy_precision": None, "buy_recall": None, "cut_loss_precision": None}
    if len(y_true) == 0:
        return {"action_accuracy": None, "macro_f1": None, "weighted_f1": None, "buy_precision": None, "buy_recall": None, "cut_loss_precision": None}
    return {
        "action_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "buy_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "buy_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "cut_loss_precision": float(precision_score(y_true, y_pred, labels=[5], average="macro", zero_division=0)),
    }


def _resolve_universe(universe: str) -> list[str]:
    if universe == "ai_train":
        raw = list(dict.fromkeys(AI_TRAIN_UNIVERSE))
    else:
        raw = [item.strip().zfill(6) for item in universe.split(",") if item.strip()]
    excluded = {str(code).zfill(6) for code in EXCLUDED_CODES}
    excluded |= {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}
    return [code for code in dict.fromkeys(raw) if code not in excluded]


def _load_test_data(sequence_length: int, universe: str, source: str):
    label_config = ActionLabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, label_mode=AI_LABEL_MODE)
    samples, all_samples = [], []
    for code in _resolve_universe(universe):
        try:
            df = fetch_intraday(code, period="120d", interval="5m", source=source, use_cache=True)
            sample = build_policy_symbol_samples(df, code, label_config, sequence_length)
            all_samples.append(sample)
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            print(f"{code}: 평가 데이터 생성 실패 - {exc}")
    if not samples:
        return None, aggregate_action_label_distribution(all_samples), None
    data = concat_policy_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(sequence_length, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
    return splits["test"], aggregate_action_label_distribution(all_samples), split_report


def _predict(model, X, device):
    model.eval()
    logits, returns, risks, holds = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start : start + 256], dtype=torch.float32, device=device)
            out = model(batch)
            logits.append(out["action_logits"].detach().cpu().numpy())
            returns.append(out["expected_return"].detach().cpu().numpy())
            risks.append(out["risk_score"].detach().cpu().numpy())
            holds.append(out["holding_time_score"].detach().cpu().numpy())
    if not logits:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    raw = np.concatenate(logits)
    shifted = raw - raw.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return probs, np.concatenate(returns), np.concatenate(risks), np.concatenate(holds)


def _confidence_bins(confidence, y_true, y_pred):
    bins = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
    rows = []
    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        rows.append(
            {
                "bin": f"{lo:.2f}~{min(hi, 1.0):.2f}",
                "count": int(mask.sum()),
                "accuracy": float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else None,
                "buy_precision": float(((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / max((y_pred[mask] == 1).sum(), 1)) if mask.any() else None,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--universe", default="ai_train")
    parser.add_argument("--source", default="cache")
    args = parser.parse_args()

    if torch is None or PatchTSTPolicyModel is None:
        print("PyTorch가 설치되어 있지 않아 policy 평가를 건너뜁니다.")
        return 0
    model_path = Path(args.model)
    metadata_path = Path(args.metadata)
    if not model_path.exists() or not metadata_path.exists():
        print(json.dumps({"status": "model_missing", "model": str(model_path)}, ensure_ascii=False, indent=2))
        return 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence_length = int(metadata.get("sequence_length", 96))
    test, distribution, split_report = _load_test_data(sequence_length, args.universe, args.source)
    if test is None or len(test["X"]) == 0:
        print(json.dumps({"status": "data_insufficient", "action_label_distribution": distribution, "split_report": split_report}, ensure_ascii=False, indent=2))
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTSTPolicyModel(num_features=int(metadata.get("num_features", len(POLICY_FEATURE_COLUMNS))), sequence_length=sequence_length).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    probs, pred_ret, pred_risk, pred_hold = _predict(model, test["X"], device)
    pred_action = probs.argmax(axis=1)
    true_action = test["action_class"].astype(int)
    metrics = _metrics(true_action, pred_action)
    confidence = probs.max(axis=1)
    bt_counts: dict[str, int] = {}
    for action_idx, conf, ret, risk in zip(pred_action, confidence, pred_ret, pred_risk):
        result = run_domestic_stock_bt_shadow(
            {
                "ai_action": ACTION_CLASSES[int(action_idx)],
                "confidence": float(conf),
                "expected_return": float(ret),
                "risk_score": float(risk),
            },
            has_position=False,
        )
        key = str(result.get("bt_would_action"))
        bt_counts[key] = bt_counts.get(key, 0) + 1
    report = {
        "status": "ok",
        **metrics,
        "expected_return_mae": float(np.mean(np.abs(pred_ret - test["expected_return_target"]))),
        "risk_score_mae": float(np.mean(np.abs(pred_risk - test["risk_target"]))),
        "holding_time_mae": float(np.mean(np.abs(pred_hold - test["holding_time_target"]))),
        "action_sample_count": {ACTION_CLASSES.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(true_action, return_counts=True))},
        "confidence_bins": _confidence_bins(confidence, true_action, pred_action),
        "bt_shadow_branch_distribution": bt_counts,
        "rule_action_vs_ai_action": "rule labels are not order decisions; compare in bt_shadow report with signal_log",
        "split_report": split_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
