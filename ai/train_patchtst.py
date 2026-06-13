#!/usr/bin/env python3
"""Train the local PatchTST-lite AI signal model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ai.calibration import fit_platt_calibrator, save_calibrator
from ai.dataset import SplitConfig, aggregate_label_distribution, build_symbol_samples, calendar_time_split, concat_samples
from ai.feature_builder import FEATURE_COLUMNS
from ai.label_builder import LabelConfig
from ai.models.patchtst_model import PatchTSTLite, torch
from config import (
    AI_LABEL_MODE,
    AI_LOSS_TYPE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    AI_TRAIN_UNIVERSE,
    WATCHLIST,
)
from core.fetcher_intraday import fetch_intraday

MODEL_DIR = Path(__file__).resolve().parent / "model_store"
MODEL_PATH = MODEL_DIR / "domestic_patchtst_model.pt"
CALIBRATOR_PATH = MODEL_DIR / "domestic_patchtst_calibrator.pkl"
METADATA_PATH = MODEL_DIR / "domestic_patchtst_metadata.json"


class FocalBCEWithLogitsLoss:
    def __init__(self, pos_weight: float, gamma: float = 2.0) -> None:
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.gamma = gamma

    def __call__(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        return ((1 - pt) ** self.gamma * bce).mean()


def _metrics(labels, probs):
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception:
        return {"auc": None, "pr_auc": None, "brier": None}
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    if len(labels) == 0:
        return {"auc": None, "pr_auc": None, "brier": None}
    if len(np.unique(labels)) < 2:
        return {"auc": None, "pr_auc": None, "brier": float(brier_score_loss(labels, probs))}
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
    }


def _predict(model, X, label_mode: str = "drop_neutral"):
    model.eval()
    if len(X) == 0:
        return np.asarray([]), np.asarray([])
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32)
        out = model(tensor)
        logits = out["logit"].detach().cpu().numpy()
        if label_mode == "three_class":
            logits_2d = np.asarray(logits)
            shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp[:, 2] / exp.sum(axis=1)
        else:
            probs = 1 / (1 + np.exp(-logits))
    return logits, probs


def _resolve_universe(universe: str | None, watchlist: str) -> list[str]:
    if universe == "ai_train":
        return list(dict.fromkeys(AI_TRAIN_UNIVERSE))
    if universe == "watchlist":
        return [str(item["code"]) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    return [item.strip() for item in watchlist.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", default="005930,000660")
    parser.add_argument("--universe", default=None, choices=["ai_train", "watchlist"])
    parser.add_argument("--period", default="120d")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--source", default="auto", choices=["auto", "kis", "yfinance", "cache"])
    parser.add_argument("--split-dry-run", action="store_true")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--pretrained-encoder", default=None)
    args = parser.parse_args()

    label_config = LabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, AI_LABEL_MODE)
    all_samples = []
    samples = []
    distributions = {}
    sample_counts = {}
    codes = _resolve_universe(args.universe, args.watchlist)
    for code in codes:
        try:
            df = fetch_intraday(code, period=args.period, interval="5m", source=args.source, use_cache=True)
            sample = build_symbol_samples(df, code, label_config, AI_SEQUENCE_LENGTH)
            all_samples.append(sample)
            distributions[code] = sample["label_distribution"]
            sample_counts[code] = int(len(sample["X"]))
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            print(f"{code}: 학습 데이터 생성 실패 - {exc}")
    print("label_distribution=", json.dumps(distributions, ensure_ascii=False))
    aggregate_distribution = aggregate_label_distribution(all_samples)
    print("label_distribution_total=", json.dumps(aggregate_distribution, ensure_ascii=False))
    if aggregate_distribution["neutral_ratio"] < 0.05:
        print("경고: neutral 비율이 5% 미만입니다. barrier(target/stop) 또는 horizon_bars 재검토를 권장합니다.")
    if not samples:
        shortage = {
            "status": "data_insufficient",
            "reason": "sequence_length와 label_mode 적용 후 학습 샘플이 없습니다.",
            "sequence_length": AI_SEQUENCE_LENGTH,
            "purge_gap_bars": AI_PURGE_GAP_BARS,
            "min_valid_samples": args.min_valid_samples,
            "min_test_samples": args.min_test_samples,
            "sample_counts": sample_counts,
        }
        print("학습 가능한 샘플이 없습니다.")
        print(json.dumps(shortage, ensure_ascii=False, indent=2))
        return 0

    data = concat_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
    print("split_report=", json.dumps(split_report, ensure_ascii=False))
    valid_size = int(split_report["sizes"]["valid"])
    test_size = int(split_report["sizes"]["test"])
    if args.split_dry_run:
        print("split dry-run 완료: production sequence_length/purge_gap 설정으로 split 크기만 계산했습니다.")
        return 0
    if valid_size < args.min_valid_samples or test_size < args.min_test_samples:
        shortage = {
            "status": "data_insufficient",
            "min_valid_samples": args.min_valid_samples,
            "min_test_samples": args.min_test_samples,
            "valid_size": valid_size,
            "test_size": test_size,
            "valid_shortage": max(args.min_valid_samples - valid_size, 0),
            "test_shortage": max(args.min_test_samples - test_size, 0),
        }
        print("데이터 부족: full 학습을 중단합니다.")
        print(json.dumps(shortage, ensure_ascii=False, indent=2))
        return 0

    if torch is None or PatchTSTLite is None:
        print("PyTorch가 설치되어 있지 않아 AI 학습을 건너뜁니다. requirements-ai.txt를 참고해 직접 설치하세요.")
        return 0

    y_train = splits["train"]["y"]
    if len(y_train) == 0 or len(splits["valid"]["y"]) == 0 or len(splits["test"]["y"]) == 0:
        print("walk-forward split 이후 train/valid/test 중 비어 있는 구간이 있어 학습을 중단합니다. period 또는 watchlist를 늘리세요.")
        return 0
    positive_count = int((y_train == 1).sum()) if AI_LABEL_MODE == "drop_neutral" else int((y_train == 2).sum())
    negative_count = int((y_train == 0).sum())
    pos_weight = negative_count / max(positive_count, 1)
    class_ratio = positive_count / max(len(y_train), 1)
    print(f"positive_count={positive_count} negative_count={negative_count} pos_weight={pos_weight:.4f} class_ratio={class_ratio:.4f}")

    num_classes = 3 if AI_LABEL_MODE == "three_class" else 1
    model = PatchTSTLite(num_features=len(FEATURE_COLUMNS), sequence_length=AI_SEQUENCE_LENGTH, num_classes=num_classes)
    if args.pretrained_encoder:
        try:
            state = torch.load(args.pretrained_encoder, map_location="cpu")
            model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
            print(f"pretrained_encoder_loaded={args.pretrained_encoder}")
        except Exception as exc:
            print(f"pretrained_encoder_load_failed={exc}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if AI_LABEL_MODE == "three_class":
        class_counts = np.bincount(y_train.astype(int), minlength=3).astype(float)
        class_weights = len(y_train) / np.maximum(class_counts, 1.0)
        class_weights[1] *= 0.5
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        print(f"class_counts={class_counts.tolist()} class_weights={class_weights.tolist()} neutral_weight_scale=0.5")
    elif AI_LOSS_TYPE == "focal":
        loss_fn = FocalBCEWithLogitsLoss(pos_weight=pos_weight)
        print("loss_type=focal gamma=2.0")
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    X_train = torch.tensor(splits["train"]["X"], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long if AI_LABEL_MODE == "three_class" else torch.float32)
    return_train_t = torch.tensor(splits["train"]["outcome_return"], dtype=torch.float32)
    risk_train_t = torch.tensor((splits["train"]["raw_label"] == "stop").astype(float), dtype=torch.float32)
    best_loss = float("inf")
    best_state = None
    patience = 3
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(X_train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            out = model(X_train[idx])
            class_loss = loss_fn(out["logit"], y_train_t[idx])
            return_loss = torch.nn.functional.mse_loss(out["expected_return"], return_train_t[idx])
            risk_loss = torch.nn.functional.binary_cross_entropy(out["risk_score"], risk_train_t[idx])
            loss = class_loss + 0.25 * return_loss + 0.10 * risk_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, valid_probs = _predict(model, splits["valid"]["X"], AI_LABEL_MODE)
        valid_labels_for_prob = (splits["valid"]["y"] == 2).astype(float) if AI_LABEL_MODE == "three_class" else splits["valid"]["y"]
        valid_loss = float(np.mean((valid_probs - valid_labels_for_prob) ** 2)) if len(valid_probs) else float("inf")
        print(f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} valid_brier_proxy={valid_loss:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    valid_logits, valid_probs = _predict(model, splits["valid"]["X"], AI_LABEL_MODE)
    valid_calibration_logits = valid_logits[:, 2] if AI_LABEL_MODE == "three_class" else valid_logits
    valid_binary_labels = (splits["valid"]["y"] == 2).astype(int) if AI_LABEL_MODE == "three_class" else splits["valid"]["y"]
    calibrator, calibration_method = fit_platt_calibrator(valid_calibration_logits, valid_binary_labels)
    save_calibrator(calibrator, CALIBRATOR_PATH)
    _, test_probs = _predict(model, splits["test"]["X"], AI_LABEL_MODE)
    test_binary_labels = (splits["test"]["y"] == 2).astype(int) if AI_LABEL_MODE == "three_class" else splits["test"]["y"]
    valid_metrics = _metrics(valid_binary_labels, valid_probs)
    test_metrics = _metrics(test_binary_labels, test_probs)
    model_version = datetime.now().strftime("patchtst_domestic_%Y%m%d_%H%M")
    metadata = {
        "model_version": model_version,
        "train_period": args.period,
        "symbols_used": codes,
        "sequence_length": AI_SEQUENCE_LENGTH,
        "horizon_bars": AI_PRED_HORIZON_BARS,
        "target_return": AI_TARGET_RETURN,
        "stop_return": AI_STOP_RETURN,
        "label_mode": AI_LABEL_MODE,
        "num_classes": num_classes,
        "feature_columns": FEATURE_COLUMNS,
        "validation_auc": valid_metrics["auc"],
        "validation_pr_auc": valid_metrics["pr_auc"],
        "test_auc": test_metrics["auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "brier_score": test_metrics["brier"],
        "calibration_method": calibration_method,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label_distribution": distributions,
        "label_distribution_total": aggregate_distribution,
        "purge_gap_bars": AI_PURGE_GAP_BARS,
        "loss_type": "cross_entropy_neutral_downweight" if AI_LABEL_MODE == "three_class" else AI_LOSS_TYPE,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "pos_weight": pos_weight,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
