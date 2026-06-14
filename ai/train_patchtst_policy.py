#!/usr/bin/env python3
"""Fine-tune PatchTST policy heads with shadow-only action labels."""

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
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    AI_TRAIN_UNIVERSE,
    EXCLUDED_CODES,
    LEVERAGE_ETN_WATCHLIST,
    WATCHLIST,
)
from core.fetcher_intraday import fetch_intraday


MODEL_DIR = Path(__file__).resolve().parent / "model_store"
MODEL_PATH = MODEL_DIR / "domestic_patchtst_policy.pt"
METADATA_PATH = MODEL_DIR / "domestic_patchtst_policy_metadata.json"
DEFAULT_PRETRAINED_ENCODER = MODEL_DIR / "patchtst_encoder_pretrained.pt"


def _resolve_universe(universe: str | None, watchlist: str | None) -> list[str]:
    if universe == "ai_train":
        raw = list(dict.fromkeys(AI_TRAIN_UNIVERSE))
    elif universe == "watchlist":
        raw = [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    else:
        raw = [item.strip().zfill(6) for item in (watchlist or "005930,000660").split(",") if item.strip()]
    excluded = {str(code).zfill(6) for code in EXCLUDED_CODES}
    excluded |= {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}
    return [code for code in dict.fromkeys(raw) if code not in excluded]


def _safe_load_encoder(model, path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return f"pretrained_encoder_missing:{path}"
    try:
        state = torch.load(candidate, map_location="cpu")
        raw_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        own_state = model.state_dict()
        filtered = {k: v for k, v in raw_state.items() if k in own_state and tuple(own_state[k].shape) == tuple(v.shape)}
        model.load_state_dict(filtered, strict=False)
        return f"pretrained_encoder_loaded:{path}:matched={len(filtered)}"
    except Exception as exc:
        return f"pretrained_encoder_load_failed:{exc}"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None, "buy_precision": None, "buy_recall": None, "cut_loss_precision": None}
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None, "buy_precision": None, "buy_recall": None, "cut_loss_precision": None}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "buy_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "buy_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "cut_loss_precision": float(precision_score(y_true, y_pred, labels=[5], average="macro", zero_division=0)),
    }


def _predict(model, X: np.ndarray, device):
    model.eval()
    if len(X) == 0:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    logits, returns, risks, holds = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start : start + 256], dtype=torch.float32, device=device)
            out = model(batch)
            logits.append(out["action_logits"].detach().cpu().numpy())
            returns.append(out["expected_return"].detach().cpu().numpy())
            risks.append(out["risk_score"].detach().cpu().numpy())
            holds.append(out["holding_time_score"].detach().cpu().numpy())
    action_logits = np.concatenate(logits, axis=0)
    shifted = action_logits - action_logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return action_logits, probs, np.concatenate(returns), np.concatenate(risks), np.concatenate(holds)


def _train_epoch(model, split, optimizer, loss_fns, batch_size: int, device):
    model.train()
    order = np.arange(len(split["X"]))
    np.random.shuffle(order)
    losses = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        x = torch.tensor(split["X"][idx], dtype=torch.float32, device=device)
        y = torch.tensor(split["action_class"][idx], dtype=torch.long, device=device)
        ret = torch.tensor(split["expected_return_target"][idx], dtype=torch.float32, device=device)
        risk = torch.tensor(split["risk_target"][idx], dtype=torch.float32, device=device)
        hold = torch.tensor(split["holding_time_target"][idx], dtype=torch.float32, device=device)
        out = model(x)
        loss = (
            loss_fns["action"](out["action_logits"], y)
            + loss_fns["return"](out["expected_return"], ret)
            + loss_fns["risk"](out["risk_score"], risk)
            + loss_fns["holding"](out["holding_time_score"], hold)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cache", choices=["auto", "kis", "yfinance", "cache"])
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--period", default="120d")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=AI_SEQUENCE_LENGTH)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--pretrained-encoder", default=str(DEFAULT_PRETRAINED_ENCODER))
    args = parser.parse_args()

    if torch is None or PatchTSTPolicyModel is None:
        print("PyTorch가 설치되어 있지 않아 policy fine-tuning을 건너뜁니다. requirements-ai.txt를 참고하세요.")
        return 0
    if args.smoke_test:
        args.sequence_length = min(args.sequence_length, 48)
        args.epochs = max(args.epochs, 2)
        args.min_valid_samples = 1
        args.min_test_samples = 1
        print("smoke-test: policy fine-tuning 파이프라인 확인용입니다. 성능 지표를 과해석하지 마세요.")

    label_config = ActionLabelConfig(
        horizon_bars=AI_PRED_HORIZON_BARS,
        target_return=AI_TARGET_RETURN,
        stop_return=AI_STOP_RETURN,
        label_mode=AI_LABEL_MODE,
    )
    samples = []
    all_samples = []
    sample_counts = {}
    for code in _resolve_universe(args.universe, args.watchlist):
        try:
            df = fetch_intraday(code, period=args.period, interval="5m", source=args.source, use_cache=True)
            sample = build_policy_symbol_samples(df, code, label_config, args.sequence_length)
            all_samples.append(sample)
            sample_counts[code] = int(len(sample["X"]))
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            print(f"{code}: policy 학습 데이터 생성 실패 - {exc}")
    distribution = aggregate_action_label_distribution(all_samples)
    print("action_label_distribution=", json.dumps(distribution, ensure_ascii=False))
    if distribution["neutral_ratio"] < 0.05:
        print("경고: neutral 비율이 5% 미만입니다. barrier/horizon 재검토를 권장합니다.")
    if not samples:
        print(json.dumps({"status": "data_insufficient", "sample_counts": sample_counts}, ensure_ascii=False, indent=2))
        return 0

    data = concat_policy_samples(samples)
    splits, split_report = calendar_time_split(data, SplitConfig(args.sequence_length, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
    print("split_report=", json.dumps(split_report, ensure_ascii=False))
    if split_report["sizes"]["valid"] < args.min_valid_samples or split_report["sizes"]["test"] < args.min_test_samples:
        print(
            json.dumps(
                {
                    "status": "data_insufficient",
                    "valid_size": split_report["sizes"]["valid"],
                    "test_size": split_report["sizes"]["test"],
                    "min_valid_samples": args.min_valid_samples,
                    "min_test_samples": args.min_test_samples,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTSTPolicyModel(num_features=len(POLICY_FEATURE_COLUMNS), sequence_length=args.sequence_length).to(device)
    encoder_status = _safe_load_encoder(model, args.pretrained_encoder)
    if encoder_status:
        print(encoder_status)

    y_train = splits["train"]["action_class"].astype(int)
    class_counts = np.bincount(y_train, minlength=len(ACTION_CLASSES)).astype(float)
    class_weights = len(y_train) / np.maximum(class_counts, 1.0)
    class_weights = np.clip(class_weights, 0.25, 20.0)
    loss_fns = {
        "action": torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device)),
        "return": torch.nn.SmoothL1Loss(),
        "risk": torch.nn.MSELoss(),
        "holding": torch.nn.SmoothL1Loss(),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_valid = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_loss = _train_epoch(model, splits["train"], optimizer, loss_fns, args.batch_size, device)
        _, valid_probs, valid_ret, valid_risk, valid_hold = _predict(model, splits["valid"]["X"], device)
        valid_pred = valid_probs.argmax(axis=1) if len(valid_probs) else np.asarray([])
        valid_action_loss = 1.0 - (_classification_metrics(splits["valid"]["action_class"], valid_pred)["accuracy"] or 0.0)
        valid_aux = float(np.mean(np.abs(valid_ret - splits["valid"]["expected_return_target"]))) if len(valid_ret) else 1.0
        valid_loss = valid_action_loss + valid_aux
        print(f"epoch={epoch + 1} train_loss={train_loss} valid_proxy_loss={valid_loss:.6f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)

    metrics = {}
    for name in ["valid", "test"]:
        _, probs, pred_ret, pred_risk, _ = _predict(model, splits[name]["X"], device)
        pred_action = probs.argmax(axis=1) if len(probs) else np.asarray([])
        metrics[name] = _classification_metrics(splits[name]["action_class"], pred_action)
        metrics[name]["expected_return_mae"] = float(np.mean(np.abs(pred_ret - splits[name]["expected_return_target"]))) if len(pred_ret) else None
        metrics[name]["risk_score_mae"] = float(np.mean(np.abs(pred_risk - splits[name]["risk_target"]))) if len(pred_risk) else None

    metadata = {
        "model_version": "patchtst_policy_domestic_" + datetime.now().strftime("%Y%m%d_%H%M"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sequence_length": args.sequence_length,
        "horizon_bars": AI_PRED_HORIZON_BARS,
        "target_return": AI_TARGET_RETURN,
        "stop_return": AI_STOP_RETURN,
        "label_mode": AI_LABEL_MODE,
        "synthetic_label_used": "generated_separately_not_mixed_into_entry_training",
        "action_class_distribution": distribution,
        "train_valid_test_sample_count": split_report["sizes"],
        "purge_gap": AI_PURGE_GAP_BARS,
        "class_weight": {str(i): float(v) for i, v in enumerate(class_weights)},
        "feature_columns": POLICY_FEATURE_COLUMNS,
        "num_features": len(POLICY_FEATURE_COLUMNS),
        "symbols_used": sorted(set(data["symbol"].astype(str).tolist())),
        "split_report": split_report,
        "valid_metrics": metrics["valid"],
        "test_metrics": metrics["test"],
        "pretrained_encoder_status": encoder_status,
        "device": str(device),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
