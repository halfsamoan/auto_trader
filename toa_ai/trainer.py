"""Policy v1 training pipeline for TOA AI."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from toa_ai.config import DEFAULT_MODEL_DIR, ModelConfig
from toa_ai.domain import ACTION_CLASSES, new_id, utc_now_iso
from toa_ai.experience import build_experience_dataset
from toa_ai.features import PolicyDataset, build_policy_dataset_for_symbol, concat_datasets
from toa_ai.policy import TOAPolicyNet, save_policy_payload, torch
from toa_ai.storage import TOAMemory


def train_policy_v1(
    memory: TOAMemory,
    config: ModelConfig,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    if torch is None or TOAPolicyNet is None:
        raise RuntimeError("torch가 설치되어 있지 않아 TOA Policy v1을 학습할 수 없습니다.")

    selected = symbols or memory.list_symbols(timeframe=config.timeframe, min_rows=config.min_rows_per_symbol)
    if max_symbols is not None:
        selected = selected[: int(max_symbols)]
    datasets = []
    skipped: dict[str, str] = {}
    for symbol in selected:
        bars = memory.load_bars(symbol, timeframe=config.timeframe)
        if len(bars) < config.min_rows_per_symbol:
            skipped[symbol] = f"rows<{config.min_rows_per_symbol}"
            continue
        dataset = build_policy_dataset_for_symbol(bars, symbol, config)
        if len(dataset.x) == 0:
            skipped[symbol] = "no_samples"
            continue
        datasets.append(dataset)

    bar_dataset_count = len(datasets)
    experience_dataset = build_experience_dataset(memory, config)
    if len(experience_dataset.x):
        datasets.append(experience_dataset)

    dataset = concat_datasets(datasets)
    if len(dataset.x) == 0:
        raise RuntimeError("학습 가능한 TOA dataset이 없습니다. 먼저 ingest-cache를 실행하세요.")

    splits = _split_dataset(dataset, config.validation_fraction)
    train = splits["train"]
    valid = splits["valid"]
    device = torch.device("cpu")
    model = TOAPolicyNet(
        num_features=len(dataset.feature_columns),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    class_weights = _class_weights(train.action, len(ACTION_CLASSES)).to(device)
    action_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    mse = torch.nn.MSELoss()

    history = []
    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        for batch in _batches(train, config.batch_size, shuffle=True):
            xb, y_action, y_return, y_risk = _batch_tensors(batch, device)
            optimizer.zero_grad()
            output = model(xb)
            loss = (
                action_loss(output["action_logits"], y_action)
                + 0.25 * mse(output["expected_return"], y_return)
                + 0.25 * mse(output["risk_score"], y_risk)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        valid_metrics = _evaluate(model, valid, config.batch_size, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else math.nan,
                **valid_metrics,
            }
        )

    final_metrics = _evaluate(model, valid, config.batch_size, device)
    model_id = f"toa-policy-v1-{utc_now_iso().replace(':', '').replace('+', 'Z')}-{new_id('m').split('_')[1][:6]}"
    model_path = Path(model_dir) / f"{model_id}.pt"
    metadata = {
        "model_id": model_id,
        "model_type": "toa_policy_v1_gru",
        "created_at": utc_now_iso(),
        "feature_columns": dataset.feature_columns,
        "num_features": len(dataset.feature_columns),
        "action_classes": {str(idx): action.value for idx, action in ACTION_CLASSES.items()},
        **asdict(config),
    }
    metrics = {
        **final_metrics,
        "train_samples": int(len(train.x)),
        "validation_samples": int(len(valid.x)),
        "total_samples": int(len(dataset.x)),
        "bar_symbol_count": int(bar_dataset_count),
        "experience_samples": int(len(experience_dataset.x)),
        "skipped_symbols": skipped,
        "action_counts": dataset.action_counts,
        "history": history,
        "profit_metrics_are_diagnostic_only": True,
    }
    save_policy_payload(model_path, model, metadata)
    memory.register_model(model_id, "toa_policy_v1_gru", model_path, metadata, metrics, status="challenger")
    return {"model_id": model_id, "model_path": str(model_path), "metadata": metadata, "metrics": metrics}


def _split_dataset(dataset: PolicyDataset, validation_fraction: float) -> dict[str, PolicyDataset]:
    n = len(dataset.x)
    valid_size = max(1, int(n * validation_fraction))
    train_end = max(1, n - valid_size)
    return {
        "train": _slice_dataset(dataset, 0, train_end),
        "valid": _slice_dataset(dataset, train_end, n),
    }


def _slice_dataset(dataset: PolicyDataset, start: int, end: int) -> PolicyDataset:
    return PolicyDataset(
        x=dataset.x[start:end],
        action=dataset.action[start:end],
        expected_return=dataset.expected_return[start:end],
        risk=dataset.risk[start:end],
        timestamps=dataset.timestamps[start:end],
        symbols=dataset.symbols[start:end],
        feature_columns=dataset.feature_columns,
        action_counts={},
    )


def _batches(dataset: PolicyDataset, batch_size: int, shuffle: bool) -> list[PolicyDataset]:
    indices = np.arange(len(dataset.x))
    if shuffle:
        np.random.default_rng(42).shuffle(indices)
    out = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        out.append(
            PolicyDataset(
                x=dataset.x[idx],
                action=dataset.action[idx],
                expected_return=dataset.expected_return[idx],
                risk=dataset.risk[idx],
                timestamps=dataset.timestamps[idx],
                symbols=dataset.symbols[idx],
                feature_columns=dataset.feature_columns,
                action_counts={},
            )
        )
    return out


def _batch_tensors(batch: PolicyDataset, device: Any) -> tuple[Any, Any, Any, Any]:
    xb = torch.tensor(batch.x, dtype=torch.float32, device=device)
    y_action = torch.tensor(batch.action, dtype=torch.long, device=device)
    y_return = torch.tensor(batch.expected_return, dtype=torch.float32, device=device)
    y_risk = torch.tensor(batch.risk, dtype=torch.float32, device=device)
    return xb, y_action, y_return, y_risk


def _class_weights(actions: np.ndarray, num_classes: int) -> Any:
    counts = np.bincount(actions, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate(model: Any, dataset: PolicyDataset, batch_size: int, device: Any) -> dict[str, Any]:
    if len(dataset.x) == 0:
        return {
            "validation_loss": math.inf,
            "action_accuracy": 0.0,
            "action_entropy": 0.0,
            "no_action_ratio": 1.0,
            "return_mae": math.inf,
            "risk_mae": math.inf,
        }
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    losses = []
    predictions = []
    action_probs = []
    return_errors = []
    risk_errors = []
    with torch.no_grad():
        for batch in _batches(dataset, batch_size, shuffle=False):
            xb, y_action, y_return, y_risk = _batch_tensors(batch, device)
            output = model(xb)
            loss = ce(output["action_logits"], y_action) + 0.25 * mse(output["expected_return"], y_return) + 0.25 * mse(output["risk_score"], y_risk)
            probs = torch.softmax(output["action_logits"], dim=1)
            pred = torch.argmax(probs, dim=1)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(pred.detach().cpu().numpy().tolist())
            action_probs.extend(probs.detach().cpu().numpy().tolist())
            return_errors.extend(torch.abs(output["expected_return"] - y_return).detach().cpu().numpy().tolist())
            risk_errors.extend(torch.abs(output["risk_score"] - y_risk).detach().cpu().numpy().tolist())
    pred_array = np.asarray(predictions, dtype=np.int64)
    probs_array = np.asarray(action_probs, dtype=np.float32)
    truth = dataset.action[: len(pred_array)]
    mean_probs = probs_array.mean(axis=0)
    entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-9)))
    pred_counts = {ACTION_CLASSES[idx].value: int((pred_array == idx).sum()) for idx in range(len(ACTION_CLASSES))}
    return {
        "validation_loss": float(np.mean(losses)),
        "action_accuracy": float((pred_array == truth).mean()) if len(pred_array) else 0.0,
        "action_entropy": entropy,
        "no_action_ratio": float((pred_array == 0).mean()) if len(pred_array) else 1.0,
        "return_mae": float(np.mean(return_errors)) if return_errors else math.inf,
        "risk_mae": float(np.mean(risk_errors)) if risk_errors else math.inf,
        "prediction_counts": pred_counts,
    }
