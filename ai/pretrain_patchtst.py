#!/usr/bin/env python3
"""Masked patch pretraining smoke-test for PatchTST policy encoder."""

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

from ai.models.patchtst_policy_model import PatchTSTPolicyModel, torch
from ai.pretrain_dataset import load_pretrain_sequences


MODEL_DIR = Path(__file__).resolve().parent / "model_store"
ENCODER_PATH = MODEL_DIR / "patchtst_encoder_pretrained.pt"
METADATA_PATH = MODEL_DIR / "patchtst_pretrain_metadata.json"


def _make_masked_batch(batch, patch_length: int, mask_ratio: float, device):
    x = torch.tensor(batch, dtype=torch.float32, device=device)
    target = x.reshape(x.shape[0], x.shape[1] // patch_length, patch_length * x.shape[2])
    mask = torch.rand(target.shape[:2], device=device) < mask_ratio
    masked = x.clone()
    for patch_idx in range(target.shape[1]):
        masked[mask[:, patch_idx], patch_idx * patch_length : (patch_idx + 1) * patch_length, :] = 0.0
    return masked, target, mask


def _train_with_batch_size(model, X, args, device, batch_size: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    split = max(int(len(X) * 0.8), 1)
    train_x = X[:split]
    valid_x = X[split:] if len(X[split:]) else X[:1]
    last_train_loss = None
    for _ in range(args.epochs):
        model.train()
        losses = []
        order = np.arange(len(train_x))
        np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = train_x[order[start : start + batch_size]]
            masked, target, mask = _make_masked_batch(batch, args.patch_length, args.mask_ratio, device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model.reconstruct_patches(masked)
                if mask.any():
                    loss = torch.nn.functional.mse_loss(pred[mask], target[mask])
                else:
                    loss = torch.nn.functional.mse_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        last_train_loss = float(np.mean(losses)) if losses else None
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for start in range(0, len(valid_x), batch_size):
            batch = valid_x[start : start + batch_size]
            masked, target, mask = _make_masked_batch(batch, args.patch_length, args.mask_ratio, device)
            pred = model.reconstruct_patches(masked)
            loss = torch.nn.functional.mse_loss(pred[mask], target[mask]) if mask.any() else torch.nn.functional.mse_loss(pred, target)
            valid_losses.append(float(loss.detach().cpu()))
    return last_train_loss, float(np.mean(valid_losses)) if valid_losses else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cache", choices=["cache"])
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--patch-length", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    args = parser.parse_args()

    if torch is None or PatchTSTPolicyModel is None:
        print("PyTorch가 설치되어 있지 않아 pretraining을 건너뜁니다. requirements-ai.txt를 참고하세요.")
        return 0
    if args.smoke_test:
        args.sequence_length = min(args.sequence_length, 48)
        args.epochs = max(args.epochs, 2)
        print("smoke-test: 성능 검증이 아니라 masked patch pretraining 파이프라인 확인용입니다.")

    dataset = load_pretrain_sequences(args.universe, args.watchlist, sequence_length=args.sequence_length)
    X = dataset["X"]
    min_sequences = 4 if args.smoke_test else 300
    print(f"total_rows={dataset['total_rows']} total_sequences={len(X)} min_required={min_sequences}")
    if len(X) < min_sequences:
        print("데이터 부족: pretraining을 중단합니다. 캐시 수집을 더 진행하세요.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTSTPolicyModel(
        num_features=X.shape[-1],
        sequence_length=args.sequence_length,
        patch_length=args.patch_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    train_loss = None
    valid_loss = None
    used_batch_size = None
    fallback_log = []
    for batch_size in [args.batch_size, 16, 8]:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            train_loss, valid_loss = _train_with_batch_size(model, X, args, device, batch_size)
            used_batch_size = batch_size
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            fallback_log.append(f"OOM batch_size={batch_size}")
            print(f"OOM 발생: batch_size={batch_size}, 더 작은 batch_size로 재시도합니다.")
    if used_batch_size is None:
        print("OOM으로 pretraining을 완료하지 못했습니다. batch_size를 더 낮춰야 합니다.")
        return 0

    metadata = {
        "symbols_used": dataset["symbols_used"],
        "total_rows": int(dataset["total_rows"]),
        "total_sequences": int(len(X)),
        "sequence_length": args.sequence_length,
        "patch_length": args.patch_length,
        "mask_ratio": args.mask_ratio,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "batch_size": used_batch_size,
        "oom_fallback_log": fallback_log,
        "model_config": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_features": int(X.shape[-1]),
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, ENCODER_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
