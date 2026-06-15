"""TOA policy model loading and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toa_ai.domain import ACTION_CLASSES, Action, PolicyOutput

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled at runtime
    torch = None
    nn = None


if nn is not None:

    class TOAPolicyNet(nn.Module):
        def __init__(
            self,
            num_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.10,
            num_actions: int = len(ACTION_CLASSES),
        ) -> None:
            super().__init__()
            self.encoder = nn.GRU(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.action_head = nn.Linear(hidden_size, num_actions)
            self.return_head = nn.Linear(hidden_size, 1)
            self.risk_head = nn.Linear(hidden_size, 1)
            self.holding_head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, hidden = self.encoder(x)
            pooled = self.norm(hidden[-1])
            return {
                "action_logits": self.action_head(pooled),
                "expected_return": self.return_head(pooled).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(pooled).squeeze(-1)),
                "holding_score": torch.sigmoid(self.holding_head(pooled).squeeze(-1)),
            }

else:
    TOAPolicyNet = None


class TOAPolicy:
    def __init__(self, model: Any, metadata: dict[str, Any], model_id: str) -> None:
        self.model = model
        self.metadata = metadata
        self.model_id = model_id
        self.feature_columns = list(metadata.get("feature_columns") or [])

    @classmethod
    def load(cls, model_path: str | Path) -> "TOAPolicy":
        if torch is None or TOAPolicyNet is None:
            raise RuntimeError("torch가 설치되어 있지 않아 TOA policy model을 로드할 수 없습니다.")
        payload = torch.load(str(model_path), map_location="cpu")
        metadata = dict(payload.get("metadata") or {})
        model = TOAPolicyNet(
            num_features=int(metadata["num_features"]),
            hidden_size=int(metadata.get("hidden_size", 64)),
            num_layers=int(metadata.get("num_layers", 1)),
            dropout=float(metadata.get("dropout", 0.10)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return cls(model=model, metadata=metadata, model_id=str(metadata.get("model_id") or Path(model_path).stem))

    def predict(self, sequence: np.ndarray) -> PolicyOutput:
        if torch is None:
            raise RuntimeError("torch가 설치되어 있지 않아 예측할 수 없습니다.")
        if sequence.ndim != 2:
            raise ValueError("sequence는 shape=(sequence_length, num_features) 이어야 합니다.")
        with torch.no_grad():
            tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
            output = self.model(tensor)
            logits = output["action_logits"].detach().cpu().numpy()[0]
            probs = _softmax(logits)
            action_idx = int(np.argmax(probs))
            expected_return = float(output["expected_return"].detach().cpu().numpy()[0])
            risk_score = float(output["risk_score"].detach().cpu().numpy()[0])
            holding_score = float(output["holding_score"].detach().cpu().numpy()[0])
        action = ACTION_CLASSES[action_idx]
        return PolicyOutput(
            action=action,
            confidence=float(probs[action_idx]),
            action_probs={ACTION_CLASSES[idx].value: float(value) for idx, value in enumerate(probs)},
            expected_return=expected_return,
            risk_score=risk_score,
            holding_score=holding_score,
            model_id=self.model_id,
            metadata={"source": "toa_policy_net"},
        )


class NoopPolicy:
    model_id = "noop"
    metadata: dict[str, Any] = {"source": "noop"}

    def predict(self, sequence: np.ndarray) -> PolicyOutput:
        return PolicyOutput(
            action=Action.NO_ACTION,
            confidence=1.0,
            action_probs={action.value: 1.0 if action == Action.NO_ACTION else 0.0 for action in Action},
            expected_return=0.0,
            risk_score=0.0,
            holding_score=0.0,
            model_id=self.model_id,
            metadata=self.metadata,
        )


def save_policy_payload(path: str | Path, model: Any, metadata: dict[str, Any]) -> None:
    if torch is None:
        raise RuntimeError("torch가 설치되어 있지 않아 모델을 저장할 수 없습니다.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, str(path))
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
