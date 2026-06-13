"""Shadow-only PatchTST policy inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai.dataset import POLICY_EXTRA_FEATURE_COLUMNS, POLICY_FEATURE_COLUMNS
from ai.feature_builder import build_feature_frame
from ai.market_breadth_features import safe_market_breadth_features
from ai.models.patchtst_policy_model import ACTION_CLASSES, PatchTSTPolicyModel, torch
from config import AI_SEQUENCE_LENGTH, ENABLE_AI_POLICY


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_store"
POLICY_MODEL_PATH = MODEL_DIR / "domestic_patchtst_policy.pt"
POLICY_METADATA_PATH = MODEL_DIR / "domestic_patchtst_policy_metadata.json"


def _latest_policy_feature_sequence(df, sequence_length: int, rule_score: float | None = None) -> tuple[np.ndarray | None, list[str]]:
    frame = build_feature_frame(df, rule_score=rule_score)
    for col in POLICY_EXTRA_FEATURE_COLUMNS:
        frame[col] = 0.0
    frame = frame[POLICY_FEATURE_COLUMNS].astype(float)
    if len(frame) < sequence_length:
        return None, POLICY_FEATURE_COLUMNS
    return frame.iloc[-sequence_length:].to_numpy(dtype=np.float32), POLICY_FEATURE_COLUMNS


def empty_policy_result(status: str, enabled: bool = ENABLE_AI_POLICY) -> dict[str, Any]:
    return {
        "ai_policy_enabled": bool(enabled),
        "ai_policy_mode": "shadow",
        "ai_status": status,
        "ai_action": "NO_ACTION",
        "action_probs": None,
        "buy_score": None,
        "hold_score": None,
        "take_profit_partial_score": None,
        "take_profit_full_score": None,
        "cut_loss_score": None,
        "trailing_exit_score": None,
        "expected_return": None,
        "risk_score": None,
        "holding_time_score": None,
        "confidence": None,
        "ai_model_name": "patchtst_policy",
        "ai_model_version": None,
        "market_breadth_features": safe_market_breadth_features(),
    }


class PolicyEngine:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.model = None
        self.status = "disabled" if not ENABLE_AI_POLICY else "not_loaded"
        if ENABLE_AI_POLICY:
            self._load()

    def _load(self) -> None:
        if torch is None or PatchTSTPolicyModel is None:
            self.status = "torch_missing"
            return
        if not POLICY_MODEL_PATH.exists() or not POLICY_METADATA_PATH.exists():
            self.status = "model_missing"
            return
        try:
            self.metadata = json.loads(POLICY_METADATA_PATH.read_text(encoding="utf-8"))
            sequence_length = int(self.metadata.get("sequence_length", AI_SEQUENCE_LENGTH))
            num_features = int(self.metadata.get("num_features", len(self.metadata.get("feature_columns", [])) or 16))
            self.model = PatchTSTPolicyModel(num_features=num_features, sequence_length=sequence_length)
            state = torch.load(POLICY_MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
            self.model.eval()
            self.status = "ok"
        except Exception as exc:
            self.status = f"load_failed:{exc}"

    def predict(self, df, rule_score: float | None = None) -> dict[str, Any]:
        if not ENABLE_AI_POLICY:
            return empty_policy_result("disabled", enabled=False)
        if self.status in {"torch_missing", "model_missing"} or self.model is None:
            return empty_policy_result(self.status)
        sequence_length = int(self.metadata.get("sequence_length", AI_SEQUENCE_LENGTH))
        seq, _ = _latest_policy_feature_sequence(df, sequence_length=sequence_length, rule_score=rule_score)
        if seq is None:
            return empty_policy_result("insufficient_sequence")
        try:
            with torch.no_grad():
                output = self.model(torch.tensor(seq[None, :, :], dtype=torch.float32))
                logits = output["action_logits"].detach().cpu().numpy()[0]
                shifted = logits - np.max(logits)
                probs = np.exp(shifted) / np.sum(np.exp(shifted))
                action_idx = int(np.argmax(probs))
                expected_return = float(output["expected_return"].detach().cpu().numpy()[0])
                risk_score = float(output["risk_score"].detach().cpu().numpy()[0])
                holding_time_score = float(output["holding_time_score"].detach().cpu().numpy()[0])
        except Exception as exc:
            return empty_policy_result(f"predict_failed:{exc}")
        action_probs = {ACTION_CLASSES[idx]: float(prob) for idx, prob in enumerate(probs)}
        return {
            "ai_policy_enabled": True,
            "ai_policy_mode": "shadow",
            "ai_status": self.status,
            "ai_action": ACTION_CLASSES[action_idx],
            "action_probs": action_probs,
            "buy_score": action_probs.get("BUY"),
            "hold_score": action_probs.get("HOLD_POSITION"),
            "take_profit_partial_score": action_probs.get("TAKE_PROFIT_PARTIAL"),
            "take_profit_full_score": action_probs.get("TAKE_PROFIT_FULL"),
            "cut_loss_score": action_probs.get("CUT_LOSS"),
            "trailing_exit_score": action_probs.get("TRAILING_EXIT"),
            "expected_return": expected_return,
            "risk_score": risk_score,
            "holding_time_score": holding_time_score,
            "confidence": float(np.max(probs)),
            "ai_model_name": "patchtst_policy",
            "ai_model_version": self.metadata.get("model_version"),
            "market_breadth_features": safe_market_breadth_features(),
        }
