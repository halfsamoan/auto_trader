"""Shadow-mode AI signal inference for domestic-stock paper watch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai.calibration import calibrated_prob, load_calibrator
from ai.feature_builder import latest_feature_sequence
from ai.models.patchtst_model import PatchTSTLite, torch
from config import (
    AI_MAX_RISK_SCORE,
    AI_MIN_EXPECTED_RETURN,
    AI_MIN_PROB_UP,
    AI_SEQUENCE_LENGTH,
    ENABLE_AI_SIGNAL,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_store"
MODEL_PATH = MODEL_DIR / "domestic_patchtst_model.pt"
CALIBRATOR_PATH = MODEL_DIR / "domestic_patchtst_calibrator.pkl"
METADATA_PATH = MODEL_DIR / "domestic_patchtst_metadata.json"


def empty_ai_result(status: str, enabled: bool = ENABLE_AI_SIGNAL) -> dict[str, Any]:
    return {
        "ai_enabled": bool(enabled),
        "ai_status": status,
        "ai_model_name": "patchtst_lite",
        "ai_model_version": None,
        "ai_prob_up": None,
        "ai_prob_up_calibrated": None,
        "ai_expected_return": None,
        "ai_risk_score": None,
        "ai_decision": "none",
        "ai_gate_passed": False,
        "ai_gate_reason": status,
        "ai_would_pass": False,
        "ai_shadow_mode": True,
    }


class AISignalEngine:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.model = None
        self.calibrator = None
        self.status = "disabled" if not ENABLE_AI_SIGNAL else "not_loaded"
        if ENABLE_AI_SIGNAL:
            self._load()

    def _load(self) -> None:
        if torch is None or PatchTSTLite is None:
            self.status = "torch_missing"
            return
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            self.status = "model_missing"
            return
        try:
            self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            feature_columns = self.metadata.get("feature_columns") or []
            sequence_length = int(self.metadata.get("sequence_length", AI_SEQUENCE_LENGTH))
            num_classes = int(self.metadata.get("num_classes", 1))
            self.model = PatchTSTLite(num_features=len(feature_columns), sequence_length=sequence_length, num_classes=num_classes)
            state = torch.load(MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
            self.model.eval()
            if CALIBRATOR_PATH.exists():
                self.calibrator = load_calibrator(CALIBRATOR_PATH)
                self.status = "ok"
            else:
                self.status = "uncalibrated"
        except Exception as exc:
            self.status = f"load_failed:{exc}"

    def predict(self, df, rule_score: float | None = None) -> dict[str, Any]:
        if not ENABLE_AI_SIGNAL:
            return empty_ai_result("disabled", enabled=False)
        if self.status in {"torch_missing", "model_missing"} or self.model is None:
            return empty_ai_result(self.status)
        sequence_length = int(self.metadata.get("sequence_length", AI_SEQUENCE_LENGTH))
        seq, _ = latest_feature_sequence(df, sequence_length=sequence_length, rule_score=rule_score)
        if seq is None:
            return empty_ai_result("insufficient_sequence")
        try:
            with torch.no_grad():
                tensor = torch.tensor(seq[None, :, :], dtype=torch.float32)
                output = self.model(tensor)
                logits = output["logit"].detach().cpu().numpy()
                if int(self.metadata.get("num_classes", 1)) == 3:
                    shifted = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(shifted)
                    raw_prob = float((exp[:, 2] / exp.sum(axis=1))[0])
                    calibration_logit = float(logits[0, 2])
                else:
                    calibration_logit = float(logits[0])
                    raw_prob = float(1 / (1 + np.exp(-calibration_logit)))
                prob = raw_prob
                status = self.status
                if self.calibrator is not None:
                    prob = float(calibrated_prob(self.calibrator, np.asarray([calibration_logit]))[0])
                elif status == "ok":
                    status = "uncalibrated"
                expected_return = float(output["expected_return"].detach().cpu().numpy()[0])
                risk_score = float(output["risk_score"].detach().cpu().numpy()[0])
        except Exception as exc:
            return empty_ai_result(f"predict_failed:{exc}")

        passed = prob >= AI_MIN_PROB_UP and expected_return >= AI_MIN_EXPECTED_RETURN and risk_score <= AI_MAX_RISK_SCORE
        reasons = []
        if prob < AI_MIN_PROB_UP:
            reasons.append("prob_up_below_min")
        if expected_return < AI_MIN_EXPECTED_RETURN:
            reasons.append("expected_return_below_min")
        if risk_score > AI_MAX_RISK_SCORE:
            reasons.append("risk_score_above_max")
        return {
            "ai_enabled": True,
            "ai_status": status,
            "ai_model_name": "patchtst_lite",
            "ai_model_version": self.metadata.get("model_version"),
            "ai_prob_up": raw_prob,
            "ai_prob_up_calibrated": prob if self.calibrator is not None else None,
            "ai_expected_return": expected_return,
            "ai_risk_score": risk_score,
            "ai_decision": "pass" if passed else "reject",
            "ai_gate_passed": passed,
            "ai_gate_reason": "pass" if passed else ",".join(reasons),
            "ai_would_pass": passed,
            "ai_shadow_mode": True,
        }
