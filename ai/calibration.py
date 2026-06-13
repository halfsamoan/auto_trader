"""Probability calibration helpers using validation predictions."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class IdentityCalibrator:
    method = "identity"

    def predict_proba(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        probs = 1 / (1 + np.exp(-values))
        return np.column_stack([1 - probs, probs])


def fit_platt_calibrator(logits: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return IdentityCalibrator(), "identity_single_class"
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return IdentityCalibrator(), "identity"
    model = LogisticRegression(max_iter=1000)
    model.fit(np.asarray(logits).reshape(-1, 1), labels)
    return model, "platt"


def calibrated_prob(calibrator, logits: np.ndarray) -> np.ndarray:
    probs = calibrator.predict_proba(np.asarray(logits).reshape(-1, 1))
    return probs[:, 1]


def save_calibrator(calibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(calibrator, handle)


def load_calibrator(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)
