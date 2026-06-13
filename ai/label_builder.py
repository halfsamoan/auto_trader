"""Build first-touch intraday labels without merging neutral into stop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.technical import ensure_ohlcv


@dataclass(frozen=True)
class LabelConfig:
    horizon_bars: int = 6
    target_return: float = 0.0035
    stop_return: float = -0.0025
    label_mode: str = "drop_neutral"


def build_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    data = ensure_ohlcv(df)
    rows = []
    for i in range(len(data)):
        entry = float(data["Close"].iloc[i])
        outcome = "neutral"
        outcome_return = 0.0
        if entry > 0:
            target = entry * (1 + config.target_return)
            stop = entry * (1 + config.stop_return)
            for j in range(i + 1, min(i + 1 + config.horizon_bars, len(data))):
                high = float(data["High"].iloc[j])
                low = float(data["Low"].iloc[j])
                if low <= stop and high >= target:
                    outcome = "stop"
                    outcome_return = config.stop_return
                    break
                if high >= target:
                    outcome = "target"
                    outcome_return = config.target_return
                    break
                if low <= stop:
                    outcome = "stop"
                    outcome_return = config.stop_return
                    break
            if outcome == "neutral":
                end_idx = min(i + config.horizon_bars, len(data) - 1)
                outcome_return = float(data["Close"].iloc[end_idx] / entry - 1) if entry else 0.0
        rows.append({"raw_label": outcome, "outcome_return": outcome_return})
    labels = pd.DataFrame(rows, index=data.index)
    if config.label_mode == "drop_neutral":
        labels["label"] = np.where(labels["raw_label"] == "target", 1, np.where(labels["raw_label"] == "stop", 0, np.nan))
    elif config.label_mode == "three_class":
        labels["label"] = np.where(labels["raw_label"] == "target", 2, np.where(labels["raw_label"] == "neutral", 1, 0))
    else:
        raise ValueError("LABEL_MODE은 drop_neutral 또는 three_class만 허용됩니다.")
    return labels


def label_distribution(labels: pd.DataFrame) -> dict[str, float | int]:
    total = max(len(labels), 1)
    target_count = int((labels["raw_label"] == "target").sum())
    stop_count = int((labels["raw_label"] == "stop").sum())
    neutral_count = int((labels["raw_label"] == "neutral").sum())
    return {
        "target_count": target_count,
        "stop_count": stop_count,
        "neutral_count": neutral_count,
        "target_ratio": target_count / total,
        "stop_ratio": stop_count / total,
        "neutral_ratio": neutral_count / total,
    }
