"""Triple-barrier action labels for PatchTST policy fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai.models.patchtst_policy_model import ACTION_TO_CLASS
from core.technical import ensure_ohlcv


NEUTRAL_ACTION = -1


@dataclass(frozen=True)
class ActionLabelConfig:
    horizon_bars: int = 6
    target_return: float = 0.0035
    stop_return: float = -0.0025
    target2_return: float = 0.0070
    trailing_return: float = -0.0015
    label_mode: str = "drop_neutral"


def _first_touch(data: pd.DataFrame, idx: int, config: ActionLabelConfig) -> tuple[str, float, int | None]:
    entry = float(data["Close"].iloc[idx])
    if entry <= 0:
        return "neutral", 0.0, None
    target = entry * (1 + config.target_return)
    stop = entry * (1 + config.stop_return)
    end = min(idx + 1 + config.horizon_bars, len(data))
    for j in range(idx + 1, end):
        high = float(data["High"].iloc[j])
        low = float(data["Low"].iloc[j])
        if low <= stop and high >= target:
            return "stop", config.stop_return, j
        if high >= target:
            return "target", config.target_return, j
        if low <= stop:
            return "stop", config.stop_return, j
    end_idx = min(idx + config.horizon_bars, len(data) - 1)
    future_return = float(data["Close"].iloc[end_idx] / entry - 1) if entry else 0.0
    return "neutral", future_return, None


def _synthetic_holding_action(data: pd.DataFrame, idx: int, config: ActionLabelConfig) -> tuple[int, str]:
    entry = float(data["Close"].iloc[idx])
    if entry <= 0:
        return ACTION_TO_CLASS["HOLD_POSITION"], "synthetic_hold_invalid_entry"
    target1 = entry * (1 + config.target_return)
    target2 = entry * (1 + config.target2_return)
    stop = entry * (1 + config.stop_return)
    trailing = entry * (1 + config.trailing_return)
    end = min(idx + 1 + config.horizon_bars, len(data))
    best_high = entry
    for j in range(idx + 1, end):
        high = float(data["High"].iloc[j])
        low = float(data["Low"].iloc[j])
        best_high = max(best_high, high)
        dynamic_trailing = max(trailing, best_high * (1 + config.trailing_return))
        if low <= stop:
            return ACTION_TO_CLASS["CUT_LOSS"], "synthetic_cut_loss"
        if high >= target2:
            return ACTION_TO_CLASS["TAKE_PROFIT_FULL"], "synthetic_take_profit_full"
        if high >= target1:
            return ACTION_TO_CLASS["TAKE_PROFIT_PARTIAL"], "synthetic_take_profit_partial"
        if best_high > entry and low <= dynamic_trailing:
            return ACTION_TO_CLASS["TRAILING_EXIT"], "synthetic_trailing_exit"
    end_idx = min(idx + config.horizon_bars, len(data) - 1)
    future_return = float(data["Close"].iloc[end_idx] / entry - 1) if entry else 0.0
    if future_return > 0:
        return ACTION_TO_CLASS["HOLD_POSITION"], "synthetic_hold_position"
    return ACTION_TO_CLASS["TRAILING_EXIT"], "synthetic_trailing_exit_time"


def build_action_labels(df: pd.DataFrame, config: ActionLabelConfig) -> pd.DataFrame:
    """Return entry action labels and separated synthetic holding labels.

    Neutral entry outcomes are never merged into NO_ACTION. In drop_neutral mode
    they receive NaN action_class and are excluded by the training dataset.
    """
    data = ensure_ohlcv(df)
    rows = []
    for i in range(len(data)):
        raw, outcome_return, touch_idx = _first_touch(data, i, config)
        if raw == "target":
            entry_action = ACTION_TO_CLASS["BUY"]
        elif raw == "stop":
            entry_action = ACTION_TO_CLASS["NO_ACTION"]
        elif config.label_mode == "drop_neutral":
            entry_action = np.nan
        elif config.label_mode == "three_class":
            entry_action = NEUTRAL_ACTION
        else:
            raise ValueError("label_mode은 drop_neutral 또는 three_class만 허용됩니다.")
        synthetic_action, synthetic_reason = _synthetic_holding_action(data, i, config)
        rows.append(
            {
                "raw_entry_label": raw,
                "entry_touch_index": touch_idx,
                "action_class": entry_action,
                "synthetic_holding_action_class": synthetic_action,
                "synthetic_holding_reason": synthetic_reason,
                "expected_return_target": outcome_return,
                "risk_target": 1.0 if raw == "stop" else 0.0,
                "holding_time_target": min(config.horizon_bars, max((touch_idx or i) - i, 0)) / max(config.horizon_bars, 1),
            }
        )
    return pd.DataFrame(rows, index=data.index)


def action_label_distribution(labels: pd.DataFrame) -> dict[str, int | float | dict[str, int]]:
    raw = labels["raw_entry_label"] if "raw_entry_label" in labels else pd.Series(dtype=object)
    action = labels["action_class"] if "action_class" in labels else pd.Series(dtype=float)
    synthetic = labels["synthetic_holding_action_class"] if "synthetic_holding_action_class" in labels else pd.Series(dtype=float)
    total = max(len(labels), 1)
    return {
        "target_count": int((raw == "target").sum()),
        "stop_count": int((raw == "stop").sum()),
        "neutral_count": int((raw == "neutral").sum()),
        "target_ratio": float((raw == "target").sum() / total),
        "stop_ratio": float((raw == "stop").sum() / total),
        "neutral_ratio": float((raw == "neutral").sum() / total),
        "action_class_distribution": {str(int(k)): int(v) for k, v in action.dropna().astype(int).value_counts().sort_index().items()},
        "synthetic_holding_distribution": {str(int(k)): int(v) for k, v in synthetic.dropna().astype(int).value_counts().sort_index().items()},
    }
