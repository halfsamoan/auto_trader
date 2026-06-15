"""Convert logged TOA experiences into learner samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from toa_ai.config import ModelConfig
from toa_ai.domain import ACTION_TO_ID, Action
from toa_ai.features import FEATURE_COLUMNS, PolicyDataset
from toa_ai.storage import TOAMemory


def build_experience_dataset(memory: TOAMemory, config: ModelConfig, limit: int | None = 50_000) -> PolicyDataset:
    rows = memory.load_experiences(limit=limit)
    xs: list[np.ndarray] = []
    actions: list[int] = []
    expected_returns: list[float] = []
    risks: list[float] = []
    timestamps: list[str] = []
    symbols: list[str] = []

    for row in rows:
        state = row.get("state") or {}
        sequence = state.get("feature_sequence")
        if not sequence:
            continue
        array = np.asarray(sequence, dtype=np.float32)
        if array.shape != (config.sequence_length, len(FEATURE_COLUMNS)):
            continue
        action = _experience_target(str(row.get("action") or Action.NO_ACTION.value), float(row.get("reward") or 0.0))
        xs.append(array)
        actions.append(ACTION_TO_ID[action])
        next_return = float(row.get("next_return") or 0.0)
        expected_returns.append(next_return)
        risks.append(float(next_return < 0 or float(row.get("reward") or 0.0) < 0))
        timestamps.append(str(row.get("timestamp") or ""))
        symbols.append(str(row.get("symbol") or ""))

    if not xs:
        return PolicyDataset(
            x=np.empty((0, 0, len(FEATURE_COLUMNS)), dtype=np.float32),
            action=np.empty((0,), dtype=np.int64),
            expected_return=np.empty((0,), dtype=np.float32),
            risk=np.empty((0,), dtype=np.float32),
            timestamps=np.empty((0,), dtype=object),
            symbols=np.empty((0,), dtype=object),
            feature_columns=FEATURE_COLUMNS,
            action_counts={},
        )
    action_array = np.asarray(actions, dtype=np.int64)
    return PolicyDataset(
        x=np.asarray(xs, dtype=np.float32),
        action=action_array,
        expected_return=np.asarray(expected_returns, dtype=np.float32),
        risk=np.asarray(risks, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=object),
        symbols=np.asarray(symbols, dtype=object),
        feature_columns=FEATURE_COLUMNS,
        action_counts=_counts(action_array),
    )


def _experience_target(raw_action: str, reward: float) -> Action:
    try:
        action = Action(raw_action)
    except ValueError:
        return Action.NO_ACTION
    if reward >= 0:
        return action
    if action == Action.OPEN_LONG:
        return Action.NO_ACTION
    if action in {Action.HOLD_LONG, Action.REDUCE_LONG}:
        return Action.CLOSE_LONG
    return action


def _counts(actions: np.ndarray) -> dict[str, int]:
    from toa_ai.domain import ACTION_CLASSES

    return {ACTION_CLASSES[int(action_id)].value: int(count) for action_id, count in zip(*np.unique(actions, return_counts=True))}
