"""Dataset assembly and time-ordered walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai.action_label_builder import ActionLabelConfig, action_label_distribution, build_action_labels
from ai.feature_builder import FEATURE_COLUMNS, build_feature_frame
from ai.label_builder import LabelConfig, build_labels, label_distribution


POLICY_EXTRA_FEATURE_COLUMNS = [
    "market_breadth_vwap_above_ratio",
    "market_breadth_positive_return_ratio",
    "market_breadth_volume_spike_ratio",
    "semiconductor_sector_strength",
    "largecap_momentum_score",
    "universe_risk_on_ratio",
    "position_qty_flag",
    "unrealized_pnl_pct",
    "holding_minutes_norm",
]

POLICY_FEATURE_COLUMNS = FEATURE_COLUMNS + POLICY_EXTRA_FEATURE_COLUMNS


@dataclass(frozen=True)
class SplitConfig:
    sequence_length: int = 96
    horizon_bars: int = 6
    purge_gap_bars: int = 96
    interval_minutes: int = 5


def build_symbol_samples(df: pd.DataFrame, symbol: str, label_config: LabelConfig, sequence_length: int) -> dict[str, object]:
    features = build_feature_frame(df)
    labels = build_labels(df, label_config)
    joined = features.join(labels, how="inner")
    xs, ys, returns, raw_labels, times, rule_scores = [], [], [], [], [], []
    for end in range(sequence_length - 1, len(joined) - label_config.horizon_bars):
        row = joined.iloc[end]
        if pd.isna(row["label"]):
            continue
        window = joined[FEATURE_COLUMNS].iloc[end - sequence_length + 1 : end + 1]
        xs.append(window.to_numpy(dtype=np.float32))
        ys.append(float(row["label"]))
        returns.append(float(row["outcome_return"]))
        raw_labels.append(str(row["raw_label"]))
        times.append(joined.index[end])
        rule_scores.append(float(row["rule_score"]) * 100.0)
    return {
        "X": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "outcome_return": np.asarray(returns, dtype=np.float32),
        "raw_label": np.asarray(raw_labels, dtype=object),
        "timestamp": np.asarray(times, dtype=object),
        "symbol": np.asarray([symbol] * len(xs), dtype=object),
        "rule_score": np.asarray(rule_scores, dtype=np.float32),
        "label_distribution": label_distribution(labels),
    }


def _with_policy_extra_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    # Historical market-breadth joins are added later. Keep deterministic neutral
    # values for smoke tests and cache-only fine-tuning.
    for col in POLICY_EXTRA_FEATURE_COLUMNS:
        out[col] = 0.0
    return out[POLICY_FEATURE_COLUMNS].astype(float)


def build_policy_symbol_samples(
    df: pd.DataFrame,
    symbol: str,
    label_config: ActionLabelConfig,
    sequence_length: int,
) -> dict[str, object]:
    features = _with_policy_extra_features(build_feature_frame(df))
    labels = build_action_labels(df, label_config)
    joined = features.join(labels, how="inner")
    xs, actions, expected_returns, risks, holding_times, raw_labels, synthetic_actions, times = [], [], [], [], [], [], [], []
    for end in range(sequence_length - 1, len(joined) - label_config.horizon_bars):
        row = joined.iloc[end]
        if pd.isna(row["action_class"]):
            continue
        window = joined[POLICY_FEATURE_COLUMNS].iloc[end - sequence_length + 1 : end + 1]
        xs.append(window.to_numpy(dtype=np.float32))
        actions.append(int(row["action_class"]))
        expected_returns.append(float(row["expected_return_target"]))
        risks.append(float(row["risk_target"]))
        holding_times.append(float(row["holding_time_target"]))
        raw_labels.append(str(row["raw_entry_label"]))
        synthetic_actions.append(int(row["synthetic_holding_action_class"]))
        times.append(joined.index[end])
    return {
        "X": np.asarray(xs, dtype=np.float32),
        "action_class": np.asarray(actions, dtype=np.int64),
        "expected_return_target": np.asarray(expected_returns, dtype=np.float32),
        "risk_target": np.asarray(risks, dtype=np.float32),
        "holding_time_target": np.asarray(holding_times, dtype=np.float32),
        "raw_entry_label": np.asarray(raw_labels, dtype=object),
        "synthetic_holding_action_class": np.asarray(synthetic_actions, dtype=np.int64),
        "timestamp": np.asarray(times, dtype=object),
        "symbol": np.asarray([symbol] * len(xs), dtype=object),
        "label_distribution": action_label_distribution(labels),
    }


def concat_policy_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "X",
        "action_class",
        "expected_return_target",
        "risk_target",
        "holding_time_target",
        "raw_entry_label",
        "synthetic_holding_action_class",
        "timestamp",
        "symbol",
    ]
    if not any(len(s["X"]) for s in samples):
        return {key: np.asarray([]) for key in keys}
    out = {key: np.concatenate([s[key] for s in samples if len(s["X"])], axis=0) for key in keys}
    order = np.argsort(out["timestamp"])
    return {key: value[order] for key, value in out.items()}


def aggregate_action_label_distribution(samples: list[dict[str, object]]) -> dict[str, object]:
    total_target = total_stop = total_neutral = 0
    action_counts: dict[str, int] = {}
    synthetic_counts: dict[str, int] = {}
    for sample in samples:
        dist = sample.get("label_distribution", {})
        total_target += int(dist.get("target_count", 0))
        total_stop += int(dist.get("stop_count", 0))
        total_neutral += int(dist.get("neutral_count", 0))
        for key, value in dict(dist.get("action_class_distribution", {})).items():
            action_counts[key] = action_counts.get(key, 0) + int(value)
        for key, value in dict(dist.get("synthetic_holding_distribution", {})).items():
            synthetic_counts[key] = synthetic_counts.get(key, 0) + int(value)
    total = max(total_target + total_stop + total_neutral, 1)
    return {
        "target_count": total_target,
        "stop_count": total_stop,
        "neutral_count": total_neutral,
        "target_ratio": total_target / total,
        "stop_ratio": total_stop / total,
        "neutral_ratio": total_neutral / total,
        "action_class_distribution": action_counts,
        "synthetic_holding_distribution": synthetic_counts,
    }


def concat_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    keys = ["X", "y", "outcome_return", "raw_label", "timestamp", "symbol", "rule_score"]
    if not any(len(s["X"]) for s in samples):
        return {key: np.asarray([]) for key in keys}
    out = {key: np.concatenate([s[key] for s in samples if len(s["X"])], axis=0) for key in keys}
    order = np.argsort(out["timestamp"])
    return {key: value[order] for key, value in out.items()}


def aggregate_label_distribution(samples: list[dict[str, object]]) -> dict[str, float | int]:
    target_count = 0
    stop_count = 0
    neutral_count = 0
    for sample in samples:
        dist = sample.get("label_distribution", {})
        target_count += int(dist.get("target_count", 0))
        stop_count += int(dist.get("stop_count", 0))
        neutral_count += int(dist.get("neutral_count", 0))
    total = max(target_count + stop_count + neutral_count, 1)
    return {
        "target_count": target_count,
        "stop_count": stop_count,
        "neutral_count": neutral_count,
        "target_ratio": target_count / total,
        "stop_ratio": stop_count / total,
        "neutral_ratio": neutral_count / total,
    }


def walk_forward_split(data: dict[str, object], split_config: SplitConfig) -> dict[str, dict[str, object]]:
    n = len(data["X"])
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    gap = max(split_config.purge_gap_bars, split_config.horizon_bars)
    ranges = {
        "train": (0, max(train_end - gap, 0)),
        "valid": (min(train_end + gap, n), max(valid_end - gap, 0)),
        "test": (min(valid_end + gap, n), n),
    }
    splits = {}
    for name, (start, end) in ranges.items():
        splits[name] = {key: value[start:end] for key, value in data.items()}
    return splits


def calendar_time_split(data: dict[str, object], split_config: SplitConfig) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    timestamps = pd.to_datetime(data["timestamp"])
    n = len(timestamps)
    if n == 0:
        empty = {name: {key: value[:0] for key, value in data.items()} for name in ["train", "valid", "test"]}
        return empty, {"total_samples": 0, "reason": "empty"}

    unique_times = pd.Index(timestamps).sort_values().unique()
    train_cutoff = unique_times[min(max(int(len(unique_times) * 0.70), 0), len(unique_times) - 1)]
    valid_cutoff = unique_times[min(max(int(len(unique_times) * 0.85), 0), len(unique_times) - 1)]
    gap_bars = max(split_config.purge_gap_bars, split_config.horizon_bars)
    purge_delta = pd.Timedelta(minutes=split_config.interval_minutes * gap_bars)

    masks = {
        "train": timestamps <= train_cutoff - purge_delta,
        "valid": (timestamps >= train_cutoff + purge_delta) & (timestamps <= valid_cutoff - purge_delta),
        "test": timestamps >= valid_cutoff + purge_delta,
    }
    splits = {}
    for name, mask in masks.items():
        mask_array = np.asarray(mask, dtype=bool)
        splits[name] = {key: value[mask_array] for key, value in data.items()}

    report = {
        "total_samples": int(n),
        "unique_calendar_times": int(len(unique_times)),
        "train_cutoff": pd.Timestamp(train_cutoff).isoformat(),
        "valid_cutoff": pd.Timestamp(valid_cutoff).isoformat(),
        "purge_gap_bars": int(gap_bars),
        "purge_gap_minutes": int(split_config.interval_minutes * gap_bars),
        "split_method": "calendar_time_cutoff",
        "sizes": {name: int(len(split["X"])) for name, split in splits.items()},
        "symbol_sizes": {
            name: {str(symbol): int((split["symbol"] == symbol).sum()) for symbol in sorted(set(split["symbol"]))}
            for name, split in splits.items()
        },
    }
    return splits, report
