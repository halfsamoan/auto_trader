# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.6)
# Dependency: numpy, pandas, scipy(optional)
# Description: Native CPCV split, Deflated Sharpe Ratio, and PBO helpers.
# ================================================================================

"""Native combinatorial purged cross-validation utilities.

The implementation is intentionally local: no mlfinlab/mlfinpy runtime
dependency is required. Splits are based on calendar timestamps and purge any
training samples around every test group window.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import erf, log, sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.cdf(x))
    except Exception:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        # Acklam's approximation is unnecessary here; binary search over cdf is
        # stable enough for DSR reporting.
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _normal_cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    purged_train_count: int
    test_time_ranges: list[dict[str, str]]


@dataclass(frozen=True)
class CombinatorialPurgedSplit:
    n_groups: int = 6
    n_test_groups: int = 2
    purge_gap_bars: int = 96
    interval_minutes: int = 5
    embargo_bars: int = 0
    max_splits: int | None = None

    def split(self, timestamps: Iterable[Any]) -> list[CPCVSplit]:
        ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
        if ts.isna().any():
            raise ValueError("timestamps contain non-datetime values")
        if len(ts) == 0:
            return []
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if self.n_test_groups < 1 or self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must satisfy 1 <= n_test_groups < n_groups")

        unique_times = pd.Index(ts).sort_values().unique()
        if len(unique_times) < self.n_groups:
            raise ValueError("not enough unique timestamps for requested CPCV groups")

        group_time_arrays = np.array_split(unique_times.to_numpy(), self.n_groups)
        group_by_time: dict[pd.Timestamp, int] = {}
        for group_id, values in enumerate(group_time_arrays):
            for value in values:
                group_by_time[pd.Timestamp(value)] = group_id

        sample_groups = np.asarray([group_by_time[pd.Timestamp(value)] for value in ts], dtype=int)
        purge_delta = pd.Timedelta(minutes=self.interval_minutes * max(self.purge_gap_bars, 0))
        embargo_delta = pd.Timedelta(minutes=self.interval_minutes * max(self.embargo_bars, 0))
        all_idx = np.arange(len(ts))
        splits: list[CPCVSplit] = []

        combo_iter = combinations(range(self.n_groups), self.n_test_groups)
        for split_id, test_groups in enumerate(combo_iter):
            if self.max_splits is not None and len(splits) >= self.max_splits:
                break
            test_mask = np.isin(sample_groups, np.asarray(test_groups, dtype=int))
            train_mask = ~test_mask
            ranges: list[dict[str, str]] = []
            for group_id in test_groups:
                group_values = pd.Index(group_time_arrays[group_id])
                start = pd.Timestamp(group_values.min())
                end = pd.Timestamp(group_values.max())
                purge_start = start - purge_delta
                purge_end = end + purge_delta + embargo_delta
                train_mask &= ~((ts >= purge_start) & (ts <= purge_end))
                ranges.append(
                    {
                        "group": str(group_id),
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "purge_start": purge_start.isoformat(),
                        "purge_end": purge_end.isoformat(),
                    }
                )
            train_idx = all_idx[np.asarray(train_mask, dtype=bool)]
            test_idx = all_idx[np.asarray(test_mask, dtype=bool)]
            purged_count = int(np.sum((~test_mask) & (~np.asarray(train_mask, dtype=bool))))
            splits.append(
                CPCVSplit(
                    split_id=len(splits),
                    test_groups=tuple(int(value) for value in test_groups),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    purged_train_count=purged_count,
                    test_time_ranges=ranges,
                )
            )
        return splits


def make_cpcv_splits(
    timestamps: Iterable[Any],
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_gap_bars: int = 96,
    interval_minutes: int = 5,
    embargo_bars: int = 0,
    max_splits: int | None = None,
) -> list[CPCVSplit]:
    return CombinatorialPurgedSplit(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_gap_bars=purge_gap_bars,
        interval_minutes=interval_minutes,
        embargo_bars=embargo_bars,
        max_splits=max_splits,
    ).split(timestamps)


def combinatorial_purged_splits(
    timestamps: Iterable[Any],
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_gap_bars: int = 96,
    interval_minutes: int = 5,
    embargo_bars: int = 0,
    max_splits: int | None = None,
) -> list[CPCVSplit]:
    """Backward-compatible named helper for prompt-specified CPCV checks.

    This is an additive alias. Existing V3.2.6 callers should keep using
    make_cpcv_splits or CombinatorialPurgedSplit unchanged.
    """

    return make_cpcv_splits(
        timestamps,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_gap_bars=purge_gap_bars,
        interval_minutes=interval_minutes,
        embargo_bars=embargo_bars,
        max_splits=max_splits,
    )


def sharpe_ratio(returns: Iterable[float]) -> float | None:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std <= 0:
        return None
    return float(np.mean(values) / std)


def deflated_sharpe_ratio(returns: Iterable[float], n_trials: int = 1) -> dict[str, Any]:
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    sr = sharpe_ratio(values)
    if n < 3 or sr is None:
        return {
            "status": "insufficient_returns",
            "sample_count": n,
            "sharpe": sr,
            "n_trials": int(max(n_trials, 1)),
        }

    demeaned = values - float(np.mean(values))
    std = float(np.std(values, ddof=1))
    skew = float(np.mean((demeaned / std) ** 3)) if std > 0 else 0.0
    kurtosis = float(np.mean((demeaned / std) ** 4)) if std > 0 else 3.0
    variance_term = max(1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr, 1e-12)
    sr_std = sqrt(variance_term / max(n - 1, 1))
    trials = int(max(n_trials, 1))
    expected_max_sr = _normal_ppf(1.0 - 1.0 / max(trials + 1, 2)) * sr_std
    z_value = (sr - expected_max_sr) / max(sr_std, 1e-12)
    probability = _normal_cdf(z_value)
    return {
        "status": "ok",
        "sample_count": n,
        "sharpe": float(sr),
        "skew": skew,
        "kurtosis": kurtosis,
        "sr_std": float(sr_std),
        "n_trials": trials,
        "expected_max_sharpe_under_multiple_testing": float(expected_max_sr),
        "dsr_z": float(z_value),
        "dsr_probability": float(probability),
    }


def probability_of_backtest_overfitting(
    in_sample_performance: Iterable[Iterable[float]],
    out_sample_performance: Iterable[Iterable[float]],
) -> dict[str, Any]:
    ins = np.asarray(list(in_sample_performance), dtype=float)
    outs = np.asarray(list(out_sample_performance), dtype=float)
    if ins.ndim != 2 or outs.ndim != 2 or ins.shape != outs.shape:
        return {"status": "invalid_shape", "pbo": None}
    if ins.shape[0] == 0 or ins.shape[1] < 2:
        return {"status": "insufficient_strategies", "pbo": None}

    logits = []
    selected_indices = []
    for row_in, row_out in zip(ins, outs):
        if not np.isfinite(row_in).any() or not np.isfinite(row_out).any():
            continue
        selected = int(np.nanargmax(row_in))
        selected_indices.append(selected)
        order = np.argsort(-np.nan_to_num(row_out, nan=-np.inf))
        rank_zero = int(np.where(order == selected)[0][0])
        if outs.shape[1] == 1:
            percentile = 1.0
        else:
            percentile = 1.0 - (rank_zero / (outs.shape[1] - 1))
        percentile = min(max(percentile, 1e-6), 1.0 - 1e-6)
        logits.append(log(percentile / (1.0 - percentile)))

    if not logits:
        return {"status": "insufficient_finite_values", "pbo": None}
    logits_array = np.asarray(logits, dtype=float)
    return {
        "status": "ok",
        "pbo": float(np.mean(logits_array < 0.0)),
        "lambda_logits": [float(value) for value in logits_array],
        "selected_strategy_indices": selected_indices,
        "split_count": int(len(logits_array)),
        "strategy_count": int(ins.shape[1]),
    }


def synthetic_overlap_self_test() -> dict[str, Any]:
    timestamps = pd.date_range("2026-01-01 09:00:00", periods=120, freq="5min", tz="Asia/Seoul")
    splitter = CombinatorialPurgedSplit(n_groups=6, n_test_groups=2, purge_gap_bars=3, interval_minutes=5)
    splits = splitter.split(timestamps)
    purge_delta = pd.Timedelta(minutes=15)
    violations = []
    ts = pd.Series(timestamps)
    for split in splits:
        train_times = ts.iloc[split.train_idx]
        for range_info in split.test_time_ranges:
            start = pd.Timestamp(range_info["start"])
            end = pd.Timestamp(range_info["end"])
            mask = (train_times >= start - purge_delta) & (train_times <= end + purge_delta)
            if bool(mask.any()):
                violations.append({"split_id": split.split_id, "test_groups": split.test_groups})
    return {
        "status": "PASS" if not violations else "FAIL",
        "split_count": len(splits),
        "n_groups": splitter.n_groups,
        "n_test_groups": splitter.n_test_groups,
        "purge_gap_bars": splitter.purge_gap_bars,
        "violations": violations,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(synthetic_overlap_self_test(), ensure_ascii=False, indent=2))
