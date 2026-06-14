# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.6)
# Dependency: ai/cpcv.py, ai/train_baseline_lgbm.py
# Description: Cost-aware meta-labeling helpers for offline validation.
# ================================================================================

"""Cost-aware meta-labeling for the domestic-stock policy dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np

from ai.cpcv import deflated_sharpe_ratio, make_cpcv_splits, probability_of_backtest_overfitting
from ai.train_baseline_lgbm import _flatten_last_step, _model_backend


@dataclass(frozen=True)
class MetaLabelingConfig:
    cost_assumption_return: float
    n_groups: int = 6
    n_test_groups: int = 2
    purge_gap_bars: int = 96
    interval_minutes: int = 5
    max_paths: int | None = None
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 31
    random_state: int = 42
    min_meta_train_samples: int = 100
    min_selected_trades: int = 20
    primary_thresholds: tuple[float, ...] = field(default_factory=lambda: (0.45, 0.50, 0.55, 0.60, 0.65))
    meta_thresholds: tuple[float, ...] = field(default_factory=lambda: (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80))


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _target(values: Any) -> np.ndarray:
    return (np.asarray(values) == 1).astype(int)


def _has_two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2


def _model_kwargs(backend_name: str, config: MetaLabelingConfig, scale_pos_weight: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"random_state": config.random_state}
    if backend_name == "lightgbm":
        kwargs.update(
            {
                "n_estimators": config.n_estimators,
                "learning_rate": config.learning_rate,
                "num_leaves": config.num_leaves,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": scale_pos_weight,
                "verbose": -1,
            }
        )
    else:
        kwargs.update(
            {
                "n_estimators": config.n_estimators,
                "learning_rate": config.learning_rate,
                "max_depth": 4,
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
            }
        )
    return kwargs


def _fit_classifier(model_cls: Any, backend_name: str, x: np.ndarray, y: np.ndarray, config: MetaLabelingConfig):
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    scale_pos_weight = negatives / max(positives, 1)
    model = model_cls(**_model_kwargs(backend_name, config, scale_pos_weight))
    model.fit(x, y)
    return model


def _predict_positive(model: Any, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    if prob.ndim == 1:
        return np.asarray(prob, dtype=float)
    if prob.shape[1] == 1:
        return np.zeros(len(x), dtype=float)
    return np.asarray(prob[:, 1], dtype=float)


def _meta_features(x: np.ndarray, primary_prob: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.asarray(primary_prob, dtype=float)])


def _selection_metrics(
    y_true: np.ndarray,
    gross_returns: np.ndarray,
    selected: np.ndarray,
    cost_return: float,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    positive = y_true == 1
    selected_count = int(np.sum(selected))
    base_rate = float(np.mean(positive)) if len(y_true) else None
    if selected_count == 0:
        return {
            "sample_count": int(len(y_true)),
            "selected_trade_count": 0,
            "selected_trade_ratio": 0.0,
            "precision": None,
            "recall": 0.0,
            "precision_lift": None,
            "gross_expected_return": None,
            "net_expected_return": None,
            "weighted_net_expected_return": None,
        }
    selected_net = gross_returns[selected] - cost_return
    selected_weights = None if weights is None else np.asarray(weights, dtype=float)[selected]
    weighted_net = None
    if selected_weights is not None and float(np.sum(selected_weights)) > 0:
        weighted_net = float(np.average(selected_net, weights=selected_weights))
    precision = float(np.mean(positive[selected]))
    recall = float(np.sum(selected & positive) / max(np.sum(positive), 1))
    return {
        "sample_count": int(len(y_true)),
        "selected_trade_count": selected_count,
        "selected_trade_ratio": float(np.mean(selected)),
        "precision": precision,
        "recall": recall,
        "precision_lift": None if base_rate is None else precision - base_rate,
        "gross_expected_return": float(np.mean(gross_returns[selected])),
        "net_expected_return": float(np.mean(selected_net)),
        "weighted_net_expected_return": weighted_net,
    }


def _threshold_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    gross_returns: np.ndarray,
    cost_return: float,
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        metrics = _selection_metrics(y_true, gross_returns, prob >= threshold, cost_return, weights=prob)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def _select_threshold(curve: list[dict[str, Any]], fallback: float, min_selected: int) -> dict[str, Any]:
    candidates = [
        row
        for row in curve
        if int(row.get("selected_trade_count") or 0) >= min_selected and row.get("net_expected_return") is not None
    ]
    if not candidates:
        return {"threshold": float(fallback), "selection_reason": "fallback_no_candidate"}
    profitable = [row for row in candidates if float(row.get("net_expected_return") or -999.0) > 0.0]
    pool = profitable or candidates
    selected = max(
        pool,
        key=lambda row: (
            float(row.get("net_expected_return") or -999.0),
            float(row.get("precision_lift") or -999.0),
            int(row.get("selected_trade_count") or 0),
        ),
    )
    return {
        **selected,
        "selection_reason": "train_only_positive_net_ev" if profitable else "train_only_best_available",
    }


def _aggregate_path_metrics(path_rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    trade_counts = np.asarray([int(row.get(f"{prefix}_metrics", {}).get("selected_trade_count") or 0) for row in path_rows], dtype=float)
    net_values = np.asarray(
        [
            float(row.get(f"{prefix}_metrics", {}).get("net_expected_return"))
            if row.get(f"{prefix}_metrics", {}).get("net_expected_return") is not None
            else np.nan
            for row in path_rows
        ],
        dtype=float,
    )
    precision_values = np.asarray(
        [
            float(row.get(f"{prefix}_metrics", {}).get("precision"))
            if row.get(f"{prefix}_metrics", {}).get("precision") is not None
            else np.nan
            for row in path_rows
        ],
        dtype=float,
    )
    if not len(path_rows):
        return {"path_count": 0}
    finite_net = net_values[np.isfinite(net_values)]
    finite_precision = precision_values[np.isfinite(precision_values)]
    return {
        "path_count": int(len(path_rows)),
        "total_selected_trades": int(np.sum(trade_counts)),
        "mean_path_selected_trades": float(np.mean(trade_counts)),
        "mean_path_net_expected_return": float(np.mean(finite_net)) if len(finite_net) else None,
        "median_path_net_expected_return": float(np.median(finite_net)) if len(finite_net) else None,
        "mean_path_precision": float(np.mean(finite_precision)) if len(finite_precision) else None,
    }


def evaluate_meta_cpcv(data: dict[str, Any], config: MetaLabelingConfig) -> dict[str, Any]:
    backend_name, model_cls, dependency_error = _model_backend()
    if model_cls is None:
        return {"status": "MODEL_DEPENDENCY_MISSING", "model_dependency_error": dependency_error}

    x_all = _flatten_last_step(np.asarray(data["X"]))
    y_all = _target(data["action_class"])
    returns_all = _as_float_array(data["expected_return_target"])
    timestamps = data["timestamp"]
    if len(x_all) == 0 or not _has_two_classes(y_all):
        return {"status": "DATA_NOT_READY", "reason": "empty samples or single-class target"}

    splits = make_cpcv_splits(
        timestamps,
        n_groups=config.n_groups,
        n_test_groups=config.n_test_groups,
        purge_gap_bars=config.purge_gap_bars,
        interval_minutes=config.interval_minutes,
        max_splits=config.max_paths,
    )
    path_rows: list[dict[str, Any]] = []
    meta_trade_returns: list[float] = []
    pbo_in: list[list[float]] = []
    pbo_out: list[list[float]] = []

    for split in splits:
        train_idx = split.train_idx
        test_idx = split.test_idx
        if len(train_idx) < config.min_meta_train_samples or len(test_idx) == 0:
            continue
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]
        if not _has_two_classes(y_train):
            continue
        primary = _fit_classifier(model_cls, backend_name, x_train, y_train, config)
        primary_train_prob = _predict_positive(primary, x_train)
        primary_test_prob = _predict_positive(primary, x_test)
        primary_curve = _threshold_curve(
            y_train,
            primary_train_prob,
            returns_all[train_idx],
            config.cost_assumption_return,
            config.primary_thresholds,
        )
        primary_selected = _select_threshold(primary_curve, fallback=0.50, min_selected=config.min_selected_trades)
        primary_threshold = float(primary_selected["threshold"])
        train_candidate_mask = primary_train_prob >= primary_threshold
        test_candidate_mask = primary_test_prob >= primary_threshold

        baseline_primary_metrics = _selection_metrics(
            y_test,
            returns_all[test_idx],
            test_candidate_mask,
            config.cost_assumption_return,
            weights=primary_test_prob,
        )
        if int(np.sum(train_candidate_mask)) < config.min_meta_train_samples:
            path_rows.append(
                {
                    "split_id": split.split_id,
                    "test_groups": list(split.test_groups),
                    "status": "INSUFFICIENT_TRADES",
                    "primary_threshold": primary_threshold,
                    "primary_metrics": baseline_primary_metrics,
                    "meta_metrics": {"selected_trade_count": 0},
                }
            )
            continue

        y_meta_train = y_train[train_candidate_mask]
        if not _has_two_classes(y_meta_train):
            continue
        meta_train_x = _meta_features(x_train[train_candidate_mask], primary_train_prob[train_candidate_mask])
        meta = _fit_classifier(model_cls, backend_name, meta_train_x, y_meta_train, config)
        meta_train_prob = _predict_positive(meta, meta_train_x)
        meta_test_prob_all = _predict_positive(meta, _meta_features(x_test, primary_test_prob))
        meta_train_curve = _threshold_curve(
            y_meta_train,
            meta_train_prob,
            returns_all[train_idx][train_candidate_mask],
            config.cost_assumption_return,
            config.meta_thresholds,
        )
        meta_selected = _select_threshold(
            meta_train_curve,
            fallback=0.60,
            min_selected=max(5, min(config.min_selected_trades, int(np.sum(train_candidate_mask)))),
        )
        meta_threshold = float(meta_selected["threshold"])
        meta_final_mask = test_candidate_mask & (meta_test_prob_all >= meta_threshold)
        meta_metrics = _selection_metrics(
            y_test,
            returns_all[test_idx],
            meta_final_mask,
            config.cost_assumption_return,
            weights=meta_test_prob_all,
        )
        if int(meta_metrics.get("selected_trade_count") or 0) > 0:
            meta_trade_returns.extend((returns_all[test_idx][meta_final_mask] - config.cost_assumption_return).astype(float).tolist())

        in_curve_values = []
        out_curve_values = []
        for threshold in config.meta_thresholds:
            train_metrics = _selection_metrics(
                y_meta_train,
                returns_all[train_idx][train_candidate_mask],
                meta_train_prob >= threshold,
                config.cost_assumption_return,
                weights=meta_train_prob,
            )
            test_metrics = _selection_metrics(
                y_test,
                returns_all[test_idx],
                test_candidate_mask & (meta_test_prob_all >= threshold),
                config.cost_assumption_return,
                weights=meta_test_prob_all,
            )
            in_curve_values.append(float(train_metrics.get("net_expected_return") if train_metrics.get("net_expected_return") is not None else -999.0))
            out_curve_values.append(float(test_metrics.get("net_expected_return") if test_metrics.get("net_expected_return") is not None else -999.0))
        pbo_in.append(in_curve_values)
        pbo_out.append(out_curve_values)
        path_rows.append(
            {
                "split_id": split.split_id,
                "test_groups": list(split.test_groups),
                "status": "ok",
                "purged_train_count": split.purged_train_count,
                "primary_threshold": primary_threshold,
                "primary_threshold_selection": primary_selected,
                "meta_threshold": meta_threshold,
                "meta_threshold_selection": meta_selected,
                "primary_metrics": baseline_primary_metrics,
                "meta_metrics": meta_metrics,
            }
        )

    baseline_summary = _aggregate_path_metrics(path_rows, "primary")
    meta_summary = _aggregate_path_metrics(path_rows, "meta")
    pbo = probability_of_backtest_overfitting(pbo_in, pbo_out)
    dsr = deflated_sharpe_ratio(meta_trade_returns, n_trials=max(len(config.meta_thresholds) * max(len(path_rows), 1), 1))
    meta_net = meta_summary.get("mean_path_net_expected_return")
    total_trades = int(meta_summary.get("total_selected_trades") or 0)
    if total_trades < config.min_selected_trades:
        edge_status = "INSUFFICIENT_TRADES"
    elif meta_net is not None and meta_net > 0 and (pbo.get("pbo") is None or float(pbo["pbo"]) < 0.50):
        edge_status = "META_EDGE_CANDIDATE"
    else:
        edge_status = "EDGE_NOT_CONFIRMED"
    primary_thresholds = [float(row["primary_threshold"]) for row in path_rows if "primary_threshold" in row]
    meta_thresholds = [float(row["meta_threshold"]) for row in path_rows if "meta_threshold" in row]
    return {
        "status": "ok" if path_rows else "DATA_NOT_READY",
        "backend": backend_name,
        "split_count": len(splits),
        "evaluated_path_count": len(path_rows),
        "config": {
            "n_groups": config.n_groups,
            "n_test_groups": config.n_test_groups,
            "purge_gap_bars": config.purge_gap_bars,
            "max_paths": config.max_paths,
            "cost_assumption_return": config.cost_assumption_return,
        },
        "selected_primary_threshold": float(median(primary_thresholds)) if primary_thresholds else 0.50,
        "selected_meta_threshold": float(median(meta_thresholds)) if meta_thresholds else 0.60,
        "baseline_primary_summary": baseline_summary,
        "meta_summary": meta_summary,
        "dsr": dsr,
        "pbo": pbo,
        "edge_status": edge_status,
        "path_results": path_rows,
    }


def fit_meta_oof_test_once(
    train_valid_data: dict[str, Any],
    test_data: dict[str, Any],
    config: MetaLabelingConfig,
    primary_threshold: float,
    meta_threshold: float,
) -> dict[str, Any]:
    backend_name, model_cls, dependency_error = _model_backend()
    if model_cls is None:
        return {"status": "MODEL_DEPENDENCY_MISSING", "model_dependency_error": dependency_error}

    x_train_valid = _flatten_last_step(np.asarray(train_valid_data["X"]))
    y_train_valid = _target(train_valid_data["action_class"])
    returns_train_valid = _as_float_array(train_valid_data["expected_return_target"])
    timestamps = train_valid_data["timestamp"]
    x_test = _flatten_last_step(np.asarray(test_data["X"]))
    y_test = _target(test_data["action_class"])
    returns_test = _as_float_array(test_data["expected_return_target"])
    if len(x_train_valid) == 0 or len(x_test) == 0 or not _has_two_classes(y_train_valid):
        return {"status": "DATA_NOT_READY", "reason": "empty train_valid/test or single-class train_valid"}

    oof_sum = np.zeros(len(x_train_valid), dtype=float)
    oof_count = np.zeros(len(x_train_valid), dtype=float)
    splits = make_cpcv_splits(
        timestamps,
        n_groups=config.n_groups,
        n_test_groups=config.n_test_groups,
        purge_gap_bars=config.purge_gap_bars,
        interval_minutes=config.interval_minutes,
        max_splits=config.max_paths,
    )
    for split in splits:
        train_idx = split.train_idx
        holdout_idx = split.test_idx
        if len(train_idx) < config.min_meta_train_samples or len(holdout_idx) == 0:
            continue
        if not _has_two_classes(y_train_valid[train_idx]):
            continue
        primary = _fit_classifier(model_cls, backend_name, x_train_valid[train_idx], y_train_valid[train_idx], config)
        oof_sum[holdout_idx] += _predict_positive(primary, x_train_valid[holdout_idx])
        oof_count[holdout_idx] += 1.0
    oof_ready = oof_count > 0
    if int(np.sum(oof_ready)) < config.min_meta_train_samples:
        return {"status": "DATA_NOT_READY", "reason": "insufficient CPCV OOF primary predictions"}
    oof_primary = np.zeros(len(x_train_valid), dtype=float)
    oof_primary[oof_ready] = oof_sum[oof_ready] / oof_count[oof_ready]
    meta_train_mask = oof_ready & (oof_primary >= primary_threshold)
    if int(np.sum(meta_train_mask)) < config.min_meta_train_samples or not _has_two_classes(y_train_valid[meta_train_mask]):
        return {
            "status": "INSUFFICIENT_TRADES",
            "oof_ready_count": int(np.sum(oof_ready)),
            "meta_train_candidate_count": int(np.sum(meta_train_mask)),
        }

    meta = _fit_classifier(
        model_cls,
        backend_name,
        _meta_features(x_train_valid[meta_train_mask], oof_primary[meta_train_mask]),
        y_train_valid[meta_train_mask],
        config,
    )
    final_primary = _fit_classifier(model_cls, backend_name, x_train_valid, y_train_valid, config)
    primary_test_prob = _predict_positive(final_primary, x_test)
    meta_test_prob = _predict_positive(meta, _meta_features(x_test, primary_test_prob))
    primary_selected = primary_test_prob >= primary_threshold
    meta_selected = primary_selected & (meta_test_prob >= meta_threshold)
    primary_metrics = _selection_metrics(y_test, returns_test, primary_selected, config.cost_assumption_return, weights=primary_test_prob)
    meta_metrics = _selection_metrics(y_test, returns_test, meta_selected, config.cost_assumption_return, weights=meta_test_prob)
    trade_returns = (returns_test[meta_selected] - config.cost_assumption_return).astype(float).tolist()
    total_trades = int(meta_metrics.get("selected_trade_count") or 0)
    meta_net = meta_metrics.get("net_expected_return")
    if total_trades < config.min_selected_trades:
        edge_status = "INSUFFICIENT_TRADES"
    elif meta_net is not None and float(meta_net) > 0.0:
        edge_status = "META_EDGE_CANDIDATE"
    else:
        edge_status = "EDGE_NOT_CONFIRMED"
    return {
        "status": "ok",
        "backend": backend_name,
        "primary_threshold": float(primary_threshold),
        "meta_threshold": float(meta_threshold),
        "oof_ready_count": int(np.sum(oof_ready)),
        "meta_train_candidate_count": int(np.sum(meta_train_mask)),
        "primary_test_metrics": primary_metrics,
        "meta_test_metrics": meta_metrics,
        "test_trade_returns": trade_returns,
        "edge_status": edge_status,
    }
