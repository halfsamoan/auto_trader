"""Leakage-aware market feature and bootstrap label generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from toa_ai.config import ModelConfig
from toa_ai.domain import ACTION_TO_ID, Action


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "log_return_1",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "volume_zscore_20",
    "volume_ratio_20",
    "dollar_volume_zscore_20",
    "vwap_distance",
    "rsi_14",
    "macd_hist_pct",
    "atr_pct_14",
    "realized_vol_6",
    "realized_vol_12",
    "realized_vol_24",
    "momentum_3",
    "momentum_6",
    "momentum_12",
    "close_position_20",
    "time_of_day_sin",
    "time_of_day_cos",
    "position_flag",
    "unrealized_pnl_pct",
    "holding_bars_norm",
]


@dataclass(frozen=True)
class PolicyDataset:
    x: np.ndarray
    action: np.ndarray
    expected_return: np.ndarray
    risk: np.ndarray
    timestamps: np.ndarray
    symbols: np.ndarray
    feature_columns: list[str]
    action_counts: dict[str, int]


def ensure_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    out = out.rename(columns={col: str(col).lower() for col in out.columns})
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return out[["open", "high", "low", "close", "volume"]]


def build_feature_frame(
    frame: pd.DataFrame,
    position_flag: float = 0.0,
    entry_price: float | None = None,
    holding_bars: int = 0,
    max_holding_bars: int = 96,
) -> pd.DataFrame:
    data = ensure_ohlcv(frame)
    if data.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    close = data["close"].replace(0, np.nan)
    high = data["high"]
    low = data["low"]
    open_ = data["open"]
    volume = data["volume"].fillna(0.0)
    typical = (high + low + close) / 3.0
    vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    dollar_volume = close * volume

    features = pd.DataFrame(index=data.index)
    for window in [1, 3, 6, 12]:
        features[f"return_{window}"] = close.pct_change(window)
    features["log_return_1"] = np.log(close / close.shift(1))
    features["range_pct"] = (high - low) / close
    features["body_pct"] = (close - open_) / close
    features["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    features["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
    features["volume_zscore_20"] = _zscore(volume, 20)
    features["volume_ratio_20"] = volume / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    features["dollar_volume_zscore_20"] = _zscore(dollar_volume, 20)
    features["vwap_distance"] = close / vwap.replace(0, np.nan) - 1.0
    features["rsi_14"] = _rsi(close, 14) / 100.0
    features["macd_hist_pct"] = _macd_hist(close) / close
    features["atr_pct_14"] = _atr(data, 14) / close
    for window in [6, 12, 24]:
        features[f"realized_vol_{window}"] = close.pct_change().rolling(window, min_periods=max(2, window // 2)).std()
    features["momentum_3"] = features["return_3"]
    features["momentum_6"] = features["return_6"]
    features["momentum_12"] = features["return_12"]
    low_20 = low.rolling(20, min_periods=5).min()
    high_20 = high.rolling(20, min_periods=5).max()
    features["close_position_20"] = (close - low_20) / (high_20 - low_20).replace(0, np.nan)
    sin, cos = _time_features(data.index)
    features["time_of_day_sin"] = sin
    features["time_of_day_cos"] = cos
    features["position_flag"] = float(position_flag)
    if entry_price and entry_price > 0:
        features["unrealized_pnl_pct"] = close / float(entry_price) - 1.0
    else:
        features["unrealized_pnl_pct"] = 0.0
    features["holding_bars_norm"] = min(max(float(holding_bars), 0.0) / max(float(max_holding_bars), 1.0), 1.0)

    features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return features[FEATURE_COLUMNS].astype(float)


def build_policy_dataset_for_symbol(frame: pd.DataFrame, symbol: str, config: ModelConfig) -> PolicyDataset:
    data = ensure_ohlcv(frame)
    if len(data) < config.sequence_length + config.horizon_bars + 1:
        return _empty_dataset()

    base_features = build_feature_frame(data, position_flag=0.0)
    base_matrix = base_features.to_numpy(dtype=np.float32)
    close_values = data["close"].to_numpy(dtype=np.float32)
    position_flag_idx = FEATURE_COLUMNS.index("position_flag")
    unrealized_idx = FEATURE_COLUMNS.index("unrealized_pnl_pct")
    holding_idx = FEATURE_COLUMNS.index("holding_bars_norm")
    holding_progress = np.linspace(0.0, 1.0, config.sequence_length, dtype=np.float32)
    xs: list[np.ndarray] = []
    actions: list[int] = []
    expected_returns: list[float] = []
    risks: list[float] = []
    timestamps: list[Any] = []
    symbols: list[str] = []

    for end in range(config.sequence_length - 1, len(data) - config.horizon_bars):
        start = end - config.sequence_length + 1
        entry_label = _entry_label(data, end, config)
        xs.append(base_matrix[start : end + 1].copy())
        actions.append(ACTION_TO_ID[entry_label["action"]])
        expected_returns.append(float(entry_label["expected_return"]))
        risks.append(float(entry_label["risk"]))
        timestamps.append(data.index[end])
        symbols.append(symbol)

        holding_window = base_matrix[start : end + 1].copy()
        holding_window[:, position_flag_idx] = 1.0
        entry_price = float(close_values[start])
        if entry_price > 0:
            holding_window[:, unrealized_idx] = close_values[start : end + 1] / entry_price - 1.0
        holding_window[:, holding_idx] = holding_progress
        holding_label = _holding_label(data, end, config)
        xs.append(holding_window)
        actions.append(ACTION_TO_ID[holding_label["action"]])
        expected_returns.append(float(holding_label["expected_return"]))
        risks.append(float(holding_label["risk"]))
        timestamps.append(data.index[end])
        symbols.append(symbol)

    action_array = np.asarray(actions, dtype=np.int64)
    return PolicyDataset(
        x=np.asarray(xs, dtype=np.float32),
        action=action_array,
        expected_return=np.asarray(expected_returns, dtype=np.float32),
        risk=np.asarray(risks, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=object),
        symbols=np.asarray(symbols, dtype=object),
        feature_columns=FEATURE_COLUMNS,
        action_counts=_action_counts(action_array),
    )


def concat_datasets(datasets: list[PolicyDataset]) -> PolicyDataset:
    valid = [dataset for dataset in datasets if len(dataset.x)]
    if not valid:
        return _empty_dataset()
    x = np.concatenate([dataset.x for dataset in valid], axis=0)
    action = np.concatenate([dataset.action for dataset in valid], axis=0)
    expected_return = np.concatenate([dataset.expected_return for dataset in valid], axis=0)
    risk = np.concatenate([dataset.risk for dataset in valid], axis=0)
    timestamps = np.concatenate([dataset.timestamps for dataset in valid], axis=0)
    symbols = np.concatenate([dataset.symbols for dataset in valid], axis=0)
    order = np.argsort(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    return PolicyDataset(
        x=x[order],
        action=action[order],
        expected_return=expected_return[order],
        risk=risk[order],
        timestamps=timestamps[order],
        symbols=symbols[order],
        feature_columns=FEATURE_COLUMNS,
        action_counts=_action_counts(action[order]),
    )


def _entry_label(data: pd.DataFrame, idx: int, config: ModelConfig) -> dict[str, Any]:
    entry = float(data["close"].iloc[idx])
    end = min(idx + config.horizon_bars, len(data) - 1)
    future = data.iloc[idx + 1 : end + 1]
    if entry <= 0 or future.empty:
        return {"action": Action.NO_ACTION, "expected_return": 0.0, "risk": 1.0}
    target = entry * (1.0 + config.target_return)
    stop = entry * (1.0 + config.stop_return)
    for _, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if low <= stop and high >= target:
            return {"action": Action.NO_ACTION, "expected_return": config.stop_return, "risk": 1.0}
        if high >= target:
            return {"action": Action.OPEN_LONG, "expected_return": config.target_return, "risk": 0.0}
        if low <= stop:
            return {"action": Action.NO_ACTION, "expected_return": config.stop_return, "risk": 1.0}
    future_return = float(data["close"].iloc[end] / entry - 1.0)
    return {"action": Action.OPEN_LONG if future_return > config.target_return * 0.5 else Action.NO_ACTION, "expected_return": future_return, "risk": float(future_return < 0)}


def _holding_label(data: pd.DataFrame, idx: int, config: ModelConfig) -> dict[str, Any]:
    entry = float(data["close"].iloc[max(0, idx - config.sequence_length + 1)])
    current = float(data["close"].iloc[idx])
    end = min(idx + config.horizon_bars, len(data) - 1)
    future = data.iloc[idx + 1 : end + 1]
    if entry <= 0 or current <= 0 or future.empty:
        return {"action": Action.HOLD_LONG, "expected_return": 0.0, "risk": 0.5}
    stop = current * (1.0 + config.stop_return)
    target = current * (1.0 + config.target_return)
    target2 = current * (1.0 + config.target_return * 2.0)
    for _, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if low <= stop:
            return {"action": Action.CLOSE_LONG, "expected_return": config.stop_return, "risk": 1.0}
        if high >= target2:
            return {"action": Action.CLOSE_LONG, "expected_return": config.target_return * 2.0, "risk": 0.0}
        if high >= target:
            return {"action": Action.REDUCE_LONG, "expected_return": config.target_return, "risk": 0.1}
    future_return = float(data["close"].iloc[end] / current - 1.0)
    action = Action.HOLD_LONG if future_return >= 0 else Action.CLOSE_LONG
    return {"action": action, "expected_return": future_return, "risk": float(future_return < 0)}


def _empty_dataset() -> PolicyDataset:
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


def _action_counts(actions: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for action_id, count in zip(*np.unique(actions, return_counts=True)):
        from toa_ai.domain import ACTION_CLASSES

        out[ACTION_CLASSES[int(action_id)].value] = int(count)
    return out


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
    return (series - mean) / std


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _macd_hist(series: pd.Series) -> pd.Series:
    fast = series.ewm(span=6, adjust=False).mean()
    slow = series.ewm(span=13, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=5, adjust=False).mean()
    return line - signal


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    high = data["high"]
    low = data["low"]
    close = data["close"]
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _time_features(index: pd.Index) -> tuple[pd.Series, pd.Series]:
    if isinstance(index, pd.DatetimeIndex):
        converted = index.tz_convert("Asia/Seoul") if index.tz is not None else index
        minute = pd.Series([ts.hour * 60 + ts.minute for ts in converted], index=index, dtype=float)
    else:
        minute = pd.Series(np.zeros(len(index)), index=index, dtype=float)
    phase = 2.0 * math.pi * minute / (24.0 * 60.0)
    return pd.Series(np.sin(phase), index=index), pd.Series(np.cos(phase), index=index)
