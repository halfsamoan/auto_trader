# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: config.py, core/technical.py, ai/action_label_builder.py
# Description: Additive futures-only cache, feature, label, and split dataset helpers.
# ================================================================================

"""Futures-only dataset helpers for MNQ/MES same-asset validation.

This module is intentionally separate from the domestic-stock feature path. It
does not import KIS clients, does not load .env, and reads only local futures
CSV caches populated from yfinance.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.action_label_builder import ActionLabelConfig, action_label_distribution, build_action_labels
from config import INTRADAY_CACHE_DIR
from core.technical import atr, bollinger_b, ensure_ohlcv, macd, rsi


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / INTRADAY_CACHE_DIR

FUTURES_FEATURE_COLUMNS = [
    "close_return",
    "volume_zscore",
    "rsi",
    "macd_hist",
    "vwap_distance",
    "atr_pct",
    "bollinger_pct_b",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "futures_session_position",
    "globex_open_flag",
    "time_of_day_sin",
    "time_of_day_cos",
    "us_rth_flag",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "momentum_15m",
    "momentum_30m",
    "momentum_60m",
    "momentum_120m",
    "ret_12_zscore",
    "ret_24_zscore",
    "realized_vol_6",
    "realized_vol_12",
    "realized_vol_24",
    "range_pct",
    "range_ma_ratio_12",
    "range_ma_ratio_24",
    "volatility_compression_12_24",
    "volume_zscore_12",
    "volume_zscore_24",
    "volume_ratio_12",
    "volume_ratio_24",
    "dollar_volume",
    "dollar_volume_zscore_24",
    "close_location_value",
    "signed_volume_proxy",
    "signed_volume_zscore_24",
    "price_volume_pressure_12",
    "price_volume_pressure_24",
    "cme_weekday_sin",
    "cme_weekday_cos",
    "minutes_to_rth_open_norm",
    "minutes_to_globex_close_norm",
]


def futures_cache_path(symbol: str, interval: str = "5m") -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{interval}.csv"


def load_futures_cache(symbol: str, interval: str = "5m") -> pd.DataFrame:
    path = futures_cache_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in raw.columns:
        return pd.DataFrame()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    frame = raw.rename(columns=rename).dropna(subset=["timestamp"]).set_index("timestamp")
    out = ensure_ohlcv(frame)
    if out.empty:
        return out
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float, default: float = 0.0) -> pd.Series:
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_count = min_periods if min_periods is not None else max(2, window // 2)
    mean = series.rolling(window, min_periods=min_count).mean()
    std = series.rolling(window, min_periods=min_count).std().replace(0, np.nan)
    return _safe_divide(series - mean, std)


def _cme_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC").tz_convert("America/Chicago")
    return index.tz_convert("America/Chicago")


def _cme_session_dates(index: pd.DatetimeIndex) -> list[date]:
    cme = _cme_index(index)
    out = []
    for ts in cme:
        value = ts.date()
        if ts.hour >= 17:
            value = value + timedelta(days=1)
        out.append(value)
    return out


def _futures_anchored_vwap(data: pd.DataFrame) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3.0
    volume = data["Volume"].fillna(0)
    session = _cme_session_dates(data.index)
    weighted = (typical * volume).groupby(session).cumsum()
    denom = volume.groupby(session).cumsum().replace(0, np.nan)
    return weighted / denom


def _session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cme = _cme_index(index)
    minutes = pd.Series(cme.hour * 60 + cme.minute, index=index, dtype=float)
    globex_open = 17 * 60
    rth_open = 8 * 60 + 30
    globex_close = 16 * 60
    session_minutes = ((minutes - globex_open) % (24 * 60)).clip(0, 23 * 60)
    session_position = (session_minutes / float(23 * 60)).clip(0, 1)
    phase = 2.0 * np.pi * session_position
    weekday = pd.Series(cme.weekday, index=index, dtype=float)
    minutes_to_rth_open = ((rth_open - minutes) % (24 * 60)) / float(24 * 60)
    minutes_to_globex_close = ((globex_close - minutes) % (24 * 60)) / float(24 * 60)
    return pd.DataFrame(
        {
            "futures_session_position": session_position,
            "globex_open_flag": (session_minutes <= 60).astype(float),
            "time_of_day_sin": np.sin(phase),
            "time_of_day_cos": np.cos(phase),
            "us_rth_flag": ((minutes >= rth_open) & (minutes <= 15 * 60 + 15)).astype(float),
            "cme_weekday_sin": np.sin(2.0 * np.pi * weekday / 7.0),
            "cme_weekday_cos": np.cos(2.0 * np.pi * weekday / 7.0),
            "minutes_to_rth_open_norm": minutes_to_rth_open,
            "minutes_to_globex_close_norm": minutes_to_globex_close,
        },
        index=index,
    )


def build_futures_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = ensure_ohlcv(df)
    if data.empty:
        return pd.DataFrame(columns=FUTURES_FEATURE_COLUMNS)
    if not isinstance(data.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=FUTURES_FEATURE_COLUMNS)
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data = data.sort_index()
    close = data["Close"]
    volume = data["Volume"].fillna(0)
    vwap = _futures_anchored_vwap(data)
    atr_series = atr(data, 10)
    _, _, macd_hist = macd(close)
    pct_b = bollinger_b(close)
    volume_mean = volume.rolling(20, min_periods=5).mean()
    volume_std = volume.rolling(20, min_periods=5).std().replace(0, np.nan)

    features = pd.DataFrame(index=data.index)
    features["close_return"] = close.pct_change()
    features["volume_zscore"] = (volume - volume_mean) / volume_std
    features["rsi"] = rsi(close, 14) / 100.0
    features["macd_hist"] = macd_hist / close.replace(0, np.nan)
    features["vwap_distance"] = (close / vwap.replace(0, np.nan)) - 1.0
    features["atr_pct"] = atr_series / close.replace(0, np.nan)
    features["bollinger_pct_b"] = pct_b
    for window in [1, 3, 6, 12, 24]:
        features[f"ret_{window}"] = close.pct_change(window)
    for window in [1, 3, 6, 12]:
        features[f"return_{window}"] = close.pct_change(window)
    features["momentum_15m"] = features["ret_3"]
    features["momentum_30m"] = features["ret_6"]
    features["momentum_60m"] = features["ret_12"]
    features["momentum_120m"] = features["ret_24"]
    features["ret_12_zscore"] = _safe_zscore(features["ret_12"], 24)
    features["ret_24_zscore"] = _safe_zscore(features["ret_24"], 48)
    ret = close.pct_change()
    for window in [6, 12, 24]:
        features[f"realized_vol_{window}"] = ret.rolling(window, min_periods=max(2, window // 2)).std()
    features["range_pct"] = _safe_divide(data["High"] - data["Low"], close.replace(0, np.nan))
    for window in [12, 24]:
        range_mean = features["range_pct"].rolling(window, min_periods=max(2, window // 2)).mean()
        features[f"range_ma_ratio_{window}"] = _safe_divide(features["range_pct"], range_mean)
    features["volatility_compression_12_24"] = _safe_divide(features["realized_vol_12"], features["realized_vol_24"])
    for window in [12, 24]:
        volume_mean_window = volume.rolling(window, min_periods=max(2, window // 2)).mean()
        features[f"volume_zscore_{window}"] = _safe_zscore(volume, window)
        features[f"volume_ratio_{window}"] = _safe_divide(volume, volume_mean_window)
    dollar_volume = close * volume
    features["dollar_volume"] = dollar_volume
    features["dollar_volume_zscore_24"] = _safe_zscore(dollar_volume, 24)
    high_low_range = (data["High"] - data["Low"]).replace(0, np.nan)
    clv = ((close - data["Low"]) - (data["High"] - close)) / high_low_range
    features["close_location_value"] = clv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    features["signed_volume_proxy"] = features["close_location_value"] * volume
    features["signed_volume_zscore_24"] = _safe_zscore(features["signed_volume_proxy"], 24)
    for window in [12, 24]:
        signed_sum = features["signed_volume_proxy"].rolling(window, min_periods=max(2, window // 2)).sum()
        volume_sum = volume.rolling(window, min_periods=max(2, window // 2)).sum()
        features[f"price_volume_pressure_{window}"] = _safe_divide(signed_sum, volume_sum)
    session = _session_features(data.index)
    for column in session.columns:
        features[column] = session[column]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return features[FUTURES_FEATURE_COLUMNS].astype(float)


def build_futures_symbol_samples(
    df: pd.DataFrame,
    symbol: str,
    label_config: ActionLabelConfig,
    sequence_length: int,
) -> dict[str, object]:
    features = build_futures_feature_frame(df)
    labels = build_action_labels(df, label_config)
    joined = features.join(labels, how="inner")
    xs, actions, expected_returns, risks, holding_times, raw_labels, synthetic_actions, times = [], [], [], [], [], [], [], []
    for end in range(sequence_length - 1, len(joined) - label_config.horizon_bars):
        row = joined.iloc[end]
        if pd.isna(row["action_class"]):
            continue
        window = joined[FUTURES_FEATURE_COLUMNS].iloc[end - sequence_length + 1 : end + 1]
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
        "symbol": np.asarray([symbol.upper()] * len(xs), dtype=object),
        "label_distribution": action_label_distribution(labels),
    }


def concat_futures_samples(samples: list[dict[str, object]]) -> dict[str, object]:
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


def aggregate_futures_label_distribution(samples: list[dict[str, object]]) -> dict[str, object]:
    total_target = total_stop = total_neutral = 0
    action_counts: dict[str, int] = {}
    for sample in samples:
        dist = sample.get("label_distribution", {})
        total_target += int(dist.get("target_count", 0))
        total_stop += int(dist.get("stop_count", 0))
        total_neutral += int(dist.get("neutral_count", 0))
        for key, value in dict(dist.get("action_class_distribution", {})).items():
            action_counts[key] = action_counts.get(key, 0) + int(value)
    total = max(total_target + total_stop + total_neutral, 1)
    return {
        "target_count": total_target,
        "stop_count": total_stop,
        "neutral_count": total_neutral,
        "target_ratio": total_target / total,
        "stop_ratio": total_stop / total,
        "neutral_ratio": total_neutral / total,
        "action_class_distribution": action_counts,
    }
