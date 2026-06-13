"""Build leak-safe 5-minute sequence features for domestic-stock AI signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import AI_EXCLUDE_LUNCH_BARS
from core.intraday_scorer import compute_intraday_details
from core.technical import atr, bollinger_b, ensure_ohlcv, macd, rsi


FEATURE_COLUMNS = [
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
    "intraday_position",
    "kospi_regime_flag",
    "time_of_day_sin",
    "time_of_day_cos",
    "rule_score",
]


@dataclass(frozen=True)
class FeaturePolicy:
    exclude_lunch_bars: bool = AI_EXCLUDE_LUNCH_BARS
    lunch_start: str = "12:00"
    lunch_end: str = "12:59"


def _drop_incomplete_5m_bar(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        return data
    last_ts = data.index[-1]
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo is not None else pd.Timestamp.now()
    if last_ts + pd.Timedelta(minutes=5) > now:
        return data.iloc[:-1].copy()
    return data


def _session_key(index: pd.DatetimeIndex) -> Iterable[object]:
    if index.tz is not None:
        return index.tz_convert("Asia/Seoul").date
    return index.date


def anchored_vwap(data: pd.DataFrame) -> pd.Series:
    bars = ensure_ohlcv(data)
    if bars.empty:
        return pd.Series(dtype=float)
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    volume = bars["Volume"].fillna(0)
    if isinstance(bars.index, pd.DatetimeIndex):
        session = _session_key(bars.index)
        weighted = (typical * volume).groupby(session).cumsum()
        denom = volume.groupby(session).cumsum().replace(0, np.nan)
        return weighted / denom
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _exclude_lunch(data: pd.DataFrame, policy: FeaturePolicy) -> pd.DataFrame:
    if not policy.exclude_lunch_bars or not isinstance(data.index, pd.DatetimeIndex):
        return data
    index = data.index.tz_convert("Asia/Seoul") if data.index.tz is not None else data.index
    mask = (index.time >= pd.Timestamp(policy.lunch_start).time()) & (index.time <= pd.Timestamp(policy.lunch_end).time())
    return data.loc[~mask].copy()


def build_feature_frame(
    df: pd.DataFrame,
    rule_score: float | None = None,
    kospi_regime_flag: float = 1.0,
    exclude_lunch_bars: bool = AI_EXCLUDE_LUNCH_BARS,
) -> pd.DataFrame:
    data = _drop_incomplete_5m_bar(ensure_ohlcv(df))
    data = _exclude_lunch(data, FeaturePolicy(exclude_lunch_bars=exclude_lunch_bars))
    if data.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    close = data["Close"]
    volume = data["Volume"].fillna(0)
    vwap = anchored_vwap(data)
    atr_series = atr(data, 10)
    _, _, macd_hist = macd(close)
    pct_b = bollinger_b(close)
    volume_mean = volume.rolling(20, min_periods=5).mean()
    volume_std = volume.rolling(20, min_periods=5).std().replace(0, np.nan)
    index = data.index.tz_convert("Asia/Seoul") if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None else data.index
    minutes = pd.Series([ts.hour * 60 + ts.minute for ts in index], index=data.index, dtype=float)
    market_open = 9 * 60
    market_close = 15 * 60 + 30
    intraday_position = ((minutes - market_open) / (market_close - market_open)).clip(0, 1)
    phase = 2 * math.pi * intraday_position

    if rule_score is None:
        try:
            rule_score = float(compute_intraday_details(data).score)
        except Exception:
            rule_score = 50.0

    features = pd.DataFrame(index=data.index)
    features["close_return"] = close.pct_change()
    features["volume_zscore"] = (volume - volume_mean) / volume_std
    features["rsi"] = rsi(close, 14) / 100.0
    features["macd_hist"] = macd_hist / close.replace(0, np.nan)
    features["vwap_distance"] = (close / vwap.replace(0, np.nan)) - 1
    features["atr_pct"] = atr_series / close.replace(0, np.nan)
    features["bollinger_pct_b"] = pct_b
    for window in [1, 3, 6, 12]:
        features[f"return_{window}"] = close.pct_change(window)
    features["intraday_position"] = intraday_position
    features["kospi_regime_flag"] = float(kospi_regime_flag)
    features["time_of_day_sin"] = np.sin(phase)
    features["time_of_day_cos"] = np.cos(phase)
    features["rule_score"] = float(rule_score) / 100.0

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return features[FEATURE_COLUMNS].astype(float)


def latest_feature_sequence(df: pd.DataFrame, sequence_length: int, rule_score: float | None = None) -> tuple[np.ndarray | None, list[str]]:
    frame = build_feature_frame(df, rule_score=rule_score)
    if len(frame) < sequence_length:
        return None, FEATURE_COLUMNS
    return frame.iloc[-sequence_length:].to_numpy(dtype=np.float32), FEATURE_COLUMNS
