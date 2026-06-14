# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-14 (V3.2.3)
# Dependency: config.py, core/intraday_scorer.py, core/technical.py
# Description: 국내주식 AI 신호용 5분봉 feature를 미래 누수 없이 append-only로 생성합니다.
# ================================================================================

"""국내주식 AI 신호용 5분봉 feature를 미래 누수 없이 생성합니다."""

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
]

ENABLE_SPECTRAL_FEATURES = False


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


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float, default: float = 0.0) -> pd.Series:
    """0 나눗셈과 무한대를 기본값으로 치환합니다."""

    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """현재 시점까지의 rolling window만 사용해 z-score를 계산합니다."""

    min_count = min_periods if min_periods is not None else max(2, window // 2)
    mean = series.rolling(window, min_periods=min_count).mean()
    std = series.rolling(window, min_periods=min_count).std().replace(0, np.nan)
    return _safe_divide(series - mean, std)


def _add_multiscale_momentum_features(features: pd.DataFrame, close: pd.Series) -> None:
    """5분봉 기준 다중 기간 수익률과 momentum feature를 뒤에 추가합니다."""

    for window in [1, 3, 6, 12, 24]:
        features[f"ret_{window}"] = close.pct_change(window)
    features["momentum_15m"] = features["ret_3"]
    features["momentum_30m"] = features["ret_6"]
    features["momentum_60m"] = features["ret_12"]
    features["momentum_120m"] = features["ret_24"]
    features["ret_12_zscore"] = _safe_zscore(features["ret_12"], 24)
    features["ret_24_zscore"] = _safe_zscore(features["ret_24"], 48)


def _add_volatility_compression_features(features: pd.DataFrame, data: pd.DataFrame) -> None:
    """방향성보다 목표/손절 도달 환경을 설명하는 변동성 feature를 추가합니다."""

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    ret = close.pct_change()
    for window in [6, 12, 24]:
        features[f"realized_vol_{window}"] = ret.rolling(window, min_periods=max(2, window // 2)).std()
    features["range_pct"] = _safe_divide(high - low, close.replace(0, np.nan))
    for window in [12, 24]:
        range_mean = features["range_pct"].rolling(window, min_periods=max(2, window // 2)).mean()
        features[f"range_ma_ratio_{window}"] = _safe_divide(features["range_pct"], range_mean)
    features["volatility_compression_12_24"] = _safe_divide(features["realized_vol_12"], features["realized_vol_24"])


def _add_volume_pressure_proxy_features(features: pd.DataFrame, data: pd.DataFrame) -> None:
    """체결 방향이 아닌 OHLCV 기반 volume/pressure proxy feature를 추가합니다."""

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"].fillna(0)
    for window in [12, 24]:
        volume_mean = volume.rolling(window, min_periods=max(2, window // 2)).mean()
        features[f"volume_zscore_{window}"] = _safe_zscore(volume, window)
        features[f"volume_ratio_{window}"] = _safe_divide(volume, volume_mean)
    dollar_volume = close * volume
    features["dollar_volume"] = dollar_volume
    features["dollar_volume_zscore_24"] = _safe_zscore(dollar_volume, 24)

    high_low_range = (high - low).replace(0, np.nan)
    close_location_value = ((close - low) - (high - close)) / high_low_range
    features["close_location_value"] = close_location_value.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    features["signed_volume_proxy"] = features["close_location_value"] * volume
    features["signed_volume_zscore_24"] = _safe_zscore(features["signed_volume_proxy"], 24)
    for window in [12, 24]:
        signed_sum = features["signed_volume_proxy"].rolling(window, min_periods=max(2, window // 2)).sum()
        volume_sum = volume.rolling(window, min_periods=max(2, window // 2)).sum()
        features[f"price_volume_pressure_{window}"] = _safe_divide(signed_sum, volume_sum)


def _add_spectral_features_disabled_by_default(features: pd.DataFrame, close: pd.Series) -> None:
    """FFT 계열 feature는 비용 검증 전까지 기본 비활성화 상태로 둡니다."""

    if not ENABLE_SPECTRAL_FEATURES:
        return
    # TODO(V3.2.x): numpy FFT 기반 24-window spectral feature는 별도 비용/누수 검증 후 연결합니다.
    # TODO(V3.3.x): wavelet feature는 PyWavelets 의존성, 계산 비용, 누수 검증 후 별도 작업으로 검토합니다.
    _ = features, close


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
    _add_multiscale_momentum_features(features, close)
    _add_volatility_compression_features(features, data)
    _add_volume_pressure_proxy_features(features, data)
    _add_spectral_features_disabled_by_default(features, close)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return features[FEATURE_COLUMNS].astype(float)


def latest_feature_sequence(df: pd.DataFrame, sequence_length: int, rule_score: float | None = None) -> tuple[np.ndarray | None, list[str]]:
    frame = build_feature_frame(df, rule_score=rule_score)
    if len(frame) < sequence_length:
        return None, FEATURE_COLUMNS
    return frame.iloc[-sequence_length:].to_numpy(dtype=np.float32), FEATURE_COLUMNS
