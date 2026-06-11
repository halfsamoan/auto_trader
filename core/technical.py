# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: None
# Description: MA, RSI, ATR, MACD, VWAP 등 기술 지표 유틸리티를 제공합니다.
# ================================================================================

"""Technical indicator utilities for daily and intraday scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with single-level OHLCV columns from yfinance output."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if len(out.columns.levels) > 0 and {"Open", "High", "Low", "Close"}.intersection(out.columns.get_level_values(0)):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    return out


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    if out.empty:
        return out
    rename = {col: str(col).title() for col in out.columns}
    out = out.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan
    out = out[needed].copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def clamp_score(value: float, default: float = 50.0) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(np.clip(value, 0.0, 100.0))


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    data = ensure_ohlcv(df)
    if data.empty:
        return pd.Series(dtype=float)
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def bollinger_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def vwap(df: pd.DataFrame) -> pd.Series:
    data = ensure_ohlcv(df)
    if data.empty:
        return pd.Series(dtype=float)
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    volume = data["Volume"].fillna(0)
    denom = volume.cumsum().replace(0, np.nan)
    return (typical * volume).cumsum() / denom
