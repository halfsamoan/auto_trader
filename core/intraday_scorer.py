# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: 5분봉 기반 장중 진입 점수와 과열 필터를 계산합니다.
# ================================================================================

"""5-minute intraday score engine."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from .technical import atr, bollinger_b, clamp_score, ensure_ohlcv, macd, sma, vwap


@dataclass
class IntradayScore:
    score: float
    regime: str
    chasing_filter: bool
    forced_hold_reason: str | None
    trend_score: float
    momentum_score: float
    macd_score: float
    bollinger_score: float
    volume_score: float
    risk_score: float
    price: float
    ma5: float
    ma20: float
    vwap: float
    pct_b: float
    return_15m: float
    return_30m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_5m(df: pd.DataFrame) -> pd.DataFrame:
    data = ensure_ohlcv(df)
    if data.empty:
        return data
    if isinstance(data.index, pd.DatetimeIndex):
        return (
            data.resample("5min")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "High", "Low", "Close"])
        )
    return data.copy()


def _latest_time(df: pd.DataFrame) -> time | None:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return None
    ts = df.index[-1]
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Seoul")
    return ts.time()


def compute_intraday_details(df: pd.DataFrame) -> IntradayScore:
    bars = to_5m(df)
    if bars.empty or len(bars) < 5:
        return IntradayScore(50.0, "insufficient_data", False, None, 50, 50, 50, 50, 50, 50, 50, np.nan, np.nan, np.nan, np.nan, 0, 0)

    close = bars["Close"]
    price = float(close.iloc[-1])
    ma5_series = sma(close, 5)
    ma20_series = sma(close, 20)
    vwap_series = vwap(bars)
    ma5 = float(ma5_series.dropna().iloc[-1]) if not ma5_series.dropna().empty else price
    ma20 = float(ma20_series.dropna().iloc[-1]) if not ma20_series.dropna().empty else price
    current_vwap = float(vwap_series.dropna().iloc[-1]) if not vwap_series.dropna().empty else price

    if ma5 > ma20 and price > current_vwap:
        regime = "intraday_uptrend"
    elif ma5 > ma20 and price <= current_vwap:
        regime = "weak_uptrend"
    else:
        regime = "intraday_downtrend"

    trend_raw = 50 + ((ma5 / ma20 - 1) * 1000 if ma20 else 0) + (8 if price > current_vwap else -8)
    trend_score = clamp_score(trend_raw)

    ret_15m = float((price / close.iloc[-4] - 1) * 100) if len(close) >= 4 and close.iloc[-4] else 0.0
    ret_30m = float((price / close.iloc[-7] - 1) * 100) if len(close) >= 7 and close.iloc[-7] else 0.0
    momentum_score = clamp_score(50 + ret_15m * 8 + ret_30m * 4)

    macd_line, signal_line, hist = macd(close)
    hist_latest = float(hist.iloc[-1]) if len(hist) else 0.0
    macd_scale = max(price * 0.002, 1.0)
    macd_score = clamp_score(50 + (hist_latest / macd_scale) * 50)

    pct_b_series = bollinger_b(close)
    pct_b = float(pct_b_series.dropna().iloc[-1]) if not pct_b_series.dropna().empty else 0.5
    bollinger_score = clamp_score(100 - abs(pct_b - 0.65) * 140)

    avg_volume = bars["Volume"].rolling(20, min_periods=5).mean()
    latest_volume = float(bars["Volume"].iloc[-1]) if np.isfinite(bars["Volume"].iloc[-1]) else 0.0
    avg_vol = float(avg_volume.dropna().iloc[-1]) if not avg_volume.dropna().empty else latest_volume
    volume_ratio = latest_volume / avg_vol if avg_vol else 1.0
    volume_score = clamp_score(45 + min(volume_ratio, 3.0) * 18)

    atr_series = atr(bars, 14)
    latest_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else price * 0.01
    atr_pct = latest_atr / price if price else 0.01
    risk_score = clamp_score(100 - atr_pct * 2500)

    score = (
        trend_score * 0.25
        + momentum_score * 0.20
        + macd_score * 0.20
        + bollinger_score * 0.15
        + volume_score * 0.10
        + risk_score * 0.10
    )

    reason = None
    current_time = _latest_time(bars)
    if ret_15m > 3.0:
        reason = "return_15m > 3%"
    elif ret_30m > 5.0:
        reason = "return_30m > 5%"
    elif pct_b > 0.95:
        reason = "pct_b > 0.95"
    elif current_time is not None and current_time < time(9, 10):
        reason = "09:10 전 신규 진입 금지"
    elif current_time is not None and current_time >= time(14, 50):
        reason = "14:50 이후 신규 진입 금지"

    return IntradayScore(
        score=clamp_score(score),
        regime=regime,
        chasing_filter=reason is not None,
        forced_hold_reason=reason,
        trend_score=trend_score,
        momentum_score=momentum_score,
        macd_score=macd_score,
        bollinger_score=bollinger_score,
        volume_score=volume_score,
        risk_score=risk_score,
        price=price,
        ma5=ma5,
        ma20=ma20,
        vwap=current_vwap,
        pct_b=pct_b,
        return_15m=ret_15m,
        return_30m=ret_30m,
    )


def compute_intraday_score(df: pd.DataFrame) -> float:
    return compute_intraday_details(df).score
