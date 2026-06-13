"""Market breadth features from AI_TRAIN_UNIVERSE cache sensors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.feature_builder import anchored_vwap
from ai.universe_builder import load_ai_universe_records
from core.fetcher_intraday import load_intraday_cache
from core.technical import ensure_ohlcv


SEMICONDUCTOR_CODES = {"005930", "000660", "009150", "042660", "402340"}


def empty_market_breadth_features() -> dict[str, float | None]:
    return {
        "kospi200_vwap_above_ratio": None,
        "kospi200_positive_return_ratio": None,
        "kospi200_volume_spike_ratio": None,
        "kosdaq50_vwap_above_ratio": None,
        "kosdaq50_positive_return_ratio": None,
        "kosdaq50_volume_spike_ratio": None,
        "semiconductor_sector_strength": None,
        "largecap_momentum_score": None,
        "universe_risk_on_ratio": None,
    }


def _empty_bucket() -> dict[str, list[float]]:
    return {"vwap": [], "positive": [], "volume": []}


def _bucket_mean(bucket: dict[str, list[float]], key: str) -> float | None:
    return float(np.mean(bucket[key])) if bucket[key] else None


def build_market_breadth_features(universe: list[str] | None = None, interval: str = "5m") -> dict[str, float | None]:
    records = load_ai_universe_records(refresh=False)
    if universe:
        records = [{"code": code, "source": "manual"} for code in universe]
    kospi_bucket = _empty_bucket()
    kosdaq_bucket = _empty_bucket()
    returns = []
    semiconductor_returns = []
    risk_on = []
    for row in records:
        code = str(row.get("code") or "").zfill(6)
        source = str(row.get("source") or "")
        bucket = kosdaq_bucket if source.startswith("KOSDAQ") else kospi_bucket
        try:
            data = ensure_ohlcv(load_intraday_cache(code, interval))
            if len(data) < 20:
                continue
            close = data["Close"]
            latest_return = float(close.pct_change().iloc[-1])
            vwap = anchored_vwap(data)
            bucket["vwap"].append(float(close.iloc[-1] > vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else 0.0)
            bucket["positive"].append(float(latest_return > 0))
            vol = data["Volume"].fillna(0)
            vol_mean = float(vol.rolling(20).mean().iloc[-1] or 0)
            bucket["volume"].append(float(vol.iloc[-1] > vol_mean * 1.5) if vol_mean > 0 else 0.0)
            ret12 = float(close.pct_change(12).iloc[-1]) if len(close) > 12 else latest_return
            returns.append(ret12)
            risk_on.append(float(ret12 > 0))
            if code in SEMICONDUCTOR_CODES:
                semiconductor_returns.append(ret12)
        except Exception:
            continue
    if not returns:
        return empty_market_breadth_features()
    return {
        "kospi200_vwap_above_ratio": _bucket_mean(kospi_bucket, "vwap"),
        "kospi200_positive_return_ratio": _bucket_mean(kospi_bucket, "positive"),
        "kospi200_volume_spike_ratio": _bucket_mean(kospi_bucket, "volume"),
        "kosdaq50_vwap_above_ratio": _bucket_mean(kosdaq_bucket, "vwap"),
        "kosdaq50_positive_return_ratio": _bucket_mean(kosdaq_bucket, "positive"),
        "kosdaq50_volume_spike_ratio": _bucket_mean(kosdaq_bucket, "volume"),
        "semiconductor_sector_strength": float(np.mean(semiconductor_returns)) if semiconductor_returns else None,
        "largecap_momentum_score": float(np.mean(returns)),
        "universe_risk_on_ratio": float(np.mean(risk_on)) if risk_on else None,
    }


def safe_market_breadth_features() -> dict[str, Any]:
    try:
        return build_market_breadth_features()
    except Exception:
        return empty_market_breadth_features()
