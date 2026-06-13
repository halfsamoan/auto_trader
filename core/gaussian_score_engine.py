# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: 일봉 기반 Gaussian 스타일 종합 점수를 계산합니다.
# ================================================================================

"""Deterministic five-axis Gaussian-style daily score engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .technical import atr, clamp_score, ensure_ohlcv, rsi, sma


class GaussianScoreEngine:
    def __init__(self) -> None:
        self.weights = {
            "trend": 0.40,
            "momentum": 0.40,
            "fundamental": 0.00,
            "risk": 0.20,
            "market": 0.00,
        }

    def _calc_trend(self, df: pd.DataFrame) -> float:
        data = ensure_ohlcv(df)
        if len(data) < 30:
            return 50.0
        close = data["Close"]
        ma20 = sma(close, 20)
        ma60 = sma(close, 60)
        latest_ma20 = ma20.dropna().iloc[-1] if not ma20.dropna().empty else close.iloc[-1]
        latest_ma60 = ma60.dropna().iloc[-1] if not ma60.dropna().empty else latest_ma20
        slope = (latest_ma20 / ma20.dropna().iloc[-6] - 1) * 100 if len(ma20.dropna()) >= 6 and ma20.dropna().iloc[-6] else 0
        alignment = (latest_ma20 / latest_ma60 - 1) * 100 if latest_ma60 else 0
        return clamp_score(50 + slope * 10 + alignment * 4)

    def _calc_momentum(self, df: pd.DataFrame) -> float:
        data = ensure_ohlcv(df)
        if len(data) < 15:
            return 50.0
        close = data["Close"]
        latest_rsi = float(rsi(close, 14).iloc[-1])
        ret_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 and close.iloc[-6] else 0
        ret_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 and close.iloc[-21] else 0
        return clamp_score(latest_rsi * 0.55 + clamp_score(50 + ret_5 * 4 + ret_20 * 1.5) * 0.45)

    def _calc_fundamental(self, ticker: str) -> float:
        try:
            info = yf.Ticker(f"{ticker}.KS").info or {}
        except Exception:
            return 50.0
        score_parts: list[float] = []
        trailing_pe = info.get("trailingPE")
        if trailing_pe and trailing_pe > 0:
            score_parts.append(clamp_score(80 - min(trailing_pe, 60)))
        price_to_book = info.get("priceToBook")
        if price_to_book and price_to_book > 0:
            score_parts.append(clamp_score(75 - min(price_to_book, 10) * 4))
        roe = info.get("returnOnEquity")
        if roe is not None:
            score_parts.append(clamp_score(50 + float(roe) * 120))
        return float(np.mean(score_parts)) if score_parts else 50.0

    def _calc_risk(self, df: pd.DataFrame) -> float:
        data = ensure_ohlcv(df)
        if len(data) < 20:
            return 50.0
        close = data["Close"]
        price = float(close.iloc[-1])
        atr_series = atr(data, 14)
        latest_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else price * 0.02
        atr_pct = latest_atr / price if price else 0.02
        recent_jump = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 and close.iloc[-4] else 0
        vol_mean = data["Volume"].rolling(20, min_periods=5).mean()
        latest_vol_mean = float(vol_mean.dropna().iloc[-1]) if not vol_mean.dropna().empty else float(data["Volume"].iloc[-1])
        volume_ratio = float(data["Volume"].iloc[-1]) / latest_vol_mean if latest_vol_mean else 1.0
        penalty = atr_pct * 1200 + max(recent_jump - 8, 0) * 3 + max(volume_ratio - 3, 0) * 8
        return clamp_score(85 - penalty)

    def _calc_market(self) -> float:
        try:
            scores = []
            for symbol in ["^VIX", "SPY", "QQQ"]:
                df = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=False)
                data = ensure_ohlcv(df)
                if len(data) < 2:
                    return 50.0
                ret = (data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1) * 100
                if symbol == "^VIX":
                    scores.append(clamp_score(50 - ret * 3))
                else:
                    scores.append(clamp_score(50 + ret * 8))
            return float(np.mean(scores))
        except Exception:
            return 50.0

    def compute(self, ticker: str, df: pd.DataFrame) -> dict[str, float]:
        trend = self._calc_trend(df)
        momentum = self._calc_momentum(df)
        fundamental = 50.0
        risk = self._calc_risk(df)
        market = 50.0
        gaussian_score = (
            trend * self.weights["trend"]
            + momentum * self.weights["momentum"]
            + fundamental * self.weights["fundamental"]
            + risk * self.weights["risk"]
            + market * self.weights["market"]
        )
        return {
            "trend": clamp_score(trend),
            "momentum": clamp_score(momentum),
            "fundamental": clamp_score(fundamental),
            "risk": clamp_score(risk),
            "market": clamp_score(market),
            "gaussian_score": clamp_score(gaussian_score),
        }
