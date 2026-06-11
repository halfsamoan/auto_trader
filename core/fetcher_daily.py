# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: yfinance로 국내 종목 일봉 데이터를 조회합니다.
# ================================================================================

"""Daily data fetcher using yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .technical import ensure_ohlcv


def _download_krx(code: str, period: str, interval: str) -> pd.DataFrame:
    suffixes = ["KS", "KQ"] if "." not in code else [""]
    for suffix in suffixes:
        ticker = code if suffix == "" else f"{code}.{suffix}"
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        data = ensure_ohlcv(df)
        if not data.empty:
            return data
    return pd.DataFrame()


def fetch_daily(code: str, period: str = "1y") -> pd.DataFrame:
    data = _download_krx(code, period=period, interval="1d")
    if data.empty:
        raise ValueError(f"{code} 일봉 데이터를 가져오지 못했습니다.")
    return data
