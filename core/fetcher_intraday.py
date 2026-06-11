# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/fetcher_daily.py, core/technical.py
# Description: yfinance 장중 데이터를 조회하고 미완성 5분봉을 제거합니다.
# ================================================================================

"""Intraday data fetcher using yfinance."""

from __future__ import annotations

import pandas as pd

from .fetcher_daily import _download_krx


# _interval_to_timedelta는 yfinance interval 문자열을 pandas Timedelta로 변환합니다.
def _interval_to_timedelta(interval: str) -> pd.Timedelta | None:
    try:
        unit = interval[-1]
        value = int(interval[:-1])
    except (ValueError, IndexError):
        return None
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    return None


# drop_incomplete_bar는 현재 시각 기준으로 아직 닫히지 않은 마지막 봉을 제거합니다.
def drop_incomplete_bar(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        return data
    delta = _interval_to_timedelta(interval)
    if delta is None:
        return data
    last_ts = data.index[-1]
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo is not None else pd.Timestamp.now()
    # yfinance intraday index는 봉 시작 시각이므로 종료 시각 전이면 최신 봉은 지표 계산에서 제외합니다.
    if last_ts + delta > now:
        return data.iloc[:-1].copy()
    return data


# fetch_intraday는 국내 종목 장중 데이터를 조회하고 옵션에 따라 미완성 봉을 제거합니다.
def fetch_intraday(code: str, period: str = "5d", interval: str = "5m", drop_incomplete: bool = True) -> pd.DataFrame:
    data = _download_krx(code, period=period, interval=interval)
    if data.empty:
        raise ValueError(f"{code} {interval} intraday 데이터를 가져오지 못했습니다.")
    return drop_incomplete_bar(data, interval) if drop_incomplete else data
