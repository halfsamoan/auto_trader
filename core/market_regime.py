# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: KOSPI 일봉 기반 시장 레짐을 판정하고 실시간 캐시를 관리합니다.
# ================================================================================

"""KOSPI market regime filter for KR-only trading."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from .technical import ensure_ohlcv, sma

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REGIME_CACHE = DATA_DIR / "regime_cache.json"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# _read_cache는 같은 날짜 반복 조회를 피하기 위해 저장된 레짐 결과를 읽습니다.
def _read_cache(today: str) -> dict[str, Any] | None:
    if not REGIME_CACHE.exists():
        return None
    try:
        data = json.loads(REGIME_CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("date") == today:
        return data
    return None


# _write_cache는 실시간 dry-run/paper-check용 KOSPI 레짐 결과를 저장합니다.
def _write_cache(record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGIME_CACHE.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


# _fetch_kospi_yfinance는 yfinance ^KS11 일봉을 우선 조회합니다.
def _fetch_kospi_yfinance(period: str = "6mo") -> pd.DataFrame:
    df = yf.download("^KS11", period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    return ensure_ohlcv(df)


# _fetch_kospi_pykrx는 yfinance 실패 시 pykrx KOSPI 지수 코드 1001로 보강 조회합니다.
def _fetch_kospi_pykrx(days: int = 220) -> pd.DataFrame:
    try:
        from pykrx import stock
    except Exception:
        return pd.DataFrame()

    end = datetime.now().date()
    start = end - timedelta(days=days)
    try:
        df = stock.get_index_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), "1001")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    renamed = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
    return ensure_ohlcv(renamed)


# fetch_kospi_daily는 KOSPI 일봉을 yfinance 우선, pykrx fallback 순서로 가져옵니다.
def fetch_kospi_daily(period: str = "6mo") -> pd.DataFrame:
    data = _fetch_kospi_yfinance(period)
    if not data.empty:
        return data
    return _fetch_kospi_pykrx()


# classify_regime은 전달된 KOSPI 일봉의 마지막 종가와 MA20으로 risk_on/off를 판정합니다.
def classify_regime(kospi_daily: pd.DataFrame) -> dict[str, Any]:
    data = ensure_ohlcv(kospi_daily)
    if len(data) < 20:
        return {"regime": "unknown", "reason": "KOSPI MA20 계산 데이터 부족", "close": None, "ma20": None}
    close = data["Close"]
    ma20 = sma(close, 20)
    latest_close = float(close.iloc[-1])
    latest_ma20 = float(ma20.iloc[-1])
    regime = "risk_on" if latest_close > latest_ma20 else "risk_off"
    return {
        "regime": regime,
        "reason": None,
        "close": latest_close,
        "ma20": latest_ma20,
        "date": str(data.index[-1].date()) if hasattr(data.index[-1], "date") else None,
    }


# get_current_market_regime은 실시간 실행에서만 캐시를 사용해 오늘의 KOSPI 레짐을 반환합니다.
def get_current_market_regime(use_cache: bool = True) -> dict[str, Any]:
    today = date.today().isoformat()
    if use_cache:
        cached = _read_cache(today)
        if cached:
            return cached
    try:
        result = classify_regime(fetch_kospi_daily())
    except Exception as exc:
        result = {"regime": "unknown", "reason": f"KOSPI 조회 실패: {exc}", "close": None, "ma20": None}
    result["date"] = today
    if use_cache:
        _write_cache(result)
    return result


# historical_regime_for_date는 백테스트 날짜 기준 직전 일봉까지만 사용해 미래 데이터 누수를 막습니다.
def historical_regime_for_date(kospi_daily: pd.DataFrame, current_dt: pd.Timestamp | datetime) -> str:
    data = ensure_ohlcv(kospi_daily)
    if data.empty:
        return "unknown"
    current_date = pd.Timestamp(current_dt).date()
    historical = data[data.index.date < current_date] if isinstance(data.index, pd.DatetimeIndex) else data
    if len(historical) < 20:
        return "unknown"
    return str(classify_regime(historical)["regime"])
