# core/technical.py
"""Technical indicator utilities.
모든 지표는 0~100 스케일을 반환하도록 설계되었습니다.
"""
import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) 계산.
    df는 반드시 ['High', 'Low', 'Close'] 컬럼을 포함해야 함.
    반환값은 0~100 스케일로 정규화됩니다.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window).mean()
    # 정규화: 최근 30일 평균 ATR 대비 현재 ATR 비율을 0~100으로 매핑
    recent_mean = atr_series.rolling(30).mean()
    normalized = (atr_series / recent_mean) * 100
    normalized = normalized.clip(lower=0, upper=100)
    return normalized
