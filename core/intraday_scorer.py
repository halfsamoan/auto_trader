# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: pandas, numpy
# Description: Intraday Scorer (5분봉 기반)
# ==============================================================================

"""intraday_scorer.py
5분봉 기준으로 intraday_score(0~100) 를 계산합니다.
간단히 가격 변화 비율을 사용하고, 0~100 로 클램프합니다.
"""
import pandas as pd
import numpy as np

def compute_intraday_score(df: pd.DataFrame) -> float:
    """5분봉 기반 intraday_score 계산.
    
    입력값:
        df: 1분 봉 데이터 (yfinance) – 반드시 'Open'과 'Close' 컬럼 포함.
    반환값:
        0~100 사이의 float 점수.
    
    알고리즘:
    1. 1분 데이터를 5분으로 리샘플링 (OHLC).
    2. 각 5분 캔들의 가격 변동 퍼센트 = (Close - Open) / Open * 100.
    3. 최신 5분 캔들의 변동을 점수로 사용합니다.
    4. 변동이 -5% 이하이면 0점, +5% 이상이면 100점, 그 사이는 선형 매핑.
    """
    if df.empty:
        return 50.0
    # 5분 리샘플링
    ohlc = df.resample('5min').agg({
        'Open': 'first',
        'Close': 'last'
    }).dropna()
    if ohlc.empty:
        return 50.0
    latest = ohlc.iloc[-1]
    change_pct = (latest['Close'] - latest['Open']) / latest['Open'] * 100
    # -5% ~ +5% 를 0~100 로 매핑
    score = np.clip((change_pct + 5) * 10, 0, 100)
    return float(score)
