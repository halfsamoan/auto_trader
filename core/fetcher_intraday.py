# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: yfinance, pandas, numpy
# Description: Intraday 데이터 fetcher (1분 봉)
# ==============================================================================

"""fetcher_intraday.py
1분 봉 intraday 데이터를 yfinance 로 가져오는 헬퍼 모듈.
"""
import yfinance as yf
import pandas as pd

def fetch_intraday(ticker: str, period: str = "5d") -> pd.DataFrame:
    """주어진 티커의 1분 봉 데이터를 반환합니다.
    
    입력값:
        ticker: 종목 코드 (예: "005930")
        period: yfinance 지원 기간 문자열 (예: "5d", "10d")
    반환값:
        pandas DataFrame, 컬럼: Open, High, Low, Close, Volume, Adj Close
    """
    full_ticker = ticker + ".KS"
    df = yf.download(tickers=full_ticker, period=period, interval="1m", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"{ticker}에 대한 1분 봉 데이터를 가져오지 못했습니다.")
    return df
