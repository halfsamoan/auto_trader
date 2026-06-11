# core/fetcher_daily.py
"""fetcher_daily.py
일일봉 데이터를 yfinance 로 가져오는 헬퍼 모듈.
"""
import yfinance as yf
import pandas as pd

def fetch_daily(ticker: str, period: str = "1y") -> pd.DataFrame:
    """주어진 티커의 일일 데이터를 반환합니다.
    
    Args:
        ticker: 종목 코드 (예: "005930")
        period: yfinance 지원 기간 문자열 (예: "1y", "2y")
    Returns:
        pandas DataFrame, 컬럼: Open, High, Low, Close, Volume, Adj Close
    """
    full_ticker = ticker + ".KS"
    df = yf.download(tickers=full_ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"{ticker}에 대한 일일 데이터를 가져오지 못했습니다.")
    return df
