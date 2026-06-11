# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: pandas, numpy, yfinance, scipy
# Description: Gaussian Score Engine (5축 점수, 0~100 스케일)
# ==============================================================================

"""gaussian_score_engine.py
Gaussian Score Engine 구현.
각 축은 0~100 스케일이며, 결합 가중치는 다음과 같습니다.
trend 0.30, momentum 0.25, fundamental 0.20, risk 0.15, market 0.10
"""
import pandas as pd
import numpy as np
import yfinance as yf
from .technical import rsi, atr

class GaussianScoreEngine:
    """Gaussian Score Engine 클래스.
    
    입력값:
        ticker: 종목 코드 (예: "005930")
        df: 가격 데이터(DataFrame) - 반드시 'Close', 'High', 'Low', 'Open' 컬럼 포함
    반환값:
        dict: 개별 축 점수(0-100)와 가중 합계 'gaussian_score'
    """
    def __init__(self):
        # 가중치 정의 (합계 1.0)
        self.weights = {
            "trend": 0.30,
            "momentum": 0.25,
            "fundamental": 0.20,
            "risk": 0.15,
            "market": 0.10,
        }

    # ---------------------------------------------------------------------
    # 개별 축 계산 함수들 (모두 0~100 스케일 반환)
    # ---------------------------------------------------------------------
    def _calc_trend(self, df: pd.DataFrame) -> float:
        """Trend 점수 계산
        20일 SMA 기울기를 구해 0~100 로 정규화합니다.
        기울기가 클수록 높은 점수이며, 데이터 부족 시 50점 반환.
        """
        sma = df['Close'].rolling(window=20).mean()
        if len(sma.dropna()) < 2:
            return 50.0
        recent = sma.dropna().iloc[-5:]
        x = np.arange(len(recent))
        y = recent.values
        slope = np.polyfit(x, y, 1)[0]
        base = recent.mean()
        if base == 0:
            return 50.0
        pct = (slope / base) * 100  # 퍼센트 변화
        score = np.clip(50 + pct * 2, 0, 100)  # 0.5% 변화당 1점 가산 예시
        return float(score)

    def _calc_momentum(self, df: pd.DataFrame) -> float:
        """Momentum 점수: RSI(14) 값을 그대로 사용 (0~100).
        최신 RSI 값이 없으면 50점 반환.
        """
        rsi_series = rsi(df['Close'], period=14)
        if rsi_series.dropna().empty:
            return 50.0
        return float(rsi_series.dropna().iloc[-1])

    def _calc_fundamental(self, ticker: str) -> float:
        """Fundamental 점수
        yfinance.info 에서 'beta' 값을 사용해 0~100 로 매핑합니다.
        beta가 없거나 조회 실패 시 50점 반환.
        """
        try:
            info = yf.Ticker(ticker + ".KS").info
            beta = info.get('beta')
            if beta is not None:
                # beta 0~2 범위 가정, 1을 중립(50점)으로 매핑
                return float(np.clip((beta - 1) * 50 + 50, 0, 100))
            else:
                return 50.0
        except Exception:
            return 50.0

    def _calc_risk(self, df: pd.DataFrame) -> float:
        """Risk 점수: ATR(14) 를 0~100 로 정규화합니다.
        ATR 계산은 technical.atr 함수 사용.
        """
        try:
            atr_series = atr(df, period=14)
            if atr_series.dropna().empty:
                return 50.0
            return float(atr_series.dropna().iloc[-1])
        except Exception:
            return 50.0

    def _calc_market(self) -> float:
        """Market 점수
        VIX, SPY, QQQ 일일 종가 변동률을 평균해 0~100 로 매핑합니다.
        조회 실패 시 50점 반환.
        """
        try:
            vix = yf.download('^VIX', period='2d', interval='1d')
            spy = yf.download('SPY', period='2d', interval='1d')
            qqq = yf.download('QQQ', period='2d', interval='1d')
            def pct_change(df):
                if len(df) < 2:
                    return 0.0
                return (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            avg_change = np.mean([pct_change(vix), pct_change(spy), pct_change(qqq)])
            # -5%~+5% 를 0~100 으로 매핑 (예시)
            return float(np.clip((avg_change + 5) * 10, 0, 100))
        except Exception:
            return 50.0

    # ---------------------------------------------------------------------
    # 메인 계산 함수
    # ---------------------------------------------------------------------
    def compute(self, ticker: str, df: pd.DataFrame) -> dict:
        """전체 Gaussian 점수와 개별 축 점수를 반환합니다.
        
        Parameters
        ----------
        ticker: str
            종목 코드 (예: "005930")
        df: pd.DataFrame
            가격 데이터 (1분 또는 일봉) - 반드시 'Close', 'High', 'Low', 'Open' 컬럼 포함
        """
        trend = self._calc_trend(df)
        momentum = self._calc_momentum(df)
        fundamental = self._calc_fundamental(ticker)
        risk = self._calc_risk(df)
        market = self._calc_market()
        # 가중합 (0~100) 계산
        gaussian_score = (
            trend * self.weights['trend'] +
            momentum * self.weights['momentum'] +
            fundamental * self.weights['fundamental'] +
            risk * self.weights['risk'] +
            market * self.weights['market']
        )
        return {
            "trend": trend,
            "momentum": momentum,
            "fundamental": fundamental,
            "risk": risk,
            "market": market,
            "gaussian_score": float(gaussian_score),
        }
