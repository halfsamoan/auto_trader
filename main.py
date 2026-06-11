# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: yfinance, pandas, numpy, scipy, python-dotenv
# Description: 진입 신호를 계산하고 dry-run 실행을 담당하는 메인 스크립트
# ==============================================================================

"""main.py
MVP 단계의 진입 신호를 계산하고 콘솔 및 JSON 로그에 기록합니다.
실제 주문은 하지 않으며, --dry-run 옵션이 기본입니다.
"""
import argparse
import os
from datetime import datetime

from core.fetcher_intraday import fetch_intraday
from core.gaussian_score_engine import GaussianScoreEngine
from core.intraday_scorer import compute_intraday_score
from logger import log_signal, log_trading
from risk_manager import RiskManager

# 간단한 전략 플래그와 위험 관리 스텁
class StrategyFlags:
    # chasing filter가 없어야 함 -> 여기서는 항상 False 로 가정
    chasing = False
    skip = False

# RiskManager imported from risk_manager module
    def allow_entry(self, ticker: str, df) -> bool:
        # MVP에서는 항상 True 반환 (실제 로직은 위험 관리 구현 필요)
        return True

def decide_signal(final_score: float, intraday_score: float, gaussian_score: float,
                 strategy: StrategyFlags, risk_mgr: RiskManager, ticker: str) -> str:
    """신호를 결정합니다.
    
    Parameters
    ----------
    final_score: float (0~100)
    intraday_score: float (0~100)
    gaussian_score: float (0~100)
    strategy: StrategyFlags 객체
    risk_mgr: RiskManager 객체
    ticker: 종목 코드
    
    Returns
    -------
    str: 'buy', 'hold', or 'avoid'
    """
    # buy 조건 복합 체크
    if (final_score >= 72 and intraday_score >= 68 and gaussian_score >= 55 and
        not strategy.chasing and not strategy.skip and risk_mgr.allow_entry(ticker, None)):
        return "buy"
    # hold/avoid 기준
    if final_score >= 55:
        return "hold"
    return "avoid"

def main():
    parser = argparse.ArgumentParser(description="KIS Open API Intraday Auto‑Trader (Dry‑Run)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="실제 주문 없이 신호만 계산합니다 (default).")
    args = parser.parse_args()

    tickers = ["005930", "000660", "035720"]  # 삼성, SK하이닉스, NAVER
    strategy = StrategyFlags()
    risk_mgr = RiskManager()
    engine = GaussianScoreEngine()

    for ticker in tickers:
        try:
            # 1분 봉 5일 데이터 가져오기
            df = fetch_intraday(ticker, period="5d")
            # intraday_score (0~100)
            intraday_score = compute_intraday_score(df)
            # gaussian 점수 (각 축 및 가중합)
            gauss = engine.compute(ticker, df)
            gaussian_score = gauss["gaussian_score"]
            # final_score 가중 합 (0~100)
            final_score = 0.70 * intraday_score + 0.30 * gaussian_score
            # 신호 결정
            signal = decide_signal(final_score, intraday_score, gaussian_score,
                                   strategy, risk_mgr, ticker)
            # 로그 기록
            timestamp = datetime.now().isoformat()
            log_signal({
                "ticker": ticker,
                "intraday_score": intraday_score,
                "gaussian_score": gaussian_score,
                "final_score": final_score,
                "signal": signal,
                "timestamp": timestamp,
            })
            # dry‑run에서는 실제 주문을 하지 않음
            if args.dry_run:
                print(f"[{timestamp}] {ticker}: 최종점수={final_score:.2f}, 신호={signal}")
            else:
                # 실거래 로직 (stub)
                from kis_trader import place_order
                place_order(ticker, signal)
                log_trading({"ticker": ticker, "action": signal, "timestamp": timestamp})
        except Exception as e:
            print(f"{ticker} 처리 중 오류: {e}")

if __name__ == "__main__":
    main()
