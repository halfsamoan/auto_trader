# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: pandas, numpy
# Description: Strategy flags and simple utilities
# ==============================================================================

"""strategy.py
전략 관련 플래그와 간단한 유틸리티를 정의합니다.
MVP에서는 chase filter와 skip flag만 구현합니다.
"""

class StrategyFlags:
    """전략 플래그 컨테이너.
    
    chasing: bool
        True이면 chase filter 적용 (현재 MVP에서는 사용되지 않음).
    skip: bool
        True이면 해당 티커에 대해 신호를 건너뜁니다.
    """
    def __init__(self, chasing: bool = False, skip: bool = False):
        self.chasing = chasing
        self.skip = skip
