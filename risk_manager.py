# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: None
# Description: 위험 관리 스텁 (MVP)
# ==============================================================================

"""risk_manager.py
위험 관리 모듈의 스텁 구현.
MVP에서는 always allow entry 반환.
"""

class RiskManager:
    """위험 관리 클래스.
    
    allow_entry: 현재는 항상 True 반환. 실제 구현 시 ATR, 변동성, 포지션 크기 등을 고려.
    """
    def allow_entry(self, ticker: str, df) -> bool:
        # MVP: 위험 관리는 항상 허용
        return True
