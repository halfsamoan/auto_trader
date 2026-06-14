# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: 해외선물 paper-sim용 계약 메타데이터를 제공합니다.
# ================================================================================

"""Futures metadata for paper simulation only."""

FUTURES_CONTRACTS = {
    "MNQ": {
        "symbol": "MNQ",
        "yfinance_symbol": "MNQ=F",
        "name": "Micro Nasdaq 100 Futures",
        "exchange": "CME",
        "tick_size": 0.25,
        "tick_value_usd": 0.50,
        "multiplier": 2.0,
        "margin_per_contract_usd": 2500.0,
        "currency": "USD",
        "default_contract_qty": 1,
        "max_contract_qty": 1,
        "trading_hours_note": "CME Globex 시간 수동 검증 필요",
        "roll_over_note": "yfinance MNQ=F는 front-month 연속 데이터라 롤오버 갭 수동 검증 필요",
        "is_real_order_enabled": False,
        "is_kis_paper_order_enabled": False,
        "is_paper_sim_enabled": True,
    },
    "MES": {
        "symbol": "MES",
        "yfinance_symbol": "MES=F",
        "name": "Micro S&P 500 Futures",
        "exchange": "CME",
        "tick_size": 0.25,
        "tick_value_usd": 1.25,
        "multiplier": 5.0,
        "margin_per_contract_usd": 1800.0,
        "currency": "USD",
        "default_contract_qty": 1,
        "max_contract_qty": 1,
        "trading_hours_note": "CME Globex 시간 수동 검증 필요",
        "roll_over_note": "yfinance MES=F는 front-month 연속 데이터라 롤오버 갭 수동 검증 필요",
        "is_real_order_enabled": False,
        "is_kis_paper_order_enabled": False,
        "is_paper_sim_enabled": True,
    },
}


# get_contract는 symbol 기준 선물 메타데이터를 반환합니다.
def get_contract(symbol: str) -> dict:
    key = symbol.upper()
    if key not in FUTURES_CONTRACTS:
        raise KeyError(f"지원하지 않는 paper-sim 해외선물 계약입니다: {symbol}")
    return FUTURES_CONTRACTS[key].copy()
