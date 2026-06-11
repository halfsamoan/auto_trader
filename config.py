# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: 자동매매 대상 종목과 핵심 리스크 파라미터를 관리합니다.
# ================================================================================

"""Central configuration for the KR + futures Paper Lab trader."""

WATCHLIST = [
    {"code": "005930", "name": "삼성전자", "asset_class": "domestic-stock"},
    {"code": "000660", "name": "SK하이닉스", "asset_class": "domestic-stock"},
]

EXCLUDED_CODES = ["035720"]

DOMESTIC_STOCK_CAPITAL_KRW = 5_000_000
FUTURES_PAPER_CAPITAL_KRW = 5_000_000
FX_RATE_USDKRW = 1350

DOMESTIC_STOCK_AMOUNT_BY_CODE = {
    "005930": 1_000_000,
    "000660": 3_000_000,
}

AMOUNT_PER_TRADE = 5_000_000
DRY_RUN_ACCOUNT_VALUE = DOMESTIC_STOCK_CAPITAL_KRW
RISK_PER_TRADE_PCT = 0.01
MAX_POSITIONS = 2

COMMISSION = 0.00015
TAX = 0.0018

FUTURES_WATCHLIST = [
    {"symbol": "MNQ", "name": "Micro Nasdaq 100 Futures", "asset_class": "futures"},
    {"symbol": "MES", "name": "Micro S&P 500 Futures", "asset_class": "futures"},
]

ENABLE_OVERSEAS_STOCK = False
ENABLE_DOMESTIC_STOCK_PAPER_ORDER = True
ENABLE_FUTURES_PAPER_SIM = True
ENABLE_FUTURES_KIS_PAPER_ORDER = False
FUTURES_KIS_ENDPOINTS_VERIFIED = False
ENABLE_REAL_ORDER = False

MAX_TRADES_PER_DAY_TOTAL = 10
MAX_TRADES_PER_DAY_DOMESTIC_STOCK = 5
MAX_TRADES_PER_DAY_FUTURES = 5
MAX_TRADES_PER_DAY = MAX_TRADES_PER_DAY_TOTAL
MAX_BUYS_PER_DAY = 5
MAX_SELLS_PER_DAY = 10
TARGET_DAILY_SIGNALS = 10
