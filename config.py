# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: 자동매매 대상 종목과 핵심 리스크 파라미터를 관리합니다.
# ================================================================================

"""Central configuration for the KR + futures Paper Lab trader."""

import json
from pathlib import Path

AI_UNIVERSE_MODE = "dynamic"
AI_UNIVERSE_KOSPI200_ENABLED = True
AI_UNIVERSE_KOSDAQ_TOP_N = 50
AI_UNIVERSE_OUTPUT_PATH = "data/ai_universe/ai_train_universe.json"

AI_TRAIN_UNIVERSE_FALLBACK = [
    "005930",
    "000660",
    "005380",
    "000270",
    "035420",
    "035720",
    "036570",
    "000810",
    "003550",
    "003670",
    "005490",
    "009150",
    "010130",
    "012330",
    "015760",
    "017670",
    "018260",
    "028260",
    "032830",
    "033780",
    "034020",
    "042660",
    "047810",
    "055550",
    "051910",
    "006400",
    "066570",
    "068270",
    "086790",
    "096770",
    "105560",
    "138040",
    "259960",
    "267260",
    "302440",
    "316140",
    "323410",
    "207940",
    "373220",
    "402340",
]


def _load_ai_universe_records() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parent / AI_UNIVERSE_OUTPUT_PATH
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            records = []
            seen = set()
            for row in rows if isinstance(rows, list) else []:
                code = str(row.get("code") or "").zfill(6)
                if not code or code in seen:
                    continue
                seen.add(code)
                records.append(
                    {
                        "code": code,
                        "name": str(row.get("name") or code),
                        "market": str(row.get("market") or "KR"),
                        "source": str(row.get("source") or "dynamic"),
                        "asset_class": "domestic-stock",
                    }
                )
            if records:
                return records
        except Exception:
            pass
    return [
        {"code": code, "name": code, "market": "KR", "source": "fallback", "asset_class": "domestic-stock"}
        for code in AI_TRAIN_UNIVERSE_FALLBACK
    ]


AI_TRAIN_UNIVERSE_RECORDS = _load_ai_universe_records()
AI_TRAIN_UNIVERSE = [row["code"] for row in AI_TRAIN_UNIVERSE_RECORDS]
WATCHLIST = [{"code": row["code"], "name": row["name"], "asset_class": "domestic-stock"} for row in AI_TRAIN_UNIVERSE_RECORDS]

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

# target1 기대수익 최소 기준. 0.0035(공격적)/0.0045(중간)/0.006(보수적) 비교 실험용.
MIN_TARGET1_PROFIT_PCT = 0.0045

# paper-watch 루프: 5분봉 마감 + offset 초 후 1회만 실행. interval은 CLI --interval-sec로 조정.
PAPER_WATCH_BAR_INTERVAL_SEC = 300
PAPER_WATCH_BAR_OFFSET_SEC = 5
PAPER_WATCH_MIN_SLEEP_SEC = 5
INTRADAY_CACHE_DIR = "data/intraday_cache"
TICK_CACHE_DIR = "data/tick_cache"
ORDERBOOK_CACHE_DIR = "data/orderbook_cache"

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

ENABLE_AI_SIGNAL = True
AI_MODEL_TYPE = "patchtst"
AI_SHADOW_MODE = True
AI_SEQUENCE_LENGTH = 96
AI_PRED_HORIZON_BARS = 6
AI_TARGET_RETURN = 0.0035
AI_STOP_RETURN = -0.0025
AI_LABEL_MODE = "drop_neutral"
AI_LOSS_TYPE = "bce_pos_weight"
AI_MIN_PROB_UP = 0.62
AI_MIN_EXPECTED_RETURN = 0.0025
AI_MAX_RISK_SCORE = 0.45
AI_PURGE_GAP_BARS = 96
AI_RETRAIN_DAYS = 120
AI_EXCLUDE_LUNCH_BARS = True
ENABLE_AI_POLICY = True
AI_POLICY_MODEL_TYPE = "patchtst_policy"
AI_POLICY_SHADOW_ONLY = True
AI_POLICY_MIN_CONFIDENCE = 0.62
AI_POLICY_MIN_EXPECTED_RETURN = 0.0025
AI_POLICY_MAX_RISK_SCORE = 0.45
ENABLE_BT_SHADOW = True
ENABLE_BT_EXECUTE = False
BT_EXECUTE_REQUIRE_PAPER = True
BT_EXECUTE_REQUIRE_AI_MODEL = True
BT_EXECUTE_REQUIRE_SHADOW_REPORT = True
BT_EXECUTE_MIN_SHADOW_DAYS = 10
BT_EXECUTE_MIN_BUY_PRECISION = 0.55
ENABLE_DYNAMIC_TRADE_UNIVERSE = True
MAX_SCAN_SYMBOLS = 300
MAX_BUY_CANDIDATES_PER_LOOP = 5
MAX_NEW_POSITIONS_PER_DAY = 3
MAX_OPEN_POSITIONS = 3
MAX_POSITION_PER_SYMBOL_KRW = 1_000_000
MAX_TOTAL_EXPOSURE_KRW = 3_000_000
MIN_AI_BUY_CONFIDENCE_FOR_CANDIDATE = 0.55
MIN_AI_BUY_CONFIDENCE_FOR_ORDER = 0.62
MIN_EXPECTED_RETURN_FOR_ORDER = 0.0025
MAX_RISK_SCORE_FOR_ORDER = 0.45
