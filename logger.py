# ==============================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0 Intraday)
# Dependency: logging, json, os, datetime
# Description: 콘솔 및 JSON 로그 기록
# ==============================================================================

"""logger.py
콘솔에 한글 로그를 출력하고, data/trading_log.json 및 data/signal_log.json 에 JSON 형태로 기록합니다.
"""
import logging
import json
import os
from datetime import datetime
from typing import Any, Dict

# 로그 디렉터리 보장 (project root 기준)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(LOG_DIR, exist_ok=True)

# 콘솔 로거 설정
logger = logging.getLogger("auto_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def _append_json(file_path: str, record: Dict[str, Any]):
    """JSON 리스트 파일에 레코드를 추가합니다.
    파일이 없으면 [] 로 초기화하고, 기존 리스트에 레코드를 append 합니다.
    """
    if not os.path.exists(file_path):
        data = []
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    data.append(record)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log_trading(record: Dict[str, Any]):
    """거래 기록을 JSON 파일에 저장하고 콘솔에 출력합니다.
    record 예시:
        {"ticker": "005930", "action": "buy", "price": 72000, "quantity": 10, "timestamp": "2026-06-11T10:00:00"}
    """
    file_path = os.path.join(LOG_DIR, "trading_log.json")
    _append_json(file_path, record)
    logger.info(f"거래 기록: {record}")

def log_signal(record: Dict[str, Any]):
    """시그널 로그를 JSON 파일에 저장하고 콘솔에 출력합니다.
    record 예시:
        {"ticker": "005930", "intraday_score": 78.5, "gaussian_score": 62.0, "final_score": 71.2, "signal": "buy", "timestamp": "2026-06-11T10:00:00"}
    """
    file_path = os.path.join(LOG_DIR, "signal_log.json")
    _append_json(file_path, record)
    logger.info(f"시그널 기록: {record}")
