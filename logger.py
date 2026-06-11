# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: 시그널과 거래 기록을 JSON 로그 파일로 저장합니다.
# ================================================================================

"""JSON list log writer for trading and signal records."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRADING_LOG = DATA_DIR / "trading_log.json"
SIGNAL_LOG = DATA_DIR / "signal_log.json"

logger = logging.getLogger("auto_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


# _ensure_json_list는 로그 파일이 없거나 깨졌을 때 빈 리스트로 복구합니다.
def _ensure_json_list(path: Path) -> list[Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# _append_json은 V3.2 공통 로그 필드를 보강한 뒤 JSON 리스트에 추가합니다.
def _append_json(path: Path, record: dict[str, Any]) -> None:
    data = _ensure_json_list(path)
    record.setdefault("logged_at", datetime.now().isoformat(timespec="seconds"))
    record.setdefault("asset_class", None)
    record.setdefault("symbol_or_code", record.get("symbol") or record.get("code"))
    record.setdefault("market", None)
    record.setdefault("exchange", None)
    record.setdefault("currency", None)
    record.setdefault("regime", None)
    record.setdefault("qty", None)
    record.setdefault("contract_qty", None)
    record.setdefault("qty_by_risk", None)
    record.setdefault("qty_by_amount", None)
    record.setdefault("highest_price", None)
    record.setdefault("mode", None)
    record.setdefault("paper_sim", False)
    record.setdefault("order_api_called", False)
    record.setdefault("is_order_allowed", False)
    record.setdefault("order_block_reason", None)
    record.setdefault("capital_krw", None)
    record.setdefault("tick_size", None)
    record.setdefault("tick_value_usd", None)
    record.setdefault("margin_per_contract_usd", None)
    record.setdefault("required_margin_krw", None)
    record.setdefault("margin_check_passed", None)
    record.setdefault("simulated_pnl_krw", None)
    record.setdefault("regime_source", None)
    data.append(record)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


# _json_default는 numpy/pandas 타입을 JSON 직렬화 가능한 기본 타입으로 변환합니다.
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# log_trading은 trading_log.json에 거래 기록을 추가합니다.
def log_trading(record: dict[str, Any]) -> None:
    _append_json(TRADING_LOG, record)
    logger.info("거래 기록 저장: %s", record)


# log_signal은 signal_log.json에 신호 기록을 추가합니다.
def log_signal(record: dict[str, Any]) -> None:
    _append_json(SIGNAL_LOG, record)
    logger.info("시그널 기록 저장: %s", record)
