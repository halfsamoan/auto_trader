# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: 보유 포지션 저장과 ATR 트레일링/기술적 청산 판단을 처리합니다.
# ================================================================================

"""Position persistence and exit decision helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.technical import atr, ensure_ohlcv, rsi, sma, vwap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POSITIONS_FILE = DATA_DIR / "positions.json"


# _ensure_file은 positions.json이 없을 때 빈 포지션 파일을 생성합니다.
def _ensure_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not POSITIONS_FILE.exists():
        POSITIONS_FILE.write_text("{}", encoding="utf-8")


# load_positions는 저장된 포지션 딕셔너리를 읽습니다.
def load_positions() -> dict[str, Any]:
    _ensure_file()
    try:
        data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# save_positions는 포지션 딕셔너리를 JSON 파일로 저장합니다.
def save_positions(positions: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POSITIONS_FILE.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")


# update_position은 신규/갱신 포지션을 표준 schema로 저장합니다.
def update_position(
    code: str,
    qty: int,
    avg_price: float,
    stop_loss: float,
    target1: float,
    target2: float,
    half_sold: bool = False,
    buy_datetime: str | None = None,
    highest_price: float | None = None,
    technical_exit_count: int = 0,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    positions = load_positions()
    positions[code] = {
        "qty": int(qty),
        "avg_price": float(avg_price),
        "stop_loss": float(stop_loss),
        "target1": float(target1),
        "target2": float(target2),
        "half_sold": bool(half_sold),
        "highest_price": float(highest_price if highest_price is not None else avg_price),
        "highest_price_since_entry": float(highest_price if highest_price is not None else avg_price),
        "technical_exit_count": int(technical_exit_count),
        "buy_datetime": buy_datetime or now,
        "last_update": now,
    }
    save_positions(positions)


# remove_position은 청산이 끝난 종목을 positions.json에서 제거합니다.
def remove_position(code: str) -> None:
    positions = load_positions()
    if code in positions:
        del positions[code]
        save_positions(positions)


# _latest_atr은 5분봉 ATR을 계산하고 부족하면 가격의 0.8%를 fallback으로 사용합니다.
def _latest_atr(data: pd.DataFrame, price: float) -> float:
    atr_series = atr(data, 14)
    clean = atr_series.dropna()
    latest = float(clean.iloc[-1]) if not clean.empty else price * 0.008
    return latest if latest > 0 else price * 0.008


# _technical_exit_signal은 MA20 이탈, RSI 약세, VWAP 이탈이 동시에 나온 봉인지 판단합니다.
def _technical_exit_signal(data: pd.DataFrame, price: float) -> bool:
    if len(data) < 20:
        return False
    ma20 = sma(data["Close"], 20).iloc[-1]
    current_rsi = rsi(data["Close"], 14).iloc[-1]
    current_vwap = vwap(data).iloc[-1]
    close_below_ma20 = float(data["Close"].iloc[-1]) < float(ma20)
    return bool(close_below_ma20 and float(current_rsi) < 45 and price < float(current_vwap))


# evaluate_exit은 현재 가격과 5분봉으로 청산 액션, 사유, 갱신할 포지션 상태를 반환합니다.
def evaluate_exit(position: dict[str, Any], price: float, intraday_df: pd.DataFrame, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now()
    avg_price = float(position["avg_price"])
    stop_loss = float(position["stop_loss"])
    target1 = float(position["target1"])
    target2 = float(position["target2"])
    half_sold = bool(position.get("half_sold", False))
    highest_price = max(float(position.get("highest_price", position.get("highest_price_since_entry", avg_price))), float(price))
    technical_count = int(position.get("technical_exit_count", 0))
    position["highest_price"] = highest_price
    position["highest_price_since_entry"] = highest_price
    buy_datetime = datetime.strptime(position["buy_datetime"], "%Y-%m-%d %H:%M:%S")
    holding_minutes = (now - buy_datetime).total_seconds() / 60
    profit_pct = (price / avg_price - 1) * 100 if avg_price else 0.0

    data = ensure_ohlcv(intraday_df)
    latest_atr = _latest_atr(data, price) if not data.empty else price * 0.008

    if half_sold:
        trailing_stop = highest_price - 1.0 * latest_atr
        stop_loss = max(stop_loss, avg_price, trailing_stop)
        position["stop_loss"] = float(stop_loss)
        # 1차 익절 후에만 본전 이상으로 올려 수익 포지션의 되돌림을 제한합니다.
        if price <= trailing_stop and profit_pct > 1.0:
            return {
                "action": "sell_all",
                "reason": "atr_trailing_stop",
                "new_stop_loss": stop_loss,
                "highest_price": highest_price,
                "technical_exit_count": technical_count,
                "updated_position": position,
            }

    if price <= stop_loss:
        return {"action": "sell_all", "reason": "stop_loss"}
    if price >= target1 and not half_sold:
        position["half_sold"] = True
        position["stop_loss"] = float(avg_price)
        return {
            "action": "sell_half",
            "reason": "target1",
            "new_stop_loss": avg_price,
            "highest_price": highest_price,
            "technical_exit_count": technical_count,
            "updated_position": position,
        }
    if price >= target2:
        return {"action": "sell_all", "reason": "target2"}
    if holding_minutes >= 240:
        return {"action": "sell_all", "reason": "time_240m"}
    if holding_minutes >= 60 and profit_pct < 0.3:
        return {"action": "sell_all", "reason": "weak_after_60m"}

    if _technical_exit_signal(data, price):
        technical_count += 1
    else:
        technical_count = 0
    position["technical_exit_count"] = technical_count
    if technical_count >= 2:
        return {
            "action": "sell_all",
            "reason": "technical_breakdown_2bars",
            "highest_price": highest_price,
            "technical_exit_count": technical_count,
            "updated_position": position,
        }

    return {
        "action": "hold",
        "reason": None,
        "highest_price": highest_price,
        "technical_exit_count": technical_count,
        "updated_position": position,
    }
