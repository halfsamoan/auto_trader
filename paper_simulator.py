# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: config.py, futures_contracts.py
# Description: 국내주식과 선물의 JSON 기반 paper-sim 체결과 손익을 관리합니다.
# ================================================================================

"""JSON-backed paper simulator for Paper Lab experiments."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config import DOMESTIC_STOCK_CAPITAL_KRW, FUTURES_PAPER_CAPITAL_KRW, FX_RATE_USDKRW, MAX_TRADES_PER_DAY_FUTURES
from futures_contracts import get_contract

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POSITIONS_FILE = DATA_DIR / "paper_positions.json"
TRADES_FILE = DATA_DIR / "paper_trades.json"
EQUITY_FILE = DATA_DIR / "paper_equity.json"


class PaperSimulator:
    # PaperSimulator 초기화: JSON 저장소가 없으면 기본 계좌 상태를 만듭니다.
    def __init__(
        self,
        domestic_capital_krw: float = DOMESTIC_STOCK_CAPITAL_KRW,
        futures_capital_krw: float = FUTURES_PAPER_CAPITAL_KRW,
        max_loss_limit_krw: float | None = None,
        max_daily_trades: int = MAX_TRADES_PER_DAY_FUTURES,
    ) -> None:
        self.domestic_capital_krw = float(domestic_capital_krw)
        self.futures_capital_krw = float(futures_capital_krw)
        self.max_loss_limit_krw = float(max_loss_limit_krw if max_loss_limit_krw is not None else futures_capital_krw * 0.03)
        self.max_daily_trades = int(max_daily_trades)
        self._ensure_files()

    # place_domestic_stock_order는 국내주식 paper-sim 포지션을 갱신합니다.
    def place_domestic_stock_order(self, code: str, side: str, qty: int, price: float) -> dict[str, Any]:
        return self._place_linear_position("domestic-stock", code, "KRW", side, qty, price)

    # place_futures_order는 선물 paper-sim 포지션을 갱신하고 tick 기반 KRW 손익 구조를 기록합니다.
    # place_futures_order는 symbol/side/qty/price와 margin 정보를 받아 KIS 없이 paper-sim 체결을 기록합니다.
    def place_futures_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        required_margin_krw: float = 0.0,
        margin_limit_krw: float | None = None,
        margin_check_passed: bool = True,
    ) -> dict[str, Any]:
        contract = get_contract(symbol)
        qty = min(int(qty), int(contract["max_contract_qty"]))
        if qty <= 0:
            return self._blocked("futures", symbol, "contract 수량 1 미만")
        if not self._can_trade_today():
            return self._blocked("futures", symbol, "일일 거래 횟수 한도 초과")
        if self._daily_realized_pnl_krw() <= -self.max_loss_limit_krw:
            return self._blocked("futures", symbol, "일일 최대 손실 한도 도달")
        if not margin_check_passed:
            return self._blocked("futures", symbol, "증거금 한도 초과")

        positions = self._read_json(POSITIONS_FILE, {})
        key = f"futures:{symbol}"
        existing = positions.get(key)
        now = datetime.now().isoformat(timespec="seconds")
        if side == "buy":
            positions[key] = {
                "asset_class": "futures",
                "symbol": symbol,
                "exchange": contract["exchange"],
                "currency": contract["currency"],
                "side": "long",
                "qty": qty,
                "avg_price": float(price),
                "tick_size": contract["tick_size"],
                "tick_value": contract["tick_value_usd"],
                "required_margin_krw": float(required_margin_krw),
                "opened_at": now,
                "last_update": now,
            }
            trade = self._trade("futures", symbol, side, qty, price, 0.0, contract, required_margin_krw, margin_limit_krw, margin_check_passed)
        elif side == "sell" and existing:
            simulated_pnl = self._futures_pnl(existing, float(price), qty)
            trade = self._trade("futures", symbol, side, qty, price, simulated_pnl, contract, required_margin_krw, margin_limit_krw, margin_check_passed)
            positions.pop(key, None)
        else:
            return self._blocked("futures", symbol, "청산할 paper-sim 포지션 없음")

        self._write_json(POSITIONS_FILE, positions)
        self._append_json(TRADES_FILE, trade)
        self._record_equity()
        return {"filled": True, "order_api_called": False, **trade}

    # mark_to_market_futures는 현재가 기준 선물 미실현 손익을 계산합니다.
    def mark_to_market_futures(self, symbol: str, current_price: float) -> dict[str, Any]:
        positions = self._read_json(POSITIONS_FILE, {})
        key = f"futures:{symbol}"
        position = positions.get(key)
        if not position:
            return {"symbol": symbol, "has_position": False, "simulated_pnl_krw": 0.0}
        pnl = self._futures_pnl(position, float(current_price), int(position["qty"]))
        return {"symbol": symbol, "has_position": True, "simulated_pnl_krw": pnl}

    # _place_linear_position은 주식형 paper-sim 포지션을 단순 현금 체결 방식으로 기록합니다.
    def _place_linear_position(self, asset_class: str, symbol: str, currency: str, side: str, qty: int, price: float) -> dict[str, Any]:
        if qty <= 0:
            return self._blocked(asset_class, symbol, "수량 1 미만")
        if not self._can_trade_today():
            return self._blocked(asset_class, symbol, "일일 거래 횟수 한도 초과")
        trade = self._trade(asset_class, symbol, side, qty, price, 0.0, {"currency": currency})
        self._append_json(TRADES_FILE, trade)
        return {"filled": True, "order_api_called": False, **trade}

    # _futures_pnl은 tick_size, tick_value, FX_RATE_USDKRW로 선물 손익을 KRW 환산합니다.
    def _futures_pnl(self, position: dict[str, Any], exit_price: float, qty: int) -> float:
        price_move = exit_price - float(position["avg_price"])
        if position.get("side") == "short":
            price_move *= -1
        ticks = price_move / float(position["tick_size"])
        pnl_usd = ticks * float(position["tick_value"]) * int(qty)
        return pnl_usd * FX_RATE_USDKRW

    # _can_trade_today는 paper-sim 일일 거래 횟수 한도를 확인합니다.
    def _can_trade_today(self) -> bool:
        today = date.today().isoformat()
        trades = self._read_json(TRADES_FILE, [])
        count = sum(1 for trade in trades if str(trade.get("timestamp", "")).startswith(today))
        return count < self.max_daily_trades

    # _daily_realized_pnl_krw는 오늘 KRW 실현손익 합계를 반환합니다.
    def _daily_realized_pnl_krw(self) -> float:
        today = date.today().isoformat()
        trades = self._read_json(TRADES_FILE, [])
        return sum(float(t.get("simulated_pnl_krw", 0.0)) for t in trades if t.get("asset_class") == "futures" and str(t.get("timestamp", "")).startswith(today))

    # _record_equity는 현재 paper-sim 기준 계좌 스냅샷을 저장합니다.
    def _record_equity(self) -> None:
        trades = self._read_json(TRADES_FILE, [])
        realized_futures_krw = sum(float(t.get("simulated_pnl_krw", 0.0)) for t in trades if t.get("asset_class") == "futures")
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "domestic_equity_krw": self.domestic_capital_krw,
            "futures_equity_krw": self.futures_capital_krw + realized_futures_krw,
        }
        self._append_json(EQUITY_FILE, snapshot)

    # _trade는 표준 paper-sim 거래 기록을 만듭니다.
    # _trade는 체결 기록과 증거금 검증 값을 표준 필드로 저장합니다.
    def _trade(
        self,
        asset_class: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        simulated_pnl: float,
        meta: dict[str, Any],
        required_margin_krw: float = 0.0,
        margin_limit_krw: float | None = None,
        margin_check_passed: bool = True,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "asset_class": asset_class,
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "price": float(price),
            "currency": meta.get("currency", "USD"),
            "tick_size": meta.get("tick_size"),
            "tick_value": meta.get("tick_value_usd") or meta.get("tick_value"),
            "margin_per_contract_usd": meta.get("margin_per_contract_usd"),
            "required_margin_krw": float(required_margin_krw),
            "margin_limit_krw": float(margin_limit_krw if margin_limit_krw is not None else 0.0),
            "margin_check_passed": bool(margin_check_passed),
            "simulated_pnl_krw": float(simulated_pnl),
            "order_api_called": False,
        }

    # _blocked는 paper-sim 차단 결과를 표준화합니다.
    def _blocked(self, asset_class: str, symbol: str, reason: str) -> dict[str, Any]:
        return {"filled": False, "asset_class": asset_class, "symbol": symbol, "reason": reason, "order_api_called": False}

    # _ensure_files는 paper-sim JSON 파일을 초기화합니다.
    def _ensure_files(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        defaults = {
            POSITIONS_FILE: {},
            TRADES_FILE: [],
            EQUITY_FILE: [
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "domestic_equity_krw": self.domestic_capital_krw,
                    "futures_equity_krw": self.futures_capital_krw,
                }
            ],
        }
        for path, default in defaults.items():
            if not path.exists():
                self._write_json(path, default)

    # _read_json은 JSON 파일을 읽고 실패하면 기본값을 반환합니다.
    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    # _write_json은 JSON 파일을 저장합니다.
    def _write_json(self, path: Path, data: Any) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # _append_json은 JSON list 파일에 record를 추가합니다.
    def _append_json(self, path: Path, record: dict[str, Any]) -> None:
        data = self._read_json(path, [])
        if not isinstance(data, list):
            data = []
        data.append(record)
        self._write_json(path, data)
