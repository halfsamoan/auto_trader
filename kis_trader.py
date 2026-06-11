# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2)
# Dependency: clients/kis_domestic_stock_client.py
# Description: 기존 import 호환을 위한 국내주식 KIS 클라이언트 래퍼입니다.
# ================================================================================

"""Backward-compatible domestic stock KIS wrapper."""

from __future__ import annotations

from typing import Any

from clients.kis_domestic_stock_client import DomesticStockClient

KISTrader = DomesticStockClient

_DEFAULT = KISTrader()


# get_access_token은 기본 국내주식 클라이언트의 access token을 반환합니다.
def get_access_token() -> str | None:
    return _DEFAULT.get_access_token()


# get_balance는 기본 국내주식 클라이언트의 잔고 조회를 실행합니다.
def get_balance() -> dict[str, Any] | None:
    return _DEFAULT.get_balance()


# get_price는 기본 국내주식 클라이언트의 현재가 조회를 실행합니다.
def get_price(code: str) -> float | None:
    return _DEFAULT.get_price(code)


# buy_limit은 기본 국내주식 클라이언트의 지정가 매수를 실행합니다.
def buy_limit(code: str, qty: int, current_price: float) -> dict[str, Any]:
    return _DEFAULT.buy_limit(code, qty, current_price)


# sell_limit은 기본 국내주식 클라이언트의 지정가 매도를 실행합니다.
def sell_limit(code: str, qty: int, current_price: float) -> dict[str, Any]:
    return _DEFAULT.sell_limit(code, qty, current_price)


# calc_qty는 금액 기준 국내주식 주문 가능 수량을 계산합니다.
def calc_qty(code: str, amount_krw: float) -> int:
    return _DEFAULT.calc_qty(code, amount_krw)


# round_price_up은 국내주식 호가 단위 올림 가격을 반환합니다.
def round_price_up(price: float) -> int:
    return _DEFAULT.round_price_up(price)


# round_price_down은 국내주식 호가 단위 내림 가격을 반환합니다.
def round_price_down(price: float) -> int:
    return _DEFAULT.round_price_down(price)
