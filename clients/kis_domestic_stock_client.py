# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: clients/kis_base_client.py
# Description: KIS 국내주식 조회와 지정가 주문 안전 게이트를 제공합니다.
# ================================================================================

"""KIS domestic stock client."""

from __future__ import annotations

import math
from typing import Any

import requests

from config import ENABLE_DOMESTIC_STOCK_PAPER_ORDER, ENABLE_REAL_ORDER

from .kis_base_client import KISBaseClient


class DomesticStockClient(KISBaseClient):
    # get_balance는 국내주식 잔고와 주문가능금액을 조회합니다.
    def get_balance(self) -> dict[str, Any] | None:
        token = self.get_access_token()
        if not token:
            return None
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        tr_id = "VTTC8434R" if self.mode == "paper" else "TTTC8434R"
        self.rate_limit()
        response = requests.get(url, headers=self.headers(token, tr_id), params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    # get_price는 국내주식 현재가를 조회합니다.
    def get_price(self, code: str) -> float | None:
        token = self.get_access_token()
        if not token:
            return None
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
        self.rate_limit()
        response = requests.get(url, headers=self.headers(token, "FHKST01010100"), params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = (data.get("output") or {}).get("stck_prpr")
        return float(price) if price else None

    # calc_qty는 금액 기준 국내주식 수량을 계산합니다.
    def calc_qty(self, code: str, amount_krw: float) -> int:
        price = self.get_price(code)
        if not price or price <= 0:
            return 0
        return int(amount_krw // price)

    # buy_limit은 국내주식 지정가 매수 주문을 안전 게이트 통과 시에만 호출합니다.
    def buy_limit(self, code: str, qty: int, current_price: float) -> dict[str, Any]:
        order_price = self.round_price_up(current_price * 1.005)
        allowed, reason = self._can_call_domestic_order()
        if not allowed:
            return self.blocked_order("buy", code, qty, order_price, reason)
        if not self.can_order():
            return self.blocked_order("buy", code, qty, order_price, "국내주식 주문 안전 게이트로 차단")
        if self.mode == "real" and not self.live:
            raise RuntimeError("KIS_MODE=real에서는 --live 없이는 주문할 수 없습니다.")
        return self._place_limit_order("buy", code, qty, order_price)

    # sell_limit은 국내주식 지정가 매도 주문을 안전 게이트 통과 시에만 호출합니다.
    def sell_limit(self, code: str, qty: int, current_price: float) -> dict[str, Any]:
        order_price = self.round_price_down(current_price * 0.995)
        allowed, reason = self._can_call_domestic_order()
        if not allowed:
            return self.blocked_order("sell", code, qty, order_price, reason)
        if not self.can_order():
            return self.blocked_order("sell", code, qty, order_price, "국내주식 주문 안전 게이트로 차단")
        if self.mode == "real" and not self.live:
            raise RuntimeError("KIS_MODE=real에서는 --live 없이는 주문할 수 없습니다.")
        return self._place_limit_order("sell", code, qty, order_price)

    # round_price_up은 국내주식 호가 단위로 가격을 올림 처리합니다.
    def round_price_up(self, price: float) -> int:
        tick = self._tick(price)
        return int(math.ceil(price / tick) * tick)

    # round_price_down은 국내주식 호가 단위로 가격을 내림 처리합니다.
    def round_price_down(self, price: float) -> int:
        tick = self._tick(price)
        return int(math.floor(price / tick) * tick)

    # _place_limit_order는 검증된 국내주식 지정가 주문 endpoint만 호출합니다.
    def _place_limit_order(self, side: str, code: str, qty: int, order_price: int) -> dict[str, Any]:
        if qty <= 0:
            raise ValueError("주문 수량은 1주 이상이어야 합니다.")
        token = self.get_access_token()
        if not token:
            raise RuntimeError("KIS access token이 없어 주문할 수 없습니다.")
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = ("VTTC0802U" if side == "buy" else "VTTC0801U") if self.mode == "paper" else ("TTTC0802U" if side == "buy" else "TTTC0801U")
        payload = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_product_code,
            "PDNO": code,
            "ORD_DVSN": "00",
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(int(order_price)),
        }
        self.rate_limit()
        response = requests.post(url, headers=self.headers(token, tr_id), json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    # _can_call_domestic_order는 config와 CLI 안전 플래그를 모두 확인합니다.
    def _can_call_domestic_order(self) -> tuple[bool, str]:
        if self.mode == "real" and (not ENABLE_REAL_ORDER or not self.live):
            return False, "실전 주문은 ENABLE_REAL_ORDER=True와 --live 없이는 차단"
        if self.mode == "paper" and not ENABLE_DOMESTIC_STOCK_PAPER_ORDER:
            return False, "ENABLE_DOMESTIC_STOCK_PAPER_ORDER=False로 국내주식 paper 주문 차단"
        if self.mode == "paper" and not self.allow_paper_order:
            return False, "--allow-paper-order가 없어 국내주식 paper 주문 차단"
        return True, ""

    # _tick은 국내주식 가격대별 호가 단위를 반환합니다.
    @staticmethod
    def _tick(price: float) -> int:
        if price < 2000:
            return 1
        if price < 5000:
            return 5
        if price < 20000:
            return 10
        if price < 50000:
            return 50
        if price < 200000:
            return 100
        if price < 500000:
            return 500
        return 1000
