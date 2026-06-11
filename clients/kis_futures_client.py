# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: clients/kis_base_client.py, config.py, futures_contracts.py
# Description: KIS 해외선물 모의투자 조회/지정가 주문 게이트를 관리합니다.
# ================================================================================

"""KIS overseas futures paper-trading client with strict safety gates."""

from __future__ import annotations

from typing import Any

from config import ENABLE_FUTURES_KIS_PAPER_ORDER, ENABLE_REAL_ORDER, FUTURES_KIS_ENDPOINTS_VERIFIED
from futures_contracts import get_contract

from .kis_base_client import KISBaseClient


KIS_FUTURES_PRICE_API_URL = "/uapi/overseas-futureoption/v1/quotations/inquire-price"
KIS_FUTURES_PRICE_TR_ID = "HHDFC55010000"
KIS_FUTURES_DEPOSIT_API_URL = "/uapi/overseas-futureoption/v1/trading/inquire-deposit"
KIS_FUTURES_DEPOSIT_TR_ID = "OTFM1411R"
KIS_FUTURES_PSAMOUNT_API_URL = "/uapi/overseas-futureoption/v1/trading/inquire-psamount"
KIS_FUTURES_PSAMOUNT_TR_ID = "OTFM3304R"
KIS_FUTURES_UNPD_API_URL = "/uapi/overseas-futureoption/v1/trading/inquire-unpd"
KIS_FUTURES_UNPD_TR_ID = "OTFM1412R"
KIS_FUTURES_ORDER_API_URL = "/uapi/overseas-futureoption/v1/trading/order"
KIS_FUTURES_ORDER_TR_ID = "OTFM3001U"


class FuturesClient(KISBaseClient):
    # get_futures_price는 KIS 선물 시세조회 검증 전까지 TODO 결과를 반환합니다.
    def get_futures_price(self, symbol: str) -> dict[str, Any]:
        # TODO: endpoint/TR ID는 KIS 샘플 기준 확인됨. MNQ/MES의 실제 SRS_CD 월물코드 매핑 검증 필요.
        contract = get_contract(symbol)
        return self._todo(
            "futures_price",
            f"{contract['exchange']}:{symbol} KIS 선물 시세조회 TODO "
            f"({KIS_FUTURES_PRICE_API_URL}, {KIS_FUTURES_PRICE_TR_ID}, SRS_CD 매핑 필요)",
        )

    # get_futures_balance는 KIS 선물 예수금/잔고조회 검증 전까지 TODO 결과를 반환합니다.
    def get_futures_balance(self) -> dict[str, Any]:
        # TODO: endpoint/TR ID는 KIS 샘플 기준 확인됨. 모의 해외선물 계좌 상품코드와 응답 필드 검증 필요.
        return self._todo(
            "futures_balance",
            f"KIS 선물 예수금/잔고조회 TODO ({KIS_FUTURES_DEPOSIT_API_URL}, {KIS_FUTURES_DEPOSIT_TR_ID})",
        )

    # get_open_positions는 KIS 선물 미결제 포지션조회 검증 전까지 TODO 결과를 반환합니다.
    def get_open_positions(self) -> dict[str, Any]:
        # TODO: endpoint/TR ID는 KIS 샘플 기준 확인됨. long/short/평단 응답 필드와 모의계좌 동작 검증 필요.
        return self._todo(
            "futures_open_positions",
            f"KIS 선물 미결제 포지션조회 TODO ({KIS_FUTURES_UNPD_API_URL}, {KIS_FUTURES_UNPD_TR_ID})",
        )

    # get_margin_info는 KIS 선물 증거금조회 검증 전까지 TODO 결과를 반환합니다.
    def get_margin_info(self, symbol: str) -> dict[str, Any]:
        # TODO: 주문가능조회 endpoint/TR ID는 KIS 샘플 기준 확인됨. 증거금/가능수량 필드 해석 검증 필요.
        return self._todo(
            "futures_margin",
            f"{symbol} KIS 선물 주문가능/증거금조회 TODO ({KIS_FUTURES_PSAMOUNT_API_URL}, {KIS_FUTURES_PSAMOUNT_TR_ID})",
        )

    # buy_futures_limit은 모든 조건이 충족될 때만 KIS 선물 paper 주문을 호출할 수 있게 구조를 열어둡니다.
    def buy_futures_limit(self, symbol: str, qty: int, limit_price: float) -> dict[str, Any]:
        allowed, reason = self._can_call_kis_futures_order(symbol)
        if not allowed:
            return self.blocked_order("buy", symbol, qty, limit_price, reason)
        return self._place_futures_limit_order("buy", symbol, qty, limit_price)

    # sell_futures_limit은 모든 조건이 충족될 때만 KIS 선물 paper 주문을 호출할 수 있게 구조를 열어둡니다.
    def sell_futures_limit(self, symbol: str, qty: int, limit_price: float) -> dict[str, Any]:
        allowed, reason = self._can_call_kis_futures_order(symbol)
        if not allowed:
            return self.blocked_order("sell", symbol, qty, limit_price, reason)
        return self._place_futures_limit_order("sell", symbol, qty, limit_price)

    # close_futures_position은 모든 조건이 충족될 때만 KIS 선물 paper 청산 주문을 호출할 수 있게 구조를 열어둡니다.
    def close_futures_position(self, symbol: str, qty: int, limit_price: float) -> dict[str, Any]:
        allowed, reason = self._can_call_kis_futures_order(symbol)
        if not allowed:
            return self.blocked_order("close", symbol, qty, limit_price, reason)
        return self._place_futures_limit_order("close", symbol, qty, limit_price)

    # _can_call_kis_futures_order는 config, contract metadata, CLI 안전 플래그를 모두 확인합니다.
    def _can_call_kis_futures_order(self, symbol: str) -> tuple[bool, str]:
        contract = get_contract(symbol)
        if self.mode == "real" and (not ENABLE_REAL_ORDER or not self.live):
            return False, "실전 선물 주문은 ENABLE_REAL_ORDER=True와 --live 없이는 차단"
        if self.mode != "paper":
            return False, "KIS 선물 주문은 paper 모드에서만 준비됨"
        if not self.allow_paper_order:
            return False, "--allow-paper-order가 없어 KIS 선물 paper 주문 차단"
        if not ENABLE_FUTURES_KIS_PAPER_ORDER:
            return False, "ENABLE_FUTURES_KIS_PAPER_ORDER=False로 KIS 선물 paper 주문 차단"
        if not contract.get("is_kis_paper_order_enabled"):
            return False, f"{symbol} contract metadata에서 KIS paper 주문 비활성"
        if not FUTURES_KIS_ENDPOINTS_VERIFIED:
            return False, "KIS 해외선물 endpoint/TR ID/body field 미검증으로 주문 호출 차단"
        return False, "KIS 해외선물 주문 호출부 TODO 상태로 주문 차단"

    # _place_futures_limit_order는 선물 주문 endpoint 검증 전까지 절대 호출되지 않도록 남겨둡니다.
    def _place_futures_limit_order(self, side: str, symbol: str, qty: int, limit_price: float) -> dict[str, Any]:
        # TODO: endpoint/TR ID는 KIS 샘플 기준 확인됨. MNQ/MES 상품번호와 paper 계좌 실호출 검증 후 구현.
        # 지정가만 허용: PRIC_DVSN_CD=1, SLL_BUY_DVSN_CD=01/02, CCLD_CNDT_CD=6 기준을 paper 계좌에서 검증해야 합니다.
        return self.blocked_order(side, symbol, qty, limit_price, "KIS 선물 주문 API 미구현")

    # _todo는 아직 API 호출하지 않는 조회 skeleton 결과를 만듭니다.
    def _todo(self, capability: str, reason: str) -> dict[str, Any]:
        return {"implemented": False, "capability": capability, "reason": reason, "order_api_called": False}
