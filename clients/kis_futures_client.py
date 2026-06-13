# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: clients/kis_base_client.py, config.py, futures_contracts.py
# Description: KIS 해외선물 모의투자 조회/지정가 주문 게이트를 관리합니다.
# ================================================================================

"""KIS overseas futures paper-trading client with strict safety gates."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests

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
    # __init__은 해외선물 전용 계좌 환경변수를 별도로 보관합니다.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.futures_account_no = os.getenv("KIS_FUTURES_ACCOUNT_NO")
        self.futures_account_product_code = os.getenv("KIS_FUTURES_ACCOUNT_PRODUCT_CODE")
        self.futures_currency_code = os.getenv("KIS_FUTURES_CRCY_CD", "TUS")

    # validate_futures_env는 해외선물 조회 실검증에 필요한 전용 계좌값을 확인합니다.
    def validate_futures_env(self) -> tuple[bool, str | None]:
        ok, msg = self.validate_env()
        if not ok:
            return ok, msg
        if self.mode != "paper":
            return False, "KIS_MODE=paper가 아니므로 해외선물 모의투자 조회를 중단합니다."
        if not self.futures_account_no:
            return False, "KIS_FUTURES_ACCOUNT_NO가 없습니다."
        if not self.futures_account_product_code:
            return False, "KIS_FUTURES_ACCOUNT_PRODUCT_CODE가 없습니다. 계좌별 상품코드를 .env에 설정하세요."
        return True, None

    # masked_futures_account는 계좌번호를 마스킹해 출력용 문자열을 만듭니다.
    def masked_futures_account(self) -> str:
        value = self.futures_account_no or ""
        if len(value) <= 4:
            return "****"
        return f"{value[:2]}{'*' * max(len(value) - 4, 0)}{value[-2:]}"

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

    # live_probe_price는 주문 없이 해외선물 현재가 조회 API만 실호출합니다.
    def live_probe_price(self, symbol: str, token: str) -> dict[str, Any]:
        product_code = self._futures_product_code(symbol)
        if not product_code:
            return self._probe_blocked(
                "futures_price",
                f"{symbol} KIS 상품번호(SRS_CD)가 없어 현재가 조회 실호출 차단",
            )
        return self._get_probe(
            "futures_price",
            KIS_FUTURES_PRICE_API_URL,
            KIS_FUTURES_PRICE_TR_ID,
            {"SRS_CD": product_code},
            token,
        )

    # live_probe_balance는 주문 없이 해외선물 예수금/증거금 조회 API만 실호출합니다.
    def live_probe_balance(self, token: str) -> dict[str, Any]:
        ok, msg = self.validate_futures_env()
        if not ok:
            return self._probe_blocked("futures_balance", msg or "해외선물 계좌 환경변수 오류")
        return self._get_probe(
            "futures_balance",
            KIS_FUTURES_DEPOSIT_API_URL,
            KIS_FUTURES_DEPOSIT_TR_ID,
            {
                "CANO": self.futures_account_no,
                "ACNT_PRDT_CD": self.futures_account_product_code,
                "CRCY_CD": self.futures_currency_code,
                "INQR_DT": datetime.now().strftime("%Y%m%d"),
            },
            token,
        )

    # live_probe_orderable은 주문 없이 해외선물 주문가능조회 API만 실호출합니다.
    def live_probe_orderable(self, symbol: str, token: str) -> dict[str, Any]:
        ok, msg = self.validate_futures_env()
        if not ok:
            return self._probe_blocked("futures_orderable", msg or "해외선물 계좌 환경변수 오류")
        product_code = self._futures_product_code(symbol)
        if not product_code:
            return self._probe_blocked(
                "futures_orderable",
                f"{symbol} KIS 상품번호(OVRS_FUTR_FX_PDNO)가 없어 주문가능조회 실호출 차단",
            )
        return self._get_probe(
            "futures_orderable",
            KIS_FUTURES_PSAMOUNT_API_URL,
            KIS_FUTURES_PSAMOUNT_TR_ID,
            {
                "CANO": self.futures_account_no,
                "ACNT_PRDT_CD": self.futures_account_product_code,
                "OVRS_FUTR_FX_PDNO": product_code,
                "SLL_BUY_DVSN_CD": "02",
                "FM_ORD_PRIC": "0",
                "ECIS_RSVN_ORD_YN": "N",
            },
            token,
        )

    # live_probe_positions는 주문 없이 해외선물 미결제 포지션 조회 API만 실호출합니다.
    def live_probe_positions(self, token: str) -> dict[str, Any]:
        ok, msg = self.validate_futures_env()
        if not ok:
            return self._probe_blocked("futures_open_positions", msg or "해외선물 계좌 환경변수 오류")
        return self._get_probe(
            "futures_open_positions",
            KIS_FUTURES_UNPD_API_URL,
            KIS_FUTURES_UNPD_TR_ID,
            {"CANO": self.futures_account_no, "ACNT_PRDT_CD": self.futures_account_product_code},
            token,
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

    # _get_probe는 KIS 조회 API를 호출하되 민감한 응답값 대신 필드명과 성공 여부만 반환합니다.
    def _get_probe(self, capability: str, api_url: str, tr_id: str, params: dict[str, Any], token: str) -> dict[str, Any]:
        safe_params = {key: ("***MASKED***" if key == "CANO" else value) for key, value in params.items()}
        self.rate_limit()
        try:
            headers = self.headers(token=token, tr_id=tr_id)
            headers["custtype"] = "P"
            headers["tr_cont"] = ""
            response = requests.get(
                f"{self.base_url}{api_url}",
                headers=headers,
                params=params,
                timeout=10,
            )
            try:
                data = response.json() if response.content else {}
            except ValueError:
                return {
                    "implemented": True,
                    "capability": capability,
                    "success": False,
                    "status_code": response.status_code,
                    "api_url": api_url,
                    "tr_id": tr_id,
                    "params": safe_params,
                    "field_names": [],
                    "output_field_names": [],
                    "content_type": response.headers.get("content-type"),
                    "error": "KIS 응답이 JSON 형식이 아님",
                    "order_api_called": False,
                }
        except Exception as exc:
            return {
                "implemented": True,
                "capability": capability,
                "success": False,
                "status_code": None,
                "api_url": api_url,
                "tr_id": tr_id,
                "params": safe_params,
                "field_names": [],
                "output_field_names": [],
                "error": str(exc),
                "order_api_called": False,
            }
        return {
            "implemented": True,
            "capability": capability,
            "success": response.ok and str(data.get("rt_cd", "0")) == "0",
            "status_code": response.status_code,
            "api_url": api_url,
            "tr_id": tr_id,
            "params": safe_params,
            "field_names": sorted(data.keys()) if isinstance(data, dict) else [],
            "output_field_names": self._output_field_names(data),
            "rt_cd": data.get("rt_cd") if isinstance(data, dict) else None,
            "msg_cd": data.get("msg_cd") if isinstance(data, dict) else None,
            "order_api_called": False,
        }

    # _output_field_names는 응답값을 노출하지 않고 output 계열 필드명만 추출합니다.
    def _output_field_names(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        names: set[str] = set()
        for key, value in data.items():
            if not str(key).startswith("output"):
                continue
            if isinstance(value, dict):
                names.update(str(item) for item in value.keys())
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                names.update(str(item) for item in value[0].keys())
        return sorted(names)

    # _futures_product_code는 env 우선으로 KIS 해외선물 상품번호를 찾습니다.
    def _futures_product_code(self, symbol: str) -> str | None:
        key = f"KIS_FUTURES_{symbol.upper()}_PRODUCT_CODE"
        env_value = os.getenv(key)
        if env_value:
            return env_value
        return get_contract(symbol).get("kis_product_code")

    # _probe_blocked는 조회 실검증이 불가능한 사유를 표준 결과로 반환합니다.
    def _probe_blocked(self, capability: str, reason: str) -> dict[str, Any]:
        return {
            "implemented": True,
            "capability": capability,
            "success": False,
            "blocked": True,
            "reason": reason,
            "field_names": [],
            "output_field_names": [],
            "order_api_called": False,
        }

    # _todo는 아직 API 호출하지 않는 조회 skeleton 결과를 만듭니다.
    def _todo(self, capability: str, reason: str) -> dict[str, Any]:
        return {"implemented": False, "capability": capability, "reason": reason, "order_api_called": False}
