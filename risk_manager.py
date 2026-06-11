# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: config.py, core/technical.py
# Description: 진입 리스크 필터와 변동성 기반 포지션 사이징을 계산합니다.
# ================================================================================

"""Conservative entry risk checks and risk-based position sizing."""

from __future__ import annotations

import pandas as pd

from config import AMOUNT_PER_TRADE, EXCLUDED_CODES, RISK_PER_TRADE_PCT
from core.technical import atr, ensure_ohlcv


class RiskManager:
    # RiskManager 초기화: 진입 필터와 기본 사이징 정책을 설정합니다.
    def __init__(
        self,
        max_atr_pct: float = 0.025,
        max_recent_jump_pct: float = 7.0,
        excluded_codes: list[str] | None = None,
        amount_per_trade: float = AMOUNT_PER_TRADE,
        risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
        use_risk_based_sizing: bool = True,
    ) -> None:
        self.max_atr_pct = max_atr_pct
        self.max_recent_jump_pct = max_recent_jump_pct
        self.excluded_codes = excluded_codes or EXCLUDED_CODES
        self.amount_per_trade = float(amount_per_trade)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.use_risk_based_sizing = use_risk_based_sizing

    # entry_check는 종목과 최근 봉 데이터 기준으로 신규 진입 가능 여부와 사유를 반환합니다.
    def entry_check(self, ticker: str, df: pd.DataFrame | None) -> tuple[bool, str | None]:
        if ticker in self.excluded_codes:
            return False, "백테스트 성능 부적합으로 제외"
        if df is None:
            return True, None
        data = ensure_ohlcv(df)
        if len(data) < 15:
            return True, None
        price = float(data["Close"].iloc[-1])
        atr_series = atr(data, 14)
        latest_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else price * 0.01
        atr_pct = latest_atr / price if price else 1.0
        recent_jump = (price / data["Close"].iloc[-4] - 1) * 100 if len(data) >= 4 and data["Close"].iloc[-4] else 0.0
        if atr_pct > self.max_atr_pct:
            return False, "ATR 변동성 한도 초과"
        if recent_jump > self.max_recent_jump_pct:
            return False, "최근 급등 한도 초과"
        return True, None

    # calculate_position_size는 손절폭 기준 리스크 수량과 금액 상한 수량 중 작은 값을 반환합니다.
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float | None = None,
        available_cash: float | None = None,
        fallback_amount: float | None = None,
    ) -> dict[str, float | int | str | bool | None]:
        reference_value = float(account_value or available_cash or 0.0)
        cash_limit = float(fallback_amount if fallback_amount is not None else self.amount_per_trade)
        if available_cash is not None and available_cash > 0:
            cash_limit = min(cash_limit, float(available_cash))

        if entry_price <= 0:
            return self._sizing_result(0, 0, 0, "entry_price가 0 이하", False)

        qty_by_amount = int(cash_limit // entry_price) if cash_limit > 0 else 0
        stop_distance = float(entry_price - stop_loss)

        if self.use_risk_based_sizing and reference_value > 0 and stop_distance > 0:
            risk_amount = reference_value * self.risk_per_trade_pct
            qty_by_risk = int(risk_amount // stop_distance)
            qty = min(qty_by_risk, qty_by_amount)
            reason = None if qty >= 1 else "리스크 기반 계산 수량 1주 미만"
            return self._sizing_result(qty, qty_by_risk, qty_by_amount, reason, True)

        # 손절폭이 비정상이거나 계좌 기준값이 없을 때만 기존 fixed amount 방식을 fallback으로 유지합니다.
        qty = qty_by_amount
        reason = None if qty >= 1 else "fixed amount 계산 수량 1주 미만"
        return self._sizing_result(qty, 0, qty_by_amount, reason, False)

    # _sizing_result는 로그에 바로 넣을 수 있는 표준 수량 계산 결과를 만듭니다.
    def _sizing_result(
        self,
        qty: int,
        qty_by_risk: int,
        qty_by_amount: int,
        reason: str | None,
        risk_based: bool,
    ) -> dict[str, float | int | str | bool | None]:
        return {
            "qty": int(qty),
            "qty_by_risk": int(qty_by_risk),
            "qty_by_amount": int(qty_by_amount),
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "risk_based": bool(risk_based),
            "sizing_skip_reason": reason,
        }

    # allow_entry는 사유가 필요 없는 호출부를 위한 boolean 래퍼입니다.
    def allow_entry(self, ticker: str, df: pd.DataFrame | None) -> bool:
        allowed, _ = self.entry_check(ticker, df)
        return allowed
