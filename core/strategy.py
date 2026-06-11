# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/technical.py
# Description: 장중 진입가, ATR 손절가, 목표가 계획을 계산합니다.
# ================================================================================

"""Intraday entry and exit level calculation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from .technical import atr, ensure_ohlcv


@dataclass
class StrategyPlan:
    entry: float
    stop_loss: float
    target1: float
    target2: float
    rr_ratio: float
    skip: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_intraday_plan(df: pd.DataFrame) -> StrategyPlan:
    data = ensure_ohlcv(df)
    if data.empty:
        return StrategyPlan(0.0, 0.0, 0.0, 0.0, 0.0, True, "데이터 없음")

    entry = float(data["Close"].iloc[-1])
    atr_series = atr(data, 14)
    atr_5m = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else entry * 0.008
    if atr_5m <= 0:
        atr_5m = entry * 0.008

    stop_loss = entry - 1.2 * atr_5m
    target1 = entry + 1.5 * atr_5m
    target2 = entry + 2.5 * atr_5m
    risk = entry - stop_loss
    rr_ratio = (target1 - entry) / risk if risk > 0 else 0.0
    stop_distance = risk / entry if entry else 1.0
    expected_profit = (target1 - entry) / entry if entry else 0.0

    skip_reason = None
    if rr_ratio < 1.2:
        skip_reason = "rr_ratio < 1.2"
    elif stop_distance > 0.015:
        skip_reason = "stop_loss distance > 1.5%"
    elif expected_profit < 0.006:
        skip_reason = "target1 expected profit < 0.6%"

    return StrategyPlan(
        entry=entry,
        stop_loss=stop_loss,
        target1=target1,
        target2=target2,
        rr_ratio=rr_ratio,
        skip=skip_reason is not None,
        skip_reason=skip_reason,
    )
