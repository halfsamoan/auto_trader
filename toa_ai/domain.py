"""Domain objects shared by TOA AI components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class Action(str, Enum):
    NO_ACTION = "NO_ACTION"
    OPEN_LONG = "OPEN_LONG"
    HOLD_LONG = "HOLD_LONG"
    REDUCE_LONG = "REDUCE_LONG"
    CLOSE_LONG = "CLOSE_LONG"


ACTION_CLASSES: dict[int, Action] = {
    0: Action.NO_ACTION,
    1: Action.OPEN_LONG,
    2: Action.HOLD_LONG,
    3: Action.REDUCE_LONG,
    4: Action.CLOSE_LONG,
}
ACTION_TO_ID = {action: idx for idx, action in ACTION_CLASSES.items()}


@dataclass(frozen=True)
class PolicyOutput:
    action: Action
    confidence: float
    action_probs: dict[str, float]
    expected_return: float
    risk_score: float
    holding_score: float
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str
    forced_action: Action | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountState:
    cash: float
    equity: float
    realized_pnl: float = 0.0
    daily_realized_pnl: float = 0.0
    positions: dict[str, "Position"] = field(default_factory=dict)

    @property
    def exposure(self) -> float:
        return sum(position.market_value for position in self.positions.values())


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_price: float
    opened_at: str
    updated_at: str

    @property
    def market_value(self) -> float:
        return max(self.qty, 0.0) * max(self.market_price, 0.0)

    @property
    def unrealized_pnl(self) -> float:
        return (self.market_price - self.avg_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return self.market_price / self.avg_price - 1.0


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    qty: float
    price: float
    decision_id: str | None = None
    order_type: str = "market"


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    fill_id: str | None
    accepted: bool
    filled: bool
    reason: str
    symbol: str
    side: str
    qty: float
    fill_price: float | None = None
    fee: float = 0.0


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    model_type: str
    path: str
    status: str
    metrics: dict[str, Any]
    metadata: dict[str, Any]
