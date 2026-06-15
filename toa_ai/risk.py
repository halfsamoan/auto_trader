"""Safety governor for TOA execution.

This module is not a trading strategy. It only enforces account survival and
operational safety around whatever action the policy model chooses.
"""

from __future__ import annotations

from dataclasses import asdict

from toa_ai.config import RiskConfig
from toa_ai.domain import AccountState, Action, GateResult, PolicyOutput


class RiskGovernor:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def review(self, policy: PolicyOutput, symbol: str, price: float, account: AccountState, live_order: bool = False) -> GateResult:
        if live_order and not self.config.allow_live_order:
            return GateResult(False, "live_order_disabled", forced_action=Action.NO_ACTION, metadata=asdict(self.config))
        if policy.confidence < self.config.min_model_confidence_for_execution:
            return GateResult(False, "model_confidence_below_execution_floor", forced_action=Action.NO_ACTION, metadata=asdict(self.config))
        if account.equity <= 0:
            return GateResult(False, "account_equity_non_positive", forced_action=Action.NO_ACTION, metadata=asdict(self.config))
        if account.daily_realized_pnl <= -account.equity * self.config.max_daily_loss_fraction:
            return GateResult(False, "daily_loss_limit_reached", forced_action=Action.NO_ACTION, metadata=asdict(self.config))

        if policy.action == Action.OPEN_LONG:
            if symbol in account.positions:
                return GateResult(False, "position_already_open", forced_action=Action.HOLD_LONG, metadata=asdict(self.config))
            if len(account.positions) >= self.config.max_open_positions:
                return GateResult(False, "max_open_positions_reached", forced_action=Action.NO_ACTION, metadata=asdict(self.config))
            projected_value = account.equity * self.config.max_position_value_fraction
            if projected_value <= 0 or price <= 0:
                return GateResult(False, "invalid_order_value_or_price", forced_action=Action.NO_ACTION, metadata=asdict(self.config))
            if account.exposure + projected_value > account.equity * self.config.max_total_exposure_fraction:
                return GateResult(False, "max_total_exposure_reached", forced_action=Action.NO_ACTION, metadata=asdict(self.config))

        if policy.action in {Action.HOLD_LONG, Action.REDUCE_LONG, Action.CLOSE_LONG} and symbol not in account.positions:
            return GateResult(False, "holding_action_without_position", forced_action=Action.NO_ACTION, metadata=asdict(self.config))

        return GateResult(True, "allowed", metadata=asdict(self.config))
