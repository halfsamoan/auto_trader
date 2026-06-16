"""One-cycle paper loop for TOA policy decisions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from toa_ai.action_selection import select_executable_action
from toa_ai.config import RiskConfig
from toa_ai.domain import Action, OrderRequest
from toa_ai.execution import PaperBroker
from toa_ai.features import build_feature_frame
from toa_ai.risk import RiskGovernor
from toa_ai.replay import _policy_json, _state
from toa_ai.storage import TOAMemory


def run_paper_once(
    memory: TOAMemory,
    policy: Any,
    symbols: list[str],
    timeframe: str = "5m",
    sequence_length: int = 64,
    initial_cash: float = 10_000_000.0,
    risk_config: RiskConfig | None = None,
) -> dict[str, Any]:
    risk = RiskGovernor(risk_config)
    broker = PaperBroker(memory, initial_cash=initial_cash, risk_config=risk.config)
    results = []
    for symbol in symbols:
        bars = memory.load_bars(symbol, timeframe=timeframe, limit=sequence_length + 1)
        if len(bars) < sequence_length:
            results.append({"symbol": symbol, "status": "skipped", "reason": "insufficient_bars"})
            continue
        price = float(bars["close"].iloc[-1])
        broker.mark(symbol, price)
        position = broker.account.positions.get(symbol)
        features = build_feature_frame(
            bars,
            position_flag=1.0 if position else 0.0,
            entry_price=position.avg_price if position else None,
            holding_bars=sequence_length,
        )
        output = policy.predict(features.iloc[-sequence_length:].to_numpy(dtype=np.float32))
        selected_output = select_executable_action(output, has_position=position is not None)
        gate = risk.review(selected_output, symbol, price, broker.account)
        action = gate.forced_action or selected_output.action
        decision_id = memory.append_decision(
            symbol=symbol,
            model_id=output.model_id,
            action=action.value,
            confidence=selected_output.confidence,
            expected_return=selected_output.expected_return,
            risk_score=selected_output.risk_score,
            accepted=gate.allowed,
            state=_state(symbol, price, broker.account, sequence=features.iloc[-sequence_length:].to_numpy(dtype=np.float32), feature_columns=list(features.columns)),
            policy=_policy_json(output, selected_output),
            gate={
                "allowed": gate.allowed,
                "reason": gate.reason,
                "metadata": gate.metadata,
                "selection": {
                    "raw_action": output.action.value,
                    "raw_confidence": output.confidence,
                    "selected_action": selected_output.action.value,
                    "selected_confidence": selected_output.confidence,
                    "selection_reason": selected_output.metadata.get("selection_reason"),
                    "allowed_actions": selected_output.metadata.get("allowed_actions", []),
                    "has_position": selected_output.metadata.get("has_position"),
                },
            },
        )
        order = None
        if gate.allowed:
            order = _execute_paper_action(broker, action, symbol, price, decision_id)
        results.append(
            {
                "symbol": symbol,
                "status": "ok",
                "decision_id": decision_id,
                "policy_action": output.action.value,
                "selected_action": selected_output.action.value,
                "executed_action": action.value,
                "policy_confidence": output.confidence,
                "selected_confidence": selected_output.confidence,
                "selection_reason": selected_output.metadata.get("selection_reason"),
                "gate": {"allowed": gate.allowed, "reason": gate.reason},
                "order": asdict(order) if order else None,
            }
        )
    return {
        "symbols": symbols,
        "results": results,
        "equity": broker.account.equity,
        "cash": broker.account.cash,
        "positions": {symbol: asdict(position) for symbol, position in broker.account.positions.items()},
    }


def _execute_paper_action(broker: PaperBroker, action: Action, symbol: str, price: float, decision_id: str):
    position = broker.account.positions.get(symbol)
    if action == Action.OPEN_LONG and position is None:
        qty = int((broker.account.equity * broker.risk_config.max_position_value_fraction) // price)
        if qty > 0:
            return broker.submit_market_order(OrderRequest(symbol, "buy", qty, price, decision_id))
    if action == Action.REDUCE_LONG and position is not None:
        return broker.submit_market_order(OrderRequest(symbol, "sell", max(int(position.qty // 2), 1), price, decision_id))
    if action == Action.CLOSE_LONG and position is not None:
        return broker.submit_market_order(OrderRequest(symbol, "sell", position.qty, price, decision_id))
    return None
