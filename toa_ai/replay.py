"""Sequential replay environment for TOA policy models."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from toa_ai.config import ReplayConfig, RiskConfig
from toa_ai.domain import Action, OrderRequest, new_id
from toa_ai.execution import PaperBroker
from toa_ai.features import build_feature_frame
from toa_ai.risk import RiskGovernor
from toa_ai.storage import TOAMemory


def run_replay_for_symbol(
    memory: TOAMemory,
    policy: Any,
    symbol: str,
    timeframe: str = "5m",
    sequence_length: int = 64,
    replay_config: ReplayConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> dict[str, Any]:
    replay_config = replay_config or ReplayConfig()
    risk = RiskGovernor(risk_config)
    broker = PaperBroker(memory, initial_cash=replay_config.initial_cash, risk_config=risk.config)
    data = memory.load_bars(symbol, timeframe=timeframe)
    if len(data) < sequence_length + 2:
        return {"symbol": symbol, "status": "skipped", "reason": "insufficient_bars", "bars": int(len(data))}

    episode_id = new_id("episode")
    start_equity = broker.account.equity
    decisions = 0
    orders = 0
    action_counts: dict[str, int] = {}
    max_end = len(data) - 1
    if replay_config.max_episode_bars is not None:
        max_end = min(max_end, sequence_length + int(replay_config.max_episode_bars))

    previous_equity = broker.account.equity
    for end in range(sequence_length - 1, max_end):
        price = float(data["close"].iloc[end])
        broker.mark(symbol, price)
        position = broker.account.positions.get(symbol)
        features = build_feature_frame(
            data.iloc[: end + 1],
            position_flag=1.0 if position else 0.0,
            entry_price=position.avg_price if position else None,
            holding_bars=end,
        )
        sequence = features.iloc[-sequence_length:].to_numpy(dtype=np.float32)
        output = policy.predict(sequence)
        gate = risk.review(output, symbol, price, broker.account)
        action = gate.forced_action or output.action
        action_counts[action.value] = action_counts.get(action.value, 0) + 1
        state = _state(symbol, price, broker.account, sequence=sequence, feature_columns=list(features.columns))
        decision_id = memory.append_decision(
            symbol=symbol,
            model_id=output.model_id,
            action=action.value,
            confidence=output.confidence,
            expected_return=output.expected_return,
            risk_score=output.risk_score,
            accepted=gate.allowed,
            state=state,
            policy=_policy_json(output),
            gate={"allowed": gate.allowed, "reason": gate.reason, "metadata": gate.metadata},
            episode_id=episode_id,
        )
        decisions += 1
        execution = {"order": None}
        if gate.allowed:
            order_result = _execute_action(broker, action, symbol, price, decision_id)
            if order_result is not None:
                orders += 1
                execution["order"] = asdict(order_result)
        next_price = float(data["close"].iloc[end + 1])
        next_return = next_price / price - 1.0 if price > 0 else 0.0
        broker.mark(symbol, next_price)
        reward = broker.account.equity / max(previous_equity, 1e-9) - 1.0
        previous_equity = broker.account.equity
        if replay_config.record_experiences:
            memory.append_experience(
                symbol=symbol,
                model_id=output.model_id,
                action=action.value,
                reward=reward,
                next_return=next_return,
                done=end == max_end - 1,
                state=state,
                next_state=_state(symbol, next_price, broker.account),
                execution=execution,
                episode_id=episode_id,
            )

    final_equity = broker.account.equity
    return {
        "symbol": symbol,
        "status": "ok",
        "episode_id": episode_id,
        "bars": int(max_end),
        "decisions": decisions,
        "orders": orders,
        "action_counts": action_counts,
        "start_equity": float(start_equity),
        "final_equity": float(final_equity),
        "diagnostic_return": float(final_equity / start_equity - 1.0) if start_equity else 0.0,
        "profit_metrics_are_diagnostic_only": True,
    }


def _execute_action(broker: PaperBroker, action: Action, symbol: str, price: float, decision_id: str):
    position = broker.account.positions.get(symbol)
    if action == Action.OPEN_LONG and position is None:
        value = broker.account.equity * broker.risk_config.max_position_value_fraction
        qty = int(value // price)
        if qty > 0:
            return broker.submit_market_order(OrderRequest(symbol, "buy", qty, price, decision_id))
    if action == Action.REDUCE_LONG and position is not None:
        qty = max(int(position.qty // 2), 1)
        return broker.submit_market_order(OrderRequest(symbol, "sell", qty, price, decision_id))
    if action == Action.CLOSE_LONG and position is not None:
        return broker.submit_market_order(OrderRequest(symbol, "sell", position.qty, price, decision_id))
    return None


def _policy_json(output: Any) -> dict[str, Any]:
    return {
        "action": output.action.value,
        "confidence": output.confidence,
        "action_probs": output.action_probs,
        "expected_return": output.expected_return,
        "risk_score": output.risk_score,
        "holding_score": output.holding_score,
        "model_id": output.model_id,
        "metadata": output.metadata,
    }


def _state(
    symbol: str,
    price: float,
    account: Any,
    sequence: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    position = account.positions.get(symbol)
    state = {
        "symbol": symbol,
        "price": float(price),
        "cash": float(account.cash),
        "equity": float(account.equity),
        "exposure": float(account.exposure),
        "position_qty": float(position.qty) if position else 0.0,
        "position_avg_price": float(position.avg_price) if position else None,
        "unrealized_pnl_pct": float(position.unrealized_pnl_pct) if position else 0.0,
    }
    if sequence is not None:
        state["feature_columns"] = feature_columns or []
        state["feature_sequence"] = sequence.astype(float).tolist()
    return state
