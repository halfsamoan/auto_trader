"""Domestic stock behavior tree shadow branch selection."""

from __future__ import annotations

from config import AI_POLICY_MAX_RISK_SCORE, AI_POLICY_MIN_CONFIDENCE, AI_POLICY_MIN_EXPECTED_RETURN

from behavior_tree.nodes import ActionNode, ConditionNode, Selector, Sequence


def _set_action(name: str):
    def inner(data: dict) -> None:
        data["bt_would_action"] = name

    return inner


def build_domestic_stock_shadow_tree() -> Selector:
    return Selector(
        "Root",
        [
            Sequence(
                "EmergencyExit",
                [
                    ConditionNode("has_position", lambda d: bool(d.get("has_position"))),
                    ConditionNode("ai_emergency_exit", lambda d: d.get("ai_action") in {"CUT_LOSS", "TRAILING_EXIT"}),
                    ActionNode("would_exit", _set_action("SELL_LIMIT")),
                ],
            ),
            Sequence(
                "TakeProfit",
                [
                    ConditionNode("has_position", lambda d: bool(d.get("has_position"))),
                    ConditionNode("ai_take_profit", lambda d: d.get("ai_action") in {"TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_FULL"}),
                    ActionNode("would_take_profit", _set_action("SELL_LIMIT")),
                ],
            ),
            Sequence(
                "HoldPosition",
                [
                    ConditionNode("has_position", lambda d: bool(d.get("has_position"))),
                    ConditionNode("ai_hold_position", lambda d: d.get("ai_action") == "HOLD_POSITION"),
                    ActionNode("would_hold", _set_action("HOLD")),
                ],
            ),
            Sequence(
                "Entry",
                [
                    ConditionNode("no_position", lambda d: not bool(d.get("has_position"))),
                    ConditionNode("ai_buy", lambda d: d.get("ai_action") == "BUY"),
                    ConditionNode("confidence_pass", lambda d: (d.get("confidence") or 0.0) >= AI_POLICY_MIN_CONFIDENCE),
                    ConditionNode("expected_return_pass", lambda d: (d.get("expected_return") or 0.0) >= AI_POLICY_MIN_EXPECTED_RETURN),
                    ConditionNode("risk_score_pass", lambda d: (d.get("risk_score") if d.get("risk_score") is not None else 1.0) <= AI_POLICY_MAX_RISK_SCORE),
                    ActionNode("would_enter", _set_action("BUY_LIMIT")),
                ],
            ),
            Sequence("Idle", [ActionNode("no_action", _set_action("NO_ACTION"))]),
        ],
    )
