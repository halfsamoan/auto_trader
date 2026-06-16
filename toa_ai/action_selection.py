"""Account-state aware policy action selection.

The model predicts over every action class, but only a subset is executable for
the current position state. This module keeps that selection step outside the
risk governor so the governor remains a safety check rather than a strategy
rewriter.
"""

from __future__ import annotations

from dataclasses import replace

from toa_ai.domain import Action, PolicyOutput


FLAT_ACTIONS = (Action.NO_ACTION, Action.OPEN_LONG)
POSITION_ACTIONS = (Action.HOLD_LONG, Action.REDUCE_LONG, Action.CLOSE_LONG)
ORDER_ACTIONS = frozenset({Action.OPEN_LONG, Action.REDUCE_LONG, Action.CLOSE_LONG})


def executable_actions(has_position: bool) -> tuple[Action, ...]:
    return POSITION_ACTIONS if has_position else FLAT_ACTIONS


def select_executable_action(output: PolicyOutput, has_position: bool) -> PolicyOutput:
    """Return a PolicyOutput whose action is executable for the position state."""
    allowed = executable_actions(has_position)
    if output.action in allowed:
        selected = output.action
        reason = "raw_action_executable"
    else:
        selected = max(allowed, key=lambda action: float(output.action_probs.get(action.value, 0.0)))
        reason = "raw_action_reselected"

    selected_confidence = float(output.action_probs.get(selected.value, output.confidence if selected == output.action else 0.0))
    metadata = {
        **output.metadata,
        "raw_action": output.action.value,
        "raw_confidence": float(output.confidence),
        "selected_action": selected.value,
        "selected_confidence": selected_confidence,
        "selection_reason": reason,
        "has_position": bool(has_position),
        "allowed_actions": [action.value for action in allowed],
    }
    return replace(output, action=selected, confidence=selected_confidence, metadata=metadata)
