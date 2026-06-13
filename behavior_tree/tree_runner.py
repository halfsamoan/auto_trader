"""Run behavior tree shadow evaluations."""

from __future__ import annotations

from typing import Any

from behavior_tree.domestic_stock_tree import build_domestic_stock_shadow_tree
from behavior_tree.nodes import TickContext
from config import ENABLE_BT_SHADOW


def empty_bt_result(enabled: bool = False, mode: str = "disabled") -> dict[str, Any]:
    return {
        "behavior_tree_enabled": enabled,
        "behavior_tree_mode": mode,
        "behavior_tree_path": [],
        "behavior_tree_result": "DISABLED",
        "bt_would_action": None,
        "bt_shadow_only": True,
    }


def run_domestic_stock_bt_shadow(record: dict[str, Any], has_position: bool) -> dict[str, Any]:
    if not ENABLE_BT_SHADOW:
        return empty_bt_result(False, "disabled")
    data = {
        "has_position": has_position,
        "ai_action": record.get("ai_action"),
        "confidence": record.get("confidence"),
        "expected_return": record.get("expected_return"),
        "risk_score": record.get("risk_score"),
        "bt_would_action": None,
    }
    context = TickContext(data=data)
    status = build_domestic_stock_shadow_tree().tick(context)
    return {
        "behavior_tree_enabled": True,
        "behavior_tree_mode": "shadow",
        "behavior_tree_path": context.path,
        "behavior_tree_result": status.value,
        "bt_would_action": data.get("bt_would_action"),
        "bt_shadow_only": True,
    }
