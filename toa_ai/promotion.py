"""Challenger to champion promotion gate for TOA models."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from toa_ai.config import PromotionConfig
from toa_ai.storage import TOAMemory


class PromotionGate:
    def __init__(self, memory: TOAMemory, config: PromotionConfig | None = None) -> None:
        self.memory = memory
        self.config = config or PromotionConfig()

    def evaluate(self, challenger_id: str) -> dict[str, Any]:
        challenger = self.memory.get_model(challenger_id)
        if challenger is None:
            return {"passed": False, "reason": "challenger_not_found", "checks": {}, "config": asdict(self.config)}
        champion = self.memory.get_champion()
        metrics = challenger.metrics
        checks = {
            "min_train_samples": int(metrics.get("train_samples", 0)) >= self.config.min_train_samples,
            "min_validation_samples": int(metrics.get("validation_samples", 0)) >= self.config.min_validation_samples,
            "max_validation_loss": float(metrics.get("validation_loss", float("inf"))) <= self.config.max_validation_loss,
            "min_action_entropy": float(metrics.get("action_entropy", 0.0)) >= self.config.min_action_entropy,
            "max_no_action_ratio": float(metrics.get("no_action_ratio", 1.0)) <= self.config.max_no_action_ratio,
        }
        if champion:
            champion_loss = float(champion.metrics.get("validation_loss", float("inf")))
            challenger_loss = float(metrics.get("validation_loss", float("inf")))
            checks["not_materially_worse_than_champion"] = challenger_loss <= champion_loss * (1.0 + self.config.challenger_loss_tolerance)
        passed = all(checks.values())
        failed = [name for name, ok in checks.items() if not ok]
        return {
            "passed": bool(passed),
            "reason": "passed" if passed else f"failed:{','.join(failed)}",
            "checks": checks,
            "challenger_id": challenger_id,
            "champion_id": champion.model_id if champion else None,
            "metrics": metrics,
            "config": asdict(self.config),
            "profit_metrics_are_diagnostic_only": True,
        }

    def promote(self, challenger_id: str) -> dict[str, Any]:
        evaluation = self.evaluate(challenger_id)
        champion_before = evaluation.get("champion_id")
        if not evaluation["passed"]:
            event_id = self.memory.append_promotion_event(
                challenger_id,
                champion_before,
                champion_before,
                "rejected",
                str(evaluation["reason"]),
                evaluation,
            )
            return {**evaluation, "promoted": False, "event_id": event_id}
        self.memory.set_champion(challenger_id)
        event_id = self.memory.append_promotion_event(
            challenger_id,
            champion_before,
            challenger_id,
            "promoted",
            "promotion_gate_passed",
            evaluation,
        )
        return {**evaluation, "promoted": True, "event_id": event_id}
