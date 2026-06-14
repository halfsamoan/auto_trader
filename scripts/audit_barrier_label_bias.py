#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.6)
# Dependency: scripts/sweep_barrier_geometry.py
# Description: Audit triple-barrier label bias without order calls.
# ================================================================================

"""Audit same-bar and timeout biases in the triple-barrier labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AI_PRED_HORIZON_BARS, AI_STOP_RETURN, AI_TARGET_RETURN
from scripts.sweep_barrier_geometry import Geometry, aggregate_label_only, excluded_codes, resolve_codes


REPORT_PATH = ROOT / "ai" / "model_store" / "barrier_label_bias_audit.json"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    codes = resolve_codes(args.universe, args.watchlist)
    geometry = Geometry(float(args.target_return), -abs(float(args.stop_return)), int(args.horizon_bars))
    label_row = aggregate_label_only(codes, geometry, args.interval)
    same_bar_ratio = float(label_row.get("same_bar_target_stop_ratio") or 0.0)
    target_ratio = float(label_row.get("target_ratio_without_neutral") or 0.0)
    warnings = []
    if same_bar_ratio > args.max_same_bar_ratio:
        warnings.append("same_bar_collision_ratio_high")
    if target_ratio < args.min_target_ratio_without_neutral or target_ratio > args.max_target_ratio_without_neutral:
        warnings.append("target_ratio_without_neutral_outside_expected_band")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "order_api_called": False,
        "env_loaded": False,
        "symbol_count": len(codes),
        "excluded_symbols": sorted(excluded_codes()),
        "geometry": {
            "target_return": geometry.target_return,
            "stop_return": geometry.stop_return,
            "horizon_bars": geometry.horizon_bars,
        },
        "label_bias_audit": {
            "same_bar_resolution": "stop_first_when_target_and_stop_hit_same_future_bar",
            "same_bar_target_stop_count": label_row.get("same_bar_target_stop_count"),
            "same_bar_target_stop_ratio": same_bar_ratio,
            "timeout_return_definition": "neutral labels use realized close-to-close return at horizon; target/stop labels use nominal barrier returns",
            "expected_return_target_definition": "mixed: nominal +/- barrier for first touch, realized timeout return for neutral",
            "target_ratio_without_neutral": target_ratio,
            "neutral_ratio": label_row.get("neutral_ratio"),
            "warnings": warnings,
        },
        "label_row": label_row,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--target-return", type=float, default=AI_TARGET_RETURN)
    parser.add_argument("--stop-return", type=float, default=abs(AI_STOP_RETURN))
    parser.add_argument("--horizon-bars", type=int, default=AI_PRED_HORIZON_BARS)
    parser.add_argument("--max-same-bar-ratio", type=float, default=0.02)
    parser.add_argument("--min-target-ratio-without-neutral", type=float, default=0.25)
    parser.add_argument("--max-target-ratio-without-neutral", type=float, default=0.75)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
