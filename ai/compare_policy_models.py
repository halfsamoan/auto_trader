# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-13 (V3.2)
# Dependency: ai/*, config.py, core/fetcher_intraday.py
# Description: PatchTST policy와 tabular baseline의 readiness를 주문 없이 비교합니다.
# ================================================================================

"""Compare policy-model readiness without enabling any order route."""

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

from ai.action_label_builder import ActionLabelConfig
from ai.dataset import (
    SplitConfig,
    aggregate_action_label_distribution,
    build_policy_symbol_samples,
    calendar_time_split,
    concat_policy_samples,
)
from ai.train_baseline_lgbm import _model_backend
from ai.universe_builder import load_ai_universe_records
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    COMMISSION,
    TAX,
    WATCHLIST,
)
from core.fetcher_intraday import fetch_intraday


POLICY_MODEL_PATH = Path(__file__).resolve().parent / "model_store" / "domestic_patchtst_policy.pt"
POLICY_METADATA_PATH = Path(__file__).resolve().parent / "model_store" / "domestic_patchtst_policy_metadata.json"


def _resolve_codes(universe: str, watchlist: str | None) -> list[str]:
    if watchlist:
        return [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    if universe == "ai_train":
        return [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    return [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]


def _data_report(args: argparse.Namespace) -> dict[str, Any]:
    label_config = ActionLabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, label_mode=AI_LABEL_MODE)
    samples = []
    all_samples = []
    failures = {}
    for code in _resolve_codes(args.universe, args.watchlist):
        try:
            df = fetch_intraday(code, period=args.period, interval="5m", source=args.source, use_cache=True)
            sample = build_policy_symbol_samples(df, code, label_config, AI_SEQUENCE_LENGTH)
            all_samples.append(sample)
            if len(sample["X"]):
                samples.append(sample)
        except Exception as exc:
            failures[code] = str(exc)
    report: dict[str, Any] = {
        "successful_symbol_count": len(samples),
        "failed_symbols": failures,
        "action_label_distribution": aggregate_action_label_distribution(all_samples),
        "split_report": None,
    }
    if samples:
        data = concat_policy_samples(samples)
        _, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
        report["split_report"] = split_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="cache", choices=["auto", "kis", "yfinance", "cache"])
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--period", default="120d")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    args = parser.parse_args()

    backend_name, _, dependency_error = _model_backend()
    data = _data_report(args)
    split = data.get("split_report") or {"sizes": {"valid": 0, "test": 0}}
    sizes = split.get("sizes", {})
    valid_size = int(sizes.get("valid", 0))
    test_size = int(sizes.get("test", 0))
    data_ready = valid_size >= args.min_valid_samples and test_size >= args.min_test_samples
    patchtst_present = POLICY_MODEL_PATH.exists() and POLICY_METADATA_PATH.exists()
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "DATA_NOT_READY" if not data_ready else "READY_FOR_OFFLINE_COMPARISON",
        "comparison_scope": "offline_shadow_report_only",
        "order_api_called": False,
        "live_order_enabled": False,
        "cost_assumption_return": 2 * COMMISSION + TAX,
        "split_report": split,
        "action_label_distribution": data["action_label_distribution"],
        "lightgbm_baseline": {
            "backend": backend_name,
            "dependency_status": "ok" if backend_name else "MODEL_DEPENDENCY_MISSING",
            "dependency_error": dependency_error,
            "result_status": "DATA_NOT_READY" if not data_ready else "NOT_RUN_BY_COMPARISON_SCRIPT",
        },
        "patchtst_policy": {
            "model_present": patchtst_present,
            "result_status": "MODEL_MISSING" if not patchtst_present else ("DATA_NOT_READY" if not data_ready else "READY_TO_EVALUATE"),
            "overfit_warning": "Do not prefer PatchTST over baseline until valid/test sizes and out-of-sample metrics are sufficient.",
        },
        "conclusion": (
            "DATA_NOT_READY - valid/test split is empty or too small; do not enable paper/live/leveraged ETN orders."
            if not data_ready
            else "Comparison can be run offline, but V3.2 still must not enable order execution."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
