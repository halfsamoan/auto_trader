#!/usr/bin/env python3
"""Report local intraday-cache coverage and AI split readiness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.dataset import SplitConfig, aggregate_label_distribution, build_symbol_samples, calendar_time_split, concat_samples
from ai.label_builder import LabelConfig
from ai.universe_builder import load_ai_universe_records
from config import (
    AI_LABEL_MODE,
    AI_PRED_HORIZON_BARS,
    AI_PURGE_GAP_BARS,
    AI_SEQUENCE_LENGTH,
    AI_STOP_RETURN,
    AI_TARGET_RETURN,
    WATCHLIST,
)
from core.fetcher_intraday import load_intraday_cache, load_intraday_cache_raw


def source_counts_for(code: str, interval: str) -> dict[str, int]:
    raw = load_intraday_cache_raw(code, interval)
    if raw.empty or "source" not in raw.columns:
        return {}
    return {str(k): int(v) for k, v in raw["source"].fillna("unknown").value_counts().to_dict().items()}


def gap_summary(data) -> dict[str, object]:
    if data.empty or len(data.index) < 2:
        return {"gap_count": 0, "max_gap_minutes": None}
    diffs = data.index.to_series().diff().dropna()
    gaps = diffs[diffs > diffs.mode().iloc[0] if not diffs.mode().empty else diffs > diffs.median()]
    return {
        "gap_count": int(len(gaps)),
        "max_gap_minutes": float(gaps.max().total_seconds() / 60) if len(gaps) else None,
    }


def resolve_universe(name: str, watchlist: str | None) -> list[str]:
    if watchlist:
        return [code.strip() for code in watchlist.split(",") if code.strip()]
    if name == "ai_train":
        return [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    if name == "watchlist":
        return [str(item["code"]) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    raise ValueError("--universe는 ai_train 또는 watchlist만 허용됩니다.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--min-valid-samples", type=int, default=300)
    parser.add_argument("--min-test-samples", type=int, default=300)
    args = parser.parse_args()

    codes = resolve_universe(args.universe, args.watchlist)
    records = load_ai_universe_records(refresh=False) if args.universe == "ai_train" and not args.watchlist else []
    source_by_code = {str(row.get("code")).zfill(6): str(row.get("source") or "") for row in records}
    label_config = LabelConfig(AI_PRED_HORIZON_BARS, AI_TARGET_RETURN, AI_STOP_RETURN, AI_LABEL_MODE)
    samples = []
    all_samples = []
    row_counts = {}
    source_counts = {}
    gap_reports = {}
    for code in codes:
        data = load_intraday_cache(code, args.interval)
        row_counts[code] = int(len(data))
        source_counts[code] = source_counts_for(code, args.interval)
        gap_reports[code] = gap_summary(data)
        if data.empty:
            continue
        sample = build_symbol_samples(data, code, label_config, AI_SEQUENCE_LENGTH)
        all_samples.append(sample)
        if len(sample["X"]):
            samples.append(sample)

    success_count = sum(1 for count in row_counts.values() if count > 0)
    total_rows = sum(row_counts.values())
    distribution = aggregate_label_distribution(all_samples)
    sequence_count = sum(int(len(sample["X"])) for sample in all_samples)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "universe": args.universe,
        "symbol_count": len(codes),
        "kospi200_count": sum(1 for code in codes if source_by_code.get(code) == "KOSPI200"),
        "kosdaq_top50_count": sum(1 for code in codes if source_by_code.get(code, "").startswith("KOSDAQ")),
        "success_symbol_count": success_count,
        "missing_cache_symbol_count": len(codes) - success_count,
        "row_counts": row_counts,
        "row_counts_top10": dict(sorted(row_counts.items(), key=lambda item: item[1], reverse=True)[:10]),
        "row_counts_bottom10": dict(sorted(row_counts.items(), key=lambda item: item[1])[:10]),
        "source_row_counts": source_counts,
        "source_row_counts_total": {},
        "missing_gap_summary": gap_reports,
        "total_rows": int(total_rows),
        "expected_sequence_count": int(sequence_count),
        "label_distribution": distribution,
        "split_report": None,
        "trainable": False,
    }
    for counts in source_counts.values():
        for key, value in counts.items():
            report["source_row_counts_total"][key] = int(report["source_row_counts_total"].get(key, 0)) + int(value)

    if samples:
        data = concat_samples(samples)
        _, split_report = calendar_time_split(data, SplitConfig(AI_SEQUENCE_LENGTH, AI_PRED_HORIZON_BARS, AI_PURGE_GAP_BARS))
        report["split_report"] = split_report
        valid_size = int(split_report["sizes"]["valid"])
        test_size = int(split_report["sizes"]["test"])
        report["trainable"] = valid_size >= args.min_valid_samples and test_size >= args.min_test_samples
        if not report["trainable"]:
            report["training_block_reason"] = (
                f"데이터 부족: valid={valid_size}/{args.min_valid_samples}, "
                f"test={test_size}/{args.min_test_samples}"
            )
    else:
        report["training_block_reason"] = "데이터 부족: sequence_length 적용 후 샘플이 없습니다."

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["trainable"]:
        print("학습 불가: valid/test 최소 샘플 기준을 충족하지 못했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
