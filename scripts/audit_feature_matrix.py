# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-14 (V3.2.3)
# Dependency: ai/feature_builder.py, ai/universe_builder.py, config.py, core/fetcher_intraday.py
# Description: intraday cache 기반 feature matrix 품질을 주문 호출 없이 점검합니다.
# ================================================================================

"""API나 주문 호출 없이 캐시 기반 intraday feature matrix 품질을 점검합니다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_builder import FEATURE_COLUMNS, build_feature_frame
from ai.universe_builder import load_ai_universe_records
from config import EXCLUDED_CODES, LEVERAGE_ETN_WATCHLIST
from core.fetcher_intraday import load_intraday_cache, load_intraday_cache_raw


def _excluded_codes() -> set[str]:
    """학습 feature matrix에서 제외할 watch-only 종목 코드를 반환합니다."""

    leveraged = {str(row.get("code")).zfill(6) for row in LEVERAGE_ETN_WATCHLIST if row.get("code")}
    configured = {str(code).zfill(6) for code in EXCLUDED_CODES}
    return leveraged | configured


def _resolve_codes(watchlist: str | None) -> tuple[list[str], list[str]]:
    """로컬 universe를 읽고 watch-only 종목을 제외한 코드 목록을 만듭니다."""

    if watchlist:
        raw = [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    else:
        raw = [str(row.get("code")).zfill(6) for row in load_ai_universe_records(refresh=False) if row.get("code")]
    excluded = _excluded_codes()
    deduped = list(dict.fromkeys(raw))
    included = [code for code in deduped if code not in excluded]
    skipped = [code for code in deduped if code in excluded]
    return included, skipped


def _source_counts(raw: pd.DataFrame) -> dict[str, int]:
    """raw cache의 source 컬럼별 row 수를 계산합니다."""

    if raw.empty or "source" not in raw.columns:
        return {}
    return {str(key): int(value) for key, value in raw["source"].fillna("unknown").value_counts().to_dict().items()}


def _feature_audit(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """feature별 NaN, inf, 상수 여부를 집계합니다."""

    result: dict[str, dict[str, Any]] = {}
    row_count = int(len(frame))
    for column in FEATURE_COLUMNS:
        series = frame[column] if column in frame.columns else pd.Series(dtype=float)
        numeric = pd.to_numeric(series, errors="coerce")
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(numeric) else 0
        nan_count = int(numeric.isna().sum())
        unique_count = int(numeric.nunique(dropna=False)) if len(numeric) else 0
        result[column] = {
            "nan_ratio": float(nan_count / row_count) if row_count else None,
            "inf_count": inf_count,
            "is_constant": bool(row_count > 0 and unique_count <= 1),
            "unique_count": unique_count,
        }
    return result


def _new_feature_columns() -> list[str]:
    """기존 V3.2.2 컬럼 뒤에 append된 신규 feature 목록을 반환합니다."""

    try:
        start = FEATURE_COLUMNS.index("rule_score") + 1
    except ValueError:
        start = 0
    return FEATURE_COLUMNS[start:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--interval", default="5m")
    args = parser.parse_args()

    codes, skipped_codes = _resolve_codes(args.watchlist)
    excluded_codes = _excluded_codes()
    feature_frames = []
    row_counts: dict[str, int] = {}
    feature_row_counts: dict[str, int] = {}
    source_row_counts: dict[str, dict[str, int]] = {}
    failures: dict[str, str] = {}
    row_decrease: dict[str, int] = {}

    for code in codes:
        raw = load_intraday_cache_raw(code, args.interval)
        data = load_intraday_cache(code, args.interval)
        row_counts[code] = int(len(data))
        source_row_counts[code] = _source_counts(raw)
        if data.empty:
            feature_row_counts[code] = 0
            row_decrease[code] = int(len(raw))
            continue
        try:
            frame = build_feature_frame(data, rule_score=50.0)
        except Exception as exc:
            failures[code] = str(exc)
            feature_row_counts[code] = 0
            row_decrease[code] = int(len(data))
            continue
        frame = frame.copy()
        frame["__symbol"] = code
        feature_frames.append(frame)
        feature_row_counts[code] = int(len(frame))
        row_decrease[code] = int(len(data) - len(frame))

    matrix = pd.concat(feature_frames, axis=0) if feature_frames else pd.DataFrame(columns=FEATURE_COLUMNS + ["__symbol"])
    source_totals: dict[str, int] = {}
    for counts in source_row_counts.values():
        for source, count in counts.items():
            source_totals[source] = int(source_totals.get(source, 0)) + int(count)

    feature_matrix_codes = set(matrix["__symbol"].astype(str).unique()) if "__symbol" in matrix.columns and not matrix.empty else set()
    feature_audit = _feature_audit(matrix[FEATURE_COLUMNS] if not matrix.empty else pd.DataFrame(columns=FEATURE_COLUMNS))
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "interval": args.interval,
        "order_api_called": False,
        "env_loaded": False,
        "symbol_count": len(codes),
        "skipped_watch_only_codes": skipped_codes,
        "excluded_codes": sorted(excluded_codes),
        "code_520100": {
            "excluded_by_policy": "520100" in excluded_codes,
            "requested_or_universe_contains": "520100" in set(codes + skipped_codes),
            "feature_matrix_contains": "520100" in feature_matrix_codes,
            "cache_file_rows": int(len(load_intraday_cache_raw("520100", args.interval))),
        },
        "row_counts": row_counts,
        "feature_row_counts": feature_row_counts,
        "feature_generation_row_decrease": row_decrease,
        "source_row_counts": source_row_counts,
        "source_row_counts_total": source_totals,
        "feature_columns": FEATURE_COLUMNS,
        "new_feature_columns": _new_feature_columns(),
        "feature_count": len(FEATURE_COLUMNS),
        "matrix_row_count": int(len(matrix)),
        "symbol_row_counts": {
            str(symbol): int(count)
            for symbol, count in (
                matrix["__symbol"].value_counts().sort_index().to_dict().items()
                if "__symbol" in matrix.columns and not matrix.empty
                else []
            )
        },
        "feature_audit": feature_audit,
        "constant_features": [column for column, audit in feature_audit.items() if audit["is_constant"]],
        "failures": failures,
        "data_status": "FEATURE_AUDIT_READY" if feature_frames and not failures else "FEATURE_AUDIT_NEEDS_REVIEW",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures and "520100" not in feature_matrix_codes else 1


if __name__ == "__main__":
    raise SystemExit(main())
