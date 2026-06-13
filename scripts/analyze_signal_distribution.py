# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-12 (V3.2)
# Dependency: data/signal_log.json (또는 logs/signal_log.json)
# Description: signal 로그의 점수/차단사유/기대수익 분포를 분석합니다. 주문 API 호출 없음.
# ================================================================================

"""Read-only analysis of domestic-stock signal log distribution."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_CANDIDATES = [ROOT / "data" / "signal_log.json", ROOT / "logs" / "signal_log.json"]
SCORE_THRESHOLDS = [60, 65, 70, 75]
PROFIT_THRESHOLDS = [0.0035, 0.0045, 0.006]


def load_rows() -> list[dict]:
    for path in LOG_CANDIDATES:
        if path.exists():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                print(f"로그 파일: {path} ({len(rows)}건)")
                return [r for r in rows if isinstance(r, dict)]
    print("signal_log.json을 찾지 못했습니다.")
    return []


def pct_stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: 데이터 없음")
        return
    ordered = sorted(values, reverse=True)
    top10 = ordered[: max(1, len(ordered) // 10)]
    print(
        f"  {label}: n={len(values)} avg={statistics.mean(values):.2f} "
        f"max={max(values):.2f} min={min(values):.2f} top10%avg={statistics.mean(top10):.2f}"
    )


def main() -> int:
    rows = load_rows()
    domestic = [r for r in rows if r.get("asset_class") == "domestic-stock"]
    scored = [r for r in domestic if isinstance(r.get("final_score"), (int, float))]
    if not scored:
        print("분석할 국내주식 점수 레코드가 없습니다.")
        return 0

    print("\n=== 종목별 점수 분포 ===")
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_code[str(r.get("code") or r.get("symbol_or_code"))].append(r)
    for code, recs in sorted(by_code.items()):
        print(f"{code}:")
        pct_stats("final_score   ", [float(r["final_score"]) for r in recs])
        pct_stats("intraday_score", [float(r["intraday_score"]) for r in recs if isinstance(r.get("intraday_score"), (int, float))])

    print("\n=== signal별 개수 ===")
    for signal, count in Counter(str(r.get("signal")) for r in domestic).most_common():
        print(f"  {signal}: {count}건")

    print("\n=== skip_reason / 차단 사유별 개수 ===")
    reason_counts: Counter[str] = Counter()
    for r in domestic:
        skip_reason = r.get("skip_reason") or (r.get("strategy") or {}).get("skip_reason")
        if skip_reason:
            reason_counts[str(skip_reason)] += 1
        for reason in r.get("reasons") or []:
            reason_counts[str(reason)] += 1
    for reason, count in reason_counts.most_common():
        print(f"  {count:4d}x {reason}")

    print("\n=== final_score threshold별 후보 수 ===")
    finals = [float(r["final_score"]) for r in scored]
    for th in SCORE_THRESHOLDS:
        hits = sum(1 for v in finals if v >= th)
        print(f"  final_score >= {th}: {hits}건 ({100 * hits / len(finals):.1f}%)")

    print("\n=== target1 기대수익률 분포 ===")
    profits: list[float] = []
    for r in scored:
        value = r.get("target1_expected_profit_pct")
        if not isinstance(value, (int, float)):
            strategy = r.get("strategy") or {}
            entry, target1 = strategy.get("entry"), strategy.get("target1")
            value = (target1 - entry) / entry if entry and target1 else None
        if isinstance(value, (int, float)):
            profits.append(float(value))
    if profits:
        print(f"  n={len(profits)} avg={statistics.mean(profits) * 100:.3f}% max={max(profits) * 100:.3f}% min={min(profits) * 100:.3f}%")
        for th in PROFIT_THRESHOLDS:
            hits = sum(1 for v in profits if v >= th)
            print(f"  >= {th * 100:.2f}%: {hits}건 ({100 * hits / len(profits):.1f}%)")
    else:
        print("  데이터 없음")

    print("\n=== 시간대별 후보 수 (final_score >= 65) ===")
    hourly: Counter[str] = Counter()
    for r in scored:
        ts = str(r.get("timestamp") or r.get("logged_at") or "")
        if len(ts) >= 13 and float(r["final_score"]) >= 65:
            hourly[ts[11:13] + "시"] += 1
    for hour, count in sorted(hourly.items()):
        print(f"  {hour}: {count}건")
    if not hourly:
        print("  해당 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
