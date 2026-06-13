#!/usr/bin/env python3
"""Validate BT shadow logs against later cache outcomes without order calls."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import AI_PRED_HORIZON_BARS, AI_STOP_RETURN, AI_TARGET_RETURN, BT_EXECUTE_MIN_BUY_PRECISION, BT_EXECUTE_MIN_SHADOW_DAYS
from core.fetcher_intraday import load_intraday_cache


SIGNAL_LOG = ROOT / "data" / "signal_log.json"
REPORT_DIR = ROOT / "reports"


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows if row.get(key) is not None))


def _rule_ai_cross(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "rule_buy_and_ai_buy": 0,
        "rule_hold_and_ai_buy": 0,
        "rule_buy_and_ai_no_action": 0,
        "rule_buy_total": 0,
        "ai_buy_total": 0,
    }
    for row in rows:
        rule = str(row.get("signal") or row.get("rule_signal") or "").lower()
        ai = str(row.get("ai_action") or "")
        is_rule_buy = rule == "buy"
        is_ai_buy = ai == "BUY"
        out["rule_buy_total"] += int(is_rule_buy)
        out["ai_buy_total"] += int(is_ai_buy)
        out["rule_buy_and_ai_buy"] += int(is_rule_buy and is_ai_buy)
        out["rule_hold_and_ai_buy"] += int((not is_rule_buy) and is_ai_buy)
        out["rule_buy_and_ai_no_action"] += int(is_rule_buy and ai == "NO_ACTION")
    return out


def _later_outcome(code: str, timestamp: str | None) -> dict[str, Any] | None:
    if not code or not timestamp:
        return None
    df = load_intraday_cache(code)
    if df.empty:
        return None
    try:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None and df.index.tz is not None:
            ts = ts.tz_localize(df.index.tz)
        future = df[df.index >= ts].head(AI_PRED_HORIZON_BARS + 1)
    except Exception:
        return None
    if len(future) < 2:
        return None
    entry = float(future["Close"].iloc[0])
    if entry <= 0:
        return None
    target = entry * (1 + AI_TARGET_RETURN)
    stop = entry * (1 + AI_STOP_RETURN)
    for _, row in future.iloc[1:].iterrows():
        if float(row["Low"]) <= stop:
            return {"outcome": "stop", "return": AI_STOP_RETURN}
        if float(row["High"]) >= target:
            return {"outcome": "target", "return": AI_TARGET_RETURN}
    return {"outcome": "neutral", "return": float(future["Close"].iloc[-1] / entry - 1)}


def _candidate_outcomes(rows: list[dict[str, Any]], action_key: str, action_value: str) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        if str(row.get(action_key)) != action_value:
            continue
        outcome = _later_outcome(str(row.get("code") or row.get("symbol_or_code") or ""), row.get("timestamp"))
        if outcome:
            outcomes.append(outcome)
    total = len(outcomes)
    return {
        "evaluated_count": total,
        "target_hit_rate": sum(1 for item in outcomes if item["outcome"] == "target") / total if total else None,
        "stop_hit_rate": sum(1 for item in outcomes if item["outcome"] == "stop") / total if total else None,
        "neutral_rate": sum(1 for item in outcomes if item["outcome"] == "neutral") / total if total else None,
        "avg_return": sum(float(item["return"]) for item in outcomes) / total if total else None,
    }


def _bin_stats(rows: list[dict[str, Any]], key: str, bins: list[tuple[float, float]]) -> list[dict[str, Any]]:
    report = []
    for lo, hi in bins:
        selected = []
        for row in rows:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if lo <= value < hi:
                selected.append(row)
        outcome = _candidate_outcomes(selected, "bt_would_action", "BUY_LIMIT")
        report.append({"bin": f"{lo:.4f}~{hi:.4f}", "count": len(selected), **outcome})
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# BT Shadow Report",
        "",
        f"- status: {report['status']}",
        f"- rows: {report['rows']}",
        f"- unique_shadow_days: {report['unique_shadow_days']}",
        f"- bt_execute_warning: {report['bt_execute_warning']}",
        "",
        "## Distributions",
        "",
        "```json",
        json.dumps(
            {
                "ai_action_distribution": report["ai_action_distribution"],
                "behavior_tree_path_distribution": report["behavior_tree_path_distribution"],
                "bt_would_action_distribution": report["bt_would_action_distribution"],
                "rule_signal_distribution": report["rule_signal_distribution"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = _read_json_list(SIGNAL_LOG)
    shadow_rows = [row for row in rows if row.get("ai_policy_enabled") or row.get("behavior_tree_enabled")]
    dates = set()
    for row in shadow_rows:
        try:
            dates.add(datetime.fromisoformat(str(row.get("timestamp"))).date().isoformat())
        except Exception:
            pass
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if shadow_rows else "data_insufficient",
        "rows": len(shadow_rows),
        "unique_shadow_days": len(dates),
        "ai_action_distribution": _counter(shadow_rows, "ai_action"),
        "behavior_tree_path_distribution": _counter(shadow_rows, "behavior_tree_result"),
        "bt_would_action_distribution": _counter(shadow_rows, "bt_would_action"),
        "rule_signal_distribution": _counter(shadow_rows, "signal"),
        "rule_ai_intersection": _rule_ai_cross(shadow_rows),
        "bt_would_enter_outcome": _candidate_outcomes(shadow_rows, "bt_would_action", "BUY_LIMIT"),
        "bt_would_exit_outcome": _candidate_outcomes(shadow_rows, "bt_would_action", "SELL_LIMIT"),
        "confidence_bins": _bin_stats(shadow_rows, "confidence", [(0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]),
        "expected_return_bins": _bin_stats(shadow_rows, "expected_return", [(-1.0, 0.0), (0.0, 0.0015), (0.0015, 0.0025), (0.0025, 0.005), (0.005, 1.0)]),
        "risk_score_bins": _bin_stats(shadow_rows, "risk_score", [(0.0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 1.01)]),
        "bt_execute_warning": "BT shadow 로그가 충분하지 않거나 BUY precision 기준 미달이면 bt-execute를 켜면 안 됩니다.",
        "bt_execute_min_shadow_days": BT_EXECUTE_MIN_SHADOW_DAYS,
        "bt_execute_min_buy_precision": BT_EXECUTE_MIN_BUY_PRECISION,
        "order_api_called": False,
    }
    if len(dates) < BT_EXECUTE_MIN_SHADOW_DAYS:
        report["status"] = "data_insufficient"
        report["shortage_reason"] = f"shadow days {len(dates)} < required {BT_EXECUTE_MIN_SHADOW_DAYS}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d")
    json_path = REPORT_DIR / f"bt_shadow_report_{suffix}.json"
    md_path = REPORT_DIR / f"bt_shadow_report_{suffix}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, report)
    print(json.dumps({"report_json": str(json_path), "report_md": str(md_path), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
