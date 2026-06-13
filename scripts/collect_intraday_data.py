#!/usr/bin/env python3
"""Collect domestic-stock 5-minute bars into the local last-wins cache."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ai.universe_builder import load_ai_universe_records
from config import WATCHLIST
from core.fetcher_intraday import (
    drop_incomplete_bar,
    fetch_intraday,
    load_intraday_cache,
    load_intraday_cache_raw,
    merge_intraday_cache,
)


def resolve_universe(name: str, watchlist: str | None, refresh: bool = False) -> list[str]:
    if watchlist:
        return [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    if name == "ai_train":
        return [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=refresh)]
    if name == "watchlist":
        return [str(item["code"]) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    raise ValueError("--universe는 ai_train 또는 watchlist만 허용됩니다.")


def collect_one(code: str, period: str, interval: str, source: str) -> dict[str, object]:
    before = load_intraday_cache(code, interval)
    fetched = []
    errors = []
    yfinance_period = "60d" if interval == "5m" else period

    source_order = {
        "auto": ["kis", "websocket", "yfinance"],
        "kis": ["kis"],
        "yfinance": ["yfinance"],
    }[source]
    for item in source_order:
        try:
            if item == "yfinance":
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    data = fetch_intraday(code, period=yfinance_period, interval=interval, source="yfinance", use_cache=False)
                source_name = "yfinance_fallback"
            elif item == "websocket":
                data = fetch_intraday(code, period=period, interval=interval, source="websocket", use_cache=False)
                source_name = "kis_websocket_cache"
            else:
                data = fetch_intraday(code, period="5d", interval=interval, source="kis", use_cache=False)
                source_name = "kis_intraday"
            data = drop_incomplete_bar(data, interval)
            if not data.empty:
                merge_intraday_cache(code, data, interval, source=source_name)
                fetched.append({"source": source_name, "rows": int(len(data))})
                if source == "auto":
                    break
        except Exception as exc:
            errors.append(f"{item}:{exc}")

    after = load_intraday_cache(code, interval)
    raw_after = load_intraday_cache_raw(code, interval)
    source_counts = raw_after["source"].fillna("unknown").value_counts().to_dict() if "source" in raw_after.columns else {}
    status = "collected" if fetched else ("cache_only" if len(after) else "skip")
    return {
        "code": code,
        "status": status,
        "success": bool(len(after)),
        "before_rows": int(len(before)),
        "after_rows": int(len(after)),
        "added_or_updated_rows": max(int(len(after) - len(before)), 0),
        "first_timestamp": after.index[0].isoformat() if isinstance(after.index, pd.DatetimeIndex) and len(after) else None,
        "last_timestamp": after.index[-1].isoformat() if isinstance(after.index, pd.DatetimeIndex) and len(after) else None,
        "fetched": fetched,
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "yfinance_seed_period": yfinance_period,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ai_train", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--source", default="auto", choices=["auto", "kis", "yfinance"])
    parser.add_argument("--refresh-universe", action="store_true")
    args = parser.parse_args()

    codes = resolve_universe(args.universe, args.watchlist, args.refresh_universe)
    print(f"수집 대상 종목 수: {len(codes)}")
    results = []
    for code in codes:
        try:
            result = collect_one(code, args.period, args.interval, args.source)
        except Exception as exc:
            result = {"code": code, "status": "skip", "success": False, "errors": [str(exc)]}
        results.append(result)
        status = result.get("status", "skip")
        print(f"{code}: {status} rows={result.get('after_rows', 0)} errors={len(result.get('errors', []))}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "universe": args.universe,
        "interval": args.interval,
        "period": args.period,
        "source": args.source,
        "symbol_count": len(codes),
        "success_count": sum(1 for row in results if row.get("success")),
        "skip_count": sum(1 for row in results if not row.get("success")),
        "total_cache_rows": sum(int(row.get("after_rows") or 0) for row in results),
        "source_counts": {},
    }
    for row in results:
        for key, value in dict(row.get("source_counts") or {}).items():
            summary["source_counts"][key] = int(summary["source_counts"].get(key, 0)) + int(value)
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
