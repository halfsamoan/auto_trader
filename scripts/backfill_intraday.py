#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-13 (V3.2.1b)
# Dependency: clients/kis_domestic_stock_client.py, config.py, core/fetcher_intraday.py
# Description: KIS/yfinance domestic-stock 5-minute quotation data is appended to local cache without order calls.
# ================================================================================

"""Backfill domestic-stock 5-minute bars through quote-only sources."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clients.kis_domestic_stock_client import DomesticStockClient
from config import LEVERAGE_ETN_WATCHLIST, WATCHLIST
from core.fetcher_intraday import load_intraday_cache, load_intraday_cache_raw, merge_intraday_cache
from core.technical import ensure_ohlcv


@dataclass(frozen=True)
class BackfillPlan:
    """Collection plan for one run.

    symbols: Six-digit domestic-stock symbols to query.
    interval: Requested bar interval; only 5m is supported by the current KIS wrapper.
    months: User-requested history horizon used for run planning and readiness guidance.
    period: yfinance download period such as 60d.
    source: Quote-only data source: kis, yfinance, or auto.
    sleep_sec: Base delay between symbols to avoid hitting KIS rate limits.
    max_retries: Per-symbol retry count for transient quotation errors.
    backoff_sec: Base exponential-backoff delay after failed requests.
    """

    symbols: list[str]
    interval: str
    months: int
    period: str
    source: str
    sleep_sec: float
    max_retries: int
    backoff_sec: float


def _normalize_symbol(value: str) -> str:
    return str(value).strip().zfill(6)


def _market_by_symbol() -> dict[str, str]:
    rows = list(WATCHLIST) + list(LEVERAGE_ETN_WATCHLIST)
    return {
        _normalize_symbol(row["code"]): str(row.get("market") or row.get("source") or "KR").upper()
        for row in rows
        if row.get("code")
    }


def resolve_symbols(symbols_arg: str, include_watch_only: bool) -> list[str]:
    """Resolve CLI symbol targets without enabling any order route.

    symbols_arg: "all" or comma-separated domestic-stock symbols.
    include_watch_only: Include watch-only leveraged ETN symbols for data collection only.
    """

    if symbols_arg.lower() != "all":
        return sorted({_normalize_symbol(item) for item in symbols_arg.split(",") if item.strip()})

    symbols = {
        _normalize_symbol(item["code"])
        for item in WATCHLIST
        if item.get("asset_class") == "domestic-stock" and item.get("code")
    }
    if include_watch_only:
        symbols.update(
            _normalize_symbol(item["code"])
            for item in LEVERAGE_ETN_WATCHLIST
            if item.get("asset_class") == "domestic-stock" and item.get("code")
        )
    return sorted(symbols)


def _cache_summary(code: str, interval: str) -> dict[str, Any]:
    cached = load_intraday_cache(code, interval)
    raw = load_intraday_cache_raw(code, interval)
    sources = raw["source"].fillna("unknown").value_counts().to_dict() if "source" in raw.columns else {}
    return {
        "rows": int(len(cached)),
        "first_timestamp": cached.index[0].isoformat() if isinstance(cached.index, pd.DatetimeIndex) and len(cached) else None,
        "last_timestamp": cached.index[-1].isoformat() if isinstance(cached.index, pd.DatetimeIndex) and len(cached) else None,
        "source_counts": {str(key): int(value) for key, value in sources.items()},
    }


def _safe_error_message(exc: Exception) -> str:
    text = str(exc)
    for needle in ["Bearer ", "appkey", "appsecret", "KIS_APP_SECRET", "KIS_APP_KEY"]:
        if needle in text:
            return "KIS quotation request failed; sensitive details redacted"
    return text[:500]


def _candidate_yfinance_tickers(code: str) -> list[str]:
    """Return yfinance ticker candidates for a Korean domestic-stock symbol.

    code: Six-digit domestic-stock code.
    """

    market = _market_by_symbol().get(code, "")
    if "KOSDAQ" in market or market == "KQ":
        return [f"{code}.KQ", f"{code}.KS"]
    return [f"{code}.KS", f"{code}.KQ"]


def _normalize_yfinance_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance OHLCV output to the local KST cache schema.

    data: Raw yfinance DataFrame.
    """

    bars = ensure_ohlcv(data)
    if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return pd.DataFrame()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert("Asia/Seoul")
    else:
        bars.index = bars.index.tz_convert("Asia/Seoul")
    valid_prices = (bars[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    bars = bars[valid_prices].sort_index()
    return bars[~bars.index.duplicated(keep="last")]


def _download_yfinance(code: str, period: str, interval: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download quote-only yfinance bars with .KS then .KQ fallback.

    code: Six-digit domestic-stock symbol.
    period: yfinance period, for example 60d.
    interval: yfinance interval, for example 5m.
    """

    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), {"status": "SOURCE_UNAVAILABLE", "error": _safe_error_message(exc)}

    attempts = []
    for ticker in _candidate_yfinance_tickers(code):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            bars = _normalize_yfinance_frame(raw)
            attempts.append({"ticker": ticker, "status": "ok" if not bars.empty else "empty", "rows": int(len(bars))})
            if not bars.empty:
                return bars, {"status": "ok", "ticker": ticker, "attempts": attempts}
        except Exception as exc:
            attempts.append({"ticker": ticker, "status": "error", "error": _safe_error_message(exc)})
    return pd.DataFrame(), {"status": "empty", "attempts": attempts}


def fetch_symbol_with_retry(client: DomesticStockClient, code: str, plan: BackfillPlan) -> dict[str, Any]:
    """Fetch and merge one symbol with retry/backoff.

    client: Dry-run KIS client used only for quotation endpoints.
    code: Six-digit domestic-stock symbol.
    plan: Backfill settings for sleep and retry behavior.
    """

    before = _cache_summary(code, plan.interval)
    attempts = []
    fetched_rows = 0
    merged_rows = before["rows"]
    status = "skip"

    for attempt in range(1, plan.max_retries + 2):
        try:
            data = client.get_intraday_5m_chart(code)
            if data.empty:
                attempts.append({"attempt": attempt, "status": "empty"})
            else:
                merged = merge_intraday_cache(code, data, plan.interval, source="kis")
                fetched_rows = int(len(data))
                merged_rows = int(len(merged))
                status = "collected"
                attempts.append({"attempt": attempt, "status": "ok", "fetched_rows": fetched_rows})
                break
        except Exception as exc:
            attempts.append({"attempt": attempt, "status": "error", "error": _safe_error_message(exc)})

        if attempt <= plan.max_retries:
            delay = plan.backoff_sec * (2 ** (attempt - 1)) + random.uniform(0, min(plan.backoff_sec, 1.0))
            time.sleep(delay)

    if status != "collected":
        status = "cache_only" if before["rows"] else "skip"

    after = _cache_summary(code, plan.interval)
    return {
        "code": code,
        "status": status,
        "before_rows": before["rows"],
        "after_rows": after["rows"],
        "added_or_updated_rows": max(int(after["rows"]) - int(before["rows"]), 0),
        "fetched_rows_last_call": fetched_rows,
        "first_timestamp": after["first_timestamp"],
        "last_timestamp": after["last_timestamp"],
        "source_counts": after["source_counts"],
        "attempts": attempts,
        "order_api_called": False,
    }


def fetch_yfinance_symbol(code: str, plan: BackfillPlan) -> dict[str, Any]:
    """Fetch and merge one symbol from yfinance without order paths.

    code: Six-digit domestic-stock symbol.
    plan: Backfill settings including yfinance period and interval.
    """

    before = _cache_summary(code, plan.interval)
    data, meta = _download_yfinance(code, plan.period, plan.interval)
    if data.empty:
        after = _cache_summary(code, plan.interval)
        return {
            "code": code,
            "status": "cache_only" if after["rows"] else "skip",
            "before_rows": before["rows"],
            "after_rows": after["rows"],
            "added_or_updated_rows": 0,
            "fetched_rows_last_call": 0,
            "first_timestamp": after["first_timestamp"],
            "last_timestamp": after["last_timestamp"],
            "source_counts": after["source_counts"],
            "yfinance": meta,
            "order_api_called": False,
        }

    merged = merge_intraday_cache(code, data, plan.interval, source="yfinance")
    after = _cache_summary(code, plan.interval)
    return {
        "code": code,
        "status": "collected",
        "before_rows": before["rows"],
        "after_rows": after["rows"],
        "added_or_updated_rows": max(int(after["rows"]) - int(before["rows"]), 0),
        "fetched_rows_last_call": int(len(data)),
        "first_timestamp": after["first_timestamp"],
        "last_timestamp": after["last_timestamp"],
        "source_counts": after["source_counts"],
        "yfinance": meta,
        "merged_rows": int(len(merged)),
        "order_api_called": False,
    }


def merge_result_steps(code: str, steps: list[dict[str, Any]], interval: str) -> dict[str, Any]:
    """Build a per-symbol summary after one or more quote-source steps."""

    after = _cache_summary(code, interval)
    before_rows = int(steps[0]["before_rows"]) if steps else after["rows"]
    return {
        "code": code,
        "status": "collected" if any(step.get("status") == "collected" for step in steps) else ("cache_only" if after["rows"] else "skip"),
        "before_rows": before_rows,
        "after_rows": after["rows"],
        "added_or_updated_rows": max(int(after["rows"]) - before_rows, 0),
        "first_timestamp": after["first_timestamp"],
        "last_timestamp": after["last_timestamp"],
        "source_counts": after["source_counts"],
        "steps": steps,
        "order_api_called": False,
    }


def make_plan(args: argparse.Namespace) -> BackfillPlan:
    """Build a conservative collection plan from CLI arguments."""

    symbols = resolve_symbols(args.symbols, include_watch_only=not args.exclude_watch_only)
    return BackfillPlan(
        symbols=symbols,
        interval=args.interval,
        months=max(int(args.months), 1),
        period=args.period,
        source=args.source,
        sleep_sec=max(float(args.sleep_sec), 0.0),
        max_retries=max(int(args.max_retries), 0),
        backoff_sec=max(float(args.backoff_sec), 0.0),
    )


def dry_run_report(plan: BackfillPlan) -> dict[str, Any]:
    """Describe what would be collected without touching KIS."""

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dry_run",
        "symbols": plan.symbols,
        "symbol_count": len(plan.symbols),
        "interval": plan.interval,
        "months_requested": plan.months,
        "period": plan.period,
        "source": plan.source,
        "yfinance_ticker_plan": {code: _candidate_yfinance_tickers(code) for code in plan.symbols},
        "kis_endpoint_scope": "current client supports latest 5m intraday chart calls; run repeatedly to accumulate history",
        "yfinance_scope": "5m intraday history is limited by yfinance availability, usually about 60 days",
        "rate_limit": {
            "sleep_sec_between_symbols": plan.sleep_sec,
            "max_retries": plan.max_retries,
            "backoff_sec": plan.backoff_sec,
        },
        "cache_mode": "idempotent append by timestamp, last value wins",
        "order_api_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all", help="'all' or comma-separated symbols such as 005930,000660")
    parser.add_argument("--interval", default="5m", choices=["5m"])
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--source", default="kis", choices=["kis", "yfinance", "auto"])
    parser.add_argument("--sleep-sec", type=float, default=0.35)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--backoff-sec", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exclude-watch-only", action="store_true")
    args = parser.parse_args()

    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(dry_run_report(plan), ensure_ascii=False, indent=2))
        return 0

    client = None
    if plan.source in {"kis", "auto"}:
        client = DomesticStockClient(
            mode="paper",
            dry_run=True,
            live=False,
            allow_paper_order=False,
            rate_limit_sleep_sec=max(plan.sleep_sec, 0.2),
        )
    if client is not None:
        ok, msg = client.validate_env()
    else:
        ok, msg = True, None
    if not ok and plan.source == "kis":
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "SKIPPED",
            "reason": "KIS quotation credentials are not configured; no traceback and no order path touched",
            "client_message": msg,
            "symbol_count": len(plan.symbols),
            "order_api_called": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    results = []
    started = datetime.now()
    for index, code in enumerate(plan.symbols, start=1):
        steps = []
        if plan.source in {"kis", "auto"} and client is not None and ok:
            steps.append(fetch_symbol_with_retry(client, code, plan))
        elif plan.source == "auto" and not ok:
            steps.append(
                {
                    "code": code,
                    "status": "skip",
                    "before_rows": _cache_summary(code, plan.interval)["rows"],
                    "after_rows": _cache_summary(code, plan.interval)["rows"],
                    "source": "kis",
                    "reason": "KIS quotation credentials unavailable; auto continues with yfinance",
                    "order_api_called": False,
                }
            )
        if plan.source in {"yfinance", "auto"}:
            steps.append(fetch_yfinance_symbol(code, plan))
        result = merge_result_steps(code, steps, plan.interval)
        results.append(result)
        print(
            f"[{index}/{len(plan.symbols)}] {code} {result['status']} "
            f"rows={result['before_rows']}->{result['after_rows']} order_api_called=False"
        )
        if index < len(plan.symbols) and plan.sleep_sec > 0:
            time.sleep(plan.sleep_sec)

    source_counts: dict[str, int] = {}
    for row in results:
        for key, value in dict(row.get("source_counts") or {}).items():
            source_counts[str(key)] = int(source_counts.get(str(key), 0)) + int(value)

    summary = {
        "created_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "symbol_count": len(plan.symbols),
        "success_count": sum(1 for row in results if row["after_rows"] > 0),
        "collected_count": sum(1 for row in results if row["status"] == "collected"),
        "skip_count": sum(1 for row in results if row["status"] == "skip"),
        "cache_only_count": sum(1 for row in results if row["status"] == "cache_only"),
        "total_cache_rows": sum(int(row["after_rows"]) for row in results),
        "total_added_or_updated_rows": sum(int(row["added_or_updated_rows"]) for row in results),
        "source_counts": source_counts,
        "interval": plan.interval,
        "months_requested": plan.months,
        "period": plan.period,
        "source": plan.source,
        "cache_mode": "idempotent append by timestamp, last value wins",
        "dedup_policy": "same timestamp keeps KIS over unknown over yfinance",
        "order_api_called": False,
    }
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
