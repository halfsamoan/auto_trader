"""Build dynamic AI/orderable universe from KOSPI200 + KOSDAQ representatives."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    AI_TRAIN_UNIVERSE_FALLBACK,
    AI_UNIVERSE_KOSDAQ_TOP_N,
    AI_UNIVERSE_OUTPUT_PATH,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / AI_UNIVERSE_OUTPUT_PATH
EXAMPLE_PATH = ROOT / "data/ai_universe/ai_train_universe.example.json"


def fallback_records() -> list[dict[str, str]]:
    return [
        {"code": code, "name": code, "market": "KR", "source": "fallback"}
        for code in AI_TRAIN_UNIVERSE_FALLBACK
    ]


def _stock_module():
    try:
        from pykrx import stock

        return stock
    except Exception:
        return None


def _latest_business_date(stock) -> str:
    today = datetime.now().strftime("%Y%m%d")
    try:
        tickers = stock.get_market_ticker_list(today, market="KOSPI")
        if tickers:
            return today
    except Exception:
        pass
    return today


def _name(stock, code: str) -> str:
    try:
        return str(stock.get_market_ticker_name(code) or code)
    except Exception:
        return code


def _kospi200_records(stock, date: str) -> tuple[list[dict[str, str]], str | None]:
    try:
        codes = stock.get_index_portfolio_deposit_file("1028", date)
        return [{"code": code, "name": _name(stock, code), "market": "KOSPI", "source": "KOSPI200"} for code in codes], None
    except Exception as exc:
        return [], f"KOSPI200 조회 실패: {exc}"


def _rank_by_value(stock, date: str, codes: list[str], market: str, top_n: int) -> list[str]:
    try:
        ohlcv = stock.get_market_ohlcv_by_ticker(date, market=market)
        if "거래대금" in ohlcv.columns:
            ranked = ohlcv.loc[[code for code in codes if code in ohlcv.index]].sort_values("거래대금", ascending=False)
            return [str(code) for code in ranked.index[:top_n]]
        if "거래량" in ohlcv.columns:
            ranked = ohlcv.loc[[code for code in codes if code in ohlcv.index]].sort_values("거래량", ascending=False)
            return [str(code) for code in ranked.index[:top_n]]
    except Exception:
        pass
    return codes[:top_n]


def _kosdaq_top_records(stock, date: str, top_n: int) -> tuple[list[dict[str, str]], str | None]:
    warnings = []
    try:
        candidates = stock.get_index_portfolio_deposit_file("2203", date)
        source = "KOSDAQ150_TOP50"
    except Exception as exc:
        warnings.append(f"KOSDAQ150 조회 실패: {exc}")
        try:
            candidates = stock.get_market_ticker_list(date, market="KOSDAQ")
            source = "KOSDAQ_TOP50"
        except Exception as inner:
            return [], "; ".join(warnings + [f"KOSDAQ 전체 조회 실패: {inner}"])
    selected = _rank_by_value(stock, date, list(candidates), "KOSDAQ", top_n)
    return [{"code": code, "name": _name(stock, code), "market": "KOSDAQ", "source": source} for code in selected], "; ".join(warnings) or None


def build_ai_universe(write: bool = True) -> dict[str, Any]:
    stock = _stock_module()
    warnings = []
    if stock is None:
        records = fallback_records()
        status = "fallback_pykrx_missing"
    else:
        date = _latest_business_date(stock)
        kospi_records, warning = _kospi200_records(stock, date)
        if warning:
            warnings.append(warning)
        kosdaq_records, warning = _kosdaq_top_records(stock, date, AI_UNIVERSE_KOSDAQ_TOP_N)
        if warning:
            warnings.append(warning)
        records = kospi_records + kosdaq_records
        status = "dynamic" if records else "fallback_query_failed"
        if not records:
            records = fallback_records()
    deduped = []
    seen = set()
    for row in records:
        code = str(row.get("code") or "").zfill(6)
        if not code or code in seen:
            continue
        seen.add(code)
        deduped.append({**row, "code": code})
    payload = {
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(deduped),
        "kospi200_count": sum(1 for row in deduped if row.get("source") == "KOSPI200"),
        "kosdaq_top_count": sum(1 for row in deduped if str(row.get("source", "")).startswith("KOSDAQ")),
        "warnings": warnings,
        "records": deduped,
    }
    if write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
        EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not EXAMPLE_PATH.exists():
            EXAMPLE_PATH.write_text(json.dumps(deduped[:5], ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_ai_universe_records(refresh: bool = False) -> list[dict[str, str]]:
    if refresh or not OUTPUT_PATH.exists():
        return build_ai_universe(write=True)["records"]
    try:
        rows = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if isinstance(rows, list) and rows:
            return rows
    except Exception:
        pass
    return fallback_records()
