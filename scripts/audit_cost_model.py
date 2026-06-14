#!/usr/bin/env python3
# ================================================================================
# Main Author: Codex
# Recently Modified Date: 2026-06-14 (V3.2.7)
# Dependency: config.py, futures_contracts.py
# Description: Offline cost-model audit without credentials or order calls.
# ================================================================================

"""Audit domestic-stock and futures paper-lab cost assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AI_STOP_RETURN, AI_TARGET_RETURN, COMMISSION, INTRADAY_CACHE_DIR, TAX
from futures_contracts import FUTURES_CONTRACTS


DEFAULT_OUTPUT = ROOT / "ai" / "model_store" / "cost_model_audit_report.json"
CACHE_DIR = ROOT / INTRADAY_CACHE_DIR


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def breakeven_precision(target_return: float, stop_return: float, cost_return: float) -> float | None:
    denominator = target_return - stop_return
    if denominator <= 0:
        return None
    return float((cost_return - stop_return) / denominator)


def cost_bps_futures(
    price: float,
    multiplier: float,
    tick_value_usd: float,
    slippage_ticks: float = 1.0,
    commission_per_side_usd: float = 0.0,
) -> float | None:
    """Return futures round-trip trading cost in basis points.

    Formula:
    (2 * slippage_ticks * tick_value_usd + 2 * commission_per_side_usd)
    / (price * multiplier) * 10_000

    This additive helper does not change the existing domestic cost audit path.
    """

    notional = float(price) * float(multiplier)
    if notional <= 0:
        return None
    cost_usd = 2.0 * float(slippage_ticks) * float(tick_value_usd) + 2.0 * float(commission_per_side_usd)
    return float(cost_usd / notional * 10_000.0)


def domestic_cost_scenarios(extra_bps_values: list[float]) -> list[dict[str, Any]]:
    base_round_trip = 2.0 * float(COMMISSION) + float(TAX)
    rows = []
    for extra_bps in extra_bps_values:
        extra_return = float(extra_bps) / 10_000.0
        total_cost = base_round_trip + extra_return
        rows.append(
            {
                "name": f"domestic_base_plus_{extra_bps:g}bps",
                "commission_round_trip_return": float(2.0 * COMMISSION),
                "tax_return": float(TAX),
                "extra_slippage_spread_return": extra_return,
                "total_round_trip_cost_return": total_cost,
                "breakeven_precision_at_config_barriers": breakeven_precision(AI_TARGET_RETURN, AI_STOP_RETURN, total_cost),
                "target_return": float(AI_TARGET_RETURN),
                "stop_return": float(AI_STOP_RETURN),
            }
        )
    return rows


def futures_cost_scenarios(round_trip_ticks: list[float], commission_usd_values: list[float]) -> list[dict[str, Any]]:
    rows = []
    for symbol, contract in sorted(FUTURES_CONTRACTS.items()):
        tick_value = float(contract["tick_value_usd"])
        tick_size = float(contract["tick_size"])
        for ticks in round_trip_ticks:
            for commission_usd in commission_usd_values:
                rows.append(
                    {
                        "symbol": symbol,
                        "name": contract.get("name"),
                        "tick_size": tick_size,
                        "tick_value_usd": tick_value,
                        "round_trip_slippage_ticks": float(ticks),
                        "round_trip_slippage_usd": float(ticks) * tick_value,
                        "round_trip_commission_usd": float(commission_usd),
                        "round_trip_total_cost_usd": float(ticks) * tick_value + float(commission_usd),
                        "status": "EDGE_CANDIDATE_REQUIRES_SAME_ASSET_VALIDATION",
                        "note": "Futures return-space cost needs same-asset price/notional validation before use.",
                    }
                )
    return rows


def _parse_symbol_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _latest_futures_cache_price(symbol: str, interval: str) -> dict[str, Any]:
    path = CACHE_DIR / f"{symbol.upper()}_{interval}.csv"
    if not path.exists():
        return {"status": "missing_cache", "cache_path": str(path), "price": None}
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        return {"status": "read_error", "cache_path": str(path), "error": str(exc), "price": None}
    if raw.empty or "close" not in raw.columns:
        return {"status": "empty_or_missing_close", "cache_path": str(path), "price": None}
    raw["timestamp"] = pd.to_datetime(raw.get("timestamp"), errors="coerce", utc=True)
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if raw.empty:
        return {"status": "no_valid_close", "cache_path": str(path), "price": None}
    latest = raw.iloc[-1]
    return {
        "status": "ok",
        "cache_path": str(path),
        "row_count": int(len(raw)),
        "price": float(latest["close"]),
        "timestamp_utc": pd.Timestamp(latest["timestamp"]).isoformat(),
    }


def futures_return_cost_scenarios(
    symbols: list[str],
    interval: str,
    slippage_ticks_values: list[float],
    commission_per_side_values: list[float],
) -> dict[str, Any]:
    rows = []
    by_symbol: dict[str, Any] = {}
    for symbol in symbols:
        contract = FUTURES_CONTRACTS.get(symbol)
        price_info = _latest_futures_cache_price(symbol, interval)
        if not contract:
            by_symbol[symbol] = {"status": "missing_contract", "price_info": price_info}
            continue
        tick_size = float(contract["tick_size"])
        tick_value = float(contract["tick_value_usd"])
        multiplier = float(contract.get("multiplier") or tick_value / tick_size)
        symbol_rows = []
        for slippage_ticks in slippage_ticks_values:
            for commission_per_side in commission_per_side_values:
                price = price_info.get("price")
                bps = (
                    cost_bps_futures(
                        float(price),
                        multiplier,
                        tick_value,
                        slippage_ticks=slippage_ticks,
                        commission_per_side_usd=commission_per_side,
                    )
                    if price is not None
                    else None
                )
                cost_return = None if bps is None else float(bps) / 10_000.0
                row = {
                    "symbol": symbol,
                    "name": contract.get("name"),
                    "price": price,
                    "price_timestamp_utc": price_info.get("timestamp_utc"),
                    "tick_size": tick_size,
                    "tick_value_usd": tick_value,
                    "multiplier": multiplier,
                    "slippage_ticks_per_side": float(slippage_ticks),
                    "commission_per_side_usd": float(commission_per_side),
                    "round_trip_slippage_usd": float(2.0 * slippage_ticks * tick_value),
                    "round_trip_commission_usd": float(2.0 * commission_per_side),
                    "round_trip_total_cost_usd": float(2.0 * slippage_ticks * tick_value + 2.0 * commission_per_side),
                    "cost_bps": bps,
                    "cost_return": cost_return,
                    "target_return": float(AI_TARGET_RETURN),
                    "stop_return": float(AI_STOP_RETURN),
                    "breakeven_precision_at_config_barriers": None
                    if cost_return is None
                    else breakeven_precision(AI_TARGET_RETURN, AI_STOP_RETURN, cost_return),
                    "tax_return": 0.0,
                }
                rows.append(row)
                symbol_rows.append(row)
        by_symbol[symbol] = {
            "status": "ok" if price_info.get("status") == "ok" else "INSUFFICIENT_FUTURES_DATA",
            "contract": contract,
            "price_info": price_info,
            "scenarios": symbol_rows,
        }
    return {
        "symbols": symbols,
        "interval": interval,
        "scenario_rows": rows,
        "by_symbol": by_symbol,
        "status": "ok" if all(row.get("status") == "ok" for row in by_symbol.values()) else "INSUFFICIENT_FUTURES_DATA",
        "formula": "(2*slippage_ticks*tick_value_usd + 2*commission_per_side_usd) / (price*multiplier)",
        "note": "TAX=0 for futures scenarios; costs are measured from latest local yfinance futures cache close.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    domestic_rows = domestic_cost_scenarios(_parse_float_list(args.domestic_extra_bps))
    futures_rows = futures_cost_scenarios(_parse_float_list(args.futures_round_trip_ticks), _parse_float_list(args.futures_commission_usd))
    futures_return_rows = futures_return_cost_scenarios(
        _parse_symbol_list(args.futures_symbols),
        args.futures_interval,
        _parse_float_list(args.futures_slippage_ticks),
        _parse_float_list(args.futures_commission_per_side_usd),
    )
    base_cost = 2.0 * float(COMMISSION) + float(TAX)
    base_breakeven = breakeven_precision(float(AI_TARGET_RETURN), float(AI_STOP_RETURN), base_cost)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "order_api_called": False,
        "env_loaded": False,
        "cost_source": "config.py constants only",
        "domestic_stock": {
            "commission": float(COMMISSION),
            "tax": float(TAX),
            "base_round_trip_cost_return": base_cost,
            "target_return": float(AI_TARGET_RETURN),
            "stop_return": float(AI_STOP_RETURN),
            "breakeven_precision": base_breakeven,
            "breakeven_precision_matches_prompt_0_767": bool(base_breakeven is not None and abs(base_breakeven - 0.767) < 0.002),
            "scenarios": domestic_rows,
        },
        "futures": {
            "contracts": futures_rows,
            "return_cost_scenarios": futures_return_rows,
            "validation_status": "EDGE_CANDIDATE_REQUIRES_SAME_ASSET_VALIDATION",
        },
        "v327_allowed_edge_statuses": [
            "EDGE_NOT_CONFIRMED",
            "COST_SENSITIVITY_POSITIVE_ONLY",
            "FUTURES_EDGE_CANDIDATE",
            "INSUFFICIENT_FUTURES_DATA",
            "NO_GEOMETRY_EDGE",
        ],
        "allowed_edge_statuses_only": [
            "EDGE_NOT_CONFIRMED",
            "COST_SENSITIVITY_POSITIVE_ONLY",
            "EDGE_CANDIDATE_REQUIRES_SAME_ASSET_VALIDATION",
            "META_EDGE_CANDIDATE",
            "GEOMETRY_CANDIDATE_DOMESTIC",
            "NO_GEOMETRY_EDGE",
            "INSUFFICIENT_TRADES",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domestic-extra-bps", default="0,5,10,20")
    parser.add_argument("--futures-round-trip-ticks", default="1,2,4")
    parser.add_argument("--futures-commission-usd", default="0,0.62,1.24")
    parser.add_argument("--futures-symbols", default="MNQ,MES")
    parser.add_argument("--futures-interval", default="5m")
    parser.add_argument("--futures-slippage-ticks", default="1,2,3")
    parser.add_argument("--futures-commission-per-side-usd", default="0,0.62,1.24")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
