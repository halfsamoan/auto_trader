#!/usr/bin/env python3
"""Safe scaffold for KIS domestic-stock tick/orderbook collection.

This script does not call order APIs. By default it runs in safe mode and only
prepares cache files/directories plus resampling helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ai.universe_builder import load_ai_universe_records
from config import ORDERBOOK_CACHE_DIR, TICK_CACHE_DIR, WATCHLIST
from core.fetcher_intraday import merge_intraday_cache


TICK_DIR = ROOT / TICK_CACHE_DIR
QUOTE_DIR = ROOT / ORDERBOOK_CACHE_DIR


def resolve_universe(universe: str, watchlist: str | None) -> list[str]:
    if watchlist:
        return [code.strip().zfill(6) for code in watchlist.split(",") if code.strip()]
    if universe == "ai_train":
        return [str(row["code"]).zfill(6) for row in load_ai_universe_records(refresh=False)]
    return [str(item["code"]).zfill(6) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]


def append_tick(code: str, price: float, volume: float = 0.0, ts: str | None = None, source: str = "kis_websocket") -> Path:
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    path = TICK_DIR / f"{code}_ticks.csv"
    row = pd.DataFrame(
        [
            {
                "timestamp": ts or datetime.now().isoformat(timespec="seconds"),
                "code": code,
                "price": float(price),
                "volume": float(volume),
                "source": source,
                "collected_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", index=False, header=header)
    return path


def append_quote(code: str, bid_price: float | None = None, ask_price: float | None = None, ts: str | None = None, source: str = "kis_websocket") -> Path:
    QUOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUOTE_DIR / f"{code}_quote.csv"
    row = pd.DataFrame(
        [
            {
                "timestamp": ts or datetime.now().isoformat(timespec="seconds"),
                "code": code,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "source": source,
                "collected_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", index=False, header=header)
    return path


def resample_ticks_to_bars(code: str, interval: str = "5m", save_intraday: bool = False) -> pd.DataFrame:
    path = TICK_DIR / f"{code}_ticks.csv"
    if not path.exists():
        return pd.DataFrame()
    ticks = pd.read_csv(path, parse_dates=["timestamp"])
    if ticks.empty or "price" not in ticks.columns:
        return pd.DataFrame()
    ticks = ticks.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], errors="coerce")
    ticks = ticks.dropna(subset=["timestamp"])
    if ticks["timestamp"].dt.tz is None:
        ticks["timestamp"] = ticks["timestamp"].dt.tz_localize("Asia/Seoul")
    else:
        ticks["timestamp"] = ticks["timestamp"].dt.tz_convert("Asia/Seoul")
    freq = "1min" if interval == "1m" else "5min"
    frame = pd.DataFrame(
        {
            "price": pd.to_numeric(ticks["price"], errors="coerce"),
            "volume": pd.to_numeric(ticks.get("volume", 0), errors="coerce").fillna(0),
        },
        index=ticks["timestamp"],
    )
    bars = frame.resample(freq, origin="start_day", label="left", closed="left").agg(
        {"price": ["first", "max", "min", "last"], "volume": "sum"}
    )
    if bars.empty:
        return pd.DataFrame()
    bars.columns = ["Open", "High", "Low", "Close", "Volume"]
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    if save_intraday and interval == "5m" and not bars.empty:
        merge_intraday_cache(code, bars, "5m", source="kis_websocket_cache")
    return bars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="watchlist", choices=["ai_train", "watchlist"])
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--safe-mode", action="store_true", default=True)
    parser.add_argument("--resample-only", action="store_true")
    parser.add_argument("--save-5m", action="store_true")
    args = parser.parse_args()

    codes = resolve_universe(args.universe, args.watchlist)
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    QUOTE_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "safe_scaffold",
        "order_api_called": False,
        "websocket_connected": False,
        "message": "KIS WebSocket 실시간 수집 연결은 아직 활성화하지 않았습니다. tick/quote cache 구조와 resample 함수만 준비했습니다.",
        "symbol_count": len(codes),
        "tick_cache_dir": str(TICK_DIR),
        "quote_cache_dir": str(QUOTE_DIR),
        "resampled": {},
    }
    if args.resample_only:
        for code in codes:
            bars = resample_ticks_to_bars(code, interval="5m", save_intraday=args.save_5m)
            result["resampled"][code] = int(len(bars))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
