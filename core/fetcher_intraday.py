# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: core/fetcher_daily.py, core/technical.py
# Description: yfinance 장중 데이터를 조회하고 미완성 5분봉을 제거합니다.
# ================================================================================

"""Intraday data fetcher with KIS primary, websocket-cache, and yfinance fallback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clients.kis_domestic_stock_client import DomesticStockClient
from config import INTRADAY_CACHE_DIR, TICK_CACHE_DIR
from core.technical import ensure_ohlcv
from .fetcher_daily import _download_krx


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / INTRADAY_CACHE_DIR
TICK_CACHE_DIR_PATH = BASE_DIR / TICK_CACHE_DIR


# _interval_to_timedelta는 yfinance interval 문자열을 pandas Timedelta로 변환합니다.
def _interval_to_timedelta(interval: str) -> pd.Timedelta | None:
    try:
        unit = interval[-1]
        value = int(interval[:-1])
    except (ValueError, IndexError):
        return None
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    return None


# drop_incomplete_bar는 현재 시각 기준으로 아직 닫히지 않은 마지막 봉을 제거합니다.
def drop_incomplete_bar(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        return data
    delta = _interval_to_timedelta(interval)
    if delta is None:
        return data
    last_ts = data.index[-1]
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo is not None else pd.Timestamp.now()
    # yfinance intraday index는 봉 시작 시각이므로 종료 시각 전이면 최신 봉은 지표 계산에서 제외합니다.
    if last_ts + delta > now:
        return data.iloc[:-1].copy()
    return data


def _cache_path(code: str, interval: str) -> Path:
    return CACHE_DIR / f"{code}_{interval}.csv"


def _tick_cache_path(code: str) -> Path:
    return TICK_CACHE_DIR_PATH / f"{code}_ticks.csv"


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    bars = ensure_ohlcv(data)
    if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return bars
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("Asia/Seoul")
    else:
        bars.index = bars.index.tz_convert("Asia/Seoul")
    return bars.sort_index()


def _read_cache_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        return data.dropna(subset=["timestamp"])
    if "Datetime" in data.columns:
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
        return data.dropna(subset=["Datetime"]).rename(columns={"Datetime": "timestamp"})
    return pd.DataFrame()


def load_intraday_cache_raw(code: str, interval: str = "5m") -> pd.DataFrame:
    return _read_cache_csv(_cache_path(code, interval))


def load_intraday_cache(code: str, interval: str = "5m") -> pd.DataFrame:
    data = load_intraday_cache_raw(code, interval)
    if "timestamp" not in data.columns:
        return pd.DataFrame()
    return _normalize_index(data.set_index("timestamp"))


def save_intraday_cache(code: str, data: pd.DataFrame, interval: str = "5m", source: str | None = None) -> None:
    bars = _normalize_index(data)
    if bars.empty:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = bars.reset_index().rename(columns={bars.index.name or "index": "timestamp"})
    out = out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    out["code"] = str(code).zfill(6)
    existing = data.copy()
    if "source" in existing.columns and source is None:
        try:
            source_map = existing["source"]
            if isinstance(existing.index, pd.DatetimeIndex):
                source_map = source_map.reindex(bars.index)
            out["source"] = source_map.to_numpy()
        except Exception:
            out["source"] = None
    else:
        out["source"] = source or "unknown"
    out["source"] = out["source"].fillna(source or "unknown")
    out["collected_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    out = out[["timestamp", "open", "high", "low", "close", "volume", "code", "source", "collected_at"]]
    out.to_csv(_cache_path(code, interval), index=False)


def merge_intraday_cache(code: str, fresh: pd.DataFrame, interval: str = "5m", source: str | None = None) -> pd.DataFrame:
    cached = load_intraday_cache(code, interval)
    fresh_bars = _normalize_index(fresh)
    if not fresh_bars.empty:
        fresh_bars["source"] = source or (str(fresh.get("source").iloc[-1]) if "source" in fresh.columns and len(fresh) else "unknown")
    frames = [frame for frame in [cached, fresh_bars] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    save_intraday_cache(code, merged, interval, source=None)
    return merged


def _fetch_kis_intraday(code: str, interval: str) -> pd.DataFrame:
    if interval != "5m":
        return pd.DataFrame()
    client = DomesticStockClient(mode="paper", dry_run=True, live=False, allow_paper_order=False)
    return client.get_intraday_5m_chart(code)


def _fetch_websocket_cache_intraday(code: str, interval: str) -> pd.DataFrame:
    tick_path = _tick_cache_path(code)
    if not tick_path.exists():
        return pd.DataFrame()
    try:
        ticks = pd.read_csv(tick_path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()
    if ticks.empty or "timestamp" not in ticks.columns or "price" not in ticks.columns:
        return pd.DataFrame()
    ticks = ticks.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], errors="coerce")
    ticks = ticks.dropna(subset=["timestamp"])
    if ticks["timestamp"].dt.tz is None:
        ticks["timestamp"] = ticks["timestamp"].dt.tz_localize("Asia/Seoul")
    else:
        ticks["timestamp"] = ticks["timestamp"].dt.tz_convert("Asia/Seoul")
    freq = "5min" if interval == "5m" else "1min" if interval == "1m" else None
    if freq is None:
        return pd.DataFrame()
    price = pd.to_numeric(ticks["price"], errors="coerce")
    volume = pd.to_numeric(ticks.get("volume", 0), errors="coerce").fillna(0)
    frame = pd.DataFrame({"price": price, "volume": volume}, index=ticks["timestamp"])
    bars = frame.resample(freq, origin="start_day", label="left", closed="left").agg(
        {"price": ["first", "max", "min", "last"], "volume": "sum"}
    )
    if bars.empty:
        return pd.DataFrame()
    bars.columns = ["Open", "High", "Low", "Close", "Volume"]
    return ensure_ohlcv(bars.dropna(subset=["Open", "High", "Low", "Close"]))


def _fetch_yfinance_intraday(code: str, period: str, interval: str) -> pd.DataFrame:
    return _download_krx(code, period=period, interval=interval)


# fetch_intraday는 국내 종목 장중 데이터를 조회하고 옵션에 따라 미완성 봉을 제거합니다.
def fetch_intraday(
    code: str,
    period: str = "5d",
    interval: str = "5m",
    drop_incomplete: bool = True,
    source: str = "auto",
    use_cache: bool = True,
) -> pd.DataFrame:
    data = pd.DataFrame()
    errors = []
    if source == "cache":
        data = load_intraday_cache(code, interval)
    if source in {"auto", "kis"}:
        try:
            data = _fetch_kis_intraday(code, interval)
            if use_cache and not data.empty:
                data = merge_intraday_cache(code, data, interval, source="kis_intraday")
        except Exception as exc:
            errors.append(f"kis_intraday:{exc}")
    if data.empty and source in {"auto", "websocket"}:
        try:
            data = _fetch_websocket_cache_intraday(code, interval)
            if use_cache and not data.empty:
                data = merge_intraday_cache(code, data, interval, source="kis_websocket_cache")
        except Exception as exc:
            errors.append(f"kis_websocket_cache:{exc}")
    if data.empty and source in {"auto", "yfinance"}:
        try:
            data = _fetch_yfinance_intraday(code, period, interval)
            if use_cache and not data.empty:
                data = merge_intraday_cache(code, data, interval, source="yfinance_fallback")
        except Exception as exc:
            errors.append(f"yfinance_fallback:{exc}")
    if data.empty and use_cache:
        data = load_intraday_cache(code, interval)
    if data.empty:
        suffix = f" ({'; '.join(errors)})" if errors else ""
        raise ValueError(f"{code} {interval} intraday 데이터를 가져오지 못했습니다.{suffix}")
    data = _normalize_index(data)
    return drop_incomplete_bar(data, interval) if drop_incomplete else data
