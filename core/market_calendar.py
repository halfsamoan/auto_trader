# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: None
# Description: KRX 장 운영 시간과 장중 여부를 판정합니다.
# ================================================================================

"""Simple KRX market time helpers without external calendar dependency."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
OPEN_TIME = time(9, 0)
CLOSE_TIME = time(15, 30)


def to_kst(dt: datetime | None = None) -> datetime:
    now = dt or datetime.now(tz=KST)
    if now.tzinfo is None:
        return now.replace(tzinfo=KST)
    return now.astimezone(KST)


def is_market_open(dt: datetime | None = None) -> bool:
    current = to_kst(dt)
    if current.weekday() >= 5:
        return False
    return OPEN_TIME <= current.time() <= CLOSE_TIME


def next_open(dt: datetime | None = None) -> datetime:
    current = to_kst(dt)
    candidate = current
    if candidate.time() >= OPEN_TIME:
        candidate = candidate + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate.replace(hour=9, minute=0, second=0, microsecond=0)
