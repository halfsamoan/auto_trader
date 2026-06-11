# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: 해외선물 paper-sim용 거래 가능 시간 필터를 제공합니다.
# ================================================================================

"""Simple overseas futures trading-time helpers."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


# is_futures_trading_time은 KST 기준 해외선물 거래 가능 시간 여부를 반환합니다.
def is_futures_trading_time(now_kst: datetime | None = None) -> bool:
    now = _kst_now(now_kst)
    if now.weekday() == 5:
        return False
    if now.weekday() == 6 and now.time() < time(18, 0):
        return False
    if time(6, 0) <= now.time() < time(7, 0):
        # CME/KIS 정확한 정산/점검 시간은 상품별로 다르므로 MVP에서는 보수적 1시간 break를 둡니다.
        return False
    return True


# is_futures_entry_time은 신규 진입 가능 여부를 반환합니다.
def is_futures_entry_time(now_kst: datetime | None = None) -> bool:
    return is_futures_trading_time(now_kst)


# _kst_now는 입력 시간이 없거나 timezone이 없을 때 Asia/Seoul 기준 datetime을 반환합니다.
def _kst_now(now_kst: datetime | None) -> datetime:
    if now_kst is None:
        return datetime.now(ZoneInfo("Asia/Seoul"))
    if now_kst.tzinfo is None:
        return now_kst.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    return now_kst.astimezone(ZoneInfo("Asia/Seoul"))
