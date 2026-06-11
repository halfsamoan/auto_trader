# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: None
# Description: KIS 공통 인증, 토큰 캐시, 헤더, 안전 게이트를 제공합니다.
# ================================================================================

"""Shared KIS Open API client primitives."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
TOKEN_FILE = DATA_DIR / "token.json"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class KISBaseClient:
    PAPER_URL = "https://openapivts.koreainvestment.com:29443"
    REAL_URL = "https://openapi.koreainvestment.com:9443"

    # KISBaseClient 초기화: 공통 환경변수와 안전 플래그를 보관합니다.
    def __init__(
        self,
        mode: str | None = None,
        dry_run: bool = True,
        live: bool = False,
        allow_paper_order: bool = False,
        rate_limit_sleep_sec: float = 0.2,
    ) -> None:
        self.mode = (mode or os.getenv("KIS_MODE", "paper")).lower()
        self.dry_run = dry_run
        self.live = live
        self.allow_paper_order = allow_paper_order
        self.rate_limit_sleep_sec = rate_limit_sleep_sec
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.account_no = os.getenv("KIS_ACCOUNT_NO")
        self.account_product_code = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "01")
        self.base_url = self.PAPER_URL if self.mode == "paper" else self.REAL_URL

    # validate_env는 민감정보를 출력하지 않고 필수 환경변수 존재 여부만 반환합니다.
    def validate_env(self) -> tuple[bool, str | None]:
        if not ENV_PATH.exists():
            return False, ".env 파일이 없습니다. .env.example을 복사해 사용자가 직접 값을 채워 주세요."
        missing = [
            name
            for name, value in {
                "KIS_APP_KEY": self.app_key,
                "KIS_APP_SECRET": self.app_secret,
                "KIS_ACCOUNT_NO": self.account_no,
            }.items()
            if not value
        ]
        if missing:
            return False, f"필수 KIS 환경변수가 없습니다: {', '.join(missing)}"
        return True, None

    # can_order는 dry-run/paper/live 안전 게이트를 모두 통과했는지 판단합니다.
    def can_order(self) -> bool:
        if self.dry_run:
            return False
        if self.mode == "paper" and not self.allow_paper_order:
            return False
        if self.mode == "real" and not self.live:
            return False
        ok, _ = self.validate_env()
        return ok

    # print_mode_summary는 민감정보 없이 실행 모드와 차단 상태만 출력합니다.
    def print_mode_summary(self) -> None:
        print(f"KIS 모드: {self.mode}")
        print(f"KIS URL: {self.base_url}")
        print(f"주문 가능 여부: {'가능' if self.can_order() else '차단'}")

    # rate_limit은 KIS 호출 사이 최소 대기시간을 둡니다.
    def rate_limit(self) -> None:
        if self.rate_limit_sleep_sec > 0:
            time.sleep(self.rate_limit_sleep_sec)

    # headers는 KIS REST 호출에 필요한 공통 헤더를 구성합니다.
    def headers(self, token: str | None = None, tr_id: str | None = None) -> dict[str, str]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.app_key or "",
            "appsecret": self.app_secret or "",
        }
        if token:
            headers["authorization"] = f"Bearer {token}"
        if tr_id:
            headers["tr_id"] = tr_id
        return headers

    # get_access_token은 캐시된 token을 재사용하거나 새 token을 발급합니다.
    def get_access_token(self) -> str | None:
        ok, msg = self.validate_env()
        if not ok:
            print(msg)
            return None

        cached = self._load_token()
        if cached and cached.get("access_token") and cached.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(cached["expires_at"])
                if expires_at > datetime.now() + timedelta(hours=1):
                    return str(cached["access_token"])
            except ValueError:
                pass

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        self.rate_limit()
        response = requests.post(url, headers={"content-type": "application/json"}, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 86400))
        if not token:
            raise RuntimeError("KIS access_token 응답이 없습니다.")
        self._save_token(token, datetime.now() + timedelta(seconds=expires_in))
        return str(token)

    # blocked_order는 주문 API를 호출하지 않고 차단 결과만 반환합니다.
    def blocked_order(self, side: str, symbol: str, qty: int, limit_price: float, reason: str) -> dict[str, Any]:
        return {
            "blocked": True,
            "side": side,
            "symbol": symbol,
            "qty": int(qty),
            "limit_price": float(limit_price),
            "reason": reason,
            "order_api_called": False,
        }

    # _load_token은 data/token.json에서 token 캐시를 읽습니다.
    def _load_token(self) -> dict[str, Any] | None:
        if not TOKEN_FILE.exists():
            return None
        try:
            data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    # _save_token은 data/token.json에 token 캐시를 저장합니다.
    def _save_token(self, token: str, expires_at: datetime) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(
            json.dumps({"access_token": token, "expires_at": expires_at.isoformat()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
