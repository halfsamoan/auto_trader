#!/usr/bin/env python3
# ================================================================================
# Description: KIS overseas futures quote parity debug script. No order API calls.
# ================================================================================

"""Run a minimal KIS overseas futures quote request for parity debugging."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
TOKEN_FILE = BASE_DIR / "data" / "token.json"
ENDPOINT = "/uapi/overseas-futureoption/v1/quotations/inquire-price"
TR_ID = "HHDFC55010000"
REQUIRED_HEADER_KEYS = ["authorization", "appkey", "appsecret", "tr_id", "tr_cont", "custtype"]
ORDER_API_CALLED = False
SENSITIVE_PREVIEW_PATTERNS = [
    re.compile(r'(?i)(access_token["\']?\s*[:=]\s*["\']?)([^"\'\s,}]+)'),
    re.compile(r'(?i)(authorization["\']?\s*[:=]\s*["\']?Bearer\s+)([^"\'\s,}]+)'),
    re.compile(r'(?i)(appkey["\']?\s*[:=]\s*["\']?)([^"\'\s,}]+)'),
    re.compile(r'(?i)(appsecret["\']?\s*[:=]\s*["\']?)([^"\'\s,}]+)'),
]


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def paper_base_url() -> str:
    return "https://openapivts.koreainvestment.com:29443"


def redact_preview(text: str, limit: int = 300) -> str:
    preview = str(text)[:limit]
    for pattern in SENSITIVE_PREVIEW_PATTERNS:
        preview = pattern.sub(r"\1<redacted>", preview)
    return preview


def load_cached_token() -> str | None:
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    token = data.get("access_token")
    expires_at_raw = data.get("expires_at")
    if not token or not expires_at_raw:
        return None
    try:
        expires_at = datetime.fromisoformat(str(expires_at_raw))
    except ValueError:
        return None
    if expires_at <= datetime.now() + timedelta(minutes=1):
        return None
    return str(token)


def save_cached_token(token: str, expires_in: int) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    expires_at = datetime.now() + timedelta(seconds=expires_in)
    TOKEN_FILE.write_text(
        json.dumps({"access_token": token, "expires_at": expires_at.isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_access_token(base_url: str) -> str | None:
    cached = load_cached_token()
    if cached:
        print("access token 확인: 성공(cache)")
        return cached

    appkey = os.getenv("KIS_APP_KEY")
    appsecret = os.getenv("KIS_APP_SECRET")
    if not appkey or not appsecret:
        print("access token 확인: 실패")
        print("missing_env_keys=['KIS_APP_KEY', 'KIS_APP_SECRET']")
        return None

    payload = {"grant_type": "client_credentials", "appkey": appkey, "appsecret": appsecret}
    try:
        response = requests.post(
            f"{base_url}/oauth2/tokenP",
            headers={"content-type": "application/json"},
            json=payload,
            timeout=10,
        )
    except requests.RequestException as exc:
        print("access token 확인: 실패")
        print(f"token_request_error={exc}")
        return None

    if not response.ok:
        print("access token 확인: 실패")
        print(f"token_http_status={response.status_code}")
        print(f"token_content_type={response.headers.get('content-type')}")
        print(f"token_response_preview={redact_preview(response.text)!r}")
        return None

    try:
        data = response.json()
    except ValueError:
        print("access token 확인: 실패")
        print("token_response_json_parse=False")
        print(f"token_content_type={response.headers.get('content-type')}")
        print(f"token_response_preview={redact_preview(response.text)!r}")
        return None

    token = data.get("access_token") if isinstance(data, dict) else None
    print(f"access token 확인: {'성공' if token else '실패'}")
    if not token:
        return None
    try:
        expires_in = int(data.get("expires_in", 86400)) if isinstance(data, dict) else 86400
    except (TypeError, ValueError):
        expires_in = 86400
    save_cached_token(str(token), expires_in)
    return str(token)


def build_headers(token: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": os.getenv("KIS_APP_KEY", ""),
        "appsecret": os.getenv("KIS_APP_SECRET", ""),
        "tr_id": TR_ID,
        "tr_cont": "",
        "custtype": "P",
    }


def print_debug_request(base_url: str, params: dict[str, str], headers: dict[str, str]) -> None:
    header_keys = sorted(headers.keys())
    header_key_presence = {key: key in headers for key in REQUIRED_HEADER_KEYS}
    print(f"base_url={base_url}")
    print(f"endpoint={ENDPOINT}")
    print(f"tr_id={TR_ID}")
    print(f"tr_cont={headers.get('tr_cont', '')!r}")
    print(f"params={params}")
    print(f"request_header_keys={header_keys}")
    print(f"required_header_key_presence={header_key_presence}")


def run_quote_request(code: str) -> int:
    base_url = paper_base_url()
    token = get_access_token(base_url)
    if not token:
        print(f"order_api_called={ORDER_API_CALLED}")
        return 1

    params = {"SRS_CD": code}
    headers = build_headers(token)
    print_debug_request(base_url, params, headers)

    try:
        response = requests.get(f"{base_url}{ENDPOINT}", headers=headers, params=params, timeout=10)
    except requests.RequestException as exc:
        print("HTTP status=None")
        print("content-type=None")
        print(f"response_preview={redact_preview(str(exc))!r}")
        print(f"order_api_called={ORDER_API_CALLED}")
        return 1

    print(f"HTTP status={response.status_code}")
    print(f"content-type={response.headers.get('content-type')}")
    print(f"response_preview={redact_preview(response.text)!r}")
    print(f"order_api_called={ORDER_API_CALLED}")
    return 0 if response.ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug KIS overseas futures quote API without order calls.")
    parser.add_argument("--code", default=os.getenv("KIS_FUTURES_MNQ_PRODUCT_CODE"), help="SRS_CD product code")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    mode = os.getenv("KIS_MODE", "paper").lower()
    if mode != "paper":
        print("KIS_MODE=paper 확인: 실패")
        print(f"KIS_MODE={mode}")
        print(f"order_api_called={ORDER_API_CALLED}")
        return 1
    print("KIS_MODE=paper 확인: 성공")

    if not args.code:
        print("SRS_CD 확인: 실패")
        print("KIS_FUTURES_MNQ_PRODUCT_CODE 또는 --code 값이 필요합니다.")
        print(f"order_api_called={ORDER_API_CALLED}")
        return 1

    return run_quote_request(str(args.code).strip().upper())


if __name__ == "__main__":
    sys.exit(main())
