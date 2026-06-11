# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: config.py, clients/*, core/*, paper_simulator.py
# Description: 국내주식과 선물 V3.2 Paper Lab 실행 모드를 제공합니다.
# ================================================================================

"""Main entry point for KR domestic stocks and futures Paper Lab."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any

import yfinance as yf

from clients.kis_domestic_stock_client import DomesticStockClient
from clients.kis_futures_client import FuturesClient
from config import (
    DOMESTIC_STOCK_AMOUNT_BY_CODE,
    DOMESTIC_STOCK_CAPITAL_KRW,
    ENABLE_DOMESTIC_STOCK_PAPER_ORDER,
    ENABLE_FUTURES_KIS_PAPER_ORDER,
    ENABLE_FUTURES_PAPER_SIM,
    ENABLE_REAL_ORDER,
    FUTURES_PAPER_CAPITAL_KRW,
    FUTURES_WATCHLIST,
    FX_RATE_USDKRW,
    WATCHLIST,
)
from core.fetcher_daily import fetch_daily
from core.fetcher_intraday import fetch_intraday
from core.futures_market_time import is_futures_entry_time
from core.gaussian_score_engine import GaussianScoreEngine
from core.intraday_scorer import compute_intraday_details
from core.market_regime import get_current_market_regime
from core.strategy import calculate_intraday_plan
from core.technical import ensure_ohlcv, sma
from futures_contracts import get_contract
from logger import log_signal
from paper_simulator import PaperSimulator
from risk_manager import RiskManager

BUY_RULES = {"final_score": 75, "intraday_score": 72, "gaussian_score": 60}


# decide_signal은 점수와 차단 조건으로 buy/hold/avoid를 반환합니다.
def decide_signal(final_score: float, intraday_score: float, gaussian_score: float, chasing_filter: bool, strategy_skip: bool, risk_allowed: bool) -> str:
    if (
        final_score >= BUY_RULES["final_score"]
        and intraday_score >= BUY_RULES["intraday_score"]
        and gaussian_score >= BUY_RULES["gaussian_score"]
        and not chasing_filter
        and not strategy_skip
        and risk_allowed
    ):
        return "buy"
    if final_score >= 55:
        return "hold"
    return "avoid"


# extract_orderable_cash는 KIS 잔고 응답에서 주문 가능 현금을 보수적으로 추출합니다.
def extract_orderable_cash(balance: dict[str, Any] | None) -> float:
    if not balance:
        return 0.0
    candidates = []
    output2 = balance.get("output2")
    if isinstance(output2, list) and output2:
        candidates.append(output2[0])
    elif isinstance(output2, dict):
        candidates.append(output2)
    output = balance.get("output")
    if isinstance(output, dict):
        candidates.append(output)
    for item in candidates:
        for key in ["ord_psbl_cash", "dnca_tot_amt", "nass_amt", "tot_evlu_amt"]:
            value = item.get(key)
            if value not in (None, ""):
                try:
                    return float(str(value).replace(",", ""))
                except ValueError:
                    continue
    return 0.0


# domestic_stock_signal은 국내주식 한 종목의 점수, 레짐, 수량을 계산합니다.
def domestic_stock_signal(code: str, mode: str, account_value: float, available_cash: float | None = None) -> dict[str, Any]:
    engine = GaussianScoreEngine()
    risk_manager = RiskManager(amount_per_trade=DOMESTIC_STOCK_AMOUNT_BY_CODE.get(code, 1_000_000))
    regime_info = get_current_market_regime(use_cache=True)
    daily_df = fetch_daily(code, period="1y")
    intraday_df = fetch_intraday(code, period="5d", interval="5m")
    gaussian = engine.compute(code, daily_df)
    intraday = compute_intraday_details(intraday_df)
    strategy = calculate_intraday_plan(intraday_df)
    final_score = 0.70 * intraday.score + 0.30 * gaussian["gaussian_score"]
    risk_allowed, risk_reason = risk_manager.entry_check(code, intraday_df)
    reasons = []
    if regime_info.get("regime") == "risk_off":
        risk_allowed = False
        risk_reason = "KOSPI risk_off 신규매수 금지"
    sizing = risk_manager.calculate_position_size(strategy.entry, strategy.stop_loss, account_value=account_value, available_cash=available_cash)
    if int(sizing["qty"]) < 1:
        risk_allowed = False
        reasons.append(str(sizing["sizing_skip_reason"] or "포지션 사이징 수량 1주 미만"))
    if intraday.forced_hold_reason:
        reasons.append(intraday.forced_hold_reason)
    if strategy.skip_reason:
        reasons.append(strategy.skip_reason)
    if risk_reason:
        reasons.append(risk_reason)
    signal = decide_signal(final_score, intraday.score, gaussian["gaussian_score"], intraday.chasing_filter, strategy.skip, risk_allowed)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "asset_class": "domestic-stock",
        "symbol_or_code": code,
        "code": code,
        "market": "KR",
        "currency": "KRW",
        "mode": mode,
        "regime": regime_info.get("regime"),
        "final_score": round(final_score, 2),
        "intraday_score": round(intraday.score, 2),
        "gaussian_score": round(gaussian["gaussian_score"], 2),
        "strategy": strategy.to_dict(),
        "qty": sizing["qty"],
        "qty_by_risk": sizing["qty_by_risk"],
        "qty_by_amount": sizing["qty_by_amount"],
        "signal": signal,
        "reasons": reasons,
        "capital_krw": account_value,
        "order_api_called": False,
        "is_order_allowed": False,
    }


# run_domestic_stock_dry_run은 국내주식 dry-run 신호를 출력합니다.
def run_domestic_stock_dry_run() -> int:
    print("국내주식 dry-run을 시작합니다.")
    for item in WATCHLIST:
        code = item["code"]
        try:
            record = domestic_stock_signal(code, "dry-run", DOMESTIC_STOCK_CAPITAL_KRW)
            record["order_block_reason"] = "dry-run 주문 없음"
            log_signal(record)
            print(f"{code}: final {record['final_score']:.2f}, qty {record['qty']}, signal {record['signal']}, regime {record['regime']}")
            for reason in record["reasons"]:
                print(f"  사유: {reason}")
        except Exception as exc:
            print(f"{code}: dry-run 실패 - {exc}")
            log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "domestic-stock", "code": code, "mode": "dry-run", "signal": "error", "error": str(exc)})
    return 0


# run_domestic_stock_paper_check는 국내주식 KIS paper 조회만 검증합니다.
def run_domestic_stock_paper_check() -> int:
    trader = DomesticStockClient(mode=None, dry_run=True, live=False, allow_paper_order=False)
    trader.print_mode_summary()
    ok, msg = trader.validate_env()
    if not ok:
        print(f"paper-check 실행 불가: {msg}")
        return 2
    if trader.mode != "paper" or os.getenv("KIS_MODE", "paper").lower() != "paper":
        print("paper-check 실행 불가: KIS_MODE=paper 인 경우만 허용됩니다.")
        return 2
    try:
        token = trader.get_access_token()
        print(f"access token 확인: {'성공' if token else '실패'}")
        balance = trader.get_balance()
        cash = extract_orderable_cash(balance)
        print(f"주문가능금액: {cash:,.0f}원")
    except Exception as exc:
        print(f"KIS 국내주식 paper-check 실패: {exc}")
        return 2
    for item in WATCHLIST:
        code = item["code"]
        try:
            price = trader.get_price(code)
            print(f"{code}: 현재가 조회 {'성공' if price else '실패'}")
        except Exception as exc:
            print(f"{code}: 현재가 조회 실패 - {exc}")
    print("paper-check 완료: 주문 API는 호출하지 않았습니다.")
    return 0


# run_domestic_stock_paper_order는 조건 충족 시 국내주식 paper 지정가 주문을 호출할 수 있습니다.
def run_domestic_stock_paper_order(allow_paper_order: bool, live: bool) -> int:
    trader = DomesticStockClient(mode="paper", dry_run=not allow_paper_order, live=live, allow_paper_order=allow_paper_order)
    if not ENABLE_DOMESTIC_STOCK_PAPER_ORDER:
        print("국내주식 paper 주문 차단: ENABLE_DOMESTIC_STOCK_PAPER_ORDER=False")
        return 0
    if not allow_paper_order:
        print("국내주식 paper 주문 차단: --allow-paper-order가 없습니다.")
        return 0
    if trader.mode != "paper":
        print("국내주식 paper 주문 차단: KIS_MODE=paper가 아닙니다.")
        return 2
    if live and not ENABLE_REAL_ORDER:
        print("실전 주문 차단: ENABLE_REAL_ORDER=False")
        return 2
    cash = DOMESTIC_STOCK_CAPITAL_KRW
    try:
        balance = trader.get_balance()
        cash = extract_orderable_cash(balance) or cash
    except Exception as exc:
        print(f"잔고 조회 실패로 주문 중단: {exc}")
        return 2
    for item in WATCHLIST:
        code = item["code"]
        record = domestic_stock_signal(code, "paper", cash, cash)
        if record["signal"] != "buy":
            print(f"{code}: paper 주문 없음, signal={record['signal']}")
            continue
        price = float(record["strategy"]["entry"])
        qty = int(record["qty"])
        result = trader.buy_limit(code, qty, price)
        record.update({"order_api_called": not result.get("blocked", False), "is_order_allowed": not result.get("blocked", False), "order_result": result})
        log_signal(record)
        print(f"{code}: paper 지정가 주문 {'호출' if record['order_api_called'] else '차단'}")
    return 0


# fetch_futures_intraday는 yfinance에서 선물 5분봉을 조회합니다.
def fetch_futures_intraday(symbol: str):
    ticker = get_contract(symbol)["yfinance_symbol"]
    df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False, threads=False)
    data = ensure_ohlcv(df)
    if data.empty:
        raise ValueError(f"{symbol} 5분봉 데이터를 가져오지 못했습니다.")
    return data


# futures_regime은 해당 선물의 일봉 MA20 기준 risk_on/off를 계산합니다.
def futures_regime(symbol: str) -> dict[str, Any]:
    contract = get_contract(symbol)
    sources = [contract["yfinance_symbol"], "QQQ" if symbol == "MNQ" else "SPY"]
    for source in sources:
        try:
            df = yf.download(source, period="6mo", interval="1d", auto_adjust=False, progress=False, threads=False)
            data = ensure_ohlcv(df)
            if len(data) < 20:
                continue
            close = data["Close"]
            ma20 = sma(close, 20)
            latest_close = float(close.iloc[-1])
            latest_ma20 = float(ma20.iloc[-1])
            return {
                "regime": "risk_on" if latest_close > latest_ma20 else "risk_off",
                "regime_source": source,
                "close": latest_close,
                "ma20": latest_ma20,
                "warning": None,
            }
        except Exception:
            continue
    return {"regime": "unknown", "regime_source": None, "close": None, "ma20": None, "warning": "선물 레짐 데이터 조회 실패"}


# futures_signal은 선물 한 계약의 신호, ATR 가격계획, 계약 수량을 계산합니다.
def futures_signal(symbol: str, mode: str, data=None) -> dict[str, Any]:
    contract = get_contract(symbol)
    data = fetch_futures_intraday(symbol) if data is None else data
    intraday = compute_intraday_details(data)
    strategy = calculate_intraday_plan(data)
    regime_info = futures_regime(symbol)
    final_score = intraday.score
    entry = float(strategy.entry)
    stop_loss = float(strategy.stop_loss)
    stop_distance_ticks = abs(entry - stop_loss) / float(contract["tick_size"])
    risk_per_contract_krw = stop_distance_ticks * float(contract["tick_value_usd"]) * FX_RATE_USDKRW
    risk_amount = FUTURES_PAPER_CAPITAL_KRW * 0.01
    qty_by_risk = int(risk_amount // risk_per_contract_krw) if risk_per_contract_krw > 0 else 0
    qty = min(qty_by_risk, int(contract["max_contract_qty"]))
    required_margin_krw = float(contract["margin_per_contract_usd"]) * qty * FX_RATE_USDKRW
    margin_limit_krw = FUTURES_PAPER_CAPITAL_KRW * 0.8
    margin_check_passed = required_margin_krw <= margin_limit_krw
    risk_allowed = qty >= 1
    reasons = []
    korean_time_reason = intraday.forced_hold_reason in {"09:10 전 신규 진입 금지", "14:50 이후 신규 진입 금지"}
    futures_chasing_filter = intraday.chasing_filter and not korean_time_reason
    if qty_by_risk < 1:
        reasons.append("선물 리스크 기준 계약 수량 1 미만")
    if required_margin_krw > margin_limit_krw:
        risk_allowed = False
        reasons.append("선물 증거금 한도 초과")
    if regime_info["regime"] == "risk_off":
        risk_allowed = False
        reasons.append("선물 risk_off 신규 진입 금지")
    if not is_futures_entry_time():
        risk_allowed = False
        reasons.append("선물 거래시간 필터 신규 진입 금지")
    if regime_info.get("warning"):
        reasons.append(str(regime_info["warning"]))
    if intraday.forced_hold_reason and not korean_time_reason:
        reasons.append(intraday.forced_hold_reason)
    if strategy.skip_reason:
        reasons.append(strategy.skip_reason)
    signal = "buy" if final_score >= 75 and intraday.score >= 72 and not futures_chasing_filter and not strategy.skip and risk_allowed else ("hold" if final_score >= 55 else "avoid")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "asset_class": "futures",
        "symbol_or_code": symbol,
        "symbol": symbol,
        "market": contract["exchange"],
        "currency": contract["currency"],
        "mode": mode,
        "regime": regime_info["regime"],
        "regime_source": regime_info["regime_source"],
        "final_score": round(final_score, 2),
        "intraday_score": round(intraday.score, 2),
        "signal": signal,
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": float(strategy.target1),
        "target2": float(strategy.target2),
        "contract_qty": qty,
        "qty": qty,
        "qty_by_risk": qty_by_risk,
        "tick_size": contract["tick_size"],
        "tick_value_usd": contract["tick_value_usd"],
        "margin_per_contract_usd": contract["margin_per_contract_usd"],
        "required_margin_krw": required_margin_krw,
        "margin_limit_krw": margin_limit_krw,
        "margin_check_passed": margin_check_passed,
        "capital_krw": FUTURES_PAPER_CAPITAL_KRW,
        "risk_per_contract_krw": risk_per_contract_krw,
        "reasons": reasons,
        "order_api_called": False,
        "is_order_allowed": False,
    }


# run_futures_dry_run은 선물 dry-run 신호를 출력합니다.
def run_futures_dry_run() -> int:
    print("선물 dry-run을 시작합니다.")
    for item in FUTURES_WATCHLIST:
        symbol = item["symbol"]
        try:
            record = futures_signal(symbol, "dry-run")
            record["order_block_reason"] = "dry-run 주문 없음"
            log_signal(record)
            print(f"{symbol}: final {record['final_score']:.2f}, contracts {record['contract_qty']}, signal {record['signal']}")
            for reason in record["reasons"]:
                print(f"  사유: {reason}")
        except Exception as exc:
            print(f"{symbol}: 선물 dry-run skip - {exc}")
            log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "futures", "symbol": symbol, "mode": "dry-run", "signal": "error", "error": str(exc)})
    return 0


# run_futures_paper_sim은 KIS 호출 없이 내부 JSON paper simulator로 선물 가상 체결을 실행합니다.
def run_futures_paper_sim() -> int:
    if not ENABLE_FUTURES_PAPER_SIM:
        print("선물 paper-sim 차단: ENABLE_FUTURES_PAPER_SIM=False")
        return 2
    simulator = PaperSimulator()
    print("선물 paper-sim을 시작합니다. KIS 주문 API는 호출하지 않습니다.")
    for item in FUTURES_WATCHLIST:
        symbol = item["symbol"]
        try:
            full_data = fetch_futures_intraday(symbol)
            if len(full_data) < 41:
                print(f"{symbol}: paper-sim skip - 5분봉 데이터 부족")
                continue
            signal_data = full_data.iloc[:-1].copy()
            next_bar = full_data.iloc[-1]
            record = futures_signal(symbol, "paper-sim", signal_data)
            record["paper_sim"] = True
            qty = int(record["contract_qty"])
            if record["signal"] != "buy":
                record["order_block_reason"] = "signal 조건 미달"
                log_signal(record)
                print(f"{symbol}: paper-sim 진입 없음 - signal {record['signal']}")
                continue
            if qty < 1:
                print(f"{symbol}: paper-sim 진입 차단 - 계약 수량 1 미만")
                log_signal(record)
                continue
            limit_price = float(record["entry"])
            if float(next_bar["Low"]) > limit_price:
                record["order_block_reason"] = "다음 봉 low가 limit_price에 닿지 않아 미체결"
                log_signal(record)
                print(f"{symbol}: paper-sim 미체결 - 다음 봉 low > limit")
                continue
            fill_price = limit_price + float(record["tick_size"])
            result = simulator.place_futures_order(
                symbol,
                "buy",
                qty,
                fill_price,
                required_margin_krw=float(record["required_margin_krw"]),
                margin_limit_krw=float(record["margin_limit_krw"]),
                margin_check_passed=bool(record["margin_check_passed"]),
            )
            record.update({"paper_sim": True, "simulated_order": result, "order_api_called": False, "simulated_pnl_krw": result.get("simulated_pnl_krw", 0.0)})
            log_signal(record)
            print(f"{symbol}: paper-sim 가상 체결 {'성공' if result.get('filled') else '차단'}, qty {qty}, price {record['entry']:.2f}")
        except Exception as exc:
            print(f"{symbol}: paper-sim 실패 - {exc}")
    return 0


# run_futures_paper_check는 KIS 선물 조회 skeleton 상태를 확인하고 주문은 호출하지 않습니다.
def run_futures_paper_check() -> int:
    client = FuturesClient(mode=None, dry_run=True, live=False, allow_paper_order=False)
    client.print_mode_summary()
    ok, msg = client.validate_env()
    if not ok:
        print(f"선물 paper-check 실행 불가: {msg}")
        return 2
    print("선물 paper-check를 시작합니다. 주문 API는 호출하지 않습니다.")
    token = client.get_access_token()
    print(f"access token 확인: {'성공' if token else '실패'}")
    balance_probe = client.get_futures_balance()
    position_probe = client.get_open_positions()
    print("KIS_MODE=paper 확인: 성공")
    print("해외선물 계좌 환경변수 로드: 성공")
    print(f"잔고/증거금 조회: TODO - {balance_probe.get('reason')}")
    print(f"미결제조회: TODO - {position_probe.get('reason')}")
    for item in FUTURES_WATCHLIST:
        symbol = item["symbol"]
        price_probe = client.get_futures_price(symbol)
        margin_probe = client.get_margin_info(symbol)
        contract = get_contract(symbol)
        print(f"{symbol} 시세조회: TODO - {price_probe.get('reason')}")
        print(f"{symbol} 증거금조회: TODO - {margin_probe.get('reason')}")
        print(f"{symbol} 주문 가능 metadata: is_kis_paper_order_enabled={contract.get('is_kis_paper_order_enabled')}")
        log_signal(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "asset_class": "futures",
                "symbol": symbol,
                "mode": "paper-check",
                "signal": "check",
                "order_api_called": False,
                "price_probe": price_probe,
                "margin_probe": margin_probe,
                "balance_probe": balance_probe,
                "position_probe": position_probe,
                "order_block_reason": "paper-check는 주문 API 호출 금지",
            }
        )
    return 0


# run_futures_paper_order는 조건 미충족 시 한국어 사유로 차단하고, 현재 단계에서는 KIS 주문을 호출하지 않습니다.
def run_futures_paper_order(allow_paper_order: bool, live: bool) -> int:
    client = FuturesClient(mode="paper", dry_run=not allow_paper_order, live=live, allow_paper_order=allow_paper_order)
    if live and not ENABLE_REAL_ORDER:
        print("선물 실전 주문 차단: ENABLE_REAL_ORDER=False")
        return 2
    if not allow_paper_order:
        print("선물 KIS paper 주문 차단: --allow-paper-order가 없습니다.")
        for item in FUTURES_WATCHLIST:
            symbol = item["symbol"]
            log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "futures", "symbol": symbol, "mode": "paper", "signal": "blocked", "order_api_called": False, "order_block_reason": "--allow-paper-order가 없어 KIS 선물 paper 주문 차단"})
        return 0
    if not ENABLE_FUTURES_KIS_PAPER_ORDER:
        print("선물 KIS paper 주문 차단: ENABLE_FUTURES_KIS_PAPER_ORDER=False")
    for item in FUTURES_WATCHLIST:
        symbol = item["symbol"]
        result = client.buy_futures_limit(symbol, 1, 0.0)
        log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "futures", "symbol": symbol, "mode": "paper", "signal": "blocked", "order_api_called": False, "order_block_reason": result["reason"]})
        print(f"{symbol}: KIS 선물 paper 주문 차단 - {result['reason']}")
    return 0


# main은 CLI 인자를 해석해 국내주식/선물 모드를 실행합니다.
def main() -> int:
    parser = argparse.ArgumentParser(description="V3.2 Paper Lab KR + Futures")
    parser.add_argument("--asset", default="domestic-stock", choices=["domestic-stock", "futures", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper-check", action="store_true")
    parser.add_argument("--paper-sim", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--allow-paper-order", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.live and not ENABLE_REAL_ORDER:
        print("실전 주문 차단: ENABLE_REAL_ORDER=False")
        return 2
    if args.asset == "all":
        if args.paper_sim:
            run_domestic_stock_dry_run()
            return run_futures_paper_sim()
        run_domestic_stock_dry_run()
        return run_futures_dry_run()
    if args.paper_check:
        return run_domestic_stock_paper_check() if args.asset == "domestic-stock" else run_futures_paper_check()
    if args.paper_sim:
        if args.asset == "domestic-stock":
            print("국내주식 paper-sim은 준비되어 있지만 기본 검증 대상은 futures입니다.")
            return 0
        return run_futures_paper_sim()
    if args.paper:
        if args.asset == "domestic-stock":
            return run_domestic_stock_paper_order(args.allow_paper_order, args.live)
        return run_futures_paper_order(args.allow_paper_order, args.live)
    if args.asset == "futures":
        return run_futures_dry_run()
    return run_domestic_stock_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
