# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: config.py, clients/*, core/*, paper_simulator.py
# Description: 국내주식과 선물 V3.2 Paper Lab 실행 모드를 제공합니다.
# ================================================================================

"""Main entry point for KR domestic stocks and futures Paper Lab."""

from __future__ import annotations

import argparse
import json
import os
import time as sleep_time
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from clients.kis_domestic_stock_client import DomesticStockClient
from clients.kis_futures_client import FuturesClient
from config import (
    AI_SHADOW_MODE,
    DOMESTIC_STOCK_AMOUNT_BY_CODE,
    DOMESTIC_STOCK_CAPITAL_KRW,
    ENABLE_DYNAMIC_TRADE_UNIVERSE,
    ENABLE_DOMESTIC_STOCK_PAPER_ORDER,
    ENABLE_BT_EXECUTE,
    ENABLE_FUTURES_KIS_PAPER_ORDER,
    ENABLE_FUTURES_PAPER_SIM,
    ENABLE_REAL_ORDER,
    FUTURES_PAPER_CAPITAL_KRW,
    FUTURES_WATCHLIST,
    FX_RATE_USDKRW,
    MAX_BUYS_PER_DAY,
    MAX_BUY_CANDIDATES_PER_LOOP,
    MAX_NEW_POSITIONS_PER_DAY,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PER_SYMBOL_KRW,
    MAX_RISK_SCORE_FOR_ORDER,
    MAX_SCAN_SYMBOLS,
    MAX_TOTAL_EXPOSURE_KRW,
    MAX_TRADES_PER_DAY_DOMESTIC_STOCK,
    MIN_AI_BUY_CONFIDENCE_FOR_CANDIDATE,
    MIN_AI_BUY_CONFIDENCE_FOR_ORDER,
    MIN_EXPECTED_RETURN_FOR_ORDER,
    PAPER_WATCH_BAR_INTERVAL_SEC,
    PAPER_WATCH_BAR_OFFSET_SEC,
    PAPER_WATCH_MIN_SLEEP_SEC,
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
from logger import TRADING_LOG, log_signal, log_trading
from paper_simulator import PaperSimulator
from position_manager import evaluate_exit, load_positions, remove_position, save_positions, update_position
from risk_manager import RiskManager

BUY_RULES = {"final_score": 75, "intraday_score": 72, "gaussian_score": 60}
DOMESTIC_ENTRY_CUTOFF = datetime.strptime("14:50", "%H:%M").time()


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
    final_score = 0.90 * intraday.score + 0.10 * gaussian["gaussian_score"]
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
        "target1_expected_profit_pct": round(strategy.target1_expected_profit_pct, 6),
        "min_target1_profit_pct": strategy.min_target1_profit_pct,
        "skip_reason": strategy.skip_reason,
        "qty": sizing["qty"],
        "qty_by_risk": sizing["qty_by_risk"],
        "qty_by_amount": sizing["qty_by_amount"],
        "signal": signal,
        "reasons": reasons,
        "capital_krw": account_value,
        "order_api_called": False,
        "is_order_allowed": False,
    }


def _today_order_counts() -> dict[str, int]:
    today = datetime.now().date()
    counts = {"total": 0, "buy": 0, "sell": 0}
    if not TRADING_LOG.exists():
        return counts
    try:
        rows = json.loads(TRADING_LOG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return counts
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if not isinstance(row, dict) or row.get("asset_class") != "domestic-stock":
            continue
        if not row.get("order_api_called"):
            continue
        raw_ts = row.get("timestamp") or row.get("logged_at")
        try:
            row_date = datetime.fromisoformat(str(raw_ts)).date()
        except ValueError:
            continue
        if row_date != today:
            continue
        side = str(row.get("side") or row.get("order_side") or "")
        counts["total"] += 1
        if side in counts:
            counts[side] += 1
    return counts


def _estimate_open_exposure(positions: dict[str, Any], broker_holdings: dict[str, dict[str, Any]]) -> float:
    exposure = 0.0
    seen = set()
    for code, position in positions.items():
        try:
            qty = int(position.get("qty") or 0)
            price = float(position.get("avg_price") or 0.0)
            exposure += max(qty, 0) * max(price, 0.0)
            seen.add(str(code))
        except (TypeError, ValueError):
            continue
    for code, holding in broker_holdings.items():
        if str(code) in seen:
            continue
        try:
            qty = int(holding.get("qty") or 0)
            price = float(holding.get("avg_price") or 0.0)
            exposure += max(qty, 0) * max(price, 0.0)
        except (TypeError, ValueError):
            continue
    return float(exposure)


def _open_position_count(positions: dict[str, Any], broker_holdings: dict[str, dict[str, Any]]) -> int:
    return len({*positions.keys(), *broker_holdings.keys()})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_market_breadth_score(record: dict[str, Any]) -> float:
    breadth = record.get("market_breadth_features") or {}
    values = [
        breadth.get("universe_risk_on_ratio"),
        breadth.get("kospi200_positive_return_ratio"),
        breadth.get("kosdaq50_positive_return_ratio"),
    ]
    clean = [_safe_float(value, -1.0) for value in values if value is not None]
    if not clean:
        return 0.0
    return max(0.0, min(sum(clean) / len(clean), 1.0))


def _liquidity_fields(code: str, intraday_df: Any | None = None) -> dict[str, Any]:
    try:
        data = ensure_ohlcv(intraday_df if intraday_df is not None else fetch_intraday(code, period="5d", interval="5m"))
    except Exception as exc:
        return {
            "liquidity_passed": False,
            "liquidity_score": 0.0,
            "volume_zscore": None,
            "liquidity_reject_reason": f"intraday_unavailable:{exc}",
            "spread_available": False,
        }
    if data.empty or len(data) < 20:
        return {
            "liquidity_passed": False,
            "liquidity_score": 0.0,
            "volume_zscore": None,
            "liquidity_reject_reason": "5분봉 유동성 판단 데이터 부족",
            "spread_available": False,
        }
    latest = data.iloc[-1]
    close = _safe_float(latest.get("Close"))
    volume = _safe_float(latest.get("Volume"))
    recent_turnover = float((data["Close"].tail(12) * data["Volume"].tail(12)).mean())
    volume_mean = float(data["Volume"].tail(20).mean())
    volume_std = float(data["Volume"].tail(20).std() or 0.0)
    volume_zscore = (volume - volume_mean) / volume_std if volume_std > 0 else 0.0
    reasons = []
    if volume <= 0:
        reasons.append("최근 5분봉 거래량 0")
    if recent_turnover < 50_000_000:
        reasons.append("최근 5분봉 평균 거래대금 부족")
    if close < 1_000 or close > 5_000_000:
        reasons.append("가격 이상치 또는 과도한 저가/고가")
    # 호가 cache/WebSocket 연결 전에는 spread를 확인할 수 없으므로 실제 주문은 보수적으로 차단합니다.
    spread_available = False
    reasons.append("스프레드/호가 정보 없음")
    liquidity_score = 0.0
    if volume > 0 and close > 0:
        turnover_score = min(recent_turnover / 1_000_000_000, 1.0)
        volume_score = max(0.0, min((volume_zscore + 3.0) / 6.0, 1.0))
        price_score = 1.0 if 1_000 <= close <= 5_000_000 else 0.0
        liquidity_score = 0.5 * turnover_score + 0.3 * volume_score + 0.2 * price_score
    return {
        "liquidity_passed": len(reasons) == 0 and spread_available,
        "liquidity_score": round(float(liquidity_score), 6),
        "volume_zscore": round(float(volume_zscore), 6),
        "recent_turnover_krw": round(float(recent_turnover), 2),
        "spread_available": spread_available,
        "liquidity_reject_reason": ";".join(reasons) if reasons else None,
    }


def _ranking_score(record: dict[str, Any]) -> float:
    confidence = _safe_float(record.get("confidence"))
    expected_return = _safe_float(record.get("expected_return"))
    risk_score = _safe_float(record.get("risk_score"), 1.0)
    normalized_expected_return = max(0.0, min(expected_return / 0.01, 1.0))
    liquidity_score = _safe_float(record.get("liquidity_score"))
    market_breadth_score = _candidate_market_breadth_score(record)
    return round(
        confidence * 0.45
        + normalized_expected_return * 0.25
        + liquidity_score * 0.15
        + market_breadth_score * 0.10
        - risk_score * 0.25,
        6,
    )


def _candidate_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": record.get("code"),
        "signal": record.get("signal"),
        "ai_action": record.get("ai_action"),
        "confidence": record.get("confidence"),
        "expected_return": record.get("expected_return"),
        "risk_score": record.get("risk_score"),
        "liquidity_score": record.get("liquidity_score"),
        "volume_zscore": record.get("volume_zscore"),
        "ranking_score": record.get("ranking_score"),
        "bt_would_action": record.get("bt_would_action"),
    }


def _extract_domestic_holdings(balance: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    holdings: dict[str, dict[str, Any]] = {}
    if not balance:
        return holdings
    output1 = balance.get("output1")
    rows = output1 if isinstance(output1, list) else []
    for item in rows:
        if not isinstance(item, dict):
            continue
        code = str(item.get("pdno") or item.get("prdt_cd") or "").strip()
        if not code:
            continue
        try:
            qty = int(float(str(item.get("hldg_qty") or item.get("ord_psbl_qty") or 0).replace(",", "")))
        except ValueError:
            qty = 0
        if qty <= 0:
            continue
        try:
            avg_price = float(str(item.get("pchs_avg_pric") or item.get("avg_prvs") or 0).replace(",", ""))
        except ValueError:
            avg_price = 0.0
        holdings[code] = {"qty": qty, "avg_price": avg_price}
    return holdings


def _after_domestic_entry_cutoff() -> bool:
    return datetime.now().time() >= DOMESTIC_ENTRY_CUTOFF


def _log_paper_watch_order(record: dict[str, Any], side: str, result: dict[str, Any], order_api_called: bool) -> None:
    order_record = {
        **record,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "paper-watch",
        "side": side,
        "order_side": side,
        "order_result": result,
        "order_api_called": order_api_called,
        "is_order_allowed": order_api_called,
        "order_block_reason": None if order_api_called else result.get("reason", "주문 조건 미충족"),
    }
    log_signal(order_record)
    if order_api_called:
        log_trading(order_record)


def _load_ai_engine():
    try:
        from ai.ai_signal_engine import AISignalEngine

        return AISignalEngine()
    except Exception:
        return None


def _load_policy_engine():
    try:
        from ai.policy_engine import PolicyEngine

        return PolicyEngine()
    except Exception:
        return None


def _policy_shadow_fields(record: dict[str, Any], code: str, policy_engine: Any | None, bt_shadow: bool, has_position: bool) -> dict[str, Any]:
    if policy_engine is None:
        from ai.policy_engine import empty_policy_result

        fields = empty_policy_result("engine_unavailable")
    else:
        try:
            intraday_df = fetch_intraday(code, period="5d", interval="5m")
            fields = policy_engine.predict(intraday_df, rule_score=float(record.get("final_score") or 0.0))
        except Exception as exc:
            from ai.policy_engine import empty_policy_result

            fields = empty_policy_result(f"predict_error:{exc}")
    if bt_shadow:
        try:
            from behavior_tree.tree_runner import run_domestic_stock_bt_shadow

            fields.update(run_domestic_stock_bt_shadow(fields, has_position=has_position))
        except Exception as exc:
            fields.update(
                {
                    "behavior_tree_enabled": True,
                    "behavior_tree_mode": "shadow",
                    "behavior_tree_path": [],
                    "behavior_tree_result": f"ERROR:{exc}",
                    "bt_would_action": None,
                    "bt_shadow_only": True,
                }
            )
    else:
        fields.update(
            {
                "behavior_tree_enabled": False,
                "behavior_tree_mode": "disabled",
                "behavior_tree_path": [],
                "behavior_tree_result": "DISABLED",
                "bt_would_action": None,
                "bt_shadow_only": True,
            }
        )
    return fields


def _ai_shadow_fields(record: dict[str, Any], code: str, ai_engine: Any | None, ai_gate: bool) -> dict[str, Any]:
    rule_score = float(record.get("final_score") or 0.0)
    fields = {
        "rule_score": rule_score,
        "rule_signal": record.get("signal"),
        "rule_threshold": BUY_RULES["final_score"],
        "final_order_signal": record.get("signal"),
    }
    if ai_engine is None:
        fields.update(
            {
                "ai_enabled": False,
                "ai_status": "engine_unavailable",
                "ai_model_name": "patchtst_lite",
                "ai_model_version": None,
                "ai_prob_up": None,
                "ai_prob_up_calibrated": None,
                "ai_expected_return": None,
                "ai_risk_score": None,
                "ai_decision": "none",
                "ai_gate_passed": False,
                "ai_gate_reason": "engine_unavailable",
                "ai_would_pass": False,
                "ai_shadow_mode": True,
            }
        )
        return fields
    try:
        intraday_df = fetch_intraday(code, period="5d", interval="5m")
        fields.update(ai_engine.predict(intraday_df, rule_score=rule_score))
    except Exception as exc:
        fields.update(
            {
                "ai_enabled": True,
                "ai_status": f"predict_error:{exc}",
                "ai_model_name": "patchtst_lite",
                "ai_model_version": None,
                "ai_prob_up": None,
                "ai_prob_up_calibrated": None,
                "ai_expected_return": None,
                "ai_risk_score": None,
                "ai_decision": "none",
                "ai_gate_passed": False,
                "ai_gate_reason": "predict_error",
                "ai_would_pass": False,
                "ai_shadow_mode": True,
            }
        )
    if ai_gate and AI_SHADOW_MODE:
        fields["ai_gate_reason"] = f"{fields.get('ai_gate_reason')};shadow_mode_no_order_effect"
    return fields


def _run_domestic_exit_checks(trader: DomesticStockClient, broker_holdings: dict[str, dict[str, Any]], bar_key: str | None = None) -> None:
    positions = load_positions()
    changed = False
    for code, position in list(positions.items()):
        try:
            intraday_df = fetch_intraday(code, period="5d", interval="5m")
            price = trader.get_price(code) or float(intraday_df["Close"].iloc[-1])
            exit_decision = evaluate_exit(position, float(price), intraday_df)
            updated_position = exit_decision.get("updated_position")
            if exit_decision["action"] == "hold":
                log_signal(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "asset_class": "domestic-stock",
                        "symbol_or_code": code,
                        "code": code,
                        "mode": "paper-watch",
                        "bar_key": bar_key,
                        "signal": "exit-hold",
                        "qty": int(position.get("qty") or 0),
                        "price": float(price),
                        "order_api_called": False,
                        "is_order_allowed": False,
                    }
                )
                if updated_position:
                    positions[code] = updated_position
                    changed = True
                continue
            held_qty = int((broker_holdings.get(code) or {}).get("qty") or position.get("qty") or 0)
            if held_qty <= 0:
                positions.pop(code, None)
                changed = True
                remove_position(code)
                print(f"{code}: 로컬 포지션 제거 - KIS 보유 수량 없음")
                continue
            sell_qty = held_qty if exit_decision["action"] == "sell_all" else max(1, held_qty // 2)
            result = trader.sell_limit(code, sell_qty, float(price))
            order_api_called = not result.get("blocked", False)
            record = {
                "asset_class": "domestic-stock",
                "symbol_or_code": code,
                "code": code,
                "bar_key": bar_key,
                "signal": "sell",
                "qty": sell_qty,
                "price": float(price),
                "exit_reason": exit_decision.get("reason"),
            }
            _log_paper_watch_order(record, "sell", result, order_api_called)
            if order_api_called and exit_decision["action"] == "sell_all":
                positions.pop(code, None)
                changed = True
                remove_position(code)
            elif order_api_called and updated_position:
                updated_position["qty"] = max(held_qty - sell_qty, 0)
                positions[code] = updated_position
                changed = True
            print(f"{code}: 청산 점검 {exit_decision['action']} - 지정가 매도 {'호출' if order_api_called else '차단'}")
        except Exception as exc:
            log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "domestic-stock", "code": code, "mode": "paper-watch", "signal": "exit-error", "error": str(exc), "order_api_called": False})
            print(f"{code}: 청산 점검 실패 - {exc}")
    if changed:
        save_positions(positions)


def _run_domestic_entry_checks(
    trader: DomesticStockClient,
    cash: float,
    broker_holdings: dict[str, dict[str, Any]],
    bar_key: str | None = None,
    ai_engine: Any | None = None,
    ai_gate: bool = False,
    policy_engine: Any | None = None,
    ai_policy: bool = False,
    bt_shadow: bool = False,
    bt_execute_requested: bool = False,
) -> None:
    positions = load_positions()
    scan_items = WATCHLIST[: max(int(MAX_SCAN_SYMBOLS), 1)] if ENABLE_DYNAMIC_TRADE_UNIVERSE else WATCHLIST
    records: list[dict[str, Any]] = []
    buy_candidates: list[dict[str, Any]] = []
    rejected_by_liquidity_count = 0
    for item in scan_items:
        code = item["code"]
        try:
            record = domestic_stock_signal(code, "paper-watch", cash, cash)
            record["bar_key"] = bar_key
            record.update(_ai_shadow_fields(record, code, ai_engine, ai_gate))
            has_position = code in positions or code in broker_holdings
            if ai_policy:
                record.update(_policy_shadow_fields(record, code, policy_engine, bt_shadow, has_position))
            record.update(_liquidity_fields(code))
            record["market_breadth_score"] = _candidate_market_breadth_score(record)
            record["ranking_score"] = _ranking_score(record)
            record["dynamic_trade_universe_enabled"] = bool(ENABLE_DYNAMIC_TRADE_UNIVERSE)
            record["bt_execute_requested"] = bool(bt_execute_requested)
            record["bt_execute_enabled"] = bool(ENABLE_BT_EXECUTE)
            record["universe_size"] = len(WATCHLIST)
            record["scanned_symbol_count"] = len(scan_items)
            if not record.get("liquidity_passed"):
                rejected_by_liquidity_count += 1
            if (
                record.get("ai_action") == "BUY"
                and _safe_float(record.get("confidence")) >= MIN_AI_BUY_CONFIDENCE_FOR_CANDIDATE
            ):
                buy_candidates.append(record)
            records.append(record)
        except Exception as exc:
            log_signal(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "asset_class": "domestic-stock",
                    "code": code,
                    "mode": "paper-watch",
                    "bar_key": bar_key,
                    "signal": "entry-scan-error",
                    "error": str(exc),
                    "order_api_called": False,
                    "is_order_allowed": False,
                }
            )
            print(f"{code}: 신규 진입 스캔 실패 - {exc}")

    ranked_candidates = sorted(buy_candidates, key=lambda row: _safe_float(row.get("ranking_score")), reverse=True)
    selected_candidates = ranked_candidates[: max(int(MAX_BUY_CANDIDATES_PER_LOOP), 0)]
    selected_codes = {str(row.get("code")) for row in selected_candidates}
    final_order_candidate_codes: list[str] = []
    rejected_by_risk_count = 0

    summary_base = {
        "universe_size": len(WATCHLIST),
        "scanned_symbol_count": len(scan_items),
        "buy_candidate_count": len(buy_candidates),
        "top_candidates": [_candidate_snapshot(row) for row in ranked_candidates[:10]],
        "selected_candidates": [_candidate_snapshot(row) for row in selected_candidates],
        "rejected_by_liquidity_count": rejected_by_liquidity_count,
        "max_buy_candidates_per_loop": MAX_BUY_CANDIDATES_PER_LOOP,
        "max_new_positions_per_day": MAX_NEW_POSITIONS_PER_DAY,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_position_per_symbol_krw": MAX_POSITION_PER_SYMBOL_KRW,
        "max_total_exposure_krw": MAX_TOTAL_EXPOSURE_KRW,
    }

    for record in records:
        code = str(record["code"])
        counts = _today_order_counts()
        block_reason = None
        price = float(record["strategy"]["entry"])
        qty = int(record["qty"])
        current_exposure = _estimate_open_exposure(positions, broker_holdings)
        open_positions = _open_position_count(positions, broker_holdings)
        per_symbol_qty_cap = int(MAX_POSITION_PER_SYMBOL_KRW // price) if price > 0 else 0
        exposure_qty_cap = int(max(MAX_TOTAL_EXPOSURE_KRW - current_exposure, 0) // price) if price > 0 else 0
        capped_qty = min(qty, per_symbol_qty_cap, exposure_qty_cap)
        if code in selected_codes:
            record["selected_by_candidate_ranking"] = True
        else:
            record["selected_by_candidate_ranking"] = False

        if ENABLE_DYNAMIC_TRADE_UNIVERSE and code not in selected_codes:
            block_reason = "candidate ranking 상위 후보 아님"
        elif ENABLE_DYNAMIC_TRADE_UNIVERSE and not bt_execute_requested:
            block_reason = "--bt-execute 없이 동적 universe 신규 주문 차단"
        elif ENABLE_DYNAMIC_TRADE_UNIVERSE and not ENABLE_BT_EXECUTE:
            block_reason = "ENABLE_BT_EXECUTE=False"
        elif not record.get("liquidity_passed"):
            block_reason = f"liquidity_reject:{record.get('liquidity_reject_reason')}"
        elif record.get("ai_action") != "BUY":
            block_reason = f"ai_action={record.get('ai_action')}"
        elif _safe_float(record.get("confidence")) < MIN_AI_BUY_CONFIDENCE_FOR_ORDER:
            block_reason = "AI 주문 confidence 기준 미달"
        elif _safe_float(record.get("expected_return")) < MIN_EXPECTED_RETURN_FOR_ORDER:
            block_reason = "AI expected_return 기준 미달"
        elif _safe_float(record.get("risk_score"), 1.0) > MAX_RISK_SCORE_FOR_ORDER:
            block_reason = "AI risk_score 기준 초과"
        elif record.get("bt_would_action") != "BUY_LIMIT":
            block_reason = f"BT entry 미통과:{record.get('bt_would_action')}"
        elif record["signal"] != "buy":
            block_reason = f"signal={record['signal']}"
        elif code in positions or code in broker_holdings:
            block_reason = "이미 보유 중인 종목 중복 매수 금지"
        elif open_positions >= MAX_OPEN_POSITIONS:
            block_reason = "MAX_OPEN_POSITIONS 초과"
        elif counts["total"] >= MAX_TRADES_PER_DAY_DOMESTIC_STOCK:
            block_reason = "daily trade limit 초과"
        elif counts["buy"] >= MAX_BUYS_PER_DAY:
            block_reason = "MAX_BUYS_PER_DAY 초과"
        elif counts["buy"] >= MAX_NEW_POSITIONS_PER_DAY:
            block_reason = "MAX_NEW_POSITIONS_PER_DAY 초과"
        elif _after_domestic_entry_cutoff():
            block_reason = "14:50 이후 신규 진입 금지"
        elif int(record["qty"]) <= 0:
            block_reason = "주문 수량 1주 미만"
        elif capped_qty <= 0:
            block_reason = "종목/총 노출 한도 기준 주문 수량 1주 미만"
        elif ai_gate and not AI_SHADOW_MODE and not record.get("ai_gate_passed"):
            block_reason = f"ai_gate_reject:{record.get('ai_gate_reason')}"
            record["final_order_signal"] = "hold"

        if code in selected_codes and block_reason:
            rejected_by_risk_count += 1
        if block_reason:
            record["order_block_reason"] = block_reason
            record["order_api_called"] = False
            record["is_order_allowed"] = False
            record.update(summary_base)
            record["rejected_by_risk_count"] = rejected_by_risk_count
            record["final_order_candidates"] = final_order_candidate_codes
            log_signal(record)
            print(f"{code}: 신규 진입 없음 - {block_reason}")
            continue

        qty = int(capped_qty)
        record["qty"] = qty
        final_order_candidate_codes.append(code)
        record.update(summary_base)
        record["rejected_by_risk_count"] = rejected_by_risk_count
        record["final_order_candidates"] = final_order_candidate_codes
        result = trader.buy_limit(code, qty, price)
        order_api_called = not result.get("blocked", False)
        _log_paper_watch_order(record, "buy", result, order_api_called)
        if order_api_called:
            update_position(
                code,
                qty,
                float(result.get("limit_price") or price),
                float(record["strategy"]["stop_loss"]),
                float(record["strategy"]["target1"]),
                float(record["strategy"]["target2"]),
            )
        print(f"{code}: 신규 지정가 매수 {'호출' if order_api_called else '차단'}")

    log_signal(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "asset_class": "domestic-stock",
            "mode": "paper-watch",
            "signal": "dynamic-universe-ranking-summary",
            "bar_key": bar_key,
            **summary_base,
            "rejected_by_risk_count": rejected_by_risk_count,
            "final_order_candidates": final_order_candidate_codes,
            "order_api_called": False,
            "is_order_allowed": False,
        }
    )


# _bar_key_str은 epoch초를 "YYYY-MM-DD HH:MM" 5분봉 키 문자열로 바꿉니다.
def _bar_key_str(epoch: int) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def _run_policy_no_order_shadow(policy_engine: Any | None, bt_shadow: bool) -> None:
    positions = load_positions()
    scan_items = WATCHLIST[: max(int(MAX_SCAN_SYMBOLS), 1)]
    records: list[dict[str, Any]] = []
    buy_candidates: list[dict[str, Any]] = []
    rejected_by_liquidity_count = 0
    for item in scan_items:
        code = item["code"]
        try:
            record = domestic_stock_signal(code, "paper-watch", DOMESTIC_STOCK_CAPITAL_KRW, DOMESTIC_STOCK_CAPITAL_KRW)
            record.update(_policy_shadow_fields(record, code, policy_engine, bt_shadow, code in positions))
            record.update(_liquidity_fields(code))
            record["market_breadth_score"] = _candidate_market_breadth_score(record)
            record["ranking_score"] = _ranking_score(record)
            record["dynamic_trade_universe_enabled"] = bool(ENABLE_DYNAMIC_TRADE_UNIVERSE)
            record["universe_size"] = len(WATCHLIST)
            record["scanned_symbol_count"] = len(scan_items)
            if not record.get("liquidity_passed"):
                rejected_by_liquidity_count += 1
            if (
                record.get("ai_action") == "BUY"
                and _safe_float(record.get("confidence")) >= MIN_AI_BUY_CONFIDENCE_FOR_CANDIDATE
            ):
                buy_candidates.append(record)
            records.append(record)
            record["order_block_reason"] = "--allow-paper-order가 없어 국내주식 paper-watch 주문 차단"
            record["order_api_called"] = False
            record["is_order_allowed"] = False
            log_signal(record)
            print(f"{code}: AI/BT shadow 기록 - 주문 차단")
        except Exception as exc:
            log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "domestic-stock", "code": code, "mode": "paper-watch", "signal": "policy-shadow-error", "error": str(exc), "order_api_called": False})
    ranked_candidates = sorted(buy_candidates, key=lambda row: _safe_float(row.get("ranking_score")), reverse=True)
    selected_candidates = ranked_candidates[: max(int(MAX_BUY_CANDIDATES_PER_LOOP), 0)]
    log_signal(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "asset_class": "domestic-stock",
            "mode": "paper-watch",
            "signal": "dynamic-universe-ranking-summary",
            "universe_size": len(WATCHLIST),
            "scanned_symbol_count": len(scan_items),
            "buy_candidate_count": len(buy_candidates),
            "top_candidates": [_candidate_snapshot(row) for row in ranked_candidates[:10]],
            "selected_candidates": [_candidate_snapshot(row) for row in selected_candidates],
            "rejected_by_risk_count": len(selected_candidates),
            "rejected_by_liquidity_count": rejected_by_liquidity_count,
            "final_order_candidates": [],
            "order_api_called": False,
            "is_order_allowed": False,
            "order_block_reason": "--allow-paper-order가 없어 ranking summary만 기록",
        }
    )


def run_domestic_stock_paper_watch(
    allow_paper_order: bool,
    interval_sec: int,
    ai_shadow: bool = False,
    ai_gate: bool = False,
    ai_policy: bool = False,
    bt_shadow: bool = False,
    bt_execute: bool = False,
) -> int:
    trader = DomesticStockClient(mode="paper", dry_run=not allow_paper_order, live=False, allow_paper_order=allow_paper_order)
    if os.getenv("KIS_MODE", "paper").lower() != "paper" or trader.mode != "paper":
        print("국내주식 paper-watch 차단: KIS_MODE=paper가 아닙니다.")
        return 2
    if ENABLE_REAL_ORDER:
        print("국내주식 paper-watch 차단: ENABLE_REAL_ORDER=True")
        return 2
    if not ENABLE_DOMESTIC_STOCK_PAPER_ORDER:
        print("국내주식 paper-watch 차단: ENABLE_DOMESTIC_STOCK_PAPER_ORDER=False")
        return 2
    if bt_execute and not ENABLE_BT_EXECUTE:
        print("BT execute는 scaffold만 준비되어 있으며 ENABLE_BT_EXECUTE=False라 주문 API를 호출하지 않습니다.")
        log_signal(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "asset_class": "domestic-stock",
                "mode": "paper-watch",
                "signal": "blocked",
                "bt_execute_requested": True,
                "bt_execute_enabled": bool(ENABLE_BT_EXECUTE),
                "bt_execute_block_reason": "ENABLE_BT_EXECUTE=False; guarded scaffold only",
                "bt_shadow_report_required": True,
                "bt_shadow_report_status": "not_checked",
                "bt_would_pass_risk_gate": False,
                "bt_would_block_reason": "BT execute default disabled",
                "bt_paper_execute_allowed": False,
                "order_api_called": False,
                "is_order_allowed": False,
            }
        )
        return 0
    if not allow_paper_order:
        print("국내주식 paper-watch 차단: --allow-paper-order가 없습니다.")
        if ai_policy or bt_shadow:
            _run_policy_no_order_shadow(_load_policy_engine() if ai_policy else None, bt_shadow)
            return 0
        log_signal(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "asset_class": "domestic-stock",
                "mode": "paper-watch",
                "signal": "blocked",
                "order_api_called": False,
                "is_order_allowed": False,
                "order_block_reason": "--allow-paper-order가 없어 국내주식 paper-watch 주문 차단",
            }
        )
        return 0

    bar_interval = max(int(PAPER_WATCH_BAR_INTERVAL_SEC), 60)
    bar_offset = max(int(PAPER_WATCH_BAR_OFFSET_SEC), 0)
    min_sleep = max(int(PAPER_WATCH_MIN_SLEEP_SEC), 1)
    interval_sec = max(int(interval_sec), min_sleep)
    last_processed_bar_key: str | None = None
    ai_engine = _load_ai_engine() if (ai_shadow or AI_SHADOW_MODE) else None
    policy_engine = _load_policy_engine() if ai_policy else None

    print(f"국내주식 paper-watch 시작: interval_sec={interval_sec}, bar_interval_sec={bar_interval}, bar_offset_sec={bar_offset}")
    try:
        while True:
            now = datetime.now()
            epoch = int(now.timestamp())
            # bar_offset초가 지난 뒤에만 직전 봉을 마감된 것으로 간주합니다.
            latest_close_epoch = ((epoch - bar_offset) // bar_interval) * bar_interval
            closed_bar_key = _bar_key_str(latest_close_epoch - bar_interval)
            current_bar_key = _bar_key_str((epoch // bar_interval) * bar_interval)

            if closed_bar_key != last_processed_bar_key:
                try:
                    balance = trader.get_balance()
                    cash = extract_orderable_cash(balance) or DOMESTIC_STOCK_CAPITAL_KRW
                    broker_holdings = _extract_domestic_holdings(balance)
                    _run_domestic_exit_checks(trader, broker_holdings, closed_bar_key)
                    _run_domestic_entry_checks(
                        trader,
                        cash,
                        broker_holdings,
                        closed_bar_key,
                        ai_engine=ai_engine,
                        ai_gate=ai_gate,
                        policy_engine=policy_engine,
                        ai_policy=ai_policy,
                        bt_shadow=bt_shadow,
                        bt_execute_requested=bt_execute,
                    )
                    last_processed_bar_key = closed_bar_key
                except Exception as exc:
                    # 처리 실패 시 같은 봉을 다음 wake에서 재시도하되, 최소 sleep으로 호출 폭주를 막습니다.
                    log_signal({"timestamp": datetime.now().isoformat(timespec="seconds"), "asset_class": "domestic-stock", "mode": "paper-watch", "bar_key": closed_bar_key, "signal": "loop-error", "error": str(exc), "order_api_called": False})
                    print(f"paper-watch 루프 실패: {exc}")

            now = datetime.now()
            next_run_epoch = latest_close_epoch + bar_interval + bar_offset
            until_next_bar = next_run_epoch - now.timestamp()
            sleep_seconds = max(min(until_next_bar, float(interval_sec)), float(min_sleep))
            next_run_at = now + timedelta(seconds=sleep_seconds)
            print(
                f"next_run_at={next_run_at.strftime('%H:%M:%S')} current_bar_key={current_bar_key} "
                f"last_processed_bar_key={last_processed_bar_key} sleep_seconds={sleep_seconds:.0f}"
            )
            sleep_time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("국내주식 paper-watch 안전 종료")
        return 0


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
    intraday = compute_intraday_details(data, apply_time_filter=False)
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
    signal = "buy" if intraday.score >= 75 and not futures_chasing_filter and not strategy.skip and risk_allowed else ("hold" if final_score >= 55 else "avoid")
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


# print_probe_result는 KIS 조회 실검증 결과를 민감값 없이 출력합니다.
def print_probe_result(label: str, result: dict[str, Any]) -> None:
    status = "성공" if result.get("success") else "실패"
    print(f"{label}: {status}")
    if result.get("blocked"):
        print(f"  차단/TODO: {result.get('reason')}")
    if result.get("api_url"):
        print(f"  endpoint: {result.get('api_url')}")
    if result.get("tr_id"):
        print(f"  TR ID: {result.get('tr_id')}")
    if result.get("status_code") is not None:
        print(f"  HTTP status: {result.get('status_code')}")
    if result.get("rt_cd") is not None:
        print(f"  rt_cd: {result.get('rt_cd')}")
    if result.get("msg_cd") is not None:
        print(f"  msg_cd: {result.get('msg_cd')}")
    if result.get("field_names"):
        print(f"  응답 최상위 필드: {', '.join(result['field_names'])}")
    if result.get("output_field_names"):
        print(f"  output 필드명: {', '.join(result['output_field_names'])}")
    if result.get("error"):
        print(f"  오류: {result.get('error')}")


# run_futures_kis_live_probe는 KIS 해외선물 조회 API만 실호출하고 주문 API는 호출하지 않습니다.
def run_futures_kis_live_probe(client: FuturesClient, token: str) -> int:
    ok, msg = client.validate_futures_env()
    if not ok:
        print(f"선물 조회 실검증 불가: {msg}")
        return 2
    print("KIS 해외선물 조회 실검증을 시작합니다. 주문 API는 절대 호출하지 않습니다.")
    print("KIS_MODE=paper 확인: 성공")
    print(f"해외선물 계좌번호: {client.masked_futures_account()}")
    print(f"해외선물 계좌 상품코드: {client.futures_account_product_code}")
    balance_probe = client.live_probe_balance(token)
    position_probe = client.live_probe_positions(token)
    print_probe_result("예수금/증거금 조회", balance_probe)
    print_probe_result("미결제 포지션 조회", position_probe)
    for item in FUTURES_WATCHLIST:
        symbol = item["symbol"]
        price_probe = client.live_probe_price(symbol, token)
        orderable_probe = client.live_probe_orderable(symbol, token)
        print_probe_result(f"{symbol} 현재가 조회", price_probe)
        print_probe_result(f"{symbol} 주문가능조회", orderable_probe)
        log_signal(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "asset_class": "futures",
                "symbol": symbol,
                "mode": "paper-check-live-probe",
                "signal": "check",
                "order_api_called": False,
                "price_probe": price_probe,
                "orderable_probe": orderable_probe,
                "balance_probe": balance_probe,
                "position_probe": position_probe,
                "order_block_reason": "kis-live-probe는 주문 API 호출 금지",
            }
        )
    return 0


# run_futures_paper_check는 KIS 선물 조회 skeleton 상태를 확인하고 주문은 호출하지 않습니다.
def run_futures_paper_check(kis_live_probe: bool = False) -> int:
    client = FuturesClient(mode=None, dry_run=True, live=False, allow_paper_order=False)
    client.print_mode_summary()
    ok, msg = client.validate_env()
    if not ok:
        print(f"선물 paper-check 실행 불가: {msg}")
        return 2
    print("선물 paper-check를 시작합니다. 주문 API는 호출하지 않습니다.")
    token = client.get_access_token()
    print(f"access token 확인: {'성공' if token else '실패'}")
    if kis_live_probe:
        if not token:
            print("선물 조회 실검증 불가: access token 발급 실패")
            return 2
        return run_futures_kis_live_probe(client, token)
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
    parser.add_argument("--paper-watch", action="store_true")
    parser.add_argument("--allow-paper-order", action="store_true")
    parser.add_argument("--kis-live-probe", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--ai-shadow", action="store_true")
    parser.add_argument("--ai-gate", action="store_true")
    parser.add_argument("--ai-policy", action="store_true")
    parser.add_argument("--bt-shadow", action="store_true")
    parser.add_argument("--bt-execute", action="store_true")
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
        if args.asset == "domestic-stock":
            if args.kis_live_probe:
                print("--kis-live-probe는 futures paper-check 전용입니다.")
                return 2
            return run_domestic_stock_paper_check()
        return run_futures_paper_check(args.kis_live_probe)
    if args.paper_sim:
        if args.asset == "domestic-stock":
            print("국내주식 paper-sim은 준비되어 있지만 기본 검증 대상은 futures입니다.")
            return 0
        return run_futures_paper_sim()
    if args.paper:
        if args.asset == "domestic-stock":
            return run_domestic_stock_paper_order(args.allow_paper_order, args.live)
        return run_futures_paper_order(args.allow_paper_order, args.live)
    if args.paper_watch:
        if args.asset != "domestic-stock":
            print("--paper-watch는 현재 domestic-stock 전용입니다.")
            return 2
        return run_domestic_stock_paper_watch(
            args.allow_paper_order,
            args.interval_sec,
            args.ai_shadow,
            args.ai_gate,
            args.ai_policy,
            args.bt_shadow,
            args.bt_execute,
        )
    if args.asset == "futures":
        return run_futures_dry_run()
    return run_domestic_stock_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
