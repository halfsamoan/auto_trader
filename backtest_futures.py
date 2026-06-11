# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.2 Paper Lab KR + Overseas Futures Final)
# Dependency: config.py, futures_contracts.py, core/*
# Description: MNQ/MES 해외선물 long-only paper-sim 전략을 백테스트합니다.
# ================================================================================

"""Futures Paper Lab backtest with next-bar execution."""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import yfinance as yf

from config import FUTURES_PAPER_CAPITAL_KRW, FX_RATE_USDKRW, RISK_PER_TRADE_PCT
from core.intraday_scorer import compute_intraday_details
from core.strategy import calculate_intraday_plan
from core.technical import ensure_ohlcv
from futures_contracts import get_contract


# fetch_data는 yfinance 선물 5분봉을 조회하고 OHLCV 형태로 정규화합니다.
def fetch_data(symbol: str, period: str):
    contract = get_contract(symbol)
    df = yf.download(contract["yfinance_symbol"], period=period, interval="5m", auto_adjust=False, progress=False, threads=False)
    data = ensure_ohlcv(df)
    if data.empty:
        raise ValueError(f"{symbol} yfinance 5분봉 데이터를 가져오지 못했습니다.")
    return data


# calc_qty는 stop 거리, tick 가치, FX, 증거금 한도로 계약 수량을 계산합니다.
def calc_qty(symbol: str, entry: float, stop_loss: float, capital: float) -> dict:
    contract = get_contract(symbol)
    stop_ticks = abs(entry - stop_loss) / float(contract["tick_size"])
    risk_per_contract_krw = stop_ticks * float(contract["tick_value_usd"]) * FX_RATE_USDKRW
    qty_by_risk = int((capital * RISK_PER_TRADE_PCT) // risk_per_contract_krw) if risk_per_contract_krw > 0 else 0
    qty = min(qty_by_risk, int(contract["max_contract_qty"]))
    required_margin_krw = qty * float(contract["margin_per_contract_usd"]) * FX_RATE_USDKRW
    margin_limit_krw = capital * 0.8
    return {
        "qty": qty,
        "qty_by_risk": qty_by_risk,
        "risk_per_contract_krw": risk_per_contract_krw,
        "required_margin_krw": required_margin_krw,
        "margin_check_passed": required_margin_krw <= margin_limit_krw,
    }


# pnl_krw는 tick 기반 선물 손익을 KRW로 환산합니다.
def pnl_krw(symbol: str, entry: float, exit_price: float, qty: int) -> float:
    contract = get_contract(symbol)
    ticks = (exit_price - entry) / float(contract["tick_size"])
    return ticks * float(contract["tick_value_usd"]) * qty * FX_RATE_USDKRW


# run_backtest는 한 선물 symbol을 미래 데이터 누수 없이 long-only 백테스트합니다.
def run_backtest(symbol: str, period: str, capital: float) -> dict:
    data = fetch_data(symbol, period)
    contract = get_contract(symbol)
    cash = float(capital)
    equity_curve = []
    trades = []
    position = None
    entry_count = 0
    tick_slip = float(contract["tick_size"])

    for i in range(40, len(data)):
        signal_data = data.iloc[max(0, i - 240) : i].copy()
        next_bar = data.iloc[i]
        now = data.index[i].to_pydatetime() if hasattr(data.index[i], "to_pydatetime") else datetime.now()

        if position:
            high = float(next_bar["High"])
            low = float(next_bar["Low"])
            exit_reason = None
            exit_price = None
            holding_minutes = (now - position["entry_time"]).total_seconds() / 60
            if low <= position["stop_loss"]:
                exit_reason = "stop_loss"
                exit_price = position["stop_loss"] - tick_slip
            elif high >= position["target2"]:
                exit_reason = "target2"
                exit_price = position["target2"] - tick_slip
            elif holding_minutes >= 240:
                exit_reason = "time_240m"
                exit_price = float(next_bar["Open"])
            elif holding_minutes >= 60 and float(next_bar["Close"]) < position["entry"]:
                exit_reason = "weak_after_60m"
                exit_price = float(next_bar["Open"])
            if exit_reason:
                pnl = pnl_krw(symbol, position["entry"], exit_price, position["qty"])
                cash += pnl
                trades.append({"pnl": pnl, "return_pct": pnl / capital * 100, "hold_minutes": holding_minutes, "reason": exit_reason})
                position = None

        if position is None:
            intraday = compute_intraday_details(signal_data)
            plan = calculate_intraday_plan(signal_data)
            final_score = intraday.score
            korean_time_reason = intraday.forced_hold_reason in {"09:10 전 신규 진입 금지", "14:50 이후 신규 진입 금지"}
            futures_chasing_filter = intraday.chasing_filter and not korean_time_reason
            if final_score >= 75 and intraday.score >= 72 and not futures_chasing_filter and not plan.skip:
                limit_price = float(next_bar["Open"])
                fill_price = None
                if float(next_bar["Low"]) <= limit_price:
                    fill_price = limit_price + tick_slip
                if fill_price is not None:
                    sizing = calc_qty(symbol, fill_price, float(plan.stop_loss), cash)
                    if sizing["qty"] >= 1 and sizing["margin_check_passed"]:
                        position = {
                            "entry": fill_price,
                            "qty": sizing["qty"],
                            "stop_loss": float(plan.stop_loss),
                            "target1": float(plan.target1),
                            "target2": float(plan.target2),
                            "entry_time": now,
                        }
                        entry_count += 1

        mark = position["qty"] * pnl_krw(symbol, position["entry"], float(next_bar["Close"]), 1) if position else 0.0
        equity_curve.append(cash + mark)

    equity = np.array(equity_curve or [capital], dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / peaks - 1) * 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_return = float(np.mean([t["return_pct"] for t in trades])) if trades else 0.0
    return {
        "symbol": symbol,
        "total_return": (cash / capital - 1) * 100,
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "mdd": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "entry_count": entry_count,
        "trade_count": len(trades),
        "avg_return": avg_return,
        "avg_holding_minutes": float(np.mean([t["hold_minutes"] for t in trades])) if trades else 0.0,
        "profit_factor": profit_factor,
        "expectancy": avg_return,
    }


# print_result는 선물 백테스트 결과를 한국어로 출력합니다.
def print_result(result: dict) -> None:
    print(f"선물 백테스트: {result['symbol']}")
    print(f"총 수익률: {result['total_return']:.2f}%")
    print(f"승률: {result['win_rate']:.2f}%")
    print(f"MDD: {result['mdd']:.2f}%")
    print(f"진입횟수: {result['entry_count']}")
    print(f"거래횟수: {result['trade_count']}")
    print(f"평균 수익률: {result['avg_return']:.4f}%")
    print(f"평균 보유시간: {result['avg_holding_minutes']:.1f}분")
    print(f"profit_factor: {result['profit_factor']:.2f}")
    print(f"expectancy: {result['expectancy']:.4f}%")


# main은 CLI 인자를 해석해 watchlist 선물 백테스트를 실행합니다.
def main() -> int:
    parser = argparse.ArgumentParser(description="V3.2 futures paper-sim backtest")
    parser.add_argument("--watchlist", default="MNQ,MES")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--capital", type=float, default=FUTURES_PAPER_CAPITAL_KRW)
    args = parser.parse_args()

    for symbol in [s.strip().upper() for s in args.watchlist.split(",") if s.strip()]:
        try:
            print_result(run_backtest(symbol, args.period, args.capital))
        except Exception as exc:
            print(f"{symbol}: 백테스트 skip - {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
