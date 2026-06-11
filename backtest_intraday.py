# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-06-11 (V3.0)
# Dependency: config.py, core/*, risk_manager.py
# Description: 국내주식 5분봉 전략을 미래 데이터 누수 없이 백테스트합니다.
# ================================================================================

"""Intraday backtest using yfinance 5-minute bars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from config import AMOUNT_PER_TRADE, COMMISSION, RISK_PER_TRADE_PCT, TAX
from core.fetcher_intraday import fetch_intraday
from core.intraday_scorer import compute_intraday_details
from core.market_regime import fetch_kospi_daily, historical_regime_for_date
from core.strategy import calculate_intraday_plan
from core.technical import atr, ensure_ohlcv, rsi, sma, vwap
from risk_manager import RiskManager


@dataclass
class Position:
    qty: int
    initial_qty: int
    avg_price: float
    stop_loss: float
    target1: float
    target2: float
    entry_time: datetime
    initial_risk: float
    highest_price: float
    realized_pnl: float = 0.0
    half_sold: bool = False
    technical_exit_count: int = 0


# _technical_exit_signal은 MA20 이탈, RSI 약세, VWAP 이탈이 동시에 나온 봉인지 판단합니다.
def _technical_exit_signal(data, price: float) -> bool:
    if len(data) < 20:
        return False
    ma20 = sma(data["Close"], 20).iloc[-1]
    current_rsi = rsi(data["Close"], 14).iloc[-1]
    current_vwap = vwap(data).iloc[-1]
    return bool(float(data["Close"].iloc[-1]) < float(ma20) and float(current_rsi) < 45 and price < float(current_vwap))


# _latest_atr은 ATR 트레일링 스탑용 5분봉 ATR 값을 반환합니다.
def _latest_atr(data, price: float) -> float:
    atr_series = atr(data, 14).dropna()
    latest = float(atr_series.iloc[-1]) if not atr_series.empty else price * 0.008
    return latest if latest > 0 else price * 0.008


# _sell는 매도 slippage, 수수료, 세금을 반영해 현금 유입과 PnL을 계산합니다.
def _sell(qty: int, raw_price: float, avg_price: float, commission: float, tax: float, slippage: float) -> tuple[float, float, float]:
    fill_price = raw_price * (1 - slippage)
    proceeds = qty * fill_price * (1 - commission - tax)
    cost_basis = qty * avg_price * (1 + commission)
    pnl = proceeds - cost_basis
    return proceeds, pnl, fill_price


# _max_consecutive_losses는 완료 거래의 최장 연속 손실 횟수를 계산합니다.
def _max_consecutive_losses(trades: list[dict]) -> int:
    max_count = 0
    current = 0
    for trade in trades:
        if trade["pnl"] < 0:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# _paper_status는 paper 투입 조건 4가지를 한국어 사유와 함께 판정합니다.
def _paper_status(total_return: float, mdd: float, avg_return: float, trade_count: int) -> tuple[bool, str]:
    failures = []
    if total_return <= 0:
        failures.append("총 수익률 0% 이하")
    if mdd <= -8:
        failures.append("MDD -8% 이하")
    if avg_return <= 0.15:
        failures.append("평균 수익률 +0.15% 이하")
    if trade_count < 20:
        failures.append("진입 횟수 20회 미만")
    if failures:
        return False, f"부적합 ({', '.join(failures)})"
    return True, "paper 투입 적합"


# run_backtest는 한 종목을 미래 데이터 누수 없이 순차 백테스트합니다.
def run_backtest(
    code: str,
    period: str,
    interval: str,
    amount: float,
    commission: float,
    tax: float,
    slippage: float,
    kospi_daily=None,
) -> dict:
    df = fetch_intraday(code, period=period, interval=interval, drop_incomplete=True)
    data = ensure_ohlcv(df)
    if len(data) < 40:
        raise RuntimeError("백테스트에 필요한 5분봉 데이터가 부족합니다.")

    kospi_daily = fetch_kospi_daily(period="1y") if kospi_daily is None else kospi_daily
    risk_manager = RiskManager(amount_per_trade=AMOUNT_PER_TRADE, risk_per_trade_pct=RISK_PER_TRADE_PCT)

    cash = float(amount)
    equity_curve = []
    trades = []
    position: Position | None = None
    signal_counts = {"buy": 0, "hold": 0, "avoid": 0, "risk_off": 0}
    stop_count = 0
    target1_partial_count = 0
    target2_exit_count = 0
    time_exit_count = 0
    weak_exit_count = 0
    technical_exit_count = 0
    trailing_exit_count = 0
    entry_count = 0

    for i in range(30, len(data)):
        # 미래 데이터 누수 방지: 시점 i의 진입 신호는 현재 봉을 제외한 df.iloc[:i]까지만 계산합니다.
        history = data.iloc[:i].copy()
        current_bar = data.iloc[i]
        now = data.index[i].to_pydatetime() if hasattr(data.index[i], "to_pydatetime") else datetime.now()
        price = float(current_bar["Close"])

        if position:
            holding_minutes = (now - position.entry_time).total_seconds() / 60
            profit_pct = (price / position.avg_price - 1) * 100 if position.avg_price else 0.0
            exit_reason = None
            sell_qty = 0
            position.highest_price = max(position.highest_price, price)

            if position.half_sold:
                trailing_stop = position.highest_price - 1.0 * _latest_atr(data.iloc[: i + 1], price)
                position.stop_loss = max(position.stop_loss, position.avg_price, trailing_stop)
                if price <= trailing_stop and profit_pct > 1.0:
                    exit_reason = "atr_trailing_stop"
                    sell_qty = position.qty
                    trailing_exit_count += 1

            if exit_reason is None and price <= position.stop_loss:
                exit_reason = "stop_loss"
                sell_qty = position.qty
                stop_count += 1
            elif exit_reason is None and price >= position.target1 and not position.half_sold:
                sell_qty = max(position.qty // 2, 1)
                position.qty -= sell_qty
                position.half_sold = True
                position.stop_loss = position.avg_price
                proceeds, pnl, _ = _sell(sell_qty, price, position.avg_price, commission, tax, slippage)
                cash += proceeds
                position.realized_pnl += pnl
                target1_partial_count += 1
                if position.qty <= 0:
                    trades.append(_trade_record(position, now, "target1_partial_only"))
                    position = None
            elif exit_reason is None and price >= position.target2:
                exit_reason = "target2"
                sell_qty = position.qty
                target2_exit_count += 1
            elif exit_reason is None and holding_minutes >= 240:
                exit_reason = "time_240m"
                sell_qty = position.qty
                time_exit_count += 1
            elif exit_reason is None and holding_minutes >= 60 and profit_pct < 0.3:
                exit_reason = "weak_after_60m"
                sell_qty = position.qty
                weak_exit_count += 1
            elif exit_reason is None:
                if _technical_exit_signal(data.iloc[: i + 1], price):
                    position.technical_exit_count += 1
                else:
                    position.technical_exit_count = 0
                if position.technical_exit_count >= 2:
                    exit_reason = "technical_breakdown_2bars"
                    sell_qty = position.qty
                    technical_exit_count += 1

            if position and exit_reason:
                proceeds, pnl, _ = _sell(sell_qty, price, position.avg_price, commission, tax, slippage)
                cash += proceeds
                position.realized_pnl += pnl
                position.qty -= sell_qty
                trades.append(_trade_record(position, now, exit_reason))
                position = None

        if position is None and i < len(data):
            regime = historical_regime_for_date(kospi_daily, data.index[i])
            if regime == "risk_off":
                signal_counts["risk_off"] += 1
                signal = "avoid"
            else:
                intraday = compute_intraday_details(history)
                plan = calculate_intraday_plan(history)
                if intraday.chasing_filter or plan.skip:
                    signal = "hold"
                elif intraday.score >= 72:
                    signal = "buy"
                elif intraday.score >= 55:
                    signal = "hold"
                else:
                    signal = "avoid"
                if signal == "buy":
                    # 미래 데이터 누수 방지: 신호 다음 봉의 Open으로 체결가를 확정합니다.
                    entry_raw = float(current_bar["Open"])
                    entry_fill = entry_raw * (1 + slippage)
                    stop_distance = max(plan.entry - plan.stop_loss, entry_fill * 0.003)
                    stop_loss = entry_fill - stop_distance
                    target1 = entry_fill + 1.5 * stop_distance
                    target2 = entry_fill + 2.5 * stop_distance
                    sizing = risk_manager.calculate_position_size(entry_fill, stop_loss, account_value=cash, available_cash=cash)
                    qty = int(sizing["qty"])
                    if qty > 0:
                        cost = qty * entry_fill * (1 + commission)
                        if cost <= cash:
                            cash -= cost
                            position = Position(qty, qty, entry_fill, stop_loss, target1, target2, now, stop_distance, entry_fill)
                            entry_count += 1
                        else:
                            signal = "hold"
                    else:
                        signal = "hold"
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        market_value = position.qty * price if position else 0.0
        equity_curve.append(cash + market_value)

    if position:
        final_price = float(data["Close"].iloc[-1])
        now = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else datetime.now()
        proceeds, pnl, _ = _sell(position.qty, final_price, position.avg_price, commission, tax, slippage)
        cash += proceeds
        position.realized_pnl += pnl
        trades.append(_trade_record(position, now, "final_close"))

    equity = np.array(equity_curve or [amount], dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / peaks - 1) * 100

    total_return = (cash / amount - 1) * 100
    mdd = float(drawdowns.min()) if len(drawdowns) else 0.0
    returns = [t["return_pct"] for t in trades]
    avg_return = float(np.mean(returns)) if returns else 0.0
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    average_win = float(np.mean([t["return_pct"] for t in wins])) if wins else 0.0
    average_loss = float(np.mean([t["return_pct"] for t in losses])) if losses else 0.0
    expectancy = avg_return
    avg_r_multiple = float(np.mean([t["r_multiple"] for t in trades])) if trades else 0.0
    paper_suitable, paper_status = _paper_status(total_return, mdd, avg_return, entry_count)

    return {
        "code": code,
        "total_return": total_return,
        "win_rate": (len(wins) / len(trades) * 100) if trades else 0.0,
        "mdd": mdd,
        "trade_count": len(trades),
        "entry_count": entry_count,
        "avg_holding_minutes": float(np.mean([t["hold_minutes"] for t in trades])) if trades else 0.0,
        "avg_return": avg_return,
        "signal_counts": signal_counts,
        "stop_count": stop_count,
        "target1_partial_count": target1_partial_count,
        "target2_exit_count": target2_exit_count,
        "time_exit_count": time_exit_count,
        "weak_exit_count": weak_exit_count,
        "technical_exit_count": technical_exit_count,
        "trailing_exit_count": trailing_exit_count,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "avg_R_multiple": avg_r_multiple,
        "paper_suitable": paper_suitable,
        "paper_status": paper_status,
    }


# _trade_record는 완료 거래의 수익률, PnL, R-multiple, 보유시간을 표준화합니다.
def _trade_record(position: Position, exit_time: datetime, reason: str) -> dict:
    hold_minutes = (exit_time - position.entry_time).total_seconds() / 60
    initial_cost = position.initial_qty * position.avg_price
    initial_risk_cash = position.initial_qty * position.initial_risk
    return_pct = (position.realized_pnl / initial_cost * 100) if initial_cost else 0.0
    r_multiple = (position.realized_pnl / initial_risk_cash) if initial_risk_cash else 0.0
    return {
        "return_pct": return_pct,
        "hold_minutes": hold_minutes,
        "reason": reason,
        "pnl": position.realized_pnl,
        "r_multiple": r_multiple,
    }


# print_result는 종목별 백테스트 결과를 한국어로 출력합니다.
def print_result(result: dict) -> None:
    print(f"백테스트 종목: {result['code']}")
    print(f"총 수익률: {result['total_return']:.2f}%")
    print(f"승률: {result['win_rate']:.2f}%")
    print(f"최대낙폭 MDD: {result['mdd']:.2f}%")
    print(f"거래횟수: {result['trade_count']} / 진입횟수: {result['entry_count']}")
    print(f"평균 보유시간: {result['avg_holding_minutes']:.1f}분")
    print(f"평균 수익률: {result['avg_return']:.2f}%")
    print(f"profit_factor: {result['profit_factor']:.2f}")
    print(f"expectancy: {result['expectancy']:.2f}%")
    print(f"average_win: {result['average_win']:.2f}%")
    print(f"average_loss: {result['average_loss']:.2f}%")
    print(f"max_consecutive_losses: {result['max_consecutive_losses']}")
    print(f"risk_per_trade_pct: {result['risk_per_trade_pct']:.4f}")
    print(f"avg_R_multiple: {result['avg_R_multiple']:.2f}")
    print(f"신호 발생 빈도: {result['signal_counts']}")
    print(f"손절 횟수: {result['stop_count']}")
    print(f"1차 목표 부분익절 횟수: {result['target1_partial_count']}")
    print(f"2차 목표 전량청산 횟수: {result['target2_exit_count']}")
    print(f"시간청산 횟수: {result['time_exit_count']}")
    print(f"약세청산 횟수: {result['weak_exit_count']}")
    print(f"기술적 청산 횟수: {result['technical_exit_count']}")
    print(f"ATR 트레일링 청산 횟수: {result['trailing_exit_count']}")
    print(f"paper 판정: {result['paper_status']}")


# print_summary_table은 watchlist 백테스트 결과와 합계를 출력합니다.
def print_summary_table(results: list[dict], amount: float) -> None:
    print("\n요약 테이블")
    print("종목 | 수익률 | MDD | 평균수익률 | 거래횟수 | profit_factor | 판정")
    print("-" * 78)
    for result in results:
        print(
            f"{result['code']} | {result['total_return']:.2f}% | {result['mdd']:.2f}% | "
            f"{result['avg_return']:.2f}% | {result['trade_count']} | {result['profit_factor']:.2f} | {result['paper_status']}"
        )
    avg_return = float(np.mean([r["total_return"] for r in results])) if results else 0.0
    total_trades = sum(r["trade_count"] for r in results)
    print(f"합계/평균 | 평균수익률 {avg_return:.2f}% | 총 거래횟수 {total_trades} | 기준 원금 {amount:,.0f}원")


# main은 CLI 인자를 해석해 단일 종목 또는 watchlist 백테스트를 실행합니다.
def main() -> int:
    parser = argparse.ArgumentParser(description="KR intraday 5m backtest")
    parser.add_argument("code", nargs="?")
    parser.add_argument("--watchlist", help="쉼표로 구분한 종목 목록 예: 005930,000660")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--amount", type=float, default=10_000_000)
    parser.add_argument("--kr-amount", type=float, help="국내주식 백테스트 원금. 지정 시 --amount보다 우선합니다.")
    parser.add_argument("--us-amount", type=float, help="V3.0 KR Final에서는 무시됩니다.")
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--tax", type=float, default=TAX)
    parser.add_argument("--slippage", type=float, default=0.0005)
    args = parser.parse_args()

    amount = float(args.kr_amount if args.kr_amount is not None else args.amount)
    codes = []
    if args.watchlist:
        codes = [code.strip() for code in args.watchlist.split(",") if code.strip()]
    elif args.code:
        codes = [args.code]
    else:
        parser.error("종목 코드 또는 --watchlist가 필요합니다.")

    if args.us_amount is not None:
        print("V3.0 KR Final은 국내주식 전용입니다. --us-amount 값은 무시합니다.")

    kospi_daily = fetch_kospi_daily(period="1y")
    results = []
    for code in codes:
        result = run_backtest(code, args.period, args.interval, amount, args.commission, args.tax, args.slippage, kospi_daily)
        results.append(result)
        print_result(result)
        if len(codes) > 1:
            print()
    if len(results) > 1:
        print_summary_table(results, amount)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
