"""Paper execution adapter for TOA AI."""

from __future__ import annotations

from toa_ai.config import CostConfig, RiskConfig
from toa_ai.domain import AccountState, OrderRequest, OrderResult, Position, new_id, utc_now_iso
from toa_ai.storage import TOAMemory


class PaperBroker:
    def __init__(
        self,
        memory: TOAMemory,
        initial_cash: float = 10_000_000.0,
        cost_config: CostConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.memory = memory
        self.cost_config = cost_config or CostConfig()
        self.risk_config = risk_config or RiskConfig()
        self.account = AccountState(cash=float(initial_cash), equity=float(initial_cash))
        self._load_positions()

    def mark(self, symbol: str, price: float) -> None:
        if symbol in self.account.positions:
            self.account.positions[symbol].market_price = float(price)
            self.account.positions[symbol].updated_at = utc_now_iso()
            pos = self.account.positions[symbol]
            self.memory.upsert_position(symbol, pos.qty, pos.avg_price, pos.market_price, {"source": "paper"})
        self._refresh_equity()

    def submit_market_order(self, request: OrderRequest) -> OrderResult:
        if request.qty <= 0 or request.price <= 0:
            order_id = self.memory.append_order(request.symbol, request.side, request.qty, request.order_type, "rejected", "invalid_qty_or_price", request.price, request.decision_id)
            return OrderResult(order_id, None, False, False, "invalid_qty_or_price", request.symbol, request.side, request.qty)

        side = request.side.lower()
        fill_price = self._fill_price(side, request.price)
        gross = request.qty * fill_price
        fee = gross * self.cost_config.commission_rate
        if side == "buy" and gross + fee > self.account.cash:
            order_id = self.memory.append_order(request.symbol, request.side, request.qty, request.order_type, "rejected", "insufficient_cash", request.price, request.decision_id)
            return OrderResult(order_id, None, False, False, "insufficient_cash", request.symbol, request.side, request.qty)
        if side == "sell":
            position = self.account.positions.get(request.symbol)
            if not position or position.qty <= 0:
                order_id = self.memory.append_order(request.symbol, request.side, request.qty, request.order_type, "rejected", "no_position", request.price, request.decision_id)
                return OrderResult(order_id, None, False, False, "no_position", request.symbol, request.side, request.qty)
            request = OrderRequest(request.symbol, request.side, min(request.qty, position.qty), request.price, request.decision_id, request.order_type)

        order_id = self.memory.append_order(request.symbol, request.side, request.qty, request.order_type, "filled", "paper_fill", request.price, request.decision_id)
        fill_id = self.memory.append_fill(order_id, request.symbol, request.side, request.qty, fill_price, fee, self.cost_config.slippage_bps)
        if side == "buy":
            self._apply_buy(request.symbol, request.qty, fill_price, fee)
        elif side == "sell":
            self._apply_sell(request.symbol, request.qty, fill_price, fee)
        else:
            return OrderResult(order_id, None, False, False, "unknown_side", request.symbol, request.side, request.qty)
        self._refresh_equity()
        return OrderResult(order_id, fill_id, True, True, "paper_fill", request.symbol, request.side, request.qty, fill_price, fee)

    def _apply_buy(self, symbol: str, qty: float, price: float, fee: float) -> None:
        now = utc_now_iso()
        current = self.account.positions.get(symbol)
        cost = qty * price + fee
        self.account.cash -= cost
        if current:
            total_qty = current.qty + qty
            avg_price = ((current.qty * current.avg_price) + (qty * price)) / max(total_qty, 1e-9)
            current.qty = total_qty
            current.avg_price = avg_price
            current.market_price = price
            current.updated_at = now
            position = current
        else:
            position = Position(symbol, qty, price, price, now, now)
            self.account.positions[symbol] = position
        self.memory.upsert_position(symbol, position.qty, position.avg_price, position.market_price, {"source": "paper"})

    def _apply_sell(self, symbol: str, qty: float, price: float, fee: float) -> None:
        current = self.account.positions[symbol]
        sell_qty = min(qty, current.qty)
        proceeds = sell_qty * price - fee
        realized = (price - current.avg_price) * sell_qty - fee
        self.account.cash += proceeds
        self.account.realized_pnl += realized
        self.account.daily_realized_pnl += realized
        current.qty -= sell_qty
        current.market_price = price
        current.updated_at = utc_now_iso()
        if current.qty <= 1e-9:
            self.account.positions.pop(symbol, None)
            self.memory.upsert_position(symbol, 0.0, current.avg_price, price, {"source": "paper"})
        else:
            self.memory.upsert_position(symbol, current.qty, current.avg_price, price, {"source": "paper"})

    def _fill_price(self, side: str, price: float) -> float:
        slip = self.cost_config.slippage_rate
        return float(price) * (1.0 + slip if side == "buy" else 1.0 - slip)

    def _refresh_equity(self) -> None:
        self.account.equity = self.account.cash + sum(pos.market_value for pos in self.account.positions.values())

    def _load_positions(self) -> None:
        for symbol, row in self.memory.load_positions().items():
            qty = float(row["qty"])
            if qty <= 0:
                continue
            position = Position(
                symbol=symbol,
                qty=qty,
                avg_price=float(row["avg_price"]),
                market_price=float(row["market_price"]),
                opened_at=str(row["opened_at"]),
                updated_at=str(row["updated_at"]),
            )
            self.account.positions[symbol] = position
        self._refresh_equity()
