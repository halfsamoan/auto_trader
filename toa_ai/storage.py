"""SQLite-backed TOA Memory.

The memory is append-first: market bars, decisions, orders, fills, experiences,
and model promotion events remain queryable for later offline learning.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from toa_ai.config import DEFAULT_DB_PATH
from toa_ai.domain import ModelRecord, new_id, utc_now_iso


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    market TEXT NOT NULL DEFAULT '',
    asset_class TEXT NOT NULL DEFAULT '',
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    collected_at TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars(symbol, timeframe, timestamp);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    model_id TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    expected_return REAL NOT NULL,
    risk_score REAL NOT NULL,
    accepted INTEGER NOT NULL,
    episode_id TEXT,
    state_json TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    gate_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol_time ON decisions(symbol, timestamp);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    decision_id TEXT,
    broker TEXT NOT NULL DEFAULT 'paper'
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL,
    slippage_bps REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    qty REAL NOT NULL,
    avg_price REAL NOT NULL,
    market_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    experience_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    model_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    next_return REAL NOT NULL,
    done INTEGER NOT NULL,
    episode_id TEXT,
    state_json TEXT NOT NULL,
    next_state_json TEXT NOT NULL,
    execution_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_symbol_time ON experiences(symbol, timestamp);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_type TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status, created_at);

CREATE TABLE IF NOT EXISTS promotion_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    challenger_id TEXT NOT NULL,
    champion_before TEXT,
    champion_after TEXT,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
"""


class TOAMemory:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def upsert_bars(
        self,
        symbol: str,
        frame: pd.DataFrame,
        timeframe: str = "5m",
        market: str = "",
        asset_class: str = "",
        source: str = "",
    ) -> int:
        if frame.empty:
            return 0
        rows = []
        data = self._normalize_bars(frame)
        collected_at = utc_now_iso()
        for ts, row in data.iterrows():
            rows.append(
                (
                    symbol,
                    market,
                    asset_class,
                    timeframe,
                    pd.Timestamp(ts).isoformat(),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    source,
                    collected_at,
                )
            )
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT INTO bars(symbol, market, asset_class, timeframe, timestamp, open, high, low, close, volume, source, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
                    market=excluded.market,
                    asset_class=excluded.asset_class,
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    source=excluded.source,
                    collected_at=excluded.collected_at
                """,
                rows,
            )
            return conn.total_changes - before

    def load_bars(self, symbol: str, timeframe: str = "5m", limit: int | None = None) -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
        """
        params: list[Any] = [symbol, timeframe]
        if limit is not None:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM (
                    SELECT timestamp, open, high, low, close, volume
                    FROM bars
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                ORDER BY timestamp
            """
            params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        return frame[["open", "high", "low", "close", "volume"]].astype(float)

    def list_symbols(self, timeframe: str = "5m", min_rows: int = 1) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, COUNT(*) AS rows
                FROM bars
                WHERE timeframe = ?
                GROUP BY symbol
                HAVING rows >= ?
                ORDER BY symbol
                """,
                (timeframe, int(min_rows)),
            ).fetchall()
        return [str(row["symbol"]) for row in rows]

    def append_decision(
        self,
        symbol: str,
        model_id: str,
        action: str,
        confidence: float,
        expected_return: float,
        risk_score: float,
        accepted: bool,
        state: dict[str, Any],
        policy: dict[str, Any],
        gate: dict[str, Any],
        episode_id: str | None = None,
    ) -> str:
        decision_id = new_id("dec")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions(decision_id, timestamp, symbol, model_id, action, confidence, expected_return, risk_score,
                                      accepted, episode_id, state_json, policy_json, gate_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    utc_now_iso(),
                    symbol,
                    model_id,
                    action,
                    float(confidence),
                    float(expected_return),
                    float(risk_score),
                    int(accepted),
                    episode_id,
                    _json(state),
                    _json(policy),
                    _json(gate),
                ),
            )
        return decision_id

    def append_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        status: str,
        reason: str,
        limit_price: float | None = None,
        decision_id: str | None = None,
        broker: str = "paper",
    ) -> str:
        order_id = new_id("ord")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders(order_id, timestamp, symbol, side, qty, order_type, limit_price, status, reason, decision_id, broker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    utc_now_iso(),
                    symbol,
                    side,
                    float(qty),
                    order_type,
                    None if limit_price is None else float(limit_price),
                    status,
                    reason,
                    decision_id,
                    broker,
                ),
            )
        return order_id

    def append_fill(self, order_id: str, symbol: str, side: str, qty: float, price: float, fee: float, slippage_bps: float) -> str:
        fill_id = new_id("fill")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO fills(fill_id, timestamp, order_id, symbol, side, qty, price, fee, slippage_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fill_id, utc_now_iso(), order_id, symbol, side, float(qty), float(price), float(fee), float(slippage_bps)),
            )
        return fill_id

    def upsert_position(self, symbol: str, qty: float, avg_price: float, market_price: float, metadata: dict[str, Any] | None = None) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute("SELECT opened_at FROM positions WHERE symbol = ?", (symbol,)).fetchone()
            opened_at = str(existing["opened_at"]) if existing else now
            if qty <= 0:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                return
            conn.execute(
                """
                INSERT INTO positions(symbol, qty, avg_price, market_price, opened_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    market_price=excluded.market_price,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (symbol, float(qty), float(avg_price), float(market_price), opened_at, now, _json(metadata or {})),
            )

    def load_positions(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
        return {str(row["symbol"]): dict(row) for row in rows}

    def append_experience(
        self,
        symbol: str,
        model_id: str,
        action: str,
        reward: float,
        next_return: float,
        done: bool,
        state: dict[str, Any],
        next_state: dict[str, Any],
        execution: dict[str, Any],
        episode_id: str | None = None,
    ) -> str:
        experience_id = new_id("exp")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiences(experience_id, timestamp, symbol, model_id, action, reward, next_return, done,
                                        episode_id, state_json, next_state_json, execution_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience_id,
                    utc_now_iso(),
                    symbol,
                    model_id,
                    action,
                    float(reward),
                    float(next_return),
                    int(done),
                    episode_id,
                    _json(state),
                    _json(next_state),
                    _json(execution),
                ),
            )
        return experience_id

    def load_experiences(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experiences ORDER BY timestamp"
        params: list[Any] = []
        if limit is not None:
            query = "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?"
            params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["state"] = _loads(str(item.pop("state_json", "{}")))
            item["next_state"] = _loads(str(item.pop("next_state_json", "{}")))
            item["execution"] = _loads(str(item.pop("execution_json", "{}")))
            out.append(item)
        return list(reversed(out)) if limit is not None else out

    def register_model(
        self,
        model_id: str,
        model_type: str,
        path: str | Path,
        metadata: dict[str, Any],
        metrics: dict[str, Any],
        status: str = "challenger",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_registry(model_id, created_at, model_type, path, metadata_json, metrics_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    model_type=excluded.model_type,
                    path=excluded.path,
                    metadata_json=excluded.metadata_json,
                    metrics_json=excluded.metrics_json,
                    status=excluded.status
                """,
                (model_id, utc_now_iso(), model_type, str(path), _json(metadata), _json(metrics), status),
            )

    def get_model(self, model_id: str) -> ModelRecord | None:
        if model_id == "champion":
            return self.get_champion()
        if model_id == "latest":
            return self.get_latest_model()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,)).fetchone()
        return _model_record(row) if row else None

    def get_champion(self) -> ModelRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_registry WHERE status = 'champion' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return _model_record(row) if row else None

    def get_latest_model(self) -> ModelRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM model_registry ORDER BY created_at DESC LIMIT 1").fetchone()
        return _model_record(row) if row else None

    def list_models(self, limit: int = 20) -> list[ModelRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM model_registry ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [_model_record(row) for row in rows if row]

    def set_champion(self, model_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'champion'")
            conn.execute("UPDATE model_registry SET status = 'champion' WHERE model_id = ?", (model_id,))

    def append_promotion_event(
        self,
        challenger_id: str,
        champion_before: str | None,
        champion_after: str | None,
        decision: str,
        reason: str,
        metrics: dict[str, Any],
    ) -> str:
        event_id = new_id("promo")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO promotion_events(event_id, timestamp, challenger_id, champion_before, champion_after, decision, reason, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, utc_now_iso(), challenger_id, champion_before, champion_after, decision, reason, _json(metrics)),
            )
        return event_id

    @staticmethod
    def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        rename = {str(col): str(col).strip().lower() for col in out.columns}
        out = out.rename(columns=rename)
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.dropna(subset=["timestamp"]).set_index("timestamp")
        needed = ["open", "high", "low", "close", "volume"]
        for col in needed:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close"])
        return out[needed].sort_index()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _model_record(row: sqlite3.Row) -> ModelRecord:
    return ModelRecord(
        model_id=str(row["model_id"]),
        model_type=str(row["model_type"]),
        path=str(row["path"]),
        status=str(row["status"]),
        metrics=_loads(str(row["metrics_json"])),
        metadata=_loads(str(row["metadata_json"])),
    )
