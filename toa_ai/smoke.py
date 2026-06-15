"""Synthetic smoke workflow for TOA AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from toa_ai.config import ModelConfig, PromotionConfig, ReplayConfig
from toa_ai.policy import TOAPolicy
from toa_ai.promotion import PromotionGate
from toa_ai.replay import run_replay_for_symbol
from toa_ai.storage import TOAMemory
from toa_ai.trainer import train_policy_v1


def run_smoke_workflow(db_path: str | Path, epochs: int = 1) -> dict[str, Any]:
    memory = TOAMemory(db_path)
    symbols = ["TOA001", "TOA002", "TOA003"]
    for idx, symbol in enumerate(symbols):
        memory.upsert_bars(symbol, _synthetic_bars(seed=idx), timeframe="5m", source="synthetic")
    model_dir = Path(db_path).parent / "models"
    train_result = train_policy_v1(
        memory,
        ModelConfig(sequence_length=32, epochs=epochs, min_rows_per_symbol=80, batch_size=64),
        model_dir=model_dir,
        symbols=symbols,
    )
    promo = PromotionGate(
        memory,
        PromotionConfig(min_train_samples=100, min_validation_samples=20, max_validation_loss=5.0, min_action_entropy=0.01, max_no_action_ratio=1.0),
    ).promote(train_result["model_id"])
    policy = TOAPolicy.load(train_result["model_path"])
    replay = run_replay_for_symbol(memory, policy, "TOA001", sequence_length=32, replay_config=ReplayConfig(max_episode_bars=80))
    return {"train": train_result, "promotion": promo, "replay": replay, "db_path": str(db_path)}


def _synthetic_bars(seed: int, rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01 09:00", periods=rows, freq="5min", tz="Asia/Seoul")
    drift = 0.00005 + seed * 0.00001
    shocks = rng.normal(drift, 0.003, size=rows)
    close = 100.0 * np.exp(np.cumsum(shocks))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * rng.uniform(0.001, 0.006, size=rows), 0.01)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(10_000, 90_000, size=rows)
    return pd.DataFrame({"timestamp": index, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
