"""Dataset utilities for masked patch pretraining."""

from __future__ import annotations

import numpy as np

from ai.feature_builder import FEATURE_COLUMNS, build_feature_frame
from config import AI_TRAIN_UNIVERSE, WATCHLIST
from core.fetcher_intraday import load_intraday_cache


def resolve_universe(universe: str | None, watchlist: str | None = None) -> list[str]:
    if watchlist:
        return [code.strip() for code in watchlist.split(",") if code.strip()]
    if universe == "ai_train":
        return list(dict.fromkeys(AI_TRAIN_UNIVERSE))
    if universe == "watchlist":
        return [str(item["code"]) for item in WATCHLIST if item.get("asset_class") == "domestic-stock"]
    return ["005930", "000660"]


def load_pretrain_sequences(
    universe: str | None = "ai_train",
    watchlist: str | None = None,
    interval: str = "5m",
    sequence_length: int = 96,
) -> dict[str, object]:
    codes = resolve_universe(universe, watchlist)
    sequences = []
    symbols_used = []
    total_rows = 0
    row_counts = {}
    for code in codes:
        data = load_intraday_cache(code, interval)
        row_counts[code] = int(len(data))
        total_rows += int(len(data))
        if len(data) < sequence_length:
            continue
        features = build_feature_frame(data)
        if len(features) < sequence_length:
            continue
        for end in range(sequence_length - 1, len(features)):
            window = features[FEATURE_COLUMNS].iloc[end - sequence_length + 1 : end + 1]
            sequences.append(window.to_numpy(dtype=np.float32))
        symbols_used.append(code)
    return {
        "X": np.asarray(sequences, dtype=np.float32),
        "feature_columns": FEATURE_COLUMNS,
        "symbols_used": symbols_used,
        "total_rows": total_rows,
        "row_counts": row_counts,
    }
