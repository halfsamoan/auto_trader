"""Command line interface for TOA AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toa_ai.config import DEFAULT_DB_PATH, DEFAULT_MODEL_DIR, ModelConfig, ReplayConfig
from toa_ai.data_collector import ingest_cache_directory
from toa_ai.external_data import ingest_external_csv
from toa_ai.paper_loop import run_paper_once
from toa_ai.policy import NoopPolicy, TOAPolicy
from toa_ai.promotion import PromotionGate
from toa_ai.replay import run_replay_for_symbol
from toa_ai.smoke import run_smoke_workflow
from toa_ai.storage import TOAMemory
from toa_ai.trainer import train_policy_v1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TOA AI model platform")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")

    ingest = sub.add_parser("ingest-cache")
    ingest.add_argument("--cache-dir", default="data/intraday_cache")
    ingest.add_argument("--timeframe", default="5m")
    ingest.add_argument("--symbols", default=None, help="comma-separated symbols")
    ingest.add_argument("--limit-symbols", type=int, default=None)

    external = sub.add_parser("ingest-external")
    external.add_argument("--path", required=True)
    external.add_argument("--symbol", required=True)
    external.add_argument("--timeframe", default="1m")
    external.add_argument("--market", default="CRYPTO")
    external.add_argument("--asset-class", default="crypto")
    external.add_argument("--format", default="auto", choices=["auto", "binance", "generic"])

    train = sub.add_parser("train")
    train.add_argument("--timeframe", default="5m")
    train.add_argument("--sequence-length", type=int, default=64)
    train.add_argument("--horizon-bars", type=int, default=6)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--max-symbols", type=int, default=None)
    train.add_argument("--symbols", default=None)
    train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))

    promote = sub.add_parser("promote")
    promote.add_argument("--model-id", default="latest")

    replay = sub.add_parser("replay")
    replay.add_argument("--model-id", default="champion")
    replay.add_argument("--symbols", default=None)
    replay.add_argument("--timeframe", default="5m")
    replay.add_argument("--sequence-length", type=int, default=64)
    replay.add_argument("--max-symbols", type=int, default=3)
    replay.add_argument("--max-episode-bars", type=int, default=200)

    paper = sub.add_parser("paper-once")
    paper.add_argument("--model-id", default="champion")
    paper.add_argument("--symbols", default=None)
    paper.add_argument("--timeframe", default="5m")
    paper.add_argument("--sequence-length", type=int, default=64)
    paper.add_argument("--max-symbols", type=int, default=5)

    sub.add_parser("status")

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args(argv)
    memory = TOAMemory(args.db_path)

    if args.command == "init-db":
        print_json({"status": "ok", "db_path": str(memory.db_path)})
        return 0

    if args.command == "ingest-cache":
        symbols = _split_symbols(args.symbols)
        result = ingest_cache_directory(memory, args.cache_dir, args.timeframe, symbols=symbols, limit_symbols=args.limit_symbols)
        print_json(result)
        return 0

    if args.command == "ingest-external":
        result = ingest_external_csv(
            memory,
            args.path,
            args.symbol,
            timeframe=args.timeframe,
            market=args.market,
            asset_class=args.asset_class,
            source_format=args.format,
        )
        print_json(result)
        return 0

    if args.command == "train":
        result = train_policy_v1(
            memory,
            ModelConfig(
                timeframe=args.timeframe,
                sequence_length=args.sequence_length,
                horizon_bars=args.horizon_bars,
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
            model_dir=args.model_dir,
            symbols=_split_symbols(args.symbols),
            max_symbols=args.max_symbols,
        )
        print_json(result)
        return 0

    if args.command == "promote":
        model_id = _resolve_model_id(memory, args.model_id)
        result = PromotionGate(memory).promote(model_id)
        print_json(result)
        return 0 if result.get("promoted") else 1

    if args.command == "replay":
        policy = _load_policy(memory, args.model_id)
        symbols = _resolve_symbols(memory, args.symbols, args.timeframe, args.max_symbols)
        results = [
            run_replay_for_symbol(
                memory,
                policy,
                symbol,
                timeframe=args.timeframe,
                sequence_length=args.sequence_length,
                replay_config=ReplayConfig(max_episode_bars=args.max_episode_bars),
            )
            for symbol in symbols
        ]
        print_json({"model_id": getattr(policy, "model_id", "unknown"), "results": results})
        return 0

    if args.command == "paper-once":
        policy = _load_policy(memory, args.model_id)
        symbols = _resolve_symbols(memory, args.symbols, args.timeframe, args.max_symbols)
        print_json(run_paper_once(memory, policy, symbols, args.timeframe, args.sequence_length))
        return 0

    if args.command == "status":
        print_json(
            {
                "db_path": str(memory.db_path),
                "symbols": memory.list_symbols(),
                "champion": _record_json(memory.get_champion()),
                "models": [_record_json(model) for model in memory.list_models()],
            }
        )
        return 0

    if args.command == "smoke-test":
        print_json(run_smoke_workflow(args.db_path, epochs=args.epochs))
        return 0

    return 2


def _load_policy(memory: TOAMemory, model_id: str):
    record = memory.get_model(model_id)
    if record is None:
        return NoopPolicy()
    return TOAPolicy.load(record.path)


def _resolve_model_id(memory: TOAMemory, value: str) -> str:
    record = memory.get_model(value)
    if record is None:
        raise RuntimeError(f"model not found: {value}")
    return record.model_id


def _resolve_symbols(memory: TOAMemory, raw: str | None, timeframe: str, max_symbols: int) -> list[str]:
    symbols = _split_symbols(raw)
    if symbols:
        return symbols
    return memory.list_symbols(timeframe=timeframe, min_rows=1)[: int(max_symbols)]


def _split_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _record_json(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "model_id": record.model_id,
        "model_type": record.model_type,
        "path": record.path,
        "status": record.status,
        "metrics": record.metrics,
        "metadata": record.metadata,
    }


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
