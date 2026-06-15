"""Configuration defaults for TOA AI.

The defaults focus on model completeness and safe iteration. Profit metrics are
recorded as diagnostics, not as the primary promotion target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOA_DATA_DIR = PROJECT_ROOT / "data" / "toa_ai"
DEFAULT_DB_PATH = TOA_DATA_DIR / "toa_memory.sqlite3"
DEFAULT_MODEL_DIR = TOA_DATA_DIR / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "toa_ai"


@dataclass(frozen=True)
class ModelConfig:
    timeframe: str = "5m"
    sequence_length: int = 64
    horizon_bars: int = 6
    target_return: float = 0.0035
    stop_return: float = -0.0025
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.10
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 4
    validation_fraction: float = 0.20
    min_rows_per_symbol: int = 120


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 1.5
    slippage_bps: float = 3.0

    @property
    def commission_rate(self) -> float:
        return self.commission_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass(frozen=True)
class RiskConfig:
    max_open_positions: int = 3
    max_position_value_fraction: float = 0.20
    max_total_exposure_fraction: float = 0.60
    max_daily_loss_fraction: float = 0.03
    min_model_confidence_for_execution: float = 0.40
    allow_live_order: bool = False


@dataclass(frozen=True)
class PromotionConfig:
    min_train_samples: int = 500
    min_validation_samples: int = 100
    max_validation_loss: float = 2.50
    min_action_entropy: float = 0.20
    max_no_action_ratio: float = 0.92
    challenger_loss_tolerance: float = 0.05


@dataclass(frozen=True)
class ReplayConfig:
    initial_cash: float = 10_000_000.0
    max_episode_bars: int | None = None
    record_experiences: bool = True
