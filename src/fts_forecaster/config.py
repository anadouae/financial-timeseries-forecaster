"""Central configuration for data, model, training, and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FTS_", env_file=".env", extra="ignore")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    artifact_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "volatility-lstm"

    # Synthetic market data
    n_days: int = 1500
    seed: int = 42
    lookback: int = 30
    forecast_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # LSTM
    input_size: int = 5  # OHLCV features (normalized)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False

    # Training
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: Literal["cpu", "cuda", "auto"] = "auto"
    smoke_epochs: int = 1
    smoke_n_days: int = 200

    # API
    model_path: Path = Field(default_factory=lambda: Path("artifacts/best_model.pt"))
    host: str = "0.0.0.0"
    port: int = 8000
    # When set (env: FTS_API_KEY), /predict requires X-API-Key or Authorization: Bearer.
    # Leave unset/empty for local unauthenticated use; production should always set it.
    api_key: Optional[str] = None


settings = Settings()
