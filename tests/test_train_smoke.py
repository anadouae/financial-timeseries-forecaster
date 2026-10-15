"""One-step training smoke test — must pass without GPU."""

from __future__ import annotations

from pathlib import Path

from fts_forecaster.config import Settings
from fts_forecaster.data import generate_ohlcv, prepare_features, temporal_split
from fts_forecaster.train import run_training


def test_synthetic_data_shapes():
    df = generate_ohlcv(n_days=120, seed=0)
    assert len(df) == 120
    assert {"open", "high", "low", "close", "volume", "target_vol"}.issubset(df.columns)
    assert (df["high"] >= df["low"]).all()
    bundle = prepare_features(df)
    assert bundle.features.shape == (120, 5)
    train, val, test = temporal_split(bundle, lookback=20, horizon=1)
    assert len(train) + len(val) + len(test) == 120 - 20


def test_train_one_epoch_smoke(tmp_path: Path):
    cfg = Settings(
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        smoke_n_days=180,
        smoke_epochs=1,
        lookback=20,
        hidden_size=16,
        num_layers=1,
        batch_size=32,
        device="cpu",
    )
    metrics = run_training(cfg, smoke=True, log_mlflow=False)
    assert "rmse" in metrics and "mae" in metrics
    assert (tmp_path / "artifacts" / "best_model.pt").exists()
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
