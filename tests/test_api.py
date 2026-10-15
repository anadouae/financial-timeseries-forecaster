"""FastAPI TestClient coverage for /health and /predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from fts_forecaster import api as api_mod
from fts_forecaster.config import Settings
from fts_forecaster.model import VolatilityLSTM
from fts_forecaster.train import run_training


def _train_tiny_checkpoint(tmp_path: Path) -> Path:
    cfg = Settings(
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        smoke_n_days=160,
        smoke_epochs=1,
        lookback=15,
        hidden_size=8,
        num_layers=1,
        batch_size=16,
        device="cpu",
        input_size=5,
    )
    run_training(cfg, smoke=True, log_mlflow=False)
    return tmp_path / "artifacts" / "best_model.pt"


def test_health_without_model(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_mod.settings, "model_path", tmp_path / "missing.pt")
    monkeypatch.setattr(api_mod.settings, "api_key", None)
    api_mod.bundle = api_mod.ModelBundle()
    with TestClient(api_mod.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False


def test_predict_roundtrip(monkeypatch, tmp_path: Path):
    ckpt_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setattr(api_mod.settings, "model_path", ckpt_path)
    monkeypatch.setattr(api_mod.settings, "api_key", None)  # auth off when unset
    api_mod.bundle = api_mod.ModelBundle()
    api_mod.bundle.load(ckpt_path)

    lookback = api_mod.bundle.lookback
    seq = np.random.randn(lookback, 5).tolist()

    with TestClient(api_mod.app) as client:
        health = client.get("/health")
        assert health.json()["model_loaded"] is True

        bad = client.post("/predict", json={"sequence": [[0.0] * 5]})
        assert bad.status_code == 422

        ok = client.post(
            "/predict",
            json={"sequence": seq, "already_normalized": True},
        )
        assert ok.status_code == 200
        payload = ok.json()
        assert "predicted_volatility" in payload
        assert isinstance(payload["predicted_volatility"], float)
        assert payload["lookback"] == lookback


def test_model_bundle_load_roundtrip(tmp_path: Path):
    """Direct checkpoint load without full train for speed on CI edge."""
    model = VolatilityLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    path = tmp_path / "m.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_size": 5,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "lookback": 12,
            },
            "norm": {
                "feature_mean": [0.0] * 5,
                "feature_std": [1.0] * 5,
                "target_mean": 0.01,
                "target_std": 0.005,
            },
            "val_rmse": 0.02,
        },
        path,
    )
    b = api_mod.ModelBundle()
    b.load(path)
    assert b.loaded
    assert b.lookback == 12
    x = torch.zeros(1, 12, 5)
    with torch.no_grad():
        y = b.model(x)
    assert y.shape == (1,)
