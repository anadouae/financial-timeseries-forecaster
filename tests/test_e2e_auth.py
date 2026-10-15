"""End-to-end: generate data → train (+ MLflow) → checkpoint → authenticated /predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from fts_forecaster import api as api_mod
from fts_forecaster.config import Settings
from fts_forecaster.train import run_training

API_KEY = "e2e-test-secret-key-not-for-prod"


def test_e2e_train_mlflow_predict_requires_api_key(monkeypatch, tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    mlflow_db = tmp_path / "mlflow.db"
    cfg = Settings(
        artifact_dir=artifact_dir,
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
        experiment_name="e2e-auth",
        smoke_n_days=180,
        smoke_epochs=2,
        lookback=15,
        hidden_size=8,
        num_layers=1,
        batch_size=16,
        device="cpu",
        input_size=5,
        api_key=API_KEY,
    )

    metrics = run_training(cfg, smoke=True, log_mlflow=True)
    assert "rmse" in metrics
    ckpt = artifact_dir / "best_model.pt"
    assert ckpt.is_file()
    # MLflow SQLite tracking store written
    assert mlflow_db.is_file()
    assert mlflow_db.stat().st_size > 0

    monkeypatch.setattr(api_mod.settings, "model_path", ckpt)
    monkeypatch.setattr(api_mod.settings, "api_key", API_KEY)
    api_mod.bundle = api_mod.ModelBundle()
    api_mod.bundle.load(ckpt)

    lookback = api_mod.bundle.lookback
    body = {
        "sequence": np.random.randn(lookback, 5).tolist(),
        "already_normalized": True,
    }

    with TestClient(api_mod.app) as client:
        # Health remains public
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        # Missing key → 401
        no_key = client.post("/predict", json=body)
        assert no_key.status_code == 401

        # Wrong key → 401
        bad = client.post(
            "/predict",
            json=body,
            headers={"X-API-Key": "wrong-key"},
        )
        assert bad.status_code == 401

        # X-API-Key → success
        ok_header = client.post(
            "/predict",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        assert ok_header.status_code == 200
        assert "predicted_volatility" in ok_header.json()

        # Bearer → success
        ok_bearer = client.post(
            "/predict",
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert ok_bearer.status_code == 200
        assert isinstance(ok_bearer.json()["predicted_volatility"], float)
