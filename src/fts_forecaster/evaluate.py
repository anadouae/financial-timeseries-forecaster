"""Evaluate a saved checkpoint on a held-out synthetic slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fts_forecaster.config import settings
from fts_forecaster.data import generate_ohlcv, make_loaders, prepare_features, temporal_split
from fts_forecaster.metrics import evaluate_arrays
from fts_forecaster.model import VolatilityLSTM
from fts_forecaster.train import predict_loader, resolve_device


def evaluate_checkpoint(model_path: Path, device_pref: str = "cpu") -> dict:
    device = resolve_device(device_pref)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = VolatilityLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    df = generate_ohlcv(n_days=settings.n_days, seed=settings.seed)
    bundle = prepare_features(df)
    _, _, test_ds = temporal_split(
        bundle,
        lookback=cfg["lookback"],
        horizon=settings.forecast_horizon,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
    )
    _, _, test_loader = make_loaders(test_ds, test_ds, test_ds, batch_size=settings.batch_size)
    y_true, y_pred = predict_loader(model, test_loader, device)
    metrics = evaluate_arrays(y_true, y_pred, bundle.target_mean, bundle.target_std)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate volatility LSTM checkpoint")
    parser.add_argument("--model", type=Path, default=settings.model_path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.device)


if __name__ == "__main__":
    main()
