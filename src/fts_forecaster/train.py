"""Training loop with MLflow tracking and checkpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fts_forecaster.config import Settings, settings
from fts_forecaster.data import generate_ohlcv, make_loaders, prepare_features, temporal_split
from fts_forecaster.metrics import evaluate_arrays
from fts_forecaster.model import VolatilityLSTM


def resolve_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        preds.append(out.cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_training(
    cfg: Optional[Settings] = None,
    *,
    smoke: bool = False,
    log_mlflow: bool = True,
) -> Dict[str, float]:
    cfg = cfg or settings
    device = resolve_device(cfg.device)

    n_days = cfg.smoke_n_days if smoke else cfg.n_days
    epochs = cfg.smoke_epochs if smoke else cfg.epochs

    df = generate_ohlcv(n_days=n_days, seed=cfg.seed)
    bundle = prepare_features(df)
    train_ds, val_ds, test_ds = temporal_split(
        bundle,
        lookback=cfg.lookback,
        horizon=cfg.forecast_horizon,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=cfg.batch_size
    )

    model = VolatilityLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=0.0 if smoke else cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / "best_model.pt"
    best_val = float("inf")
    history = []

    if log_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        run_ctx = mlflow.start_run(run_name="smoke" if smoke else "train")
    else:
        run_ctx = None

    try:
        if log_mlflow:
            mlflow.log_params(
                {
                    "n_days": n_days,
                    "epochs": epochs,
                    "lookback": cfg.lookback,
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "device": str(device),
                    "n_params": model.count_parameters(),
                    "smoke": smoke,
                }
            )

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, cfg.grad_clip
            )
            y_true, y_pred = predict_loader(model, val_loader, device)
            val_metrics = evaluate_arrays(
                y_true, y_pred, bundle.target_mean, bundle.target_std
            )
            val_rmse = val_metrics["rmse"]
            scheduler.step(val_rmse)
            row = {"epoch": epoch, "train_mse": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            if log_mlflow:
                mlflow.log_metrics(
                    {"train_mse": train_loss, "val_rmse": val_rmse, "val_mae": val_metrics["mae"]},
                    step=epoch,
                )
            print(
                f"epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"val_rmse={val_rmse:.6f} val_mae={val_metrics['mae']:.6f}"
            )
            if val_rmse < best_val:
                best_val = val_rmse
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": {
                            "input_size": cfg.input_size,
                            "hidden_size": cfg.hidden_size,
                            "num_layers": cfg.num_layers,
                            "dropout": cfg.dropout,
                            "lookback": cfg.lookback,
                        },
                        "norm": {
                            "feature_mean": bundle.feature_mean.tolist(),
                            "feature_std": bundle.feature_std.tolist(),
                            "target_mean": bundle.target_mean,
                            "target_std": bundle.target_std,
                        },
                        "val_rmse": best_val,
                    },
                    best_path,
                )

        # Final test evaluation with best checkpoint
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        y_true, y_pred = predict_loader(model, test_loader, device)
        test_metrics = evaluate_arrays(
            y_true, y_pred, bundle.target_mean, bundle.target_std
        )
        print(
            f"test_rmse={test_metrics['rmse']:.6f} test_mae={test_metrics['mae']:.6f} "
            f"test_mape={test_metrics['mape']:.2f}%"
        )

        meta = {
            "history": history,
            "test_metrics": test_metrics,
            "best_val_rmse": best_val,
            "n_params": model.count_parameters(),
        }
        (artifact_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
        if log_mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_artifact(str(best_path))
            mlflow.log_artifact(str(artifact_dir / "metrics.json"))

        return test_metrics
    finally:
        if run_ctx is not None:
            mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train volatility LSTM")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU-friendly smoke run")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-days", type=int, default=None)
    args = parser.parse_args()

    cfg = settings.model_copy(deep=True)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_days is not None:
        cfg.n_days = args.n_days

    run_training(cfg, smoke=args.smoke, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
