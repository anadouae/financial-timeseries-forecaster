"""Volatility forecast metrics in original (de-normalized) scale."""

from __future__ import annotations

from typing import Dict

import numpy as np


def denormalize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y * std + mean


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_arrays(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    target_mean: float,
    target_std: float,
) -> Dict[str, float]:
    yt = denormalize(y_true_norm, target_mean, target_std)
    yp = denormalize(y_pred_norm, target_mean, target_std)
    return {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "mape": mape(yt, yp),
        "rmse_norm": rmse(y_true_norm, y_pred_norm),
        "mae_norm": mae(y_true_norm, y_pred_norm),
    }
