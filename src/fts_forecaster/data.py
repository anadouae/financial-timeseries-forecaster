"""Synthetic OHLCV + realized-volatility series generation and windowing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SeriesBundle:
    """Normalized feature matrix X and target volatility y with metadata."""

    features: np.ndarray  # (T, F)
    target: np.ndarray  # (T,)
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    raw: pd.DataFrame


def generate_ohlcv(
    n_days: int = 1500,
    seed: int = 42,
    start_price: float = 100.0,
    base_vol: float = 0.015,
    vol_of_vol: float = 0.35,
) -> pd.DataFrame:
    """
    Simulate a geometric Brownian motion with stochastic volatility (CIR-like).

    Produces daily OHLCV bars and a next-day realized-volatility target proxy
    (absolute log-return rolling RMS), suitable for supervised forecasting demos.
    """
    rng = np.random.default_rng(seed)

    # Latent log-volatility mean-reverting process
    log_vol = np.log(base_vol)
    reversion = 0.08
    shocks = rng.normal(0.0, vol_of_vol * 0.1, size=n_days)
    vols = np.empty(n_days)
    for t in range(n_days):
        log_vol = log_vol + reversion * (np.log(base_vol) - log_vol) + shocks[t]
        vols[t] = np.exp(log_vol)

    # Correlated return / vol innovations (leverage effect)
    z = rng.standard_normal(n_days)
    w = -0.4 * z + np.sqrt(1 - 0.4**2) * rng.standard_normal(n_days)
    mu = 0.0002
    log_returns = mu + vols * z

    close = start_price * np.exp(np.cumsum(log_returns))
    # Intraday range scaled by instantaneous vol
    spread = vols * np.abs(w) * close
    high = close + spread * rng.uniform(0.4, 1.0, size=n_days)
    low = close - spread * rng.uniform(0.4, 1.0, size=n_days)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    # Ensure OHLC consistency
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.lognormal(mean=14.0, sigma=0.4, size=n_days) * (1.0 + 2.0 * vols / base_vol)

    # Realized vol proxy: 5-day RMS of absolute log returns (shifted for next-day target)
    abs_lr = np.abs(log_returns)
    window = 5
    kernel = np.ones(window) / window
    rv = np.sqrt(np.convolve(abs_lr**2, kernel, mode="same") * window)
    # Forecast target: next-day realized vol
    target_rv = np.roll(rv, -1)
    target_rv[-1] = rv[-1]

    dates = pd.bdate_range("2018-01-02", periods=n_days)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "log_return": log_returns,
            "instant_vol": vols,
            "realized_vol": rv,
            "target_vol": target_rv,
        }
    )
    return df


def prepare_features(df: pd.DataFrame) -> SeriesBundle:
    """Build model features from OHLCV and z-score normalize (fit on full series for demo)."""
    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    hl_range = (df["high"] - df["low"]).to_numpy() / np.maximum(close, 1e-8)
    oc_gap = (df["open"] - np.roll(close, 1)) / np.maximum(close, 1e-8)
    oc_gap[0] = 0.0
    log_vol = np.log(np.maximum(df["volume"].to_numpy(), 1.0))
    # Parkinson-style range vol proxy
    park = np.sqrt(
        np.maximum(
            (1.0 / (4.0 * np.log(2.0))) * (np.log(df["high"] / df["low"]).to_numpy() ** 2),
            0.0,
        )
    )

    features = np.column_stack([log_ret, hl_range, oc_gap, log_vol, park]).astype(np.float64)
    target = df["target_vol"].to_numpy(dtype=np.float64)

    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0)
    f_std = np.where(f_std < 1e-8, 1.0, f_std)
    t_mean = float(target.mean())
    t_std = float(target.std())
    if t_std < 1e-8:
        t_std = 1.0

    return SeriesBundle(
        features=(features - f_mean) / f_std,
        target=(target - t_mean) / t_std,
        feature_mean=f_mean,
        feature_std=f_std,
        target_mean=t_mean,
        target_std=t_std,
        raw=df,
    )


class SequenceDataset(Dataset):
    """Sliding windows of length `lookback` predicting horizon-step volatility."""

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lookback: int = 30,
        horizon: int = 1,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("features must be (T, F)")
        self.lookback = lookback
        self.horizon = horizon
        # Align: window ends at t, predict target at t + horizon - 1 (default: next day)
        max_start = len(features) - lookback - horizon + 1
        if max_start <= 0:
            raise ValueError("Series too short for given lookback/horizon")
        self.x = []
        self.y = []
        for i in range(max_start):
            self.x.append(features[i : i + lookback])
            self.y.append(target[i + lookback + horizon - 1])
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])


def temporal_split(
    bundle: SeriesBundle,
    lookback: int,
    horizon: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Chronological train / val / test split on the windowed series index."""
    full = SequenceDataset(bundle.features, bundle.target, lookback, horizon)
    n = len(full)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Split too small: train={n_train}, val={n_val}, test={n_test}")

    def slice_ds(start: int, end: int) -> SequenceDataset:
        ds = SequenceDataset.__new__(SequenceDataset)
        ds.lookback = lookback
        ds.horizon = horizon
        ds.x = full.x[start:end]
        ds.y = full.y[start:end]
        return ds

    return (
        slice_ds(0, n_train),
        slice_ds(n_train, n_train + n_val),
        slice_ds(n_train + n_val, n),
    )


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
