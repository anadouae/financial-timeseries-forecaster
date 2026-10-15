"""FastAPI inference service for volatility forecasts."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from fts_forecaster.config import settings
from fts_forecaster.model import VolatilityLSTM

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    x_api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> None:
    """Enforce API key when FTS_API_KEY is configured; no-op otherwise."""
    expected = (settings.api_key or "").strip()
    if not expected:
        return

    provided = (x_api_key or "").strip()
    if not provided and bearer is not None and bearer.credentials:
        provided = bearer.credentials.strip()

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


class PredictRequest(BaseModel):
    """Window of OHLCV-derived features (already in model feature space or raw)."""

    sequence: List[List[float]] = Field(
        ...,
        description="Shape (lookback, n_features). Default features: log_ret, hl_range, oc_gap, log_vol, parkinson.",
    )
    already_normalized: bool = Field(
        False,
        description="If false, apply checkpoint feature mean/std normalization.",
    )


class PredictResponse(BaseModel):
    predicted_volatility: float
    lookback: int
    n_features: int
    model_val_rmse: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    lookback: Optional[int] = None


class ModelBundle:
    def __init__(self) -> None:
        self.model: Optional[VolatilityLSTM] = None
        self.lookback: int = settings.lookback
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        self.val_rmse: Optional[float] = None
        self.loaded: bool = False

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = VolatilityLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.model = model
        self.lookback = int(cfg["lookback"])
        norm = ckpt["norm"]
        self.feature_mean = np.asarray(norm["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(norm["feature_std"], dtype=np.float64)
        self.target_mean = float(norm["target_mean"])
        self.target_std = float(norm["target_std"])
        self.val_rmse = float(ckpt.get("val_rmse", float("nan")))
        self.loaded = True


bundle = ModelBundle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(settings.model_path)
    if path.exists():
        bundle.load(path)
    yield


app = FastAPI(
    title="Financial Time Series Forecaster",
    description="Custom LSTM volatility prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=bundle.loaded,
        lookback=bundle.lookback if bundle.loaded else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    _: None = Depends(require_api_key),
) -> PredictResponse:
    if not bundle.loaded or bundle.model is None:
        # Lazy-load for tests / late artifact placement
        path = Path(settings.model_path)
        if path.exists():
            bundle.load(path)
        else:
            raise HTTPException(status_code=503, detail="Model not loaded; train first.")

    arr = np.asarray(req.sequence, dtype=np.float64)
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="sequence must be 2D (lookback, features)")
    if arr.shape[0] != bundle.lookback:
        raise HTTPException(
            status_code=422,
            detail=f"Expected lookback={bundle.lookback}, got {arr.shape[0]}",
        )
    expected_f = bundle.model.input_size
    if arr.shape[1] != expected_f:
        raise HTTPException(
            status_code=422,
            detail=f"Expected n_features={expected_f}, got {arr.shape[1]}",
        )

    if not req.already_normalized:
        assert bundle.feature_mean is not None and bundle.feature_std is not None
        arr = (arr - bundle.feature_mean) / bundle.feature_std

    x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, T, F)
    with torch.no_grad():
        pred_norm = float(bundle.model(x).item())
    pred = pred_norm * bundle.target_std + bundle.target_mean
    return PredictResponse(
        predicted_volatility=pred,
        lookback=bundle.lookback,
        n_features=expected_f,
        model_val_rmse=bundle.val_rmse,
    )


def main() -> None:
    uvicorn.run(
        "fts_forecaster.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
