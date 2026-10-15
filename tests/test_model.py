"""Model shape and forward-pass invariants (CPU-only)."""

from __future__ import annotations

import torch

from fts_forecaster.model import LSTMCell, VolatilityLSTM


def test_lstm_cell_shapes():
    cell = LSTMCell(input_size=5, hidden_size=16)
    x = torch.randn(8, 5)
    h = torch.zeros(8, 16)
    c = torch.zeros(8, 16)
    h2, c2 = cell(x, (h, c))
    assert h2.shape == (8, 16)
    assert c2.shape == (8, 16)
    assert torch.isfinite(h2).all()
    assert torch.isfinite(c2).all()


def test_volatility_lstm_output_shape():
    model = VolatilityLSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.0)
    x = torch.randn(4, 30, 5)
    y = model(x)
    assert y.shape == (4,)
    assert model.count_parameters() > 0


def test_gradients_flow():
    model = VolatilityLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0)
    x = torch.randn(2, 10, 5, requires_grad=False)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
