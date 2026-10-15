"""
Custom LSTM built from an explicit LSTMCell (gates written out).

Complexity notes (also in README Mermaid):
  Per timestep for one layer with input dim D, hidden H:
    - Input/hidden projections: 4 linear maps → O(4 H (D + H)) matmul work
    - Elementwise gates + cell update: O(H)
  Sequence length T, L layers, batch B:
    Forward ≈ O(B · T · L · H · (D + H))  (D→H only on layer 0; deeper layers use H)
  Backward (BPTT) is of the same asymptotic order with a constant factor ~2–3×
  from storing activations / duals through the T-step chain.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    Single-step LSTM with explicit forget / input / output / candidate gates.

    Gates (standard Hochreiter & Schmidhuber / Greff formulation):
        f_t = σ(W_f x_t + U_f h_{t-1} + b_f)
        i_t = σ(W_i x_t + U_i h_{t-1} + b_i)
        g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)
        o_t = σ(W_o x_t + U_o h_{t-1} + b_o)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # One fused linear for efficiency; we still document gates explicitly below.
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias init → 1 for better long-horizon retention
                hidden = self.hidden_size
                param.data[hidden : 2 * hidden].fill_(1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.x2h(x_t) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LSTMLayer(nn.Module):
    """Unroll an LSTMCell over time; optional residual dropout on outputs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, D)
        batch, seq_len, _ = x.shape
        if state is None:
            h = x.new_zeros(batch, self.cell.hidden_size)
            c = x.new_zeros(batch, self.cell.hidden_size)
        else:
            h, c = state

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        out = torch.stack(outputs, dim=1)  # (B, T, H)
        out = self.dropout(out)
        return out, (h, c)


class VolatilityLSTM(nn.Module):
    """Stacked custom LSTM → linear head for scalar volatility forecast."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_size if layer_idx == 0 else hidden_size
            # Dropout between layers only (not after last), matching nn.LSTM convention
            drop = dropout if layer_idx < num_layers - 1 else 0.0
            layers.append(LSTMLayer(in_dim, hidden_size, dropout=drop))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) feature windows
        Returns:
            (B,) predicted normalized volatility
        """
        h = x
        for layer in self.layers:
            h, _ = layer(h)
        last = h[:, -1, :]  # last timestep hidden state
        return self.head(last).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
