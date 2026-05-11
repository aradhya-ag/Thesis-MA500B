from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GRUConfig:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20


class GRURegressor(nn.Module):

    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.cfg = cfg
        inter_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=inter_dropout,
        )
        self.final_dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        last_hidden = self.final_dropout(last_hidden)
        return self.head(last_hidden).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_recurrent_model(
    cell: str,
    n_features: int,
    *,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    cell = cell.lower()
    if cell == "lstm":
        from .lstm_model import LSTMConfig, LSTMRegressor
        return LSTMRegressor(LSTMConfig(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ))
    if cell == "gru":
        return GRURegressor(GRUConfig(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ))
    raise ValueError(f"Unknown cell type: {cell!r}")
