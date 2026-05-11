from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20


class LSTMRegressor(nn.Module):

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        inter_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=inter_dropout,
        )
        self.final_dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _c_n) = self.lstm(x)
        last_hidden = h_n[-1]               
        last_hidden = self.final_dropout(last_hidden)
        y = self.head(last_hidden).squeeze(-1) 
        return y

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
