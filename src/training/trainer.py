from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..data.sequence_builder import SequenceDataset
from ..models.gru_model import build_recurrent_model
from ..utils.seed import set_global_seed

@dataclass
class TrainerConfig:
    cell: str
    n_features: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 300
    grad_clip_norm: Optional[float] = 1.0
    patience: int = 30
    lr_scheduler: str = "plateau"
    lr_factor: float = 0.5
    lr_patience: int = 10
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True
    tensorboard_dir: Optional[str] = None


class RecurrentTrainer:

    def __init__(self, cfg: TrainerConfig):
        set_global_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = build_recurrent_model(
            cfg.cell,
            n_features=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self._build_scheduler()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_loss: float = math.inf
        self.best_epoch: int = -1
        self._writer = None

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor=cfg.lr_factor, patience=cfg.lr_patience,
            )
            self._scheduler_step_uses_metric = True
        elif cfg.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.max_epochs)
            self._scheduler_step_uses_metric = False
        elif cfg.lr_scheduler == "none":
            self.scheduler = None
            self._scheduler_step_uses_metric = False
        else:
            raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")

    def _loader(self, ds: SequenceDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def fit(
        self,
        train_ds: SequenceDataset,
        val_ds: SequenceDataset,
        *,
        trial=None,
    ) -> Dict[str, List[float]]:
        if len(train_ds) == 0:
            raise ValueError("Empty training dataset.")
        train_loader = self._loader(train_ds, shuffle=True)
        val_loader = self._loader(val_ds, shuffle=False)

        epochs_since_improve = 0
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._eval_loss(val_loader) if len(val_ds) > 0 else math.nan
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            if self._writer is not None:
                self._writer.add_scalar("loss/train", train_loss, epoch)
                self._writer.add_scalar("loss/val", val_loss, epoch)
                self._writer.add_scalar("lr", current_lr, epoch)
            if self.scheduler is not None:
                if self._scheduler_step_uses_metric:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

            improved = val_loss < self.best_val_loss - 1e-8
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            yhat = self.model(X)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            if self.cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
            bsz = X.size(0)
            total += float(loss.item()) * bsz
            n += bsz
        return total / max(n, 1)

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            yhat = self.model(X)
            loss = self.loss_fn(yhat, y)
            bsz = X.size(0)
            total += float(loss.item()) * bsz
            n += bsz
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, ds: SequenceDataset) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        if len(ds) == 0:
            return np.empty(0), np.empty(0), pd.DatetimeIndex([])
        self.model.eval()
        loader = self._loader(ds, shuffle=False)
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            yhat = self.model(X).detach().cpu().numpy()
            preds.append(yhat)
            trues.append(y.numpy())
        return np.concatenate(preds), np.concatenate(trues), ds.dates

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: Path, *, device: Optional[str] = None) -> "RecurrentTrainer":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg_dict = dict(ckpt["config"])
        if device is not None:
            cfg_dict["device"] = device
        cfg = TrainerConfig(**cfg_dict)
        trainer = cls(cfg)
        trainer.model.load_state_dict(ckpt["state_dict"])
        return trainer
