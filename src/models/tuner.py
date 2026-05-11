from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler

from ..data.preprocess import MinMaxScaler1D
from ..data.sequence_builder import SplitDatasets, build_split_datasets
from ..data.splits import SplitSpec
from ..training.trainer import RecurrentTrainer, TrainerConfig
from ..utils.metrics import mape


def _sample(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    if kind == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search-space type for {name!r}: {kind!r}")


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_tuning(
    *,
    cell: str,
    feature_frame: pd.DataFrame,
    splits: SplitSpec,
    target_scaler: MinMaxScaler1D,
    base_hparams: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
    n_trials: int,
    timeout_seconds: Optional[float],
    pruner: str = "median",
    seed: int = 42,
    device: str = "cpu",
) -> TuningResult:
    sampler = TPESampler(seed=seed)
    pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=5) if pruner == "median" else NopPruner()

    def objective(trial: optuna.Trial) -> float:
        hp = copy.deepcopy(base_hparams)
        for name, spec in search_space.items():
            hp[name] = _sample(trial, name, spec)

        datasets = build_split_datasets(feature_frame, splits, lookback=hp["lookback"])
        if len(datasets.val) == 0:
            raise optuna.TrialPruned("Validation set is empty for this lookback.")

        tcfg = TrainerConfig(
            cell=cell,
            n_features=datasets.n_features,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp.get("weight_decay", 0.0),
            batch_size=hp["batch_size"],
            max_epochs=hp.get("max_epochs", 200),
            grad_clip_norm=hp.get("grad_clip_norm", 1.0),
            patience=hp.get("patience", 20),
            lr_scheduler=hp.get("lr_scheduler", "plateau"),
            lr_factor=hp.get("lr_factor", 0.5),
            lr_patience=hp.get("lr_patience", 5),
            device=device,
            seed=seed,
            verbose=False,
        )

        trainer = RecurrentTrainer(tcfg)
        history = trainer.fit(datasets.train, datasets.val, trial=trial)

        y_pred_scaled, y_true_scaled, _ = trainer.predict(datasets.val)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_true_scaled)
        val_mape = mape(y_true, y_pred)
        return val_mape

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner_obj)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)

    summary = study.trials_dataframe()
    return TuningResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        study_summary=summary,
    )
