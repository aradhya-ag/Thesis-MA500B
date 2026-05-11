from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from _common import setup
from src.data.macro import build_macro_features
from src.data.preprocess import (
    MinMaxScaler1D,
    assemble_feature_frame,
    fit_target_scaler,
)
from src.data.sequence_builder import build_split_datasets
from src.data.splits import SplitSpec
from src.models.tuner import run_tuning
from src.training.trainer import RecurrentTrainer, TrainerConfig
from src.utils.io_utils import ensure_dir, save_json
from src.utils.metrics import all_metrics
from src.utils.plotting import plot_actual_vs_predicted, plot_residuals, plot_train_curves
from src.utils.seed import derive_seed



# ---------------------------------------------------------------------------
def _load_inputs(cfg):
    cache_dir = cfg.resolve_path("paths.cache_dir")
    rr = pd.read_parquet(cache_dir / "roll_rate.parquet")
    rr.index = pd.DatetimeIndex(rr.index)
    target = rr["roll_rate_filtered"].astype(float)

    macro_raw = pd.read_parquet(cache_dir / "macro_panel_raw.parquet")
    macro_raw.index = pd.DatetimeIndex(macro_raw.index)

    splits = SplitSpec.from_strings(
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        test_end=cfg["splits"]["test_end"],
    )

    target_scaler = fit_target_scaler(target, splits.train_end)

    macro_std, _scaler = build_macro_features(
        macro_raw,
        yoy_transform=cfg["macros"]["yoy_transform"],
        publication_lag_months=int(cfg["macros"]["publication_lag_months"]),
        train_end=splits.train_end,
        adf_pvalue_threshold=float(cfg["macros"]["adf_pvalue_threshold"]),
    )
    macro_std = macro_std.reindex(target.index)
    return target, macro_std, target_scaler, splits


# ---------------------------------------------------------------------------
def _train_variant(
    *, name: str, cell: str, use_macro: bool, cfg,
    target: pd.Series, macro_std: pd.DataFrame, target_scaler: MinMaxScaler1D,
    splits: SplitSpec,
) -> Dict[str, Any]:
    macros = macro_std if use_macro else None
    feature_frame = assemble_feature_frame(target, macros, target_scaler=target_scaler)
    feature_frame = feature_frame.dropna(how="any")

    base_hp: Dict[str, Any] = dict(cfg["neural"]["default_hparams"])

    #Hyperparameter tuning 
    tuning_cfg = cfg["tuning"]
    if tuning_cfg.get("enabled", False):
        result = run_tuning(
            cell=cell,
            feature_frame=feature_frame,
            splits=splits,
            target_scaler=target_scaler,
            base_hparams=base_hp,
            search_space=tuning_cfg["search_space"],
            n_trials=int(tuning_cfg["n_trials"]),
            timeout_seconds=tuning_cfg.get("timeout_seconds"),
            pruner=tuning_cfg.get("pruner", "median"),
            seed=int(cfg.get("seed", 42)),
            device=cfg.get_dotted("device", "cpu"),
        )
        chosen_hp = {**base_hp, **result.best_params}
        tuning_summary = {
            "best_params": result.best_params,
            "best_value": result.best_value,
            "n_trials": result.n_trials,
        }
    else:
        chosen_hp = base_hp
        tuning_summary = {"enabled": False}


    seeds: List[int] = list(cfg["neural"]["seeds"])
    metric_records: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    best_run = None

    for run_idx, seed in enumerate(seeds):
        actual_seed = derive_seed(int(cfg.get("seed", 42)), run_idx) if seed is None else int(seed)
        datasets = build_split_datasets(feature_frame, splits, lookback=int(chosen_hp["lookback"]))
        tcfg = TrainerConfig(
            cell=cell,
            n_features=datasets.n_features,
            hidden_size=int(chosen_hp["hidden_size"]),
            num_layers=int(chosen_hp["num_layers"]),
            dropout=float(chosen_hp["dropout"]),
            learning_rate=float(chosen_hp["learning_rate"]),
            weight_decay=float(chosen_hp.get("weight_decay", 0.0)),
            batch_size=int(chosen_hp["batch_size"]),
            max_epochs=int(chosen_hp.get("max_epochs", 300)),
            grad_clip_norm=float(chosen_hp.get("grad_clip_norm", 1.0)),
            patience=int(chosen_hp.get("patience", 30)),
            lr_scheduler=str(chosen_hp.get("lr_scheduler", "plateau")),
            lr_factor=float(chosen_hp.get("lr_factor", 0.5)),
            lr_patience=int(chosen_hp.get("lr_patience", 10)),
            device=cfg.get_dotted("device", "cpu"),
            seed=actual_seed,
            verbose=True,
        )
        trainer = RecurrentTrainer(tcfg)
        history = trainer.fit(datasets.train, datasets.val)

        y_pred_scaled, y_true_scaled, dates = trainer.predict(datasets.test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_true_scaled)
        m = all_metrics(y_true, y_pred)
        m.update({"seed": actual_seed, "best_epoch": trainer.best_epoch, "n_parameters": trainer.model.n_parameters})
        metric_records.append(m)

        frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "seed": actual_seed}, index=dates)
        prediction_frames.append(frame)

        if best_run is None or m["mape"] < best_run["mape"]:
            best_run = {"seed": actual_seed, "trainer": trainer, "history": history,
                        "predictions": frame.copy(), "metrics": m}

    pred_dir = ensure_dir(cfg.resolve_path("paths.prediction_dir"))
    model_dir = ensure_dir(cfg.resolve_path("paths.model_dir"))
    fig_dir = ensure_dir(cfg.resolve_path("paths.figure_dir"))

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(pred_dir / f"{name}_per_seed_metrics.csv", index=False)
    agg = {
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
        "mae_mean": float(metrics_df["mae"].mean()),
        "mae_std": float(metrics_df["mae"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
        "mape_mean": float(metrics_df["mape"].mean()),
        "mape_std": float(metrics_df["mape"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
    }
    save_json(
        {
            "variant": name, "cell": cell, "use_macro": use_macro,
            "hyperparameters": chosen_hp,
            "tuning": tuning_summary,
            "aggregate_metrics": agg,
            "best_seed_metrics": best_run["metrics"],
        },
        pred_dir / f"{name}_summary.json",
    )

    best_run["predictions"].to_csv(pred_dir / f"{name}_test_predictions.csv")
    best_run["trainer"].save(model_dir / f"{name}.pt")

    pd.concat(prediction_frames).to_csv(pred_dir / f"{name}_all_seed_predictions.csv")

    plot_train_curves(
        {"train": best_run["history"]["train_loss"], "val": best_run["history"]["val_loss"]},
        fig_dir / f"{name}_train_curves.png",
    )
    plot_actual_vs_predicted(
        best_run["predictions"]["y_true"],
        best_run["predictions"]["y_pred"],
        fig_dir / f"{name}_actual_vs_predicted.png",
        title=f"{name} — test set",
    )
    plot_residuals(
        best_run["predictions"]["y_true"],
        best_run["predictions"]["y_pred"],
        fig_dir / f"{name}_residuals.png",
        title=f"{name} residuals",
    )

    return {"variant": name, "aggregate": agg, "hyperparameters": chosen_hp}

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM / GRU variants.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated variant names to run (default: all configured).")
    args, cfg = setup(parser)

    target, macro_std, target_scaler, splits = _load_inputs(cfg)

    variants = cfg["neural"]["variants"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        variants = [v for v in variants if v["name"] in wanted]
        if not variants:
            raise ValueError(f"--only filter matched no variants: {args.only!r}")

    summaries = []
    for v in variants:
        s = _train_variant(
            name=v["name"], cell=v["cell"], use_macro=bool(v["use_macro"]),
            cfg=cfg,
            target=target, macro_std=macro_std, target_scaler=target_scaler, splits=splits,
        )
        summaries.append(s)

    pred_dir = ensure_dir(cfg.resolve_path("paths.prediction_dir"))
    save_json({"variants": summaries}, pred_dir / "neural_variant_summaries.json")


if __name__ == "__main__":
    main()
