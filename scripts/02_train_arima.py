from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import setup
from src.data.splits import SplitSpec
from src.models.arima_baseline import ArimaForecaster, adf_test
from src.utils.io_utils import ensure_dir, save_json
from src.utils.metrics import all_metrics
from src.utils.plotting import plot_acf_pacf, plot_actual_vs_predicted, plot_residuals




def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ARIMA baseline.")
    args, cfg = setup(parser)

    cache_dir = cfg.resolve_path("paths.cache_dir")
    model_dir = ensure_dir(cfg.resolve_path("paths.model_dir"))
    pred_dir = ensure_dir(cfg.resolve_path("paths.prediction_dir"))
    fig_dir = ensure_dir(cfg.resolve_path("paths.figure_dir"))

    rr = pd.read_parquet(cache_dir / "roll_rate.parquet")
    series = rr["roll_rate_filtered"].astype(float)
    series.index = pd.DatetimeIndex(rr.index)

    splits = SplitSpec.from_strings(
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        test_end=cfg["splits"]["test_end"],
    )
    train = splits.slice(series, "train")
    val = splits.slice(series, "val")
    test = splits.slice(series, "test")

    plot_acf_pacf(train, fig_dir / "arima_diagnostics")
    stat, p = adf_test(train)

    fitting_series = pd.concat([train, val])
    arima_cfg = cfg["arima"]
    forecaster = ArimaForecaster(
        p_range=arima_cfg["p_range"],
        q_range=arima_cfg["q_range"],
        d_max=int(arima_cfg["d_max"]),
        order=tuple(arima_cfg["order"]) if arima_cfg.get("order") else None,
        trend=arima_cfg["trend"],
        enforce_stationarity=arima_cfg["enforce_stationarity"],
        enforce_invertibility=arima_cfg["enforce_invertibility"],
    )
    forecaster.fit(fitting_series)
    forecaster.save(model_dir / "arima.pkl")

    preds = forecaster.one_step_forecast(test)
    df_out = pd.DataFrame({"y_true": test.values, "y_pred": preds.values}, index=test.index)
    df_out.to_csv(pred_dir / "arima_test_predictions.csv")

    metrics = all_metrics(test.values, preds.values)
    save_json(
        {"order": list(forecaster.fitted_order), "aic": forecaster.aic, "metrics": metrics},
        pred_dir / "arima_test_metrics.json",
    )

    plot_actual_vs_predicted(
        pd.Series(test.values, index=test.index, name="y_true"),
        pd.Series(preds.values, index=test.index, name="y_pred"),
        fig_dir / "arima_actual_vs_predicted.png",
        title=f"ARIMA{forecaster.fitted_order} — test set",
    )
    plot_residuals(
        pd.Series(test.values, index=test.index),
        pd.Series(preds.values, index=test.index),
        fig_dir / "arima_residuals.png",
        title="ARIMA test residuals",
    )


if __name__ == "__main__":
    main()
