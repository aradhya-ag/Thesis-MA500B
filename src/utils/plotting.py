from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Periods shaded on every time-series plot for context.
DEFAULT_REGIMES: Dict[str, tuple[str, str, str]] = {
    "2008 Financial Crisis": ("2007-12", "2009-06", "lightgrey"),
    "2017 US Hurricanes":    ("2017-08", "2018-02", "moccasin"),
    "COVID-19 Pandemic":     ("2020-03", "2021-12", "plum"),
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_regimes(ax: plt.Axes, regimes: Mapping[str, tuple[str, str, str]]) -> None:
    for label, (start, end, color) in regimes.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.4, label=label)


def plot_roll_rate_series(series: pd.Series, out_path: Path, *, title: str = "Monthly roll rate $r_{12}(t)$ (Bucket 1 → Bucket 2)") -> None:
    """Recreate Figure 3.1 from the report."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_regimes(ax, DEFAULT_REGIMES)
    ax.plot(series.index, series.values, color="red", linewidth=1.2)
    ax.set_xlabel("Reporting Period")
    ax.set_ylabel("Monthly Roll Rates (Bucket 1 -> Bucket 2)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_acf_pacf(series: pd.Series, out_dir: Path, *, lags: int = 16) -> None:
    """Recreate Figures 8.1 and 8.2 (ACF / PACF stem plots)."""
    from statsmodels.tsa.stattools import acf, pacf

    acf_vals = acf(series.dropna().values, nlags=lags, fft=True)
    pacf_vals = pacf(series.dropna().values, nlags=lags, method="ywm")
    n = len(series.dropna())
    band = 1.96 / np.sqrt(n)

    for kind, vals, label in [("acf", acf_vals, "ACF (Determines q)"), ("pacf", pacf_vals, "PACF (Determines p)")]:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.stem(range(len(vals)), vals, basefmt=" ")
        ax.axhspan(-band, band, color="C0", alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Lags (Months)")
        ax.set_ylabel("Correlation")
        ax.set_title(label)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.3)
        _save(fig, out_dir / f"{kind}.png")


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    out_path: Path,
    *,
    title: str = "Actual vs Predicted",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true.index, y_true.values, label="Actual", color="black", linewidth=1.3)
    ax.plot(y_pred.index, y_pred.values, label="Predicted", color="red", linewidth=1.3, alpha=0.8)
    ax.set_xlabel("Reporting Period")
    ax.set_ylabel("Roll rate $r_{12}(t)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, out_path: Path, *, title: str = "Residuals") -> None:
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(resid.index, resid.values, color="steelblue", linewidth=1.0)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_title(f"{title} — time series")
    axes[0].grid(alpha=0.3)
    axes[1].hist(resid.values, bins=30, color="steelblue", edgecolor="black")
    axes[1].set_title(f"{title} — distribution")
    axes[1].grid(alpha=0.3)
    _save(fig, out_path)


def plot_rolling_mape(
    series_by_model: Mapping[str, pd.Series],
    out_path: Path,
    *,
    title: str = "Rolling MAPE (3-year window)",
) -> None:
    """Generalisation of Figure 8.3 to multiple models."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_regimes(ax, DEFAULT_REGIMES)
    for name, s in series_by_model.items():
        ax.plot(s.index, s.values, label=name, linewidth=1.2)
    ax.set_xlabel("Reporting Period")
    ax.set_ylabel("Forecasting Error (MAPE %)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: Path, *, metric: str = "mape") -> None:
    """Bar chart comparing one metric across models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df = metrics_df.sort_values(metric)
    ax.barh(metrics_df.index, metrics_df[metric], color="steelblue", edgecolor="black")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model comparison ({metric.upper()})")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_train_curves(history: Mapping[str, Sequence[float]], out_path: Path) -> None:
    """Training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)
