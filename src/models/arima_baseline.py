from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series) -> Tuple[float, float]:
    clean = series.dropna().values
    stat, pvalue, *_ = adfuller(clean, autolag="AIC")
    return float(stat), float(pvalue)


def choose_d(series: pd.Series, *, d_max: int = 2, alpha: float = 0.05) -> int:
    work = series.copy()
    for d in range(d_max + 1):
        _, p = adf_test(work)
        if p < alpha:
            return d
        work = work.diff().dropna()
    return d_max

def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int],
    *,
    trend: str = "c",
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
):
    model = ARIMA(
        series.astype(float),
        order=order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    return model.fit()


@dataclass
class ArimaForecaster:

    p_range: Sequence[int] = (0, 1, 2, 3, 4, 5)
    q_range: Sequence[int] = (0, 1, 2, 3, 4, 5)
    d_max: int = 2
    order: Optional[Tuple[int, int, int]] = None
    trend: str = "c"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False

    fitted_order: Optional[Tuple[int, int, int]] = None
    aic: Optional[float] = None
    train_series: Optional[pd.Series] = field(default=None, repr=False)

    def fit(self, train_series: pd.Series) -> "ArimaForecaster":
        train_series = train_series.dropna()
        if self.order is not None:
            d = self.order[1]
        else:
            d = choose_d(train_series, d_max=self.d_max)

        if self.order is not None:
            best_order = tuple(self.order)
            best_aic = float("inf")
        else:
            best_aic = float("inf")
            best_order: Optional[Tuple[int, int, int]] = None
            for p, q in itertools.product(self.p_range, self.q_range):
                cand = (int(p), int(d), int(q))
                try:
                    fit = fit_arima(
                        train_series, cand,
                        trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                    )
                    aic = float(fit.aic)
                except Exception as e:
                    continue
                if aic < best_aic:
                    best_aic = aic
                    best_order = cand
            if best_order is None:
                raise RuntimeError("No ARIMA order from the search grid converged.")

        self.fitted_order = best_order
        self.aic = best_aic
        self.train_series = train_series
        return self

    def one_step_forecast(self, future_series: pd.Series) -> pd.Series:
        if self.fitted_order is None or self.train_series is None:
            raise RuntimeError("Forecaster has not been fit() yet.")

        history = list(self.train_series.values.astype(float))
        predictions: List[float] = []

        for ts, _y in future_series.items():
            tmp = pd.Series(history)
            fit = fit_arima(
                tmp, self.fitted_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            yhat = float(fit.forecast(steps=1).iloc[0])
            predictions.append(yhat)
            history.append(float(future_series.loc[ts]))

        return pd.Series(predictions, index=future_series.index, name="prediction")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "p_range": list(self.p_range),
                    "q_range": list(self.q_range),
                    "d_max": self.d_max,
                    "trend": self.trend,
                    "enforce_stationarity": self.enforce_stationarity,
                    "enforce_invertibility": self.enforce_invertibility,
                    "fitted_order": self.fitted_order,
                    "aic": self.aic,
                    "train_series": self.train_series,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "ArimaForecaster":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            p_range=state["p_range"],
            q_range=state["q_range"],
            d_max=state["d_max"],
            trend=state["trend"],
            enforce_stationarity=state["enforce_stationarity"],
            enforce_invertibility=state["enforce_invertibility"],
        )
        obj.fitted_order = state["fitted_order"]
        obj.aic = state["aic"]
        obj.train_series = state["train_series"]
        return obj
