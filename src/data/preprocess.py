from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MinMaxScaler1D:

    rmin: float
    rmax: float

    def transform(self, x: np.ndarray | pd.Series) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        denom = self.rmax - self.rmin
        if denom == 0:
            return np.zeros_like(x)
        return (x - self.rmin) / denom

    def inverse_transform(self, x: np.ndarray | pd.Series) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x * (self.rmax - self.rmin) + self.rmin

    def to_dict(self) -> dict:
        return {"rmin": float(self.rmin), "rmax": float(self.rmax)}

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler1D":
        return cls(rmin=float(d["rmin"]), rmax=float(d["rmax"]))


def fit_target_scaler(series: pd.Series, train_end: pd.Timestamp) -> MinMaxScaler1D:
    train = series.loc[:train_end].dropna()
    if train.empty:
        raise ValueError("Empty training window when fitting target scaler.")
    return MinMaxScaler1D(rmin=float(train.min()), rmax=float(train.max()))


def assemble_feature_frame(
    target_series: pd.Series,
    macro_panel: Optional[pd.DataFrame],
    *,
    target_scaler: MinMaxScaler1D,
) -> pd.DataFrame:
    df = pd.DataFrame(index=target_series.index)
    df["y"] = target_scaler.transform(target_series.values)
    if macro_panel is not None and not macro_panel.empty:
        macros = macro_panel.reindex(df.index)
        for col in macros.columns:
            df[col] = macros[col].values
    return df
