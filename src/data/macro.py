from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd



def _read_single_macro(path: Path, variable_name: str) -> pd.Series:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}, got {df.shape[1]}")

    date_col = next((c for c in df.columns if c.lower() in {"date", "observation_date"}), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[1])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = df[value_col].rename(variable_name)

    s = s.resample("ME").last()
    return s


def load_macro_panel(
    macro_files: Mapping[str, str | Path],
    enabled: Sequence[str],
    *,
    project_root: Path | None = None,
) -> pd.DataFrame:
    series: List[pd.Series] = []
    for name in enabled:
        if name not in macro_files:
            raise KeyError(f"Macro file path missing for {name}")
        raw = Path(macro_files[name])
        if project_root is not None and not raw.is_absolute():
            raw = (project_root / raw).resolve()
        if not raw.exists():
            raise FileNotFoundError(f"Macro file not found: {raw}")
        s = _read_single_macro(raw, name)
        series.append(s)
    df = pd.concat(series, axis=1).sort_index()
    return df


def to_yoy_pct(series: pd.Series) -> pd.Series:
    return 100.0 * (series / series.shift(12) - 1.0)


@dataclass
class MacroScaler:
    means: Dict[str, float]
    stds: Dict[str, float]
    differenced: List[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if col in self.differenced:
                out[col] = out[col].diff()
            mu = self.means.get(col, 0.0)
            sigma = self.stds.get(col, 1.0)
            if sigma == 0:
                sigma = 1.0
            out[col] = (out[col] - mu) / sigma
        return out


def _adf_pvalue(s: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller

    clean = s.dropna()
    if len(clean) < 20:
        return 1.0  
    try:
        _, pvalue, *_ = adfuller(clean, autolag="AIC")
        return float(pvalue)
    except Exception: 
        return 1.0


def build_macro_features(
    raw_panel: pd.DataFrame,
    *,
    yoy_transform: Sequence[str],
    publication_lag_months: int,
    train_end: pd.Timestamp,
    adf_pvalue_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, MacroScaler]:
    panel = raw_panel.copy()
    for col in yoy_transform:
        if col in panel.columns:
            panel[col] = to_yoy_pct(panel[col])

    if publication_lag_months > 0:
        panel = panel.shift(publication_lag_months)

    differenced: List[str] = []
    train_window = panel.loc[:train_end]
    for col in panel.columns:
        p = _adf_pvalue(train_window[col])
        if p > adf_pvalue_threshold:
            differenced.append(col)

    for col in differenced:
        panel[col] = panel[col].diff()
    train_window = panel.loc[:train_end]

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for col in panel.columns:
        mu = float(train_window[col].mean())
        sigma = float(train_window[col].std(ddof=0))
        means[col] = mu
        stds[col] = sigma if sigma > 0 else 1.0

    standardised = panel.copy()
    for col in panel.columns:
        standardised[col] = (panel[col] - means[col]) / stds[col]

    scaler = MacroScaler(means=means, stds=stds, differenced=differenced)
    return standardised, scaler
