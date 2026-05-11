from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .loaders import iter_performance_records



@dataclass
class RollRateConfig:

    state_map: Mapping[str, int]
    drop_status_values: Sequence[str]
    target_from_state: int
    target_to_state: int
    min_source_count: int
    series_start: str           # "YYYY-MM"
    series_end: str             # "YYYY-MM"


def _accumulate_quarter(
    df: pd.DataFrame,
    cfg: RollRateConfig,
    num_acc: Dict[pd.Period, int],
    den_acc: Dict[pd.Period, int],
) -> None:
    df = df[~df["delq_status"].isin(list(cfg.drop_status_values))]
    if df.empty:
        return

    state = df["delq_status"].map(dict(cfg.state_map))
    if state.isna().any():
        max_state = max(cfg.state_map.values())
        state = state.fillna(max_state)
    df = df.assign(state=state.astype("int8"))

    df = df.sort_values(["loan_id", "period"], kind="mergesort")

    period_int = df["period"].dt.year.astype(np.int64) * 12 + df["period"].dt.month.astype(np.int64)
    df = df.assign(_period_int=period_int)

    grp = df.groupby("loan_id", sort=False)
    df["next_state"] = grp["state"].shift(-1)
    df["next_period_int"] = grp["_period_int"].shift(-1)

    gap_ok = (df["next_period_int"] - df["_period_int"]) == 1

    src_mask = df["state"] == cfg.target_from_state
    src_counts = df.loc[src_mask, "period"].value_counts()
    for p, n in src_counts.items():
        den_acc[p] = den_acc.get(p, 0) + int(n)

    num_mask = src_mask & gap_ok & (df["next_state"] == cfg.target_to_state)
    num_counts = df.loc[num_mask, "period"].value_counts()
    for p, n in num_counts.items():
        num_acc[p] = num_acc.get(p, 0) + int(n)


def compute_roll_rate_series(
    performance_dir: Path,
    column_rename: Mapping[str, str],
    cfg: RollRateConfig,
    *,
    quarters: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    needed_cols = ["loan_id", "period", "delq_status"]
    num_acc: Dict[pd.Period, int] = {}
    den_acc: Dict[pd.Period, int] = {}

    iterator = iter_performance_records(
        performance_dir,
        column_rename,
        columns=needed_cols,
        quarters=quarters,
    )
    if show_progress:
        iterator = tqdm(iterator, desc="Aggregating performance files", unit="quarter")

    for quarter, df in iterator:
        _accumulate_quarter(df, cfg, num_acc, den_acc)

    full_index = pd.period_range(start=cfg.series_start, end=cfg.series_end, freq="M")
    num = pd.Series(num_acc).reindex(full_index, fill_value=0).astype(np.int64)
    den = pd.Series(den_acc).reindex(full_index, fill_value=0).astype(np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(den > 0, num / den, np.nan)

    out = pd.DataFrame({"numerator": num.values, "denominator": den.values, "roll_rate": rate}, index=full_index)
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"

    low_conf = out["denominator"] < cfg.min_source_count
    out["low_confidence"] = low_conf
    filtered = out["roll_rate"].copy()
    filtered[low_conf] = np.nan
    filtered = filtered.interpolate(method="time").bfill().ffill()
    out["roll_rate_filtered"] = filtered.values

    return out
