from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  
    Dataset = object  
    DataLoader = None  
    _TORCH_AVAILABLE = False


from .splits import SplitSpec

def _windows_from_frame(
    frame: pd.DataFrame,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    vals = frame.values.astype(np.float32)
    n_rows, n_feats = vals.shape

    n_windows = n_rows - lookback 
    if n_windows <= 0:
        return (
            np.empty((0, lookback, n_feats), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    X = np.empty((n_windows, lookback, n_feats), dtype=np.float32)
    y = np.empty((n_windows,), dtype=np.float32)
    for i in range(n_windows):
        X[i] = vals[i : i + lookback]
        y[i] = vals[i + lookback, 0]

    target_dates = frame.index[lookback : lookback + n_windows]
    return X, y, target_dates


class SequenceDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use SequenceDataset.")
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.dates = pd.DatetimeIndex(dates)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])


@dataclass
class SplitDatasets:

    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    lookback: int
    n_features: int


def build_split_datasets(
    feature_frame: pd.DataFrame,
    splits: SplitSpec,
    *,
    lookback: int,
) -> SplitDatasets:
    if "y" not in feature_frame.columns:
        raise ValueError("feature_frame must have a 'y' column as its scaled target.")

    full = feature_frame.dropna(how="any").copy()
    idx = full.index

    train_mask = idx <= splits.train_end
    val_mask = (idx > splits.train_end) & (idx <= splits.val_end)
    test_mask = (idx > splits.val_end) & (idx <= splits.test_end)

    def carve(start_mask: np.ndarray) -> SequenceDataset:
        target_dates = full.index[start_mask]
        if len(target_dates) == 0:
            return SequenceDataset(
                np.empty((0, lookback, full.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                pd.DatetimeIndex([]),
            )

        positions = np.where(np.isin(full.index, target_dates))[0]
        valid = positions[positions >= lookback]
        if len(valid) == 0:
            return SequenceDataset(
                np.empty((0, lookback, full.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                pd.DatetimeIndex([]),
            )

        vals = full.values.astype(np.float32)
        X = np.stack([vals[i - lookback : i] for i in valid], axis=0)
        y = vals[valid, 0]
        dates = full.index[valid]
        return SequenceDataset(X, y, dates)

    train_ds = carve(train_mask.values if hasattr(train_mask, "values") else train_mask)
    val_ds = carve(val_mask.values if hasattr(val_mask, "values") else val_mask)
    test_ds = carve(test_mask.values if hasattr(test_mask, "values") else test_mask)

    return SplitDatasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        lookback=lookback,
        n_features=full.shape[1],
    )


def build_loaders(
    datasets: SplitDatasets,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, "DataLoader"]:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build data loaders.")
    return {
        "train": DataLoader(
            datasets.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
    }
