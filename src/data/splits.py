from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def from_strings(cls, train_end: str, val_end: str, test_end: str) -> "SplitSpec":
        return cls(
            train_end=pd.Timestamp(train_end) + pd.offsets.MonthEnd(0),
            val_end=pd.Timestamp(val_end) + pd.offsets.MonthEnd(0),
            test_end=pd.Timestamp(test_end) + pd.offsets.MonthEnd(0),
        )

    def mask(self, index: pd.DatetimeIndex, which: str) -> pd.Series:
        idx = pd.DatetimeIndex(index)
        if which == "train":
            return pd.Series(idx <= self.train_end, index=idx)
        if which == "val":
            return pd.Series((idx > self.train_end) & (idx <= self.val_end), index=idx)
        if which == "test":
            return pd.Series((idx > self.val_end) & (idx <= self.test_end), index=idx)
        raise ValueError(f"Unknown split: {which}")

    def slice(self, frame: pd.DataFrame | pd.Series, which: str) -> pd.DataFrame | pd.Series:
        m = self.mask(frame.index, which)
        return frame.loc[m.values]
