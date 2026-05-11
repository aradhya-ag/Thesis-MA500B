from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq




_QUARTER_RE = re.compile(r"(\d{4})Q([1-4])")


def iter_performance_files(performance_dir: Path) -> Iterator[tuple[Path, str]]:
    files: list[tuple[str, Path]] = []
    for f in performance_dir.glob("*.parquet"):
        m = _QUARTER_RE.search(f.stem)
        if m is None:
            continue
        files.append((f"{m.group(1)}Q{m.group(2)}", f))
    files.sort(key=lambda x: x[0])
    for q, p in files:
        yield p, q


def load_performance_quarter(
    path: Path,
    column_rename: Mapping[str, str],
    *,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = [c for c in pf.schema_arrow.names]

    name_map = {c: column_rename.get(str(c)) for c in available if str(c) in column_rename}
    if not name_map:
        raise RuntimeError(
            f"None of the columns in {path.name} match the configured rename map. "
            f"Found columns: {available}"
        )
    df = pf.read(columns=list(name_map.keys())).to_pandas()
    df = df.rename(columns={c: name_map[c] for c in df.columns})

    if "delq_status" in df.columns:
        # Some files store as int, some as string.  Standardise to str.
        df["delq_status"] = df["delq_status"].astype("string")

    if "period" in df.columns:
        # Period is YYYYMM int; convert to pandas Period["M"] for safe arithmetic.
        df["period"] = pd.to_datetime(df["period"].astype("int64").astype(str), format="%Y%m").dt.to_period("M")

    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df


def iter_performance_records(
    performance_dir: Path,
    column_rename: Mapping[str, str],
    columns: Optional[Sequence[str]] = None,
    *,
    quarters: Optional[Iterable[str]] = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    selected = set(quarters) if quarters is not None else None
    for path, q in iter_performance_files(performance_dir):
        if selected is not None and q not in selected:
            continue
        df = load_performance_quarter(path, column_rename, columns=columns)
        yield q, df


ORIGINATION_FIELDS: list[str] = [
    "credit_score",                 # 1
    "first_payment_date",           # 2
    "first_time_homebuyer_flag",    # 3
    "maturity_date",                # 4
    "msa",                          # 5
    "mi_pct",                       # 6
    "num_units",                    # 7
    "occupancy_status",             # 8
    "ocltv",                        # 9
    "dti",                          # 10
    "orig_upb",                     # 11
    "oltv",                         # 12
    "orig_interest_rate",           # 13
    "channel",                      # 14
    "ppmt_penalty_flag",            # 15
    "amortization_type",            # 16
    "property_state",               # 17
    "property_type",                # 18
    "postal_code",                  # 19
    "loan_id",                      # 20
    "loan_purpose",                 # 21
    "orig_loan_term",               # 22
    "num_borrowers",                # 23
    "seller_name",                  # 24
    "servicer_name",                # 25
    "super_conforming_flag",        # 26
]


def load_origination_quarter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=ORIGINATION_FIELDS,
        dtype={"loan_id": "string", "property_state": "string", "msa": "string"},
        low_memory=False,
    )
    return df
