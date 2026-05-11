from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def derive_seed(master_seed: int, run_index: int) -> int:
    rng = np.random.default_rng(np.uint64(master_seed) * np.uint64(0x9E3779B1) + np.uint64(run_index))
    return int(rng.integers(0, 2**31 - 1))


def select_device(preferred: str = "auto") -> str:
    if preferred not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unknown device preference: {preferred!r}")
    if preferred == "cpu":
        return "cpu"
    try:
        import torch

        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
