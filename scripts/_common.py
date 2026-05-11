from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import Config, load_config, parse_cli_overrides                     
from src.utils.seed import select_device, set_global_seed            


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Dotted-key override, e.g. --override neural.default_hparams.lookback=18",
    )


def setup(parser: argparse.ArgumentParser) -> Tuple[argparse.Namespace, Config]:
    add_common_args(parser)
    args = parser.parse_args()
    overrides = parse_cli_overrides(args.override)
    cfg = load_config(args.config, overrides=overrides)
    set_global_seed(cfg.get("seed", 42))
    cfg.set_dotted("device", select_device(cfg.get("device", "auto")))
    return args, cfg