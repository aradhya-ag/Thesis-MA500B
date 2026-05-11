from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"


STAGES = [
    ("01_build_roll_rate_series.py", "Build roll-rate series"),
    ("02_train_arima.py",             "Train ARIMA baseline"),
    ("03_train_neural.py",            "Train LSTM / GRU variants"),
    ("04_evaluate_all.py",            "Build cross-model comparison table"),
    ("05_make_plots.py",              "Render figures"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll-rate forecasting pipeline.")
    parser.add_argument(
        "--from-stage", type=int, default=1,
        help="Start from this stage number (1-5).",
    )
    parser.add_argument(
        "--to-stage", type=int, default=len(STAGES),
        help="Stop after this stage number (1-5).",
    )
    parser.add_argument(
        "--config", "-c", type=str,
        default=str(HERE / "configs" / "config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Dotted-key config override forwarded to every stage.",
    )
    args = parser.parse_args()

    for idx, (script, label) in enumerate(STAGES, start=1):
        if idx < args.from_stage or idx > args.to_stage:
            continue
        cmd = [sys.executable, str(SCRIPTS / script), "--config", args.config]
        for ov in args.override:
            cmd.extend(["--override", ov])
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
