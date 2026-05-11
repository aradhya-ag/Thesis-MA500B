from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import setup
from src.data.loaders import iter_performance_files
from src.data.macro import load_macro_panel
from src.data.roll_rate import RollRateConfig, compute_roll_rate_series
from src.utils.io_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the roll-rate series.")
    args, cfg = setup(parser)

    perf_dir = cfg.resolve_path("paths.performance_dir")
    cache_dir = ensure_dir(cfg.resolve_path("paths.cache_dir"))

    files = list(iter_performance_files(perf_dir))
    if not files:
        raise FileNotFoundError(f"No performance .parquet files under {perf_dir}")

    rr_cfg = RollRateConfig(
        state_map=cfg["roll_rate"]["state_map"],
        drop_status_values=cfg["roll_rate"]["drop_status_values"],
        target_from_state=int(cfg["roll_rate"]["target_from_state"]),
        target_to_state=int(cfg["roll_rate"]["target_to_state"]),
        min_source_count=int(cfg["roll_rate"]["min_source_count"]),
        series_start=cfg["roll_rate"]["series_start"],
        series_end=cfg["roll_rate"]["series_end"],
    )

    rr_df = compute_roll_rate_series(
        performance_dir=perf_dir,
        column_rename=cfg["roll_rate"]["performance_columns"],
        cfg=rr_cfg,
        show_progress=True,
    )

    out_path = cache_dir / "roll_rate.parquet"
    rr_df.to_parquet(out_path)

    # ----------------------------------------------------------------- macros
    macro_panel = load_macro_panel(
        macro_files=cfg["paths"]["macro_files"],
        enabled=cfg["macros"]["enabled"],
        project_root=Path(args.config).parent.parent,
    )
    macro_out = cache_dir / "macro_panel_raw.parquet"
    macro_panel.to_parquet(macro_out)


if __name__ == "__main__":
    main()
