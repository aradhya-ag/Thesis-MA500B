from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from _common import setup
from src.utils.io_utils import ensure_dir, save_json
from src.utils.metrics import all_metrics, mape


def _load_predictions(pred_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    arima_csv = pred_dir / "arima_test_predictions.csv"
    if arima_csv.exists():
        df = pd.read_csv(arima_csv, index_col=0, parse_dates=True)
        out["ARIMA"] = df

    for csv in pred_dir.glob("*_test_predictions.csv"):
        if csv.name == "arima_test_predictions.csv":
            continue
        name = csv.stem.replace("_test_predictions", "")
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        out[name] = df
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the comparison table.")
    args, cfg = setup(parser)
    pred_dir = cfg.resolve_path("paths.prediction_dir")

    preds = _load_predictions(pred_dir)
    if not preds:
        raise FileNotFoundError(f"No predictions found in {pred_dir}; run stages 2 and 3 first.")

    rows: List[Dict[str, float]] = []
    arima_mape = None
    for name, df in preds.items():
        m = all_metrics(df["y_true"].values, df["y_pred"].values)
        rows.append({"model": name, **m})
        if name == "ARIMA":
            arima_mape = m["mape"]

    table = pd.DataFrame(rows).set_index("model").sort_values("mape")
    if arima_mape is not None:
        table["improvement_vs_arima_pct"] = 100.0 * (arima_mape - table["mape"]) / arima_mape
    table.to_csv(pred_dir / "comparison_table.csv")

    regimes = cfg["evaluation"]["regimes"]
    regime_rows: List[Dict[str, float]] = []
    for name, df in preds.items():
        record = {"model": name}
        for rname, span in regimes.items():
            mask = (df.index >= pd.Timestamp(span["start"])) & (df.index <= pd.Timestamp(span["end"]))
            sub = df.loc[mask]
            if len(sub) == 0:
                record[f"{rname}_mape"] = float("nan")
            else:
                record[f"{rname}_mape"] = mape(sub["y_true"].values, sub["y_pred"].values)
        if "pre_shock_mape" in record and "shock_mape" in record and record["pre_shock_mape"] > 0:
            record["degradation_pct"] = 100.0 * (record["shock_mape"] - record["pre_shock_mape"]) / record["pre_shock_mape"]
        regime_rows.append(record)

    regime_df = pd.DataFrame(regime_rows).set_index("model")
    regime_df.to_csv(pred_dir / "regime_decomposed_mape.csv")

    save_json(
        {"headline": table.reset_index().to_dict(orient="records"),
         "regime": regime_df.reset_index().to_dict(orient="records")},
        pred_dir / "evaluation_summary.json",
    )


if __name__ == "__main__":
    main()
