# Deep Neural Analysis of Roll Rate Forecasting

Trains an ARIMA baseline plus LSTM and GRU recurrent networks (with and without macroeconomic inputs) on the monthly Bucket-1 → Bucket-2 roll-rate series constructed from the Freddie Mac Single-Family Loan-Level Dataset (2004–2021).

## Project layout

```
project/
├── configs/config.yaml          Single source of truth for paths & hparams
├── requirements.txt
├── main.py                      End-to-end orchestrator
├── scripts/                     Pipeline-stage CLI entrypoints
│   ├── 01_build_roll_rate_series.py
│   ├── 02_train_arima.py
│   ├── 03_train_neural.py
│   ├── 04_evaluate_all.py
├── src/
│   ├── data/                    Loaders, roll-rate construction, scaling
│   ├── models/                  ARIMA, LSTM, GRU, Optuna tuner
│   ├── training/                Trainer with early stopping & LR schedules
│   └── utils/                   Seeding, IO, metrics
└── artifacts/                   
```

## Data layout

The pipeline expects the following sibling directories of `project/`:

```
performance/    *.parquet (9 selected Freddie Mac fields)
origination/    *.txt     (raw Freddie Mac origination files)
UNRATE.csv  FEDFUNDS.csv  MORTGAGE30US.csv  CPIAUCSL.csv  HPIPONM226S.csv
```

