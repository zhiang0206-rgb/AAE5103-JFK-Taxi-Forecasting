# JFK Taxi Pickup Forecasting — AAE5103

Five forecasting methods for recorded yellow-taxi pickups at JFK Airport: a neural network, decision tree, linear regression, seasonal time series and a three-week historical mean.

At each New York local prediction time `t`, the target is recorded pickups in `(t + 10 minutes, t + 40 minutes]`; records exactly at the next day's midnight are excluded. Predictions are generated every minute from 06:00 to 23:20. Train: January–May 2025; validation and model selection: June; fixed-model test: July–December. Windows overlap. Forecasts estimate completed recorded taxi trips, not passengers or unmet demand.

## Project layout

| Folder | Contents |
| --- | --- |
| `annual_neural/` | 12 → 16 → 1 ReLU neural network; training, validation, prediction and frozen checkpoint |
| `annual_tree/` | Decision tree model and validation |
| `linear_regression/` | Linear regression model and validation |
| `time_series_forecast/` | Seasonal 10-minute AR(1) method and time series preparation |
| `historical_mean/` | Three-week historical mean baseline (fixed rule; no training) |
| `generalization_2025/` | TLC data collection and sample construction |
| `final_test_2025/` | Shared frozen-model test and summary code |
| `neural_diagnostics/` | 80-epoch, three-seed neural network diagnostics |

`results/` includes frozen model files and selected metrics. The data, large prediction tables and generated charts are not included in this code repository. Original project files have been left unchanged.

## Data and reproducibility

Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 2025 yellow taxis; JFK is pickup location ID 132. Set up Python 3.9 and `pip install -r requirements.txt`. The original experiment used Python 3.9.6.

To use the supplied project data, copy the original `generalization_2025/monthly_jfk/` directory and its `shared_samples/train.csv`, `validation.csv`, and `test.csv` into the same paths in this checkout. Copy `time_series_forecast/data/train_minutes.csv` and `validation_minutes.csv`, and `annual_neural/results/validation_predictions.csv` from the original project for the final test's input hash and June forecast checks. These files are omitted from this repository. The test script checks SHA-256 hashes, so regenerated files with different bytes may fail those checks even if their content is similar.

Alternatively, `generalization_2025/collect_jfk.py` downloads and filters the TLC data; for March it expects the original March parquet at `data/yellow_tripdata_2025-03.parquet`. Then run `generalization_2025/prepare_2025.py` and `time_series_forecast/prepare_series.py`. Downloading all months requires substantial bandwidth. See `generalization_2025/shared_samples/README_全年数据说明.md` for fields, split and window rules. Regeneration and exact hash verification have not been tested in this code-only checkout.

With the original inputs in place, `python final_test_2025/test_all.py` checks June forecasts, verifies test inputs against raw records and writes test results. Running training or evaluation scripts writes to `results/`; back up the supplied checkpoints before retraining. The individual method `test.py` scripts call the same shared test. Model files use PyTorch/joblib serialization; load only files from a trusted source.

The three-seed experiment is a development diagnostic, not test-set model selection. Repeatedly tuning against July–December would invalidate that period as an independent test.
