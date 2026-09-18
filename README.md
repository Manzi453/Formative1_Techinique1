# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## 1. Project Overview

Mobile network operators need accurate short-term forecasts of network
traffic to plan capacity, detect anomalies, and optimize resource
allocation. This project is an empirical investigation into **one-step-ahead
Internet-traffic forecasting** using the Telecom Italia "Milan CDR" dataset
(telecommunications activity across 10,000 geographical grid squares in
Milan, recorded at 10-minute intervals) [1].

**Research question:** *How do different sequential models compare for
one-step-ahead mobile network traffic forecasting, and how does their
performance vary across geographical areas with different traffic
characteristics?*

Three model families are implemented, tuned, and compared: a statistical
baseline (SARIMAX with Fourier seasonal regressors), a recurrent deep
learning model (LSTM), and a gradient-boosted tree ensemble (XGBoost).

## 2. Data Availability Note (important)

The assignment specifies a ~two-month archive and asks for the forecasting
evaluation to be run on the week of **Dec 16–22, 2013**. The raw archive
available in `data/raw/` only contains the contiguous block **2013-11-01 →
2013-11-30** (plus two disjoint single-day files, `2013-12-31` and
`2014-01-01`, which are excluded — see `config.yaml` for the reasoning).

**All results in this repository are therefore computed over the available
November window**, with the evaluation week substituted to the **last full
week of that window, 2013-11-24 → 2013-11-30** (mirroring the brief's
placement of the evaluation week near the end of the observation period).
If the missing December files are obtained (see references [2]/[3] below),
only `data.observation_period` and `data.eval_week` in `config.yaml` need to
change to reproduce the exact Dec 16–22 setup — no code changes required.

## 3. Dataset

Each daily raw file (`data/raw/sms-call-internet-mi-YYYY-MM-DD.txt`) is a
tab-separated, headerless file with one row per
`(square_id, time_interval, country_code)`:

```
square_id  timestamp_ms  country_code  sms_in  sms_out  call_in  call_out  internet
```

A single square/interval is split across many rows (one per country code
active in that cell at that time), so raw row counts are ~20-30x the number
of (square, interval) pairs actually needed. This project focuses on the
`internet` traffic column, matching the assignment's scope.

## 4. Data Handling & Memory Management

Daily files are 300-400 MB each (~10 GB for the full November archive) on
an 8 GB RAM machine — naively loading the whole archive is not viable. The
strategy implemented in [`src/ingest.py`](src/ingest.py) is a **two-pass,
chunked, column-pruned, dtype-downcast aggregation**:

- **Pass 1** (`compute_square_totals`): stream every file in fixed-size
  chunks, reading only the 2 columns needed, collapsing country codes with a
  per-chunk `groupby(square_id).sum()`. Only a 10,000-length running total is
  kept in memory for the whole archive.
- **Pass 2** (`extract_target_series`): once the top-traffic squares are
  known from Pass 1, stream the files again reading 3 columns, filtering to
  the ~5 target squares *before* aggregating (>99.9% row reduction), then
  collapsing country codes per `(square, timestamp)`.

Measured evidence (`python scripts/benchmark_memory.py`, results in
[`results/tables/memory_benchmark.csv`](results/tables/memory_benchmark.csv)):
on a single day file, the naive full load (`pd.read_csv`, all 8 columns,
default dtypes) peaks at **~950 MB** resident memory; the optimized loader
(chunked, 2 columns, `int32`/`float32`, early aggregation) peaks at
**~305 MB** (a ~68% reduction) while also collapsing 4.84M raw rows down to
10,000. Applied end-to-end, both passes process the full 30-day / ~10 GB
archive in **under 90 seconds combined**, with peak memory bounded by chunk
size rather than file or archive size — see the full discussion in the
report.

## 5. Models Implemented and Justification

| # | Model | File | Justification |
|---|-------|------|----------------|
| 1 | **SARIMAX + Fourier terms** | [`src/models/arima_model.py`](src/models/arima_model.py) | Classical linear statistical baseline. Daily seasonality (period=144 at 10-min resolution) is modeled with sin/cos (Fourier) regressors rather than a full seasonal-ARIMA term, since fitting `SARIMAX(seasonal_order=(P,D,Q,144))` directly is computationally impractical to tune (see report/Methodology). |
| 2 | **LSTM** | [`src/models/lstm_model.py`](src/models/lstm_model.py) | Recurrent architecture capable of learning nonlinear, longer-range temporal dependencies (daily/weekly rhythms, bursty spikes) that a linear model cannot. |
| 3 | **XGBoost** | [`src/models/xgboost_model.py`](src/models/xgboost_model.py) | Gradient-boosted trees on a lag/rolling/calendar feature representation. Fast to train, robust to noise and outliers/spikes, and gives feature-importance insight into which lags drive predictions. |

Full justification (grounded in exploratory analysis + literature) is in
the report, Section 3.

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (developed/tested on 3.11)
- ~4 GB free disk for the November raw archive, ~8 GB RAM recommended

### Setup

```bash
git clone <your-repo-url>
cd Formative1_Techinique1
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw daily files (`sms-call-internet-mi-YYYY-MM-DD.txt`) in
`data/raw/` — see references [1]-[3] below for where to obtain the dataset.

## 7. How to Run the Pipeline

The pipeline has 4 stages, each independently runnable (see each script's
docstring under `scripts/`). `main.py` runs them all in order:

```bash
python main.py --config config.yaml
```

| Stage | Script | Produces |
|---|---|---|
| 1. Ingest | `scripts/build_dataset.py` | `data/processed/square_totals.csv`, `data/processed/target_squares_timeseries.csv`, `data/processed/target_squares.yaml` |
| 2. EDA | `scripts/run_eda.py` | `results/plots/exploratory_analysis/*`, `results/plots/time_series_decomposition/*`, `results/tables/data_summary.csv`, `results/tables/stationarity_tests.csv` |
| 3. Tuning | `scripts/tune_hyperparameters.py` | `experiments/hyperparameter_tuning/tuning_results.csv`, `experiments/hyperparameter_tuning/best_params.yaml` |
| 4. Forecasting | `scripts/run_forecasting.py` | `results/plots/model_predictions/square_<id>_<model>.png` (9 plots), `results/tables/model_performance_square_<id>.csv` (3 tables), `results/tables/timing_stats.csv`, `experiments/experiment_log.csv` |

Stages 1 and 3 are the most expensive; skip them once their outputs already
exist with e.g. `python main.py --skip-ingest --skip-tuning`.

Regenerate the cross-square/cross-model comparison plot from saved results
without retraining:
```bash
python results/visualizations.py
```

Also see the notebooks (`notebooks/01_exploratory_analysis.ipynb`,
`notebooks/02_preprocessing.ipynb`, `notebooks/03_model_comparison.ipynb`)
for an interactive walkthrough of the same steps.

## 8. Results Summary

See [`results/tables/`](results/tables/) for the full per-square performance
tables and [`reports/`](reports/) for the complete write-up, methodology,
and discussion.

## 9. Repository Structure

```
├── data/
│   ├── raw/               # daily CDR files (gitignored, see section 3)
│   └── processed/         # square_totals.csv, target_squares_timeseries.csv
├── scripts/                # the 4 pipeline stages + memory benchmark
├── src/                    # ingest, preprocessing, models, evaluation, utils
├── notebooks/               # interactive EDA / preprocessing / comparison
├── experiments/             # experiment log + hyperparameter tuning results
├── results/                 # plots and summary tables
├── reports/                 # final report and bibliography
├── main.py                  # orchestrates all 4 stages
├── config.yaml               # single source of truth for paths/hyperparameters/seed
└── requirements.txt
```

## 10. Reproducibility

Random seeds (NumPy, TensorFlow, XGBoost) are fixed via `config.yaml` and
set centrally in [`src/utils.py`](src/utils.py)'s `set_seed()`.

## 11. References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city
of Milan and the Province of Trentino," *Sci Data*, vol. 2, 150055, 2015.
https://doi.org/10.1038/sdata.2015.55
[2] https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
[3] https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

## 12. Author and Date

- **Author:** Manzi Ivan
- **Date:** September 2026
- **Course/Assignment:** Formative 1 — Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting
