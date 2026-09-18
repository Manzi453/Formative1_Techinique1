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
baseline (SARIMA/SARIMAX with Fourier seasonal regressors), a recurrent
deep learning model (LSTM), and a Temporal Convolutional Network (TCN).

The full write-up — methodology, results, and discussion — is in
[`report/final_report.pdf`](report/final_report.pdf).

## 2. Data Window

The assignment specifies a ~two-month archive and asks for the forecasting
evaluation to be run on the week of **Dec 16–22, 2013**. The raw archive in
`data/raw/` covers **2013-11-01 → 2014-01-01**; the observation period used
for all EDA/ranking is the full two months, **2013-11-01 → 2013-12-31**
(`OBSERVATION_START`/`OBSERVATION_END` in `common.py`), and forecasting is
evaluated on the literal assignment week, **2013-12-16 → 2013-12-22**
(`EVAL_WEEK_START`/`EVAL_WEEK_END`), with models trained on
2013-11-01 → 2013-12-15.

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

Daily files are 300-400 MB each (~20 GB for the full 2-month archive) on
an 8 GB RAM machine — naively loading the whole archive is not viable. The
strategy implemented in
[`notebooks/00_data_pipeline.ipynb`](notebooks/00_data_pipeline.ipynb) is a
**two-pass, chunked, column-pruned, dtype-downcast aggregation**:

- **Pass 1**: stream every file in fixed-size chunks, reading only the 2
  columns needed, collapsing country codes with a per-chunk
  `groupby(square_id).sum()`. Only a 10,000-length running total is kept in
  memory for the whole archive.
- **Pass 2**: once the top-traffic squares are known from Pass 1, stream
  the files again reading 3 columns, filtering to the ~5 target squares
  *before* aggregating (>99.9% row reduction), then collapsing country
  codes per `(square, timestamp)`.

Measured evidence (peak RSS via `/usr/bin/time -l`, results in
[`results/memory_benchmark.csv`](results/memory_benchmark.csv)): on a
single day file, the naive full load (`pd.read_csv`, all 8 columns, default
dtypes) peaks at **~635 MB** resident memory; the optimized loader
(chunked, 2 columns, `int32`/`float32`, early aggregation) peaks at
**~292 MB** (a ~54% reduction) while also collapsing 4.84M raw rows down to
10,000. Applied end-to-end, both passes process the full ~2-month / ~20 GB
archive in a few minutes, with peak memory bounded by chunk size rather
than file or archive size — see the full discussion in the report.

## 5. Models Implemented and Justification

| # | Model | Notebook | Justification |
|---|-------|----------|----------------|
| 1 | **SARIMA + Fourier terms** | [`02_sarima.ipynb`](notebooks/02_sarima.ipynb) | Classical linear statistical baseline. Daily seasonality (period=144 at 10-min resolution) is modeled with sin/cos (Fourier) regressors rather than a full seasonal-ARIMA term, since fitting `SARIMAX(seasonal_order=(P,D,Q,144))` directly is computationally impractical to tune (see report, Methodology). |
| 2 | **LSTM** | [`03_lstm_tcn.ipynb`](notebooks/03_lstm_tcn.ipynb) | Recurrent architecture capable of learning nonlinear, longer-range temporal dependencies (daily/weekly rhythms, bursty spikes) that a linear model cannot; its sequential hidden-state update naturally privileges the most recent observations. |
| 3 | **TCN** | [`03_lstm_tcn.ipynb`](notebooks/03_lstm_tcn.ipynb) | Dilated causal 1D convolutions with residual connections — a nonlinear, non-recurrent sequence model, structurally distinct from LSTM (parallel convolution vs. sequential recurrence, no built-in recency bias), that tests whether recurrence is actually necessary and is substantially faster to train. |

Full justification (grounded in exploratory analysis + literature) is in
the report, Section 2.

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (developed/tested on 3.11)
- ~20 GB free disk for the raw archive, ~8 GB RAM recommended
- Jupyter (`jupyter nbconvert` / `jupyter lab`) to run the notebooks

### Setup

```bash
git clone https://github.com/Manzi453/Formative1_Techinique1
cd Formative1_Techinique1
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw daily files (`sms-call-internet-mi-YYYY-MM-DD.txt`) in
`data/raw/` — see references [1]-[3] below for where to obtain the dataset.

## 7. How to Run the Pipeline

The pipeline is 5 notebooks, run **in order**. Each notebook picks up where
the previous one left off by reading files the earlier notebook wrote to
`data/processed/`, `results/`, or `figures/` — there is no separate
orchestration script; run them from `jupyter lab`/`jupyter notebook`, or
headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_sarima.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_lstm_tcn.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_comparison.ipynb
```

| Stage | Notebook | Produces |
|---|---|---|
| 1. Data pipeline | `00_data_pipeline.ipynb` | `data/processed/{square_totals.csv, target_squares_timeseries.csv, target_squares.yaml}`, `results/memory_benchmark.csv` |
| 2. EDA | `01_eda.ipynb` | `figures/eda_*.png`, `results/{data_summary.csv, stationarity_tests.csv}` |
| 3. SARIMA | `02_sarima.ipynb` | `results/tuning_results_sarima.csv`, `results/best_params.yaml` (`sarima` key), `results/pred_sarima_sq*.pkl` |
| 4. LSTM + TCN | `03_lstm_tcn.ipynb` | `results/tuning_results_lstm_tcn.csv`, `results/best_params.yaml` (`lstm`/`tcn` keys), `results/pred_{lstm,tcn}_sq*.pkl` |
| 5. Comparison | `04_model_comparison.ipynb` | `figures/pred_sq*_*.png` (9 plots), `figures/model_comparison.png`, `results/model_performance_square_*.csv`, `results/timing_stats.csv`, `results/model_performance_comparison_all.csv` |

Notebook 00 (full-archive ingestion) takes a few minutes; notebook 02
(SARIMA grid search + walk-forward eval) a few minutes; notebook 03
(LSTM/TCN tuning + walk-forward training on 3 squares) is the most
expensive, ~20–30 min on CPU. Notebook 04 does **not** retrain anything —
it only reads the `results/pred_*.pkl` files written by notebooks 02 and 03.

`common.py` holds the constants and data/feature-engineering functions
shared identically across every notebook (path constants, seed,
`load_square_series`, preprocessing/Fourier helpers, `compute_metrics`) —
see its module docstring for exactly what is and isn't there.

**If you edit `common.py` and re-run a notebook from an already-open Jupyter
kernel** (e.g. in VS Code / JupyterLab), restart that kernel first. Python
caches imported modules, so a kernel started before the edit will keep using
the old `common.py` values in memory and silently regenerate outputs with
stale constants even though the file on disk is correct — the symptom is
`data/processed/`/`results/` files reverting to old numbers after a notebook
you didn't intend to touch gets re-run. `jupyter nbconvert --execute` (as
used above) is unaffected, since it always starts a fresh kernel.

## 8. Results Summary

See [`results/`](results/) for the full per-square performance tables and
[`report/final_report.pdf`](report/final_report.pdf) for the complete
write-up, methodology, and discussion. Headline finding: **SARIMA leads on
MAE/RMSE on every square** (LSTM edges it out on MAPE on one square, a
near-tie); **LSTM is a clear second, well ahead of TCN on every square and
metric**; **TCN is the weakest model everywhere**, most dramatically on
square 5161 (the most volatile of the top-3) — see the report's
failure-case analysis (Section 6.3) for why, and why more training data
alone didn't close that gap.

## 9. Repository Structure

```
├── data/
│   ├── raw/               # daily CDR files (gitignored, see section 3)
│   └── processed/         # square_totals.csv, target_squares_timeseries.csv (gitignored, regenerate via notebook 00)
├── notebooks/              # the 5-stage pipeline (run in order 00 -> 04)
├── common.py                # shared constants + data/feature-engineering functions
├── figures/                 # all plots produced by the notebooks
├── results/                 # all tables + saved predictions produced by the notebooks
├── report/                  # final_report.pdf, references.bib, video_script.md
├── requirements.txt
└── README.md
```

## 10. Reproducibility

Random seeds (NumPy, TensorFlow) are fixed via `common.py`'s `set_seed()`;
XGBoost is no longer used, and TCN/LSTM's own `random_state`/Keras seeding
is covered by the same call. SARIMAX and the grid-search tuning are
otherwise deterministic given fixed data.

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
