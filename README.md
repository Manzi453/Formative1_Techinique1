# Mobile Network Traffic Prediction — Time Series Forecasting

## 1. Project Overview

Mobile network operators need accurate short-term forecasts of network traffic
(e.g. cellular data volume, call/SMS volume, or per-cell throughput) to plan
capacity, detect anomalies, and optimize resource allocation. This project
frames mobile network traffic prediction as a **univariate/multivariate time
series forecasting problem** and compares three modeling paradigms —
statistical, deep learning, and gradient-boosted trees — under a common,
reproducible evaluation pipeline.

**Problem statement:** Given historical mobile network traffic measurements
sampled at regular intervals (e.g. hourly), predict future traffic volume over
a forecast horizon, so that network capacity planning and anomaly detection
can be performed proactively rather than reactively.

## 2. Dataset Description

The pipeline expects a time-indexed CSV file containing mobile network traffic
measurements (e.g. total call, SMS, and internet traffic per grid cell / base
station, aggregated over fixed time intervals). See
[`data/raw/README.md`](data/raw/README.md) for the expected schema, the
source used for this project, and licensing notes.

Raw and processed data files are **not** version-controlled (see
[`.gitignore`](.gitignore)) to keep the repository lightweight and to respect
dataset licensing terms. Place the raw file(s) in `data/raw/` before running
the pipeline.

## 3. Models Implemented and Justification

Three complementary forecasting approaches were selected to cover the main
families of time series models used in the literature:

| # | Model | File | Justification |
|---|-------|------|----------------|
| 1 | **ARIMA / SARIMAX** | [`src/models/arima_model.py`](src/models/arima_model.py) | Classical statistical baseline. Strong on linear, stationary (or differenced) series; provides an interpretable benchmark and confidence intervals against which more complex models are judged. |
| 2 | **LSTM (Long Short-Term Memory)** | [`src/models/lstm_model.py`](src/models/lstm_model.py) | Recurrent deep learning architecture capable of learning long-range temporal dependencies and non-linear patterns (e.g. daily/weekly seasonality, bursty traffic) that ARIMA cannot capture. |
| 3 | **XGBoost** | [`src/models/xgboost_model.py`](src/models/xgboost_model.py) | Gradient-boosted trees applied to a lag/feature-engineered representation of the series. Fast to train, robust to noise, and gives feature-importance insight into which lags/features drive predictions. |

Comparing a linear statistical model, a recurrent neural network, and a
tree-based ensemble gives a broad, defensible view of the accuracy/complexity
trade-offs for this forecasting task.

## 4. Installation & Setup

### Prerequisites
- Python 3.10+
- pip (or conda)

### Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Formative1_Techinique1

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 5. Dependencies

All dependencies and pinned versions are listed in
[`requirements.txt`](requirements.txt) (pandas, numpy, scikit-learn,
statsmodels, tensorflow, xgboost, matplotlib, seaborn, pyyaml, jupyter).

## 6. How to Run the Project

1. **Add data** — place the raw dataset in `data/raw/` (see
   [`data/raw/README.md`](data/raw/README.md)).
2. **Configure** — edit [`config.yaml`](config.yaml) to set data paths,
   model hyperparameters, training settings, and the random seed.
3. **Explore (optional)** — run the notebooks in order for exploratory
   analysis, preprocessing rationale, and model comparison:
   ```bash
   jupyter notebook notebooks/01_exploratory_analysis.ipynb
   jupyter notebook notebooks/02_preprocessing.ipynb
   jupyter notebook notebooks/03_model_comparison.ipynb
   ```
4. **Run the full pipeline** (load → preprocess → train all 3 models →
   evaluate → save results/plots):
   ```bash
   python main.py --config config.yaml
   ```
5. **Inspect results** — trained-model artifacts and metrics are written to
   `experiments/results/`, comparison tables to `results/tables/`, and plots
   to `results/plots/`. Regenerate plots only (without retraining) with:
   ```bash
   python results/visualizations.py
   ```

## 7. Results Summary

Populate this section after running `main.py`. A template comparison table is
maintained at [`results/tables/model_performance.csv`](results/tables/model_performance.csv)
and the full experiment history at
[`experiments/experiment_log.csv`](experiments/experiment_log.csv).

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| ARIMA | — | — | — |
| LSTM | — | — | — |
| XGBoost | — | — | — |

The full write-up, methodology, and discussion are in
[`reports/final_report.pdf`](reports/final_report.pdf).

## 8. Repository Structure

```
├── data/            # raw/ (gitignored) and processed data
├── notebooks/       # EDA, preprocessing, model comparison notebooks
├── src/             # data loading, preprocessing, models, training, evaluation
├── experiments/     # experiment log + hyperparameter tuning results
├── results/         # plots and summary tables
├── reports/         # final report and bibliography
├── main.py          # pipeline entry point
├── config.yaml      # single source of truth for paths/hyperparameters/seed
└── requirements.txt
```

## 9. Reproducibility

All random seeds (NumPy, TensorFlow, XGBoost, train/test splitting) are fixed
via `config.yaml` and set centrally in [`src/utils.py`](src/utils.py)'s
`set_seed()` to ensure results are reproducible across runs.

## 10. Author and Date

- **Author:** Manzi Ivan
- **Date:** September 2026
- **Course/Assignment:** Formative 1 — Technique 1
