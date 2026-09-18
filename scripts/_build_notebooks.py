#!/usr/bin/env python3
"""One-off generator for notebooks/*.ipynb (thin, interactive wrappers around
the scripts/ pipeline stages -- not a duplicate implementation)."""

import nbformat as nbf


def cell(nb, kind, src):
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- 01: EDA ---
nb1 = nbf.v4.new_notebook()
cell(nb1, "md", """# 01 — Exploratory Data Analysis

Interactive walkthrough of Stage 1 (ingestion) and Stage 2 (EDA). The heavy
lifting lives in `src/ingest.py` and `scripts/run_eda.py`; this notebook
calls into that code and displays the results rather than duplicating it.

Run `python scripts/build_dataset.py` and `python scripts/run_eda.py` from
the repo root first if `data/processed/` is empty.""")

cell(nb1, "code", """import os
import sys

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, ".")

import pandas as pd
import yaml
from IPython.display import Image, display

from src.utils import load_config

cfg = load_config("config.yaml")""")

cell(nb1, "md", "## 1. Square totals and target squares")
cell(nb1, "code", """totals = pd.read_csv(cfg["data"]["square_totals_path"], index_col="square_id")["internet_total"]
with open(cfg["data"]["target_squares_meta_path"]) as f:
    meta = yaml.safe_load(f)

print("Observation period:", meta["observation_period"])
print("Top-3 squares by total internet traffic:", meta["top3_totals"])
print("Fixed squares (assignment brief):", meta["fixed_squares"])
totals.describe()""")

cell(nb1, "md", "## 2. Distribution of total traffic across all 10,000 squares")
cell(nb1, "code", """display(Image(filename="results/plots/exploratory_analysis/traffic_distribution.png"))
pd.read_csv("results/tables/data_summary.csv")""")

cell(nb1, "md", "## 3. First two weeks: top-3 squares + squares 4159, 4556")
cell(nb1, "code", 'display(Image(filename="results/plots/exploratory_analysis/five_squares_first_two_weeks.png"))')

cell(nb1, "md", "## 4. Additional analyses on the highest-traffic square: ACF/PACF and STL decomposition")
cell(nb1, "code", """display(Image(filename="results/plots/exploratory_analysis/acf_pacf_top1.png"))
display(Image(filename="results/plots/time_series_decomposition/stl_decomposition_top1.png"))
pd.read_csv("results/tables/stationarity_tests.csv")""")

# --------------------------------------------------------- 02: Preprocessing
nb2 = nbf.v4.new_notebook()
cell(nb2, "md", """# 02 — Data Ingestion and Preprocessing

Demonstrates the memory-efficient two-pass raw-data ingestion
(`src/ingest.py`) and the per-square feature engineering used by the
tree-based model (`src/preprocessor.py`). See `scripts/build_dataset.py` for
the full, non-interactive pipeline run over the whole archive.""")

cell(nb2, "code", """import os
import sys

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, ".")

import pandas as pd

from src.utils import load_config
from src.ingest import list_raw_files, naive_load_single_file, optimized_load_single_file

cfg = load_config("config.yaml")""")

cell(nb2, "md", "## 1. Memory-efficient loading: naive vs. optimized (single day file)")
cell(nb2, "code", """pd.read_csv("results/tables/memory_benchmark.csv")
# Peak RSS was measured out-of-process with `/usr/bin/time -l`; see
# scripts/benchmark_memory.py and the README/report for methodology.
# The cell below just demonstrates output-shape difference, not peak memory.
path = f'{cfg["data"]["raw_dir"]}/sms-call-internet-mi-2013-11-01.txt'
naive_df = naive_load_single_file(path)
optimized_df = optimized_load_single_file(path)
print("naive shape:", naive_df.shape)
print("optimized (aggregated) shape:", optimized_df.shape)""")

cell(nb2, "md", "## 2. Target-square time series (already extracted by Stage 1)")
cell(nb2, "code", """series_df = pd.read_csv(cfg["data"]["target_series_path"], parse_dates=["timestamp"])
series_df.head()""")

cell(nb2, "md", "## 3. Feature engineering for the tree-based model (XGBoost)")
cell(nb2, "code", """from src.preprocessor import add_lag_features, add_rolling_features, add_time_features, fourier_terms

square_id = series_df["square_id"].iloc[0]
s = series_df[series_df["square_id"] == square_id].set_index("timestamp")["internet"].asfreq(cfg["data"]["frequency"])
s = s.interpolate("linear").bfill().ffill()

feat_df = s.to_frame("internet")
feat_df = add_time_features(feat_df)
feat_df = add_lag_features(feat_df, "internet", cfg["preprocessing"]["lag_features"])
feat_df = add_rolling_features(feat_df, "internet", cfg["preprocessing"]["rolling_windows"])
feat_df = feat_df.dropna()
feat_df.head()""")

cell(nb2, "md", "## 4. Fourier seasonal regressors used by the SARIMAX baseline")
cell(nb2, "code", """fourier = fourier_terms(s.index[:288], period=cfg["models"]["arima"]["fourier_period"],
                        n_harmonics=cfg["models"]["arima"]["fourier_harmonics"])
fourier.plot(figsize=(10, 3), title="Fourier daily-seasonality regressors (2 days shown)")""")

# ----------------------------------------------------- 03: Model comparison
nb3 = nbf.v4.new_notebook()
cell(nb3, "md", """# 03 — Hyperparameter Tuning and Model Comparison

Loads the logged tuning experiments and the final per-square forecasting
results produced by `scripts/tune_hyperparameters.py` and
`scripts/run_forecasting.py`. Run those scripts first if the tables below
are empty.""")

cell(nb3, "code", """import os
import sys

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, ".")

import pandas as pd
import yaml
from IPython.display import Image, display

from src.utils import load_config

cfg = load_config("config.yaml")""")

cell(nb3, "md", "## 1. Hyperparameter tuning trajectory")
cell(nb3, "code", """tuning_df = pd.read_csv(cfg["paths"]["tuning_results"])
tuning_df""")

cell(nb3, "code", """with open("experiments/hyperparameter_tuning/best_params.yaml") as f:
    best_params = yaml.safe_load(f)
best_params""")

cell(nb3, "md", "## 2. Per-square performance tables (MAE / MAPE / RMSE)")
cell(nb3, "code", """with open(cfg["data"]["target_squares_meta_path"]) as f:
    meta = yaml.safe_load(f)

for sq in meta["top3_squares"]:
    print(f"--- Square {sq} ---")
    display(pd.read_csv(f"results/tables/model_performance_square_{sq}.csv", index_col=0))""")

cell(nb3, "md", "## 3. Prediction plots (9 total: 3 squares x 3 models)")
cell(nb3, "code", """for sq in meta["top3_squares"]:
    for model in ["ARIMA", "LSTM", "XGBoost"]:
        display(Image(filename=f"results/plots/model_predictions/square_{sq}_{model}.png"))""")

cell(nb3, "md", "## 4. Timing statistics")
cell(nb3, "code", 'pd.read_csv("results/tables/timing_stats.csv")')

cell(nb3, "md", "## 5. Cross-square, cross-model comparison")
cell(nb3, "code", """display(Image(filename="results/plots/comparative_analysis/metric_comparison.png"))
pd.read_csv("results/tables/model_performance_comparison_all.csv")""")

for path, nb in [
    ("notebooks/01_exploratory_analysis.ipynb", nb1),
    ("notebooks/02_preprocessing.ipynb", nb2),
    ("notebooks/03_model_comparison.ipynb", nb3),
]:
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)
