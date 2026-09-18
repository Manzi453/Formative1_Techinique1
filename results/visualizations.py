#!/usr/bin/env python3
"""Regenerate the cross-square, cross-model comparison plot/table from saved
experiment results, without retraining any models.

Usage:
    python results/visualizations.py --config config.yaml
"""

import argparse
import glob
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dirs, get_logger, load_config, save_fig

logger = get_logger(__name__)

MODEL_FILE_PREFIX = {"model_1": "ARIMA", "model_2": "LSTM", "model_3": "XGBoost"}
RESULT_PATTERN = re.compile(r"(model_\d)_results_sq(\d+)\.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate cross-square/model comparison plot and table")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    ensure_dirs(f"{paths_cfg['plots_dir']}/comparative_analysis")

    rows = []
    for path in sorted(glob.glob(os.path.join(paths_cfg["results_dir"], "model_*_results_sq*.pkl"))):
        match = RESULT_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        model_name, square_id = MODEL_FILE_PREFIX[match.group(1)], int(match.group(2))
        with open(path, "rb") as f:
            saved = pickle.load(f)
        rows.append({"square_id": square_id, "model": model_name, **saved["metrics"]})

    if not rows:
        logger.error("No saved per-square results found. Run `python scripts/run_forecasting.py` first.")
        return

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(paths_cfg["tables_dir"], "model_performance_comparison_all.csv"), index=False)

    metrics = ["MAE", "MAPE", "RMSE"]
    squares = sorted(comparison_df["square_id"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    for ax, metric in zip(axes, metrics):
        pivot = comparison_df.pivot(index="square_id", columns="model", values=metric).loc[squares]
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Square ID")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Model performance across squares and models", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(paths_cfg["plots_dir"], "comparative_analysis", "metric_comparison.png"))

    logger.info("Regenerated cross-square comparison table and plot from saved results.")


if __name__ == "__main__":
    main()
