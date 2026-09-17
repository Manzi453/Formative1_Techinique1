#!/usr/bin/env python3
"""Regenerate comparison plots/tables from saved experiment results
without retraining any models.

Usage:
    python results/visualizations.py --config config.yaml
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_processed_data
from src.evaluation import compare_models, plot_metric_comparison, plot_predictions
from src.preprocessor import chronological_split
from src.utils import get_logger, load_config, save_fig

logger = get_logger(__name__)

MODEL_FILES = {
    "ARIMA": "model_1_results.pkl",
    "LSTM": "model_2_results.pkl",
    "XGBoost": "model_3_results.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate result plots/tables")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]

    predictions, metrics = {}, {}
    for model_name, filename in MODEL_FILES.items():
        result_path = os.path.join(paths_cfg["results_dir"], filename)
        if not os.path.exists(result_path):
            logger.warning("No saved results for %s at %s, skipping", model_name, result_path)
            continue
        with open(result_path, "rb") as f:
            saved = pickle.load(f)
        predictions[model_name] = saved["predictions"]
        metrics[model_name] = saved["metrics"]

    if not predictions:
        logger.error("No saved results found. Run `python main.py` first.")
        return

    features_df = load_processed_data(data_cfg["processed_path"])
    _, _, test_df = chronological_split(features_df, data_cfg["test_size"], data_cfg["validation_size"])
    y_true = test_df[data_cfg["target_column"]].values

    comparison_df = compare_models(metrics)
    comparison_df.to_csv(os.path.join(paths_cfg["tables_dir"], "model_performance.csv"))

    fig1 = plot_predictions(y_true, predictions)
    save_fig(fig1, os.path.join(paths_cfg["plots_dir"], "model_predictions", "all_models_comparison.png"))

    fig2 = plot_metric_comparison(comparison_df)
    save_fig(fig2, os.path.join(paths_cfg["plots_dir"], "comparative_analysis", "metric_comparison.png"))

    logger.info("Regenerated plots and tables from saved results.")


if __name__ == "__main__":
    main()
