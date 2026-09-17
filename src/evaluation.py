"""Forecast accuracy metrics, model comparison, and result visualizations."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build a tidy comparison table from {model_name: metrics_dict}."""
    return pd.DataFrame(results).T.sort_values("RMSE")


def plot_predictions(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Model Predictions vs Actual",
) -> plt.Figure:
    """Plot actual vs. predicted values for one or more models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label="Actual", color="black", linewidth=1.5)
    for name, y_pred in predictions.items():
        ax.plot(y_pred, label=name, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Traffic volume")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing MAE / RMSE / MAPE across models."""
    fig, axes = plt.subplots(1, len(comparison_df.columns), figsize=(4 * len(comparison_df.columns), 4))
    if len(comparison_df.columns) == 1:
        axes = [axes]
    for ax, metric in zip(axes, comparison_df.columns):
        comparison_df[metric].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(metric)
        ax.set_ylabel(metric)
    fig.tight_layout()
    return fig
