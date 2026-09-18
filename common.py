"""Shared constants and data/feature utilities used identically across notebooks.

Model classes and training/tuning loops are NOT here -- each lives in its own
notebook.
"""

import os
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42

RAW_DIR = "data/raw"
RAW_GLOB = "sms-call-internet-mi-*.txt"
SQUARE_TOTALS_PATH = "data/processed/square_totals.csv"
TARGET_SERIES_PATH = "data/processed/target_squares_timeseries.csv"
TARGET_SQUARES_META_PATH = "data/processed/target_squares.yaml"

FREQUENCY = "10min"
CHUNKSIZE = 2_000_000

OBSERVATION_START = "2013-11-01"
OBSERVATION_END = "2013-12-31"

FIXED_SQUARES = [4159, 4556]

EVAL_WEEK_START = "2013-12-16"
EVAL_WEEK_END = "2013-12-22"
TRAIN_START = "2013-11-01"


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds across libraries for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_target_squares_meta(path: str = TARGET_SQUARES_META_PATH) -> dict:
    """Read the top-3/fixed/target square IDs written by 00_data_pipeline.ipynb
    (data-derived, so read from disk rather than hardcoded here)."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_square_series(square_id: int, path: str = TARGET_SERIES_PATH, freq: str = FREQUENCY) -> pd.Series:
    """Reconstruct one square's regular, gap-filled 'internet' traffic series
    (dedup -> asfreq -> linear interpolation -> edge fill)."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet"].sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq(freq)
    return s.interpolate(method="linear").bfill().ffill()


def fourier_terms(index: pd.DatetimeIndex, period: int, n_harmonics: int = 2, freq_minutes: int = 10) -> pd.DataFrame:
    """Deterministic sin/cos regressors at `period` steps (dynamic harmonic
    regression). Phase is derived from absolute epoch time, not row position,
    so terms stay phase-consistent across disjoint slices (train vs. a single
    future timestamp during walk-forward forecasting)."""
    cycle_seconds = period * freq_minutes * 60
    epoch_seconds = index.asi8 // 10**9
    phase = (epoch_seconds % cycle_seconds) / cycle_seconds
    data = {}
    for k in range(1, n_harmonics + 1):
        data[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * phase)
        data[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * phase)
    return pd.DataFrame(data, index=index)


class SeriesScaler:
    """Thin wrapper around sklearn scalers for a single target column."""

    def __init__(self, method: str = "minmax"):
        if method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        elif method == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        if self.scaler is None:
            return series.values.reshape(-1, 1)
        return self.scaler.fit_transform(series.values.reshape(-1, 1))

    def transform(self, series: pd.Series) -> np.ndarray:
        if self.scaler is None:
            return series.values.reshape(-1, 1)
        return self.scaler.transform(series.values.reshape(-1, 1))

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return values
        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, and MAPE for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
