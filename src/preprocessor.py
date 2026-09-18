"""Feature engineering and scaling helpers for per-square time series."""

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from the datetime index."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_column: str, lags: List[int]) -> pd.DataFrame:
    """Add lagged copies of the target column."""
    df = df.copy()
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_column: str, windows: List[int]) -> pd.DataFrame:
    """Add rolling mean/std features computed on the (lagged) target column."""
    df = df.copy()
    shifted = df[target_column].shift(1)
    for window in windows:
        df[f"{target_column}_roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"{target_column}_roll_std_{window}"] = shifted.rolling(window).std()
    return df


def fourier_terms(
    index: pd.DatetimeIndex, period: int, n_harmonics: int = 2, freq_minutes: int = 10
) -> pd.DataFrame:
    """Deterministic sin/cos regressors at `period` steps, for dynamic harmonic
    regression (Hyndman & Athanasopoulos, ch. 12): a cheap, numerically stable
    substitute for a full seasonal-ARIMA term when the seasonal period is long
    (here, 144 steps = 1 day at 10-minute resolution), since fitting SARIMAX
    with seasonal_order=(P,D,Q,144) directly is computationally impractical.

    Phase is derived from absolute epoch time (not row position), so terms
    computed on disjoint slices of the same index (e.g. train vs. a single
    future timestamp during walk-forward forecasting) remain phase-consistent.
    """
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
