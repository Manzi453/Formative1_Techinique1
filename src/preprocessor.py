"""Data cleaning, feature engineering, and scaling for time series data."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_series(df: pd.DataFrame, target_column: str, fill_method: str = "linear") -> pd.DataFrame:
    """Interpolate missing values and drop duplicate timestamps."""
    df = df[~df.index.duplicated(keep="first")].copy()
    df[target_column] = df[target_column].interpolate(method=fill_method).bfill().ffill()
    return df


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


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed dataframe chronologically (no shuffling).

    Returns (train, validation, test). If validation_size is 0, the
    validation dataframe will be empty.
    """
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - validation_size))

    train = df.iloc[:val_start]
    validation = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]

    return train, validation, test


def build_feature_pipeline(
    df: pd.DataFrame,
    target_column: str,
    fill_method: str,
    lags: List[int],
    rolling_windows: List[int],
    add_time: bool = True,
) -> pd.DataFrame:
    """Run the full cleaning + feature engineering pipeline used by tree/DL models."""
    df = clean_series(df, target_column, fill_method)
    if add_time:
        df = add_time_features(df)
    df = add_lag_features(df, target_column, lags)
    df = add_rolling_features(df, target_column, rolling_windows)
    df = df.dropna()
    return df
