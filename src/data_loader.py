"""Load raw mobile network traffic data efficiently and safely."""

import os

import pandas as pd


def load_raw_data(
    path: str,
    datetime_column: str = "timestamp",
    frequency: str = "H",
) -> pd.DataFrame:
    """Load the raw CSV, parse the datetime index, and enforce a fixed frequency.

    Uses chunked reading for large files and downcasts numeric columns to
    reduce memory footprint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. See data/raw/README.md "
            "for where to obtain the dataset and where to place it."
        )

    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000, parse_dates=[datetime_column]):
        for col in chunk.select_dtypes(include=["float64"]).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast="float")
        for col in chunk.select_dtypes(include=["int64"]).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(datetime_column).set_index(datetime_column)

    if frequency:
        df = df.asfreq(frequency)

    return df


def load_processed_data(path: str) -> pd.DataFrame:
    """Load a previously preprocessed dataset (already indexed by time)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data file not found at '{path}'. Run the preprocessing "
            "step (src/preprocessor.py or notebooks/02_preprocessing.ipynb) first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Persist a processed dataframe to disk, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
