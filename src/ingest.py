"""Memory-efficient ingestion of the raw Milan CDR (Call Detail Record) archive.

Each daily raw file (`sms-call-internet-mi-YYYY-MM-DD.txt`) is a tab-separated,
headerless file with one row per (square_id, time_interval, country_code):

    square_id  timestamp_ms  country_code  sms_in  sms_out  call_in  call_out  internet

A single square/interval is split across many rows (one per country code), so
the raw row count is roughly 20-30x the number of (square, interval) pairs we
actually need. Daily files are ~300-400MB each; loading a month naively with
default dtypes would not fit comfortably in 8GB of RAM. The strategy used here
is a two-pass, chunked, column-pruned, dtype-downcast aggregation:

Pass 1 (`compute_square_totals`): stream every file in fixed-size chunks,
reading only the 2 columns needed (square_id, internet), and immediately
collapse country codes with a per-chunk `groupby(square_id).sum()`. Only a
10,000-length running total is kept in memory across the whole archive.

Pass 2 (`extract_target_series`): once the target squares are known from pass
1's ranking, stream the files again, reading 3 columns (square_id, timestamp,
internet), filter to the ~5 target squares *before* aggregating (a >99.9%
row-count reduction), then collapse country codes per (square, timestamp).

This keeps peak memory bounded by one chunk (not one file, not the archive),
regardless of how many days are processed.
"""

import glob
import os
from typing import Iterable, List

import numpy as np
import pandas as pd

RAW_COLUMN_NAMES = [
    "square_id", "timestamp_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]
COL_INDEX = {name: i for i, name in enumerate(RAW_COLUMN_NAMES)}


def list_raw_files(raw_dir: str, raw_glob: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(raw_dir, raw_glob)))
    if not files:
        raise FileNotFoundError(f"No raw files matching '{raw_glob}' found in '{raw_dir}'.")
    return files


def _filter_files_by_date(files: List[str], start: str, end: str) -> List[str]:
    """Keep only files whose embedded YYYY-MM-DD date falls within [start, end]."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    kept = []
    for f in files:
        date_str = os.path.basename(f).replace("sms-call-internet-mi-", "").replace(".txt", "")
        try:
            file_date = pd.Timestamp(date_str)
        except ValueError:
            continue
        if start_ts <= file_date <= end_ts:
            kept.append(f)
    return kept


def compute_square_totals(
    files: List[str],
    chunksize: int = 2_000_000,
) -> pd.Series:
    """Pass 1: total internet traffic per square_id across the given files.

    Only the square_id and internet columns are read; country codes are
    collapsed with an additive groupby per chunk. Memory footprint is
    O(chunksize) for the read buffer plus O(n_squares) for the accumulator,
    never O(file size) or O(archive size).
    """
    totals = pd.Series(dtype="float64")

    usecols = [COL_INDEX["square_id"], COL_INDEX["internet"]]
    dtypes = {COL_INDEX["square_id"]: "int32", COL_INDEX["internet"]: "float32"}

    for path in files:
        for chunk in pd.read_csv(
            path, sep="\t", header=None, usecols=usecols, dtype=dtypes,
            names=["square_id", "internet"], chunksize=chunksize,
        ):
            chunk_sum = chunk.groupby("square_id", sort=False)["internet"].sum()
            totals = totals.add(chunk_sum, fill_value=0.0)

    totals.index.name = "square_id"
    return totals.sort_index()


def extract_target_series(
    files: List[str],
    target_squares: Iterable[int],
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Pass 2: full 10-minute internet time series for a handful of squares.

    Filtering to `target_squares` happens before the groupby, so each chunk
    is reduced to a tiny fraction of its original size before any aggregation
    or concatenation, keeping the accumulated result small regardless of
    archive size.
    """
    target_squares = set(int(s) for s in target_squares)
    usecols = [COL_INDEX["square_id"], COL_INDEX["timestamp_ms"], COL_INDEX["internet"]]
    dtypes = {
        COL_INDEX["square_id"]: "int32",
        COL_INDEX["timestamp_ms"]: "int64",
        COL_INDEX["internet"]: "float32",
    }

    day_frames = []
    for path in files:
        chunk_frames = []
        for chunk in pd.read_csv(
            path, sep="\t", header=None, usecols=usecols, dtype=dtypes,
            names=["square_id", "timestamp_ms", "internet"], chunksize=chunksize,
        ):
            chunk = chunk[chunk["square_id"].isin(target_squares)]
            if len(chunk):
                chunk_frames.append(
                    chunk.groupby(["square_id", "timestamp_ms"], sort=False)["internet"].sum().reset_index()
                )
        if chunk_frames:
            day_df = pd.concat(chunk_frames, ignore_index=True)
            day_df = day_df.groupby(["square_id", "timestamp_ms"], sort=False)["internet"].sum().reset_index()
            day_frames.append(day_df)

    result = pd.concat(day_frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp_ms"], unit="ms")
    result = result.drop(columns="timestamp_ms").sort_values(["square_id", "timestamp"])
    return result.reset_index(drop=True)


def naive_load_single_file(path: str) -> pd.DataFrame:
    """Unoptimized baseline loader (whole file, all columns, default dtypes).

    Used only to produce a controlled before/after memory comparison on a
    single day file; never used on the full archive.
    """
    return pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMN_NAMES)


def optimized_load_single_file(path: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Optimized loader applied to a single file, for the same comparison."""
    usecols = [COL_INDEX["square_id"], COL_INDEX["internet"]]
    dtypes = {COL_INDEX["square_id"]: "int32", COL_INDEX["internet"]: "float32"}
    chunk_sums = []
    for chunk in pd.read_csv(
        path, sep="\t", header=None, usecols=usecols, dtype=dtypes,
        names=["square_id", "internet"], chunksize=chunksize,
    ):
        chunk_sums.append(chunk.groupby("square_id", sort=False)["internet"].sum())
    return pd.concat(chunk_sums, axis=1).sum(axis=1).reset_index()
