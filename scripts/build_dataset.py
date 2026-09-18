#!/usr/bin/env python3
"""Stage 1 of the pipeline: turn the raw Milan CDR archive into two small,
analysis-ready artifacts (see src/ingest.py for the memory-management
rationale):

  data/processed/square_totals.csv           -- total internet traffic per
                                                 square_id over the observation
                                                 period (all 10,000 squares).
  data/processed/target_squares_timeseries.csv -- full 10-minute internet
                                                 series for the target squares
                                                 only (top-3 by total traffic,
                                                 plus the two fixed squares
                                                 required by the brief).
  data/processed/target_squares.yaml          -- records which squares were
                                                 selected and why, so later
                                                 stages don't need to
                                                 re-derive the ranking.

Usage:
    python scripts/build_dataset.py --config config.yaml
"""

import argparse
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.ingest import compute_square_totals, extract_target_series, list_raw_files, _filter_files_by_date
from src.utils import ensure_dirs, get_logger, load_config

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    ensure_dirs(data_cfg["processed_dir"])

    all_files = list_raw_files(data_cfg["raw_dir"], data_cfg["raw_glob"])
    obs_files = _filter_files_by_date(
        all_files, data_cfg["observation_period"]["start"], data_cfg["observation_period"]["end"]
    )
    logger.info(
        "Observation period %s -> %s: %d files selected out of %d in data/raw/",
        data_cfg["observation_period"]["start"], data_cfg["observation_period"]["end"],
        len(obs_files), len(all_files),
    )

    # ---- Pass 1: total internet traffic per square (all 10,000 squares) ----
    t0 = time.perf_counter()
    totals = compute_square_totals(obs_files, chunksize=data_cfg["chunksize"])
    logger.info("Pass 1 (square totals) done in %.1fs over %d files -> %d squares",
                time.perf_counter() - t0, len(obs_files), len(totals))
    totals.to_csv(data_cfg["square_totals_path"], header=["internet_total"])

    top3 = totals.sort_values(ascending=False).head(3)
    fixed = data_cfg["fixed_squares"]
    target_squares = sorted(set(int(s) for s in top3.index) | set(int(s) for s in fixed))
    logger.info("Top-3 squares by total internet traffic: %s", top3.to_dict())
    logger.info("Target squares for time-series extraction (top-3 U fixed): %s", target_squares)

    with open(data_cfg["target_squares_meta_path"], "w") as f:
        yaml.safe_dump({
            "top3_squares": [int(s) for s in top3.index],
            "top3_totals": {int(k): float(v) for k, v in top3.items()},
            "fixed_squares": [int(s) for s in fixed],
            "target_squares": target_squares,
            "observation_period": data_cfg["observation_period"],
        }, f)

    # ---- Pass 2: full 10-min series for the target squares only ----
    t0 = time.perf_counter()
    series_df = extract_target_series(obs_files, target_squares, chunksize=data_cfg["chunksize"])
    logger.info("Pass 2 (target series) done in %.1fs -> %d rows for %d squares",
                time.perf_counter() - t0, len(series_df), series_df["square_id"].nunique())
    series_df.to_csv(data_cfg["target_series_path"], index=False)

    logger.info("Saved: %s, %s, %s",
                data_cfg["square_totals_path"], data_cfg["target_series_path"], data_cfg["target_squares_meta_path"])


if __name__ == "__main__":
    main()
