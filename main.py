#!/usr/bin/env python3
"""Entry point: run the full pipeline end to end.

    Stage 1 (scripts/build_dataset.py)      raw archive -> square totals +
                                             target-square time series
    Stage 2 (scripts/run_eda.py)            exploratory analysis (Task 2)
    Stage 3 (scripts/tune_hyperparameters.py) hyperparameter tuning (grid
                                             search) on the top-traffic square
    Stage 4 (scripts/run_forecasting.py)    one-step-ahead forecasting
                                             experiments for all 3 squares x
                                             3 models (Task 4)

Each stage is also independently runnable (see the header of each script
under scripts/) -- useful when iterating on a single stage without re-running
the expensive raw-data ingestion. Stages 1 and 3 are the most expensive and
can be skipped with --skip-ingest / --skip-tuning once their outputs already
exist in data/processed/ and experiments/hyperparameter_tuning/.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --skip-ingest --skip-tuning
"""

import argparse
import subprocess
import sys

from src.utils import get_logger

logger = get_logger(__name__)

STAGES = [
    ("skip_ingest", "scripts/build_dataset.py", "Stage 1: raw data ingestion"),
    ("skip_eda", "scripts/run_eda.py", "Stage 2: exploratory analysis"),
    ("skip_tuning", "scripts/tune_hyperparameters.py", "Stage 3: hyperparameter tuning"),
    ("skip_forecasting", "scripts/run_forecasting.py", "Stage 4: forecasting experiments"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile Network Traffic Forecasting pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Stage 1 (raw ingestion)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip Stage 2 (EDA)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Stage 3 (hyperparameter tuning)")
    parser.add_argument("--skip-forecasting", action="store_true", help="Skip Stage 4 (forecasting)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flags = {
        "skip_ingest": args.skip_ingest,
        "skip_eda": args.skip_eda,
        "skip_tuning": args.skip_tuning,
        "skip_forecasting": args.skip_forecasting,
    }

    for flag_name, script, label in STAGES:
        if flags[flag_name]:
            logger.info("Skipping %s (%s)", label, script)
            continue
        logger.info("Running %s (%s)", label, script)
        result = subprocess.run([sys.executable, script, "--config", args.config])
        if result.returncode != 0:
            logger.error("%s failed with exit code %d", label, result.returncode)
            sys.exit(result.returncode)

    logger.info("Pipeline complete. See results/plots/, results/tables/, experiments/, data/processed/.")


if __name__ == "__main__":
    main()
