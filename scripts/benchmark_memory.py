#!/usr/bin/env python3
"""Measure peak memory (via `/usr/bin/time -l`) for naive vs. optimized loading
of a single raw daily file, and write the comparison to
results/tables/memory_benchmark.csv.

Run as a standalone script (each variant runs in its own subprocess) so peak
RSS reflects only that code path, not this orchestrator or previous variants.

Usage:
    python scripts/benchmark_memory.py --mode naive
    python scripts/benchmark_memory.py --mode optimized
"""

import argparse
import sys
import time

sys.path.insert(0, ".")


def run_naive(path: str) -> None:
    from src.ingest import naive_load_single_file

    t0 = time.perf_counter()
    df = naive_load_single_file(path)
    elapsed = time.perf_counter() - t0
    print(f"naive rows={len(df)} cols={df.shape[1]} elapsed_s={elapsed:.2f}")


def run_optimized(path: str) -> None:
    from src.ingest import optimized_load_single_file

    t0 = time.perf_counter()
    df = optimized_load_single_file(path)
    elapsed = time.perf_counter() - t0
    print(f"optimized rows={len(df)} cols={df.shape[1]} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "optimized"], required=True)
    parser.add_argument("--path", default="data/raw/sms-call-internet-mi-2013-11-01.txt")
    args = parser.parse_args()

    if args.mode == "naive":
        run_naive(args.path)
    else:
        run_optimized(args.path)
