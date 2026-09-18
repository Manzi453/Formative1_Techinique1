#!/usr/bin/env python3
"""Stage 2: exploratory analysis required by Task 2 of the assignment.

Produces:
  results/plots/exploratory_analysis/traffic_distribution.png
  results/plots/exploratory_analysis/five_squares_first_two_weeks.png
  results/plots/exploratory_analysis/acf_pacf_top1.png
  results/plots/time_series_decomposition/stl_decomposition_top1.png
  results/tables/data_summary.csv
  results/tables/stationarity_tests.csv

Usage:
    python scripts/run_eda.py --config config.yaml
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, ".")

from src.utils import ensure_dirs, get_logger, load_config, save_fig

logger = get_logger(__name__)


def load_target_series(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["target_series_path"], parse_dates=["timestamp"])
    return df


def square_series(df: pd.DataFrame, square_id: int, freq: str) -> pd.Series:
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet"].sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq(freq)
    return s.interpolate(method="linear").bfill().ffill()


def plot_distribution(totals: pd.Series, plots_dir: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(totals.values, bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of total Internet traffic per square")
    axes[0].set_xlabel("Total internet traffic (observation period)")
    axes[0].set_ylabel("Number of squares")

    axes[1].boxplot(totals.values, vert=True)
    axes[1].set_title("Boxplot of total Internet traffic per square")
    axes[1].set_ylabel("Total internet traffic")
    fig.tight_layout()
    save_fig(fig, f"{plots_dir}/exploratory_analysis/traffic_distribution.png")

    stats = {
        "n_squares": int(totals.shape[0]),
        "mean": float(totals.mean()),
        "median": float(totals.median()),
        "std": float(totals.std()),
        "min": float(totals.min()),
        "max": float(totals.max()),
        "skewness": float(totals.skew()),
        "kurtosis": float(totals.kurtosis()),
        "p90": float(totals.quantile(0.90)),
        "p99": float(totals.quantile(0.99)),
        "share_top_1pct_of_total": float(totals.sort_values(ascending=False).head(int(len(totals) * 0.01)).sum() / totals.sum()),
    }
    return stats


def plot_five_squares(df: pd.DataFrame, target_squares: list, freq: str, plots_dir: str, two_weeks_end: str) -> None:
    fig, axes = plt.subplots(len(target_squares), 1, figsize=(11, 2.3 * len(target_squares)), sharex=True)
    for ax, sq in zip(axes, target_squares):
        s = square_series(df, sq, freq)
        s = s[s.index < two_weeks_end]
        ax.plot(s.index, s.values, linewidth=0.8, color="darkorange")
        ax.set_ylabel(f"Sq {sq}", rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Internet traffic, first two weeks of the observation period", y=1.01)
    fig.tight_layout()
    save_fig(fig, f"{plots_dir}/exploratory_analysis/five_squares_first_two_weeks.png")


def analyze_top_square(df: pd.DataFrame, top_square: int, freq: str, plots_dir: str, tables_dir: str) -> None:
    s = square_series(df, top_square, freq)

    # Analysis 1: ACF / PACF -- reveals temporal dependence structure and periodicity.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(s, lags=288, ax=axes[0])   # 2 days of 10-min lags
    axes[0].set_title(f"ACF - square {top_square}")
    plot_pacf(s, lags=288, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF - square {top_square}")
    fig.tight_layout()
    save_fig(fig, f"{plots_dir}/exploratory_analysis/acf_pacf_top1.png")

    # Analysis 2: STL decomposition (daily period = 144 steps) + ADF stationarity test.
    stl = STL(s, period=144, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(s.index, s.values, linewidth=0.7); axes[0].set_ylabel("Observed")
    axes[1].plot(s.index, stl.trend, linewidth=0.9, color="green"); axes[1].set_ylabel("Trend")
    axes[2].plot(s.index, stl.seasonal, linewidth=0.7, color="purple"); axes[2].set_ylabel("Seasonal (daily)")
    axes[3].plot(s.index, stl.resid, linewidth=0.5, color="gray"); axes[3].set_ylabel("Residual")
    fig.suptitle(f"STL decomposition (period=144, i.e. 1 day) - square {top_square}", y=1.01)
    fig.tight_layout()
    save_fig(fig, f"{plots_dir}/time_series_decomposition/stl_decomposition_top1.png")

    adf_level = adfuller(s.dropna(), autolag="AIC")
    adf_diff = adfuller(s.diff().dropna(), autolag="AIC")
    stationarity_df = pd.DataFrame([
        {"series": "level", "adf_stat": adf_level[0], "p_value": adf_level[1],
         "n_lags_used": adf_level[2], "critical_5pct": adf_level[4]["5%"],
         "stationary_at_5pct": adf_level[1] < 0.05},
        {"series": "first_difference", "adf_stat": adf_diff[0], "p_value": adf_diff[1],
         "n_lags_used": adf_diff[2], "critical_5pct": adf_diff[4]["5%"],
         "stationary_at_5pct": adf_diff[1] < 0.05},
    ])
    stationarity_df.to_csv(f"{tables_dir}/stationarity_tests.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    plots_dir, tables_dir = cfg["paths"]["plots_dir"], cfg["paths"]["tables_dir"]
    ensure_dirs(f"{plots_dir}/exploratory_analysis", f"{plots_dir}/time_series_decomposition", tables_dir)

    totals = pd.read_csv(cfg["data"]["square_totals_path"], index_col="square_id")["internet_total"]
    with open(cfg["data"]["target_squares_meta_path"]) as f:
        meta = yaml.safe_load(f)

    df = load_target_series(cfg)
    freq = cfg["data"]["frequency"]

    logger.info("Plotting distribution of total traffic across %d squares", len(totals))
    dist_stats = plot_distribution(totals, plots_dir)
    pd.DataFrame([dist_stats]).to_csv(f"{tables_dir}/data_summary.csv", index=False)
    logger.info("Distribution stats: %s", dist_stats)

    five_squares = sorted(set(meta["top3_squares"]) | set(meta["fixed_squares"]))
    logger.info("Plotting first-two-weeks series for squares: %s", five_squares)
    two_weeks_end = pd.Timestamp(cfg["data"]["observation_period"]["start"]) + pd.Timedelta(days=14)
    plot_five_squares(df, five_squares, freq, plots_dir, str(two_weeks_end))

    top_square = meta["top3_squares"][0]
    logger.info("Running ACF/PACF + STL decomposition + ADF test on top square %s", top_square)
    analyze_top_square(df, top_square, freq, plots_dir, tables_dir)

    logger.info("EDA complete. Top-3 squares: %s | Fixed squares: %s", meta["top3_squares"], meta["fixed_squares"])


if __name__ == "__main__":
    main()
