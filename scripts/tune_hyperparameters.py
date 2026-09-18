#!/usr/bin/env python3
"""Stage 3: hyperparameter tuning for the three forecasting models.

Tuning is run on the single highest-traffic square (most representative,
most data) using a validation slice carved out of the training window --
strictly *before* the eval week, so the final Section-4 evaluation week is
never touched during tuning. Every combination tried is logged to
experiments/hyperparameter_tuning/tuning_results.csv; the best combination
per model is written to experiments/hyperparameter_tuning/best_params.yaml.

Usage:
    python scripts/tune_hyperparameters.py --config config.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.evaluation import compute_metrics
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel, create_sequences
from src.models.xgboost_model import XGBoostModel
from src.preprocessor import add_lag_features, add_rolling_features, add_time_features, fourier_terms, SeriesScaler
from src.utils import ensure_dirs, get_logger, load_config, set_seed

logger = get_logger(__name__)


def load_square_series(cfg: dict, square_id: int) -> pd.Series:
    df = pd.read_csv(cfg["data"]["target_series_path"], parse_dates=["timestamp"])
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet"].sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq(cfg["data"]["frequency"])
    return s.interpolate(method="linear").bfill().ffill()


def write_rows(path: str, rows: list) -> None:
    """Write all logged rows in one shot, so the CSV has a single, consistent
    header (the union of every model's parameter columns) rather than one
    call per model appending mismatched column sets."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def tune_arima(train: pd.Series, val: pd.Series, grid_cfg: dict, fourier_period: int) -> dict:
    rows, best = [], {"score": float("inf")}
    for order in grid_cfg["order"]:
        for n_harm in grid_cfg["fourier_harmonics"]:
            t0 = time.perf_counter()
            try:
                exog_train = fourier_terms(train.index, fourier_period, n_harm) if n_harm else None
                model = ARIMAModel(tuple(order), (0, 0, 0, 0)).fit(train, exog=exog_train)
                preds = []
                for t in val.index:
                    exog_step = fourier_terms(pd.DatetimeIndex([t]), fourier_period, n_harm) if n_harm else None
                    preds.append(model.results.forecast(steps=1, exog=exog_step).iloc[0])
                    new_obs = pd.Series([val.loc[t]], index=[t], name=train.name)
                    model.results = model.results.append(new_obs, exog=exog_step, refit=False)
                metrics = compute_metrics(val.values, np.array(preds))
                fit_time = time.perf_counter() - t0
                row = {"model": "ARIMA", "order": str(order), "fourier_harmonics": n_harm,
                       "rmse": metrics["RMSE"], "mae": metrics["MAE"], "mape": metrics["MAPE"],
                       "elapsed_s": fit_time,
                       "notes": "walk-forward one-step over validation slice, params fixed after fit; "
                                "daily seasonality via Fourier regressors, not seasonal-ARIMA (s=144 impractical)"}
            except Exception as exc:  # noqa: BLE001 - log and continue the grid
                row = {"model": "ARIMA", "order": str(order), "fourier_harmonics": n_harm,
                       "rmse": np.nan, "mae": np.nan, "mape": np.nan, "elapsed_s": np.nan,
                       "notes": f"failed: {exc}"}
            rows.append(row)
            logger.info("ARIMA order=%s fourier_harmonics=%s -> %s", order, n_harm, row)
            if not np.isnan(row["rmse"]) and row["rmse"] < best["score"]:
                best = {"order": order, "fourier_harmonics": n_harm, "score": row["rmse"]}
    return rows, best


def tune_lstm(train: pd.Series, val: pd.Series, grid_cfg: dict, seed: int) -> dict:
    rows, best = [], {"score": float("inf")}
    for seq_len in grid_cfg["sequence_length"]:
        for units in grid_cfg["units"]:
            for lr in grid_cfg["learning_rate"]:
                t0 = time.perf_counter()
                set_seed(seed)
                scaler = SeriesScaler("minmax")
                train_scaled = scaler.fit_transform(train).flatten()
                full_val_input = pd.concat([train.tail(seq_len), val])
                val_scaled = scaler.transform(full_val_input).flatten()

                X_train, y_train = create_sequences(train_scaled, seq_len)
                X_val, y_val = create_sequences(val_scaled, seq_len)

                lstm = LSTMModel(seq_len, units, dropout=0.2, learning_rate=lr)
                lstm.fit(X_train, y_train, batch_size=64, epochs=15, early_stopping_patience=3)
                preds_scaled = lstm.predict(X_val)
                preds = scaler.inverse_transform(preds_scaled)
                metrics = compute_metrics(val.values, preds)
                elapsed = time.perf_counter() - t0
                row = {"model": "LSTM", "sequence_length": seq_len, "units": str(units), "learning_rate": lr,
                       "rmse": metrics["RMSE"], "mae": metrics["MAE"], "mape": metrics["MAPE"],
                       "elapsed_s": elapsed, "notes": "15 tuning epochs w/ early stopping (patience=3)"}
                rows.append(row)
                logger.info("LSTM seq=%d units=%s lr=%s -> %s", seq_len, units, lr, row)
                if row["rmse"] < best["score"]:
                    best = {"sequence_length": seq_len, "units": units, "learning_rate": lr, "score": row["rmse"]}
    return rows, best


def build_xgb_features(series: pd.Series, prep_cfg: dict) -> pd.DataFrame:
    df = series.to_frame("internet")
    df = add_time_features(df)
    df = add_lag_features(df, "internet", prep_cfg["lag_features"])
    df = add_rolling_features(df, "internet", prep_cfg["rolling_windows"])
    return df.dropna()


def tune_xgboost(full_series: pd.Series, train_end, val_end, grid_cfg: dict, prep_cfg: dict, seed: int) -> dict:
    feats = build_xgb_features(full_series, prep_cfg)
    train_df = feats[feats.index < train_end]
    val_df = feats[(feats.index >= train_end) & (feats.index < val_end)]
    feature_cols = [c for c in feats.columns if c != "internet"]

    rows, best = [], {"score": float("inf")}
    for max_depth in grid_cfg["max_depth"]:
        for n_estimators in grid_cfg["n_estimators"]:
            for lr in grid_cfg["learning_rate"]:
                t0 = time.perf_counter()
                model = XGBoostModel(n_estimators, max_depth, lr, 0.8, 0.8, seed)
                model.fit(train_df[feature_cols], train_df["internet"], val_df[feature_cols], val_df["internet"], 20)
                preds = model.predict(val_df[feature_cols])
                metrics = compute_metrics(val_df["internet"].values, preds)
                elapsed = time.perf_counter() - t0
                row = {"model": "XGBoost", "max_depth": max_depth, "n_estimators": n_estimators,
                       "learning_rate": lr, "rmse": metrics["RMSE"], "mae": metrics["MAE"],
                       "mape": metrics["MAPE"], "elapsed_s": elapsed,
                       "notes": "early_stopping_rounds=20 on validation slice"}
                rows.append(row)
                logger.info("XGB depth=%d n_est=%d lr=%s -> %s", max_depth, n_estimators, lr, row)
                if row["rmse"] < best["score"]:
                    best = {"max_depth": max_depth, "n_estimators": n_estimators, "learning_rate": lr,
                            "score": row["rmse"]}
    return rows, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--val-days", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(os.path.dirname(cfg["paths"]["tuning_results"]))

    with open(cfg["data"]["target_squares_meta_path"]) as f:
        meta = yaml.safe_load(f)
    tune_square = meta["top3_squares"][0]

    series = load_square_series(cfg, tune_square)
    eval_start = pd.Timestamp(cfg["data"]["eval_week"]["start"])
    train_end = eval_start - pd.Timedelta(days=args.val_days)

    train = series[series.index < train_end]
    val = series[(series.index >= train_end) & (series.index < eval_start)]
    logger.info("Tuning on square %d: train=%d pts, val=%d pts (%d days)",
                tune_square, len(train), len(val), args.val_days)

    log_path = cfg["paths"]["tuning_results"]
    tuning_cfg = cfg["tuning"]

    fourier_period = cfg["models"]["arima"]["fourier_period"]
    arima_rows, best_arima = tune_arima(train, val, tuning_cfg["arima"], fourier_period)
    lstm_rows, best_lstm = tune_lstm(train, val, tuning_cfg["lstm"], cfg["seed"])
    xgb_rows, best_xgb = tune_xgboost(series, train_end, eval_start, tuning_cfg["xgboost"], cfg["preprocessing"],
                                       cfg["seed"])
    write_rows(log_path, arima_rows + lstm_rows + xgb_rows)

    best_params = {
        "tuned_on_square": tune_square,
        "validation_window_days": args.val_days,
        "arima": {"order": list(best_arima["order"]), "fourier_harmonics": best_arima["fourier_harmonics"],
                  "fourier_period": fourier_period},
        "lstm": {"sequence_length": best_lstm["sequence_length"], "units": list(best_lstm["units"]),
                 "learning_rate": best_lstm["learning_rate"]},
        "xgboost": {"max_depth": best_xgb["max_depth"], "n_estimators": best_xgb["n_estimators"],
                    "learning_rate": best_xgb["learning_rate"]},
    }
    best_path = os.path.join(os.path.dirname(log_path), "best_params.yaml")
    with open(best_path, "w") as f:
        yaml.safe_dump(best_params, f)
    logger.info("Best params written to %s: %s", best_path, best_params)


if __name__ == "__main__":
    main()
