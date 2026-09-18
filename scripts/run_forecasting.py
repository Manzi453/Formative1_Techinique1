#!/usr/bin/env python3
"""Stage 4: the core forecasting experiment required by Section 4.

For each of the 3 target squares (top-3 by total internet traffic) and each
of the 3 models (SARIMAX, LSTM, XGBoost), this script:
  1. trains on the history preceding the evaluation week,
  2. produces a true one-step-ahead, walk-forward forecast across the whole
     evaluation week (using tuned hyperparameters from Stage 3 if available),
  3. records MAE / RMSE / MAPE, and training/inference time,
  4. saves a superposed actual-vs-predicted plot.

Produces:
  results/plots/model_predictions/square_<id>_<model>.png   (9 plots)
  results/tables/model_performance_square_<id>.csv           (3 tables)
  results/tables/timing_stats.csv
  experiments/results/model_*_results.pkl
  experiments/experiment_log.csv (one row per model/square run)

Usage:
    python scripts/run_forecasting.py --config config.yaml
"""

import argparse
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.evaluation import compute_metrics
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel, create_sequences
from src.models.xgboost_model import XGBoostModel
from src.preprocessor import add_lag_features, add_rolling_features, add_time_features, fourier_terms, SeriesScaler
from src.training import log_experiment
from src.utils import ensure_dirs, get_logger, load_config, save_fig, set_seed

logger = get_logger(__name__)


def load_square_series(cfg: dict, square_id: int) -> pd.Series:
    df = pd.read_csv(cfg["data"]["target_series_path"], parse_dates=["timestamp"])
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet"].sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq(cfg["data"]["frequency"])
    return s.interpolate(method="linear").bfill().ffill()


def load_best_params(cfg: dict) -> dict:
    path = os.path.join(os.path.dirname(cfg["paths"]["tuning_results"]), "best_params.yaml")
    if os.path.exists(path):
        with open(path) as f:
            best = yaml.safe_load(f)
        logger.info("Using tuned hyperparameters from %s (tuned on square %s)", path, best.get("tuned_on_square"))
        return best
    logger.warning("No tuned hyperparameters found at %s; falling back to config.yaml defaults", path)
    return None


def run_arima(train: pd.Series, eval_series: pd.Series, order, fourier_period: int, n_harm: int) -> tuple:
    t0 = time.perf_counter()
    exog_train = fourier_terms(train.index, fourier_period, n_harm) if n_harm else None
    model = ARIMAModel(tuple(order), (0, 0, 0, 0)).fit(train, exog=exog_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = []
    n_fallback = 0
    last_valid = train.iloc[-1]
    for t in eval_series.index:
        exog_step = fourier_terms(pd.DatetimeIndex([t]), fourier_period, n_harm) if n_harm else None
        pred = model.results.forecast(steps=1, exog=exog_step).iloc[0]
        if not np.isfinite(pred):
            # SARIMAX occasionally fails to converge (see ConvergenceWarning) on some
            # squares/orders, producing a degenerate NaN/inf forecast. Fall back to
            # persistence (last observed true value) for that step rather than
            # letting one bad step crash the whole walk-forward evaluation; the
            # fallback rate is reported and discussed as a robustness finding.
            pred = last_valid
            n_fallback += 1
        preds.append(pred)
        new_obs = pd.Series([eval_series.loc[t]], index=[t], name=train.name)
        model.results = model.results.append(new_obs, exog=exog_step, refit=False)
        last_valid = eval_series.loc[t]
    predict_time = time.perf_counter() - t0
    if n_fallback:
        logger.warning("ARIMA: %d/%d steps used persistence fallback due to non-finite forecasts",
                        n_fallback, len(eval_series))
    return np.array(preds), fit_time, predict_time, n_fallback


def run_lstm(train: pd.Series, eval_series: pd.Series, seq_len, units, lr, seed, batch_size=64, epochs=40) -> tuple:
    set_seed(seed)
    scaler = SeriesScaler("minmax")
    train_scaled = scaler.fit_transform(train).flatten()
    full_eval_input = pd.concat([train.tail(seq_len), eval_series])
    eval_scaled = scaler.transform(full_eval_input).flatten()

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_eval, y_eval = create_sequences(eval_scaled, seq_len)

    t0 = time.perf_counter()
    lstm = LSTMModel(seq_len, units, dropout=0.2, learning_rate=lr)
    lstm.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, early_stopping_patience=5)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_scaled = lstm.predict(X_eval)
    predict_time = time.perf_counter() - t0
    preds = scaler.inverse_transform(preds_scaled)
    return preds, fit_time, predict_time


def build_xgb_features(series: pd.Series, prep_cfg: dict) -> pd.DataFrame:
    df = series.to_frame("internet")
    df = add_time_features(df)
    df = add_lag_features(df, "internet", prep_cfg["lag_features"])
    df = add_rolling_features(df, "internet", prep_cfg["rolling_windows"])
    return df.dropna()


def run_xgboost(full_series: pd.Series, train_end, eval_start, eval_end, max_depth, n_estimators, lr,
                 prep_cfg: dict, seed: int) -> tuple:
    feats = build_xgb_features(full_series, prep_cfg)
    train_df = feats[feats.index < train_end]
    eval_df = feats[(feats.index >= eval_start) & (feats.index <= eval_end)]
    feature_cols = [c for c in feats.columns if c != "internet"]

    t0 = time.perf_counter()
    model = XGBoostModel(n_estimators, max_depth, lr, 0.8, 0.8, seed)
    model.fit(train_df[feature_cols], train_df["internet"])
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(eval_df[feature_cols])
    predict_time = time.perf_counter() - t0
    return preds, fit_time, predict_time, eval_df.index


def plot_square_model(actual: pd.Series, preds: np.ndarray, index: pd.DatetimeIndex,
                       square_id: int, model_name: str, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=1.2)
    ax.plot(index, preds, label=f"Predicted ({model_name})", color="crimson", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Square {square_id} - {model_name}: actual vs. one-step-ahead predicted internet traffic")
    ax.set_xlabel("Time"); ax.set_ylabel("Internet traffic")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"{plots_dir}/model_predictions/square_{square_id}_{model_name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    plots_dir, tables_dir = cfg["paths"]["plots_dir"], cfg["paths"]["tables_dir"]
    ensure_dirs(f"{plots_dir}/model_predictions", tables_dir, cfg["paths"]["results_dir"])

    with open(cfg["data"]["target_squares_meta_path"]) as f:
        meta = yaml.safe_load(f)
    squares = meta["top3_squares"]

    best = load_best_params(cfg)
    arima_order = best["arima"]["order"] if best else cfg["models"]["arima"]["order"]
    arima_harmonics = best["arima"]["fourier_harmonics"] if best else cfg["models"]["arima"]["fourier_harmonics"]
    fourier_period = cfg["models"]["arima"]["fourier_period"]
    lstm_seq = best["lstm"]["sequence_length"] if best else cfg["models"]["lstm"]["sequence_length"]
    lstm_units = best["lstm"]["units"] if best else cfg["models"]["lstm"]["units"]
    lstm_lr = best["lstm"]["learning_rate"] if best else cfg["models"]["lstm"]["learning_rate"]
    xgb_depth = best["xgboost"]["max_depth"] if best else cfg["models"]["xgboost"]["max_depth"]
    xgb_n_est = best["xgboost"]["n_estimators"] if best else cfg["models"]["xgboost"]["n_estimators"]
    xgb_lr = best["xgboost"]["learning_rate"] if best else cfg["models"]["xgboost"]["learning_rate"]

    eval_start = pd.Timestamp(cfg["data"]["eval_week"]["start"])
    eval_end = pd.Timestamp(cfg["data"]["eval_week"]["end"]) + pd.Timedelta(hours=23, minutes=50)
    train_start = pd.Timestamp(cfg["data"]["train_start"])

    timing_rows = []

    for square_id in squares:
        logger.info("=== Square %d ===", square_id)
        series = load_square_series(cfg, square_id)
        train = series[(series.index >= train_start) & (series.index < eval_start)]
        eval_series = series[(series.index >= eval_start) & (series.index <= eval_end)]
        logger.info("train=%d pts, eval=%d pts", len(train), len(eval_series))

        metrics_table = {}

        # ---- ARIMA ----
        preds, fit_t, pred_t, n_fallback = run_arima(train, eval_series, arima_order, fourier_period, arima_harmonics)
        metrics = compute_metrics(eval_series.values, preds)
        metrics["fallback_steps"] = n_fallback
        metrics_table["ARIMA"] = metrics
        plot_square_model(eval_series, preds, eval_series.index, square_id, "ARIMA", plots_dir)
        timing_rows.append({"square_id": square_id, "model": "ARIMA", "fit_time_s": fit_t,
                             "predict_time_total_s": pred_t, "predict_time_per_step_ms": pred_t / len(eval_series) * 1000})
        with open(f"{cfg['paths']['results_dir']}/model_1_results_sq{square_id}.pkl", "wb") as f:
            pickle.dump({"predictions": preds, "metrics": metrics}, f)
        log_experiment(cfg["paths"]["experiment_log"], f"exp_arima_sq{square_id}", "ARIMA",
                        {"order": arima_order, "fourier_harmonics": arima_harmonics, "square_id": square_id,
                         "fallback_steps": n_fallback},
                        metrics)
        logger.info("ARIMA sq=%d metrics=%s fit=%.1fs predict=%.1fs fallback_steps=%d",
                    square_id, metrics, fit_t, pred_t, n_fallback)

        # ---- LSTM ----
        preds, fit_t, pred_t = run_lstm(train, eval_series, lstm_seq, lstm_units, lstm_lr, cfg["seed"],
                                         batch_size=cfg["models"]["lstm"]["batch_size"],
                                         epochs=cfg["models"]["lstm"]["epochs"])
        metrics = compute_metrics(eval_series.values, preds)
        metrics_table["LSTM"] = metrics
        plot_square_model(eval_series, preds, eval_series.index, square_id, "LSTM", plots_dir)
        timing_rows.append({"square_id": square_id, "model": "LSTM", "fit_time_s": fit_t,
                             "predict_time_total_s": pred_t, "predict_time_per_step_ms": pred_t / len(eval_series) * 1000})
        with open(f"{cfg['paths']['results_dir']}/model_2_results_sq{square_id}.pkl", "wb") as f:
            pickle.dump({"predictions": preds, "metrics": metrics}, f)
        log_experiment(cfg["paths"]["experiment_log"], f"exp_lstm_sq{square_id}", "LSTM",
                        {"sequence_length": lstm_seq, "units": lstm_units, "learning_rate": lstm_lr, "square_id": square_id},
                        metrics)
        logger.info("LSTM sq=%d metrics=%s fit=%.1fs predict=%.1fs", square_id, metrics, fit_t, pred_t)

        # ---- XGBoost ----
        preds, fit_t, pred_t, xgb_index = run_xgboost(series, eval_start, eval_start, eval_end, xgb_depth,
                                                        xgb_n_est, xgb_lr, cfg["preprocessing"], cfg["seed"])
        actual_aligned = series.loc[xgb_index]
        metrics = compute_metrics(actual_aligned.values, preds)
        metrics_table["XGBoost"] = metrics
        plot_square_model(eval_series, preds, xgb_index, square_id, "XGBoost", plots_dir)
        timing_rows.append({"square_id": square_id, "model": "XGBoost", "fit_time_s": fit_t,
                             "predict_time_total_s": pred_t, "predict_time_per_step_ms": pred_t / len(xgb_index) * 1000})
        with open(f"{cfg['paths']['results_dir']}/model_3_results_sq{square_id}.pkl", "wb") as f:
            pickle.dump({"predictions": preds, "metrics": metrics}, f)
        log_experiment(cfg["paths"]["experiment_log"], f"exp_xgboost_sq{square_id}", "XGBoost",
                        {"max_depth": xgb_depth, "n_estimators": xgb_n_est, "learning_rate": xgb_lr, "square_id": square_id},
                        metrics)
        logger.info("XGBoost sq=%d metrics=%s fit=%.1fs predict=%.1fs", square_id, metrics, fit_t, pred_t)

        table = pd.DataFrame(metrics_table).T[["MAE", "MAPE", "RMSE"]]
        table.to_csv(f"{tables_dir}/model_performance_square_{square_id}.csv")
        logger.info("Square %d comparison:\n%s", square_id, table.to_string())

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(f"{tables_dir}/timing_stats.csv", index=False)
    logger.info("Saved timing stats to %s/timing_stats.csv", tables_dir)
    logger.info("Forecasting experiments complete for squares: %s", squares)


if __name__ == "__main__":
    main()
