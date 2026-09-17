#!/usr/bin/env python3
"""Entry point: run the full pipeline — load -> preprocess -> train -> evaluate.

Usage:
    python main.py --config config.yaml
"""

import argparse
import os
import pickle

import pandas as pd

from src.data_loader import load_raw_data, save_processed_data
from src.evaluation import compare_models, compute_metrics, plot_predictions
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel, create_sequences
from src.models.xgboost_model import XGBoostModel
from src.preprocessor import build_feature_pipeline, chronological_split, SeriesScaler
from src.training import log_experiment
from src.utils import ensure_dirs, get_logger, load_config, save_fig, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile Network Traffic Prediction pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]
    model_cfg = cfg["models"]
    paths_cfg = cfg["paths"]
    target_col = data_cfg["target_column"]

    ensure_dirs(
        paths_cfg["results_dir"],
        paths_cfg["plots_dir"],
        paths_cfg["tables_dir"],
        os.path.dirname(data_cfg["processed_path"]),
    )

    # ---- 1. Load ----
    logger.info("Loading raw data from %s", data_cfg["raw_path"])
    raw_df = load_raw_data(
        data_cfg["raw_path"], data_cfg["datetime_column"], data_cfg["frequency"]
    )

    # ---- 2. Preprocess ----
    logger.info("Running preprocessing / feature engineering pipeline")
    features_df = build_feature_pipeline(
        raw_df,
        target_col,
        prep_cfg["fill_method"],
        prep_cfg["lag_features"],
        prep_cfg["rolling_windows"],
        prep_cfg["add_time_features"],
    )
    save_processed_data(features_df, data_cfg["processed_path"])

    train_df, val_df, test_df = chronological_split(
        features_df, data_cfg["test_size"], data_cfg["validation_size"]
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )

    feature_cols = [c for c in features_df.columns if c != target_col]
    predictions = {}
    all_metrics = {}

    # ---- 3a. ARIMA ----
    logger.info("Training ARIMA")
    arima_cfg = model_cfg["arima"]
    arima = ARIMAModel(tuple(arima_cfg["order"]), tuple(arima_cfg["seasonal_order"]))
    arima.fit(train_df[target_col])
    arima_preds = arima.predict(steps=len(test_df))
    predictions["ARIMA"] = arima_preds
    all_metrics["ARIMA"] = compute_metrics(test_df[target_col].values, arima_preds)
    with open(os.path.join(paths_cfg["results_dir"], "model_1_results.pkl"), "wb") as f:
        pickle.dump({"predictions": arima_preds, "metrics": all_metrics["ARIMA"]}, f)
    log_experiment(paths_cfg["experiment_log"], "exp_arima_001", "ARIMA", arima_cfg, all_metrics["ARIMA"])

    # ---- 3b. LSTM ----
    logger.info("Training LSTM")
    lstm_cfg = model_cfg["lstm"]
    scaler = SeriesScaler(prep_cfg["scaling_method"])
    train_scaled = scaler.fit_transform(train_df[target_col]).flatten()
    test_scaled = scaler.transform(test_df[target_col]).flatten()

    X_train, y_train = create_sequences(train_scaled, lstm_cfg["sequence_length"])
    full_test_input = pd.concat([train_df[target_col].tail(lstm_cfg["sequence_length"]), test_df[target_col]])
    test_scaled_full = scaler.transform(full_test_input).flatten()
    X_test, y_test = create_sequences(test_scaled_full, lstm_cfg["sequence_length"])

    lstm = LSTMModel(
        lstm_cfg["sequence_length"], lstm_cfg["units"], lstm_cfg["dropout"], lstm_cfg["learning_rate"]
    )
    lstm.fit(
        X_train,
        y_train,
        batch_size=lstm_cfg["batch_size"],
        epochs=lstm_cfg["epochs"],
        early_stopping_patience=lstm_cfg["early_stopping_patience"],
    )
    lstm_preds_scaled = lstm.predict(X_test)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
    predictions["LSTM"] = lstm_preds
    all_metrics["LSTM"] = compute_metrics(test_df[target_col].values, lstm_preds)
    with open(os.path.join(paths_cfg["results_dir"], "model_2_results.pkl"), "wb") as f:
        pickle.dump({"predictions": lstm_preds, "metrics": all_metrics["LSTM"]}, f)
    log_experiment(paths_cfg["experiment_log"], "exp_lstm_001", "LSTM", lstm_cfg, all_metrics["LSTM"])

    # ---- 3c. XGBoost ----
    logger.info("Training XGBoost")
    xgb_cfg = model_cfg["xgboost"]
    xgb = XGBoostModel(
        xgb_cfg["n_estimators"],
        xgb_cfg["max_depth"],
        xgb_cfg["learning_rate"],
        xgb_cfg["subsample"],
        xgb_cfg["colsample_bytree"],
        cfg["seed"],
    )
    xgb.fit(
        train_df[feature_cols],
        train_df[target_col],
        val_df[feature_cols] if len(val_df) else None,
        val_df[target_col] if len(val_df) else None,
        xgb_cfg["early_stopping_rounds"],
    )
    xgb_preds = xgb.predict(test_df[feature_cols])
    predictions["XGBoost"] = xgb_preds
    all_metrics["XGBoost"] = compute_metrics(test_df[target_col].values, xgb_preds)
    with open(os.path.join(paths_cfg["results_dir"], "model_3_results.pkl"), "wb") as f:
        pickle.dump({"predictions": xgb_preds, "metrics": all_metrics["XGBoost"]}, f)
    log_experiment(paths_cfg["experiment_log"], "exp_xgboost_001", "XGBoost", xgb_cfg, all_metrics["XGBoost"])

    # ---- 4. Compare & save ----
    comparison_df = compare_models(all_metrics)
    comparison_df.to_csv(os.path.join(paths_cfg["tables_dir"], "model_performance.csv"))
    logger.info("\n%s", comparison_df.to_string())

    fig = plot_predictions(test_df[target_col].values, predictions)
    save_fig(fig, os.path.join(paths_cfg["plots_dir"], "model_predictions", "all_models_comparison.png"))

    logger.info("Pipeline complete. Results saved under %s and %s", paths_cfg["results_dir"], paths_cfg["tables_dir"])


if __name__ == "__main__":
    main()
