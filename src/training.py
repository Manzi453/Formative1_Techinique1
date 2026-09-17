"""Training loops, hyperparameter tuning, and experiment logging."""

import csv
import os
from datetime import datetime
from typing import Callable, Dict, Iterable

from sklearn.model_selection import ParameterGrid


def log_experiment(
    log_path: str,
    experiment_id: str,
    model_name: str,
    parameters: dict,
    metrics: dict,
) -> None:
    """Append one row to the experiment log CSV, creating it with a header if needed."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)

    row = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "parameters": str(parameters),
        "metrics": str(metrics),
        "date": datetime.now().isoformat(timespec="seconds"),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def grid_search(
    param_grid: Dict[str, Iterable],
    train_and_eval_fn: Callable[[dict], float],
    tuning_log_path: str = None,
) -> dict:
    """Run a simple grid search over `param_grid`, scoring with `train_and_eval_fn`.

    `train_and_eval_fn` receives a single parameter combination and must
    return a scalar validation error (lower is better). Every combination
    tried is optionally appended to `tuning_log_path`.
    """
    best_params, best_score = None, float("inf")
    results = []

    for params in ParameterGrid(param_grid):
        score = train_and_eval_fn(params)
        results.append({**params, "score": score})

        if score < best_score:
            best_score, best_params = score, params

    if tuning_log_path:
        os.makedirs(os.path.dirname(tuning_log_path), exist_ok=True)
        file_exists = os.path.exists(tuning_log_path)
        with open(tuning_log_path, "a", newline="") as f:
            fieldnames = list(results[0].keys()) if results else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    return {"best_params": best_params, "best_score": best_score, "all_results": results}
