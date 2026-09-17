"""Helper utilities: reproducibility, config loading, logging, plotting."""

import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across libraries for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create a console (and optionally file) logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ensure_dirs(*paths: str) -> None:
    """Create one or more directories if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
