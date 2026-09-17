"""LSTM deep learning forecasting model."""

from typing import List, Tuple

import numpy as np


def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1D scaled series into overlapping (X, y) supervised sequences."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


class LSTMModel:
    """Wrapper around a Keras Sequential LSTM for a consistent fit/predict API."""

    def __init__(
        self,
        sequence_length: int = 24,
        units: List[int] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.sequence_length = sequence_length
        self.units = list(units)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None

    def _build(self) -> None:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        for i, n_units in enumerate(self.units):
            return_sequences = i < len(self.units) - 1
            if i == 0:
                model.add(
                    LSTM(
                        n_units,
                        return_sequences=return_sequences,
                        input_shape=(self.sequence_length, 1),
                    )
                )
            else:
                model.add(LSTM(n_units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        self.model = model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 5,
    ) -> "LSTMModel":
        from tensorflow.keras.callbacks import EarlyStopping

        if self.model is None:
            self._build()

        X_train = X_train.reshape((*X_train.shape, 1))
        validation_data = None
        callbacks = []

        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val = X_val.reshape((*X_val.shape, 1))
            validation_data = (X_val, y_val)
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X = X.reshape((*X.shape, 1))
        return self.model.predict(X, verbose=0).flatten()
