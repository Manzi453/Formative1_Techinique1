"""ARIMA / SARIMAX baseline forecasting model."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """Wrapper around statsmodels SARIMAX for a consistent fit/predict API."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.model = None
        self.results = None

    def fit(self, train_series: pd.Series) -> "ARIMAModel":
        self.model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = self.model.fit(disp=False)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self.results.forecast(steps=steps)
        return forecast.values

    def predict_in_sample(self, start: int, end: int) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Call fit() before predict_in_sample().")
        return self.results.predict(start=start, end=end).values

    def summary(self) -> Optional[str]:
        return None if self.results is None else str(self.results.summary())
