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

    def fit(self, train_series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "ARIMAModel":
        # enforce_stationarity/invertibility=True constrains the optimizer to
        # AR/MA roots outside the unit circle. Leaving both False (statsmodels'
        # example default) let the optimizer converge to explosive AR
        # coefficients on some squares, producing forecasts that diverge to
        # +-inf within a few walk-forward steps; enforcing both keeps the
        # fitted model's recursive dynamics bounded.
        self.model = SARIMAX(
            train_series,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        self.results = self.model.fit(disp=False)
        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self.results.forecast(steps=steps, exog=exog)
        return forecast.values

    def predict_in_sample(self, start: int, end: int) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Call fit() before predict_in_sample().")
        return self.results.predict(start=start, end=end).values

    def summary(self) -> Optional[str]:
        return None if self.results is None else str(self.results.summary())
