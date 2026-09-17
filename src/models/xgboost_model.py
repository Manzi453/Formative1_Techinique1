"""XGBoost forecasting model on engineered lag/rolling features."""

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostModel:
    """Wrapper around XGBRegressor for a consistent fit/predict API."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 20,
    ) -> "XGBoostModel":
        eval_set = None
        fit_kwargs = {}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            fit_kwargs["eval_set"] = eval_set
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        self.model.fit(X_train, y_train, verbose=False, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
