from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


@dataclass
class TargetwiseBaselineRegressor:
    name: str
    estimator_factory: Any
    models: dict[str, Any] | None = None

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> "TargetwiseBaselineRegressor":
        self.models = {}
        for column in targets.columns:
            mask = targets[column].notna().to_numpy()
            estimator = self.estimator_factory()
            estimator.fit(features.loc[mask].to_numpy(dtype=float), targets.loc[mask, column].to_numpy(dtype=float))
            self.models[column] = estimator
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.models is None:
            raise ValueError("Baseline model must be fitted before prediction.")
        predictions = {
            column: np.asarray(model.predict(features.to_numpy(dtype=float)), dtype=float).reshape(-1)
            for column, model in self.models.items()
        }
        return pd.DataFrame(predictions, index=features.index)


def _multioutput_factory(base_estimator: Any) -> Any:
    return MultiOutputRegressor(base_estimator)


def baseline_factories(random_state: int) -> dict[str, Any]:
    return {
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=random_state),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=random_state),
        "svr": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "kernel_ridge": lambda: KernelRidge(alpha=1.0, kernel="rbf"),
        "mlp": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
    }


def build_baselines(random_state: int) -> dict[str, TargetwiseBaselineRegressor]:
    return {
        name: TargetwiseBaselineRegressor(name=name, estimator_factory=factory)
        for name, factory in baseline_factories(random_state).items()
    }


def fit_multioutput_reference(
    name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    random_state: int,
) -> Any:
    factories = baseline_factories(random_state)
    if name == "random_forest":
        estimator = factories[name]()
        estimator.fit(features.to_numpy(dtype=float), targets.to_numpy(dtype=float))
        return estimator
    estimator = _multioutput_factory(factories[name]())
    estimator.fit(features.to_numpy(dtype=float), targets.to_numpy(dtype=float))
    return estimator
