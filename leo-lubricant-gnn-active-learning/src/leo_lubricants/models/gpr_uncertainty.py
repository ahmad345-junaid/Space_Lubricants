from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


@dataclass(frozen=True)
class GPRConfig:
    noise_level: float = 1.0e-4
    normalize_y: bool = True
    max_fit_points: int | None = None
    random_state: int = 7


@dataclass
class MultiOutputGPRUncertainty:
    config: GPRConfig = field(default_factory=GPRConfig)
    target_names: list[str] = field(default_factory=list)
    models: dict[str, GaussianProcessRegressor] = field(default_factory=dict)

    def fit(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        target_names: list[str],
    ) -> "MultiOutputGPRUncertainty":
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if targets.ndim != 2:
            raise ValueError("Targets must be a 2D array.")
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("Embeddings and targets must have the same number of rows.")
        self.target_names = list(target_names)
        fit_embeddings = embeddings
        fit_targets = targets
        if self.config.max_fit_points is not None and embeddings.shape[0] > self.config.max_fit_points:
            fit_embeddings = embeddings[: self.config.max_fit_points]
            fit_targets = targets[: self.config.max_fit_points]
        self.models = {}
        for target_index, target_name in enumerate(self.target_names):
            length_scale = np.ones(fit_embeddings.shape[1], dtype=float)
            kernel = RBF(length_scale=length_scale, length_scale_bounds=(1.0e-3, 1.0e3)) + WhiteKernel(
                noise_level=self.config.noise_level,
                noise_level_bounds=(1.0e-8, 1.0e1),
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.config.noise_level,
                normalize_y=self.config.normalize_y,
                random_state=self.config.random_state,
            )
            model.fit(fit_embeddings, fit_targets[:, target_index])
            self.models[target_name] = model
        return self

    def predict(self, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.models:
            raise ValueError("The GPR model must be fitted before prediction.")
        mean_columns: dict[str, np.ndarray] = {}
        std_columns: dict[str, np.ndarray] = {}
        for target_name in self.target_names:
            mean_values, std_values = self.models[target_name].predict(embeddings, return_std=True)
            mean_columns[target_name] = mean_values
            std_columns[target_name] = std_values
        return pd.DataFrame(mean_columns), pd.DataFrame(std_columns)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "MultiOutputGPRUncertainty":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__} artifact at {path}.")
        return loaded

