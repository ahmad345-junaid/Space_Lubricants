from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


@dataclass
class MultiTargetGaussianProcessRegressor:
    max_points: int = 128
    noise: float = 1.0e-4
    pca_components: int = 8
    target_models: list[GaussianProcessRegressor] = field(default_factory=list)
    pca: PCA | None = None

    def fit(self, embeddings: np.ndarray, targets: np.ndarray) -> "MultiTargetGaussianProcessRegressor":
        subset = embeddings[: self.max_points]
        target_subset = targets[: self.max_points]
        components = min(self.pca_components, subset.shape[0], subset.shape[1])
        self.pca = PCA(n_components=max(1, components))
        reduced = self.pca.fit_transform(subset)
        self.target_models = []
        for index in range(target_subset.shape[1]):
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise)
            model = GaussianProcessRegressor(kernel=kernel, alpha=self.noise, normalize_y=True)
            model.fit(reduced, target_subset[:, index])
            self.target_models.append(model)
        return self

    def predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.pca is None or not self.target_models:
            raise ValueError("Gaussian-process models must be fitted before prediction.")
        reduced = self.pca.transform(embeddings)
        means = []
        stds = []
        for model in self.target_models:
            mean, std = model.predict(reduced, return_std=True)
            means.append(mean)
            stds.append(std)
        return np.stack(means, axis=1), np.stack(stds, axis=1)

