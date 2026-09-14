"""
model.py
========

Quantitative blend-composition regressors. The task: predict cotton mass
fraction (polyester = 1 - cotton) from a preprocessed NIR spectrum.

Two models, deliberately:
  * PLSRegression  -> the chemometrics standard; this is the SOTA-style BASELINE
                      the brief's Phase 0 requires us to reproduce and beat.
  * RandomForest   -> a non-linear ML comparator.

Predictions are clipped to [0, 1] so they are physically valid fractions.
"""

from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor


class BlendRegressor:
    """Thin wrapper that clips predictions to a valid fraction range."""

    def __init__(self, kind: str = "pls", **kwargs):
        self.kind = kind
        if kind == "pls":
            self.model = PLSRegression(n_components=kwargs.get("n_components", 12))
        elif kind == "rf":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                random_state=kwargs.get("random_state", 0),
                n_jobs=-1,
            )
        else:
            raise ValueError(kind)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        pred = np.asarray(self.model.predict(X)).ravel()
        return np.clip(pred, 0.0, 1.0)
