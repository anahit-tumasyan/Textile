"""
preprocessing.py
================

Standard chemometric preprocessing for NIR spectra. These transforms remove the
physical nuisances the simulator injects (and that real sensors produce), leaving
the chemical information the regressor needs.

- SNV (Standard Normal Variate): per-spectrum centre+scale — removes multiplicative
  scatter and additive offset.
- Savitzky-Golay derivative: smooths, then differentiates — removes sloping
  baselines and sharpens overlapping bands (key for blend separation).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate, applied row-wise (per spectrum)."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mean) / sd


def savgol_derivative(X: np.ndarray, window: int = 15, poly: int = 2,
                      deriv: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing derivative, row-wise."""
    if window % 2 == 0:
        window += 1
    return savgol_filter(X, window_length=window, polyorder=poly,
                         deriv=deriv, axis=1)


def preprocess(X: np.ndarray, method: str = "snv+sg1") -> np.ndarray:
    """
    Apply a named preprocessing pipeline.

    method options:
        "none"    : raw absorbance
        "snv"     : SNV only
        "sg1"     : 1st-derivative only
        "snv+sg1" : SNV then 1st derivative (recommended default)
    """
    if method == "none":
        return np.asarray(X, dtype=float)
    if method == "snv":
        return snv(X)
    if method == "sg1":
        return savgol_derivative(X, deriv=1)
    if method == "snv+sg1":
        return savgol_derivative(snv(X), deriv=1)
    raise ValueError(f"unknown preprocessing method: {method}")
