"""
fabric_simulator.py
===================

A deliberately crude procedural fabric renderer, in the same spirit — and with the
same honesty caveat — as ``src/spectral_simulator.py``.

Its job is NOT to look like fabric. Its job is to make this repo's vision branch
**runnable and testable end-to-end before a single real swatch exists**, and to
encode the physical prior the architecture is built around:

    the blend fraction shows up as the *mixing proportion of two micro-texture
    populations*, under nuisances (dye colour, illumination, weave construction,
    sensor noise) that are uncorrelated with the label.

Cotton    -> matte, irregular, high-frequency, broad width distribution.
Polyester -> smoother filaments, near-constant diameter, sparse specular highlights.

Every absolute number below is invented. Do not report a metric measured on this
simulator as a result; use it to prove the pipeline runs and that the model can
recover a label that is genuinely present.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["render_swatch"]


def _band_noise(rng, shape, sigma):
    """Zero-mean band-limited noise, unit variance."""
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / (n.std() + 1e-8)


def render_swatch(
    cotton_area_fraction: float,
    size: int = 512,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render one fabric swatch as float RGB in [0, 1], shape (size, size, 3).

    ``cotton_area_fraction`` is the *visible area* fraction of cotton — the
    quantity an image can actually observe. See ``data.area_to_mass`` for the
    conversion to the mass fraction that a recycler is billed on.
    """
    rng = rng or np.random.default_rng()
    a = float(np.clip(cotton_area_fraction, 0.0, 1.0))

    # --- nuisance 1: weave construction (uncorrelated with blend) --------------
    yarns = rng.integers(18, 42)          # yarns across the frame
    theta = rng.uniform(-0.15, 0.15)      # small on-camera rotation
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size * (2 * np.pi * yarns)
    u = xx * np.cos(theta) - yy * np.sin(theta)
    v = xx * np.sin(theta) + yy * np.cos(theta)
    weave = 0.5 * (np.sin(u) * np.sin(v) + 0.6 * np.cos(2 * u))

    # --- the actual signal: two fibre populations -----------------------------
    # A soft spatial partition assigns each location to a fibre type at the
    # required area proportion; the threshold is set from the field's own
    # quantile so the realised area fraction matches `a` closely.
    field = _band_noise(rng, (size, size), sigma=1.5)
    thresh = np.quantile(field, 1.0 - a)
    cotton_mask = (field > thresh).astype(np.float32)
    cotton_mask = gaussian_filter(cotton_mask, 0.8)  # fibres are not hard-edged

    cotton_tex = 0.85 * _band_noise(rng, (size, size), sigma=0.7)   # rough, matte
    pet_tex = 0.35 * _band_noise(rng, (size, size), sigma=1.9)      # smooth

    # Specular highlights: sparse, small, and their density scales with PET area.
    spec_seed = rng.random((size, size)).astype(np.float32)
    spec = (spec_seed > (1.0 - 0.010 * (1.0 - a))).astype(np.float32)
    spec = gaussian_filter(spec, 0.9) * 6.0

    micro = cotton_mask * cotton_tex + (1.0 - cotton_mask) * (pet_tex + spec)

    # --- nuisance 2: dye colour, uncorrelated with composition ----------------
    dye = rng.uniform(0.25, 0.9, size=3).astype(np.float32)
    base = 0.55 + 0.10 * weave + 0.12 * micro
    rgb = base[..., None] * dye[None, None, :]

    # Specular reflection carries the illuminant, not the dye -> add it white.
    rgb += (0.10 * spec * (1.0 - cotton_mask))[..., None]

    # --- nuisance 3: illumination gradient, gamma, sensor noise ---------------
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    illum = 1.0 + rng.uniform(-0.25, 0.25) * gx + rng.uniform(-0.25, 0.25) * gy
    rgb *= illum[..., None]
    rgb = np.clip(rgb, 0.0, 1.0) ** rng.uniform(0.85, 1.2)
    rgb += rng.standard_normal(rgb.shape).astype(np.float32) * 0.012
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
