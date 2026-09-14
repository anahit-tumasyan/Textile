"""
spectral_simulator.py
=====================

Physically-grounded near-infrared (NIR) spectral simulator for cotton/polyester
textile blends.

WHY THIS EXISTS
---------------
Labelled NIR datasets of *quantitative* textile blends are scarce and fragmented
(the core Phase-0 bottleneck named in the research brief). To de-risk the method
before real samples are collected, we simulate reflectance spectra from the
*published* characteristic absorption bands of cellulose (cotton) and PET
(polyester), then mix them by mass fraction using a Beer-Lambert-style additive
model in the absorbance domain, and finally corrupt them with the artifacts a
real handheld/line NIR sensor introduces (multiplicative light scatter, sloping
baselines, detector noise, small wavelength drift).

This is a *stand-in* for real spectra, not a replacement. Its job is to prove the
regression method works and to quantify how accuracy degrades with noise / band
overlap, so the SOTA bar (Phase 0 of the brief) is established before investing
in sample collection (Phase 1).

BAND POSITIONS (from the NIR-spectroscopy literature)
-----------------------------------------------------
Cotton / cellulose:  ~1130-1200 nm (2nd overtone C-H), ~1480 nm (1st overtone
    O-H + C-O comb.), ~1660 nm (C-H comb.), ~1940 nm (O-H comb.), ~2100 nm
    (O-H + C-O comb.).
Polyester / PET:     ~1120 nm (2nd overtone C-H), ~1410 nm, ~1660 nm (aromatic
    C=C / C=O of terephthalate — the most distinctive PET band), ~1900 nm,
    ~2140 nm (aromatic combination bands).

The two fibres share overlapping regions around 1120 and 1660 nm, which is
exactly the spectral-overlap difficulty that makes *quantitative* blend analysis
hard — the simulator reproduces this on purpose.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Wavelength grid: a typical NIR range for textile sorting (1000-2200 nm).
# ----------------------------------------------------------------------------
WL_MIN, WL_MAX, WL_STEP = 1000.0, 2200.0, 2.0
WAVELENGTHS = np.arange(WL_MIN, WL_MAX + WL_STEP, WL_STEP)


@dataclass
class Band:
    """A single Gaussian absorption band: centre (nm), width (nm), height (a.u.)."""
    centre: float
    width: float
    height: float


# Published characteristic bands, expressed as Gaussian absorbance peaks.
COTTON_BANDS = [
    Band(1160, 28, 0.25),   # 2nd overtone C-H
    Band(1480, 45, 0.90),   # 1st overtone O-H + C-O combination (strong for cellulose)
    Band(1660, 32, 0.30),   # C-H combination
    Band(1940, 42, 1.00),   # O-H combination (very strong — water/hydroxyl rich)
    Band(2100, 40, 0.70),   # O-H + C-O combination
]

POLYESTER_BANDS = [
    Band(1120, 26, 0.35),   # 2nd overtone C-H
    Band(1410, 40, 0.55),   # overtone/combination C-H
    Band(1660, 30, 0.85),   # aromatic C=C / C=O of terephthalate (distinctive PET)
    Band(1900, 38, 0.45),   # combination
    Band(2140, 44, 0.80),   # aromatic combination bands
]


def _absorbance_from_bands(bands: list[Band], wl: np.ndarray) -> np.ndarray:
    """Sum of Gaussian absorption peaks -> pure-fibre absorbance spectrum."""
    a = np.zeros_like(wl)
    for b in bands:
        a += b.height * np.exp(-0.5 * ((wl - b.centre) / b.width) ** 2)
    return a


def pure_endmembers(wl: np.ndarray = WAVELENGTHS):
    """Return (cotton_absorbance, polyester_absorbance) on the wavelength grid."""
    cotton = _absorbance_from_bands(COTTON_BANDS, wl)
    poly = _absorbance_from_bands(POLYESTER_BANDS, wl)
    return cotton, poly


@dataclass
class SimConfig:
    """Controls how much real-world messiness to inject."""
    noise_sd: float = 0.020          # additive detector noise (absorbance units)
    scatter_sd: float = 0.14         # multiplicative scatter (fraction)
    baseline_slope_sd: float = 0.10  # random sloping baseline magnitude
    baseline_offset_sd: float = 0.06 # random constant offset
    wl_shift_sd: float = 2.0         # wavelength drift (nm)
    nonlinear: float = 0.30          # non-linear mixing term (0 = perfectly linear)
    band_jitter_sd: float = 3.0      # per-sample band-centre jitter (nm), fabric variability
    dye_tilt_sd: float = 0.12        # broadband slope from colour/dye
    # --- dark / carbon-black fabric -------------------------------------------
    # `darkness` is 0 for a light fabric and 1 for a deeply carbon-black one.
    # A float applies to every sample; a (lo, hi) tuple is sampled per sample.
    #
    # WHY THIS IS MODELLED AS ATTENUATION, NOT AN OFFSET: carbon black is a
    # broadband NIR absorber, so far less light returns to the detector. A pure
    # additive offset would be removed by SNV/baseline correction in one line and
    # would make dark fabric look easy. The damaging effect is that the
    # fibre-specific band CONTRAST is attenuated while detector noise stays at its
    # absolute level -- signal-to-noise collapses, and no amount of preprocessing
    # recovers information the detector never captured.
    darkness: float | tuple = 0.0
    dark_contrast_loss: float = 0.92  # fraction of band contrast lost at darkness=1
    dark_offset: float = 0.80         # broadband absorbance added at darkness=1
    seed: int = 42
    rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)


def _apply_wavelength_shift(spectrum: np.ndarray, wl: np.ndarray, shift_nm: float) -> np.ndarray:
    """Interpolate the spectrum onto a shifted grid (simulates sensor drift)."""
    return np.interp(wl, wl + shift_nm, spectrum)


def simulate_blend(cotton_fraction: float,
                   cfg: SimConfig,
                   wl: np.ndarray = WAVELENGTHS,
                   endmembers=None) -> np.ndarray:
    """
    Simulate ONE measured absorbance spectrum for a blend with the given
    cotton mass fraction (polyester fraction = 1 - cotton_fraction).

    Mixing model: additive in the absorbance domain (Beer-Lambert), with a small
    optional non-linear cross term to mimic real blends never being perfectly
    linear. Then multiplicative scatter, sloping baseline, wavelength drift and
    additive noise are applied — the standard nuisances a chemometric model must
    survive.
    """
    rng = cfg.rng

    # Per-sample band jitter: real fabrics shift/broaden bands slightly
    # (moisture, weave, temperature). Rebuild endmembers with jittered centres.
    if getattr(cfg, "band_jitter_sd", 0.0) and endmembers is None:
        j_cot = [Band(b.centre + rng.normal(0, cfg.band_jitter_sd), b.width, b.height)
                 for b in COTTON_BANDS]
        j_poly = [Band(b.centre + rng.normal(0, cfg.band_jitter_sd), b.width, b.height)
                  for b in POLYESTER_BANDS]
        cotton = _absorbance_from_bands(j_cot, wl)
        poly = _absorbance_from_bands(j_poly, wl)
    elif endmembers is None:
        cotton, poly = pure_endmembers(wl)
    else:
        cotton, poly = endmembers

    f = float(np.clip(cotton_fraction, 0.0, 1.0))

    # Per-sample darkness draw
    dk = getattr(cfg, "darkness", 0.0)
    if isinstance(dk, (tuple, list)):
        dk = float(rng.uniform(dk[0], dk[1]))
    dk = float(np.clip(dk, 0.0, 1.0))

    # --- linear (Beer-Lambert) mixing + small non-linear interaction term ---
    mixed = f * cotton + (1.0 - f) * poly
    if cfg.nonlinear:
        mixed = mixed + cfg.nonlinear * f * (1.0 - f) * (cotton * poly)

    x = (wl - wl.min()) / (wl.max() - wl.min())

    # --- broadband dye/colour tilt (mimics dark-fabric absorption drift) ---
    if getattr(cfg, "dye_tilt_sd", 0.0):
        mixed = mixed * (1.0 + rng.normal(0.0, cfg.dye_tilt_sd) * x)

    # --- carbon-black attenuation (dark fabric) ---
    # Applied BEFORE scatter/baseline/noise, because the physical order matters:
    # the dye attenuates the light that carries the fibre signature, and the
    # detector's own noise is then added to that weakened signal.
    if dk > 0.0:
        keep = 1.0 - cfg.dark_contrast_loss * dk
        mixed = mixed * keep + cfg.dark_offset * dk

    # --- multiplicative scatter (path-length / particle-size effects) ---
    scatter = 1.0 + rng.normal(0.0, cfg.scatter_sd)
    mixed = mixed * scatter

    # --- sloping baseline + offset ---
    baseline = (rng.normal(0.0, cfg.baseline_slope_sd) * x
                + rng.normal(0.0, cfg.baseline_offset_sd))
    mixed = mixed + baseline

    # --- wavelength drift ---
    if cfg.wl_shift_sd:
        mixed = _apply_wavelength_shift(mixed, wl, rng.normal(0.0, cfg.wl_shift_sd))

    # --- additive detector noise ---
    mixed = mixed + rng.normal(0.0, cfg.noise_sd, size=mixed.shape)

    return mixed


def make_dataset(n_samples: int = 1200,
                 cfg: SimConfig | None = None,
                 wl: np.ndarray = WAVELENGTHS,
                 fraction_dist: str = "uniform"):
    """
    Build a labelled dataset of blend spectra.

    Returns
    -------
    X : (n_samples, n_wavelengths) absorbance spectra
    y : (n_samples,) cotton mass fraction in [0, 1]
    wl : wavelength grid
    """
    if cfg is None:
        cfg = SimConfig()
    rng = cfg.rng
    # Note: endmembers left as None per-sample so band-jitter (fabric
    # variability) is applied independently to each simulated spectrum.

    if fraction_dist == "uniform":
        y = rng.uniform(0.0, 1.0, size=n_samples)
    elif fraction_dist == "realistic":
        # Real streams cluster near common blends (100/0, 0/100, 65/35, 50/50).
        modes = np.array([0.0, 0.35, 0.5, 0.65, 1.0])
        pick = rng.choice(len(modes), size=n_samples, p=[0.22, 0.18, 0.2, 0.18, 0.22])
        y = np.clip(modes[pick] + rng.normal(0.0, 0.05, size=n_samples), 0.0, 1.0)
    else:
        raise ValueError(fraction_dist)

    X = np.vstack([simulate_blend(f, cfg, wl, endmembers=None) for f in y])
    return X, y, wl


if __name__ == "__main__":
    X, y, wl = make_dataset(5)
    print("dataset:", X.shape, "labels:", np.round(y, 2))
    c, p = pure_endmembers()
    print("endmember peaks — cotton max @ %.0f nm, polyester max @ %.0f nm"
          % (wl[c.argmax()], wl[p.argmax()]))
