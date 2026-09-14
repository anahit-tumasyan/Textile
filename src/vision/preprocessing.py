"""
preprocessing.py  (vision branch)
=================================

Frequency-domain front end for fabric RGB patches — the imaging analogue of the
SNV + Savitzky-Golay stage in ``src/preprocessing.py``.

Why a front end at all (full argument in docs/vision_architecture_decision.md §3):

  * A woven fabric's dominant image energy is the **weave lattice** — a strong,
    near-periodic carrier fixed by yarn count, twist and weave pattern. That is a
    property of the fabric *construction*, not of the fibre *blend*. Left in, it is
    the single most salient thing in the image, it is nearly constant within a
    swatch, and it therefore lets the network identify the swatch and memorise its
    label. It is a nuisance variable, and it is separable: it is discrete peaks in
    the 2-D Fourier magnitude.

  * The blend signal is **aperiodic micro-texture**. Cotton is a flat, convoluted,
    matte ribbon (~12-20 um, irregular width, high diffuse albedo). PET staple is a
    smooth, round, near-specular filament of near-constant diameter. Their
    discriminative statistics are broadband, plus a distinct highlight population
    from specular filaments.

So we notch the lattice, keep the residual, and hand the network an explicit
specular-highlight channel it would otherwise have to rediscover from data it
does not have.

Everything here is numpy + scipy, runs in dataloader workers on CPU, and costs
~2 ms per 384x384 patch — deliberately off the GPU's critical path.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter, median_filter

__all__ = [
    "to_luminance",
    "radial_profile_detrend",
    "weave_peak_mask",
    "weave_notch",
    "highlight_map",
    "build_channels",
    "N_CHANNELS",
]

#: RGB (3) + weave-suppressed luminance residual (1) + specular highlight map (1).
N_CHANNELS = 5


def to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance from a float RGB image in [0, 1], shape (H, W, 3)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


def radial_profile_detrend(log_mag: np.ndarray) -> np.ndarray:
    """
    Remove the 1/f^a radial falloff from a (fftshifted) log-magnitude spectrum.

    Natural-image spectra decay steeply with radius, so a raw peak search just
    returns the low-frequency blob. Subtracting the mean log-magnitude at each
    integer radius flattens that envelope and leaves the *lattice* peaks — which
    are what stands proud of the local background — as the maxima.
    """
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    nbins = int(r.max()) + 1
    counts = np.bincount(r.ravel(), minlength=nbins)
    sums = np.bincount(r.ravel(), weights=log_mag.ravel(), minlength=nbins)
    profile = sums / np.maximum(counts, 1)
    return log_mag - profile[r]


def weave_peak_mask(
    gray: np.ndarray,
    n_peaks: int = 16,
    sigma: float = 2.5,
    min_radius: int = 4,
    z_thresh: float = 2.5,
) -> np.ndarray:
    """
    Build a multiplicative Gaussian stop-band mask (fftshifted) over the dominant
    periodic peaks of ``gray``.

    Parameters
    ----------
    n_peaks
        Maximum number of conjugate peak *pairs* to suppress. A plain weave needs
        ~2 fundamentals + harmonics; 16 comfortably covers twill/satin.
    sigma
        Gaussian stop-band radius in frequency bins. Too small leaves ringing
        residue of the carrier; too large starts eating broadband signal.
    min_radius
        Frequency bins around DC that are never notched — that region carries
        illumination gradient, not weave.
    z_thresh
        A candidate must exceed this many robust std-devs of the detrended
        spectrum to count as a real lattice peak. Knitted / nonwoven / heavily
        napped fabrics have no sharp lattice, and for those this correctly
        notches nothing.
    """
    F = np.fft.fftshift(np.fft.fft2(gray))
    log_mag = np.log1p(np.abs(F))
    detr = radial_profile_detrend(log_mag)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Robust z-score (MAD-based): the spectrum is heavy-tailed, so a plain std
    # is dragged upward by the very peaks we are trying to detect.
    med = np.median(detr)
    mad = np.median(np.abs(detr - med)) + 1e-8
    z = (detr - med) / (1.4826 * mad)

    local_max = detr >= maximum_filter(detr, size=5)
    cand = local_max & (rad > min_radius) & (z > z_thresh)

    mask = np.ones_like(gray, dtype=np.float32)
    if not cand.any():
        return mask

    ys, xs = np.nonzero(cand)
    order = np.argsort(-z[ys, xs])
    # Half-plane only: each peak's conjugate partner is suppressed explicitly, so
    # taking both would waste half the budget on duplicates.
    taken = 0
    two_sig_sq = 2.0 * sigma * sigma
    for idx in order:
        py, px = int(ys[idx]), int(xs[idx])
        if (py - cy, px - cx) < (0, 0):  # lexicographic half-plane test
            continue
        for sy, sx in ((py, px), (2 * cy - py, 2 * cx - px)):
            if 0 <= sy < h and 0 <= sx < w:
                d2 = (yy - sy) ** 2 + (xx - sx) ** 2
                mask *= 1.0 - np.exp(-d2 / two_sig_sq)
        taken += 1
        if taken >= n_peaks:
            break
    return mask.astype(np.float32)


def weave_notch(gray: np.ndarray, **kwargs) -> np.ndarray:
    """
    Return the weave-suppressed residual of a single-channel image.

    The output is robustly standardised (median / MAD) so that downstream
    normalisation does not have to cope with per-image contrast swings — the same
    role SNV plays for spectra.
    """
    mask = weave_peak_mask(gray, **kwargs)
    F = np.fft.fftshift(np.fft.fft2(gray))
    resid = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-8
    return ((resid - med) / (1.4826 * mad)).astype(np.float32)


def highlight_map(rgb: np.ndarray, size: int = 9) -> np.ndarray:
    """
    Specular-highlight channel.

    A specular lobe off a round PET filament is (a) brighter than its local
    neighbourhood and (b) less saturated than the diffuse body colour, because it
    carries the illuminant's spectrum rather than the dye's. Cotton, being matte
    and irregular, produces far fewer such pixels. Combining the local brightness
    excess with a desaturation term isolates that population and — importantly —
    is invariant to the dye colour, which is the biggest confound in a real
    waste stream.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    lum = to_luminance(rgb)
    local = median_filter(lum, size=size)
    excess = np.clip(lum - local, 0.0, None)

    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    desat = np.clip(1.0 - sat, 0.0, 1.0)

    hl = excess * desat
    scale = np.percentile(hl, 99.5) + 1e-8
    return np.clip(hl / scale, 0.0, 1.0).astype(np.float32)


def build_channels(
    rgb: np.ndarray,
    notch: bool = True,
    notch_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Assemble the network input for one patch.

    Parameters
    ----------
    rgb : (H, W, 3) float array in [0, 1]

    Returns
    -------
    (N_CHANNELS, H, W) float32:
        [0:3] RGB, mean-centred to roughly [-0.5, 0.5]
        [3]   weave-suppressed luminance residual (0 if ``notch`` is False)
        [4]   specular-highlight map

    Ablating this to plain RGB is a one-flag change (``notch=False`` plus a
    3-channel stem) — see the ablation table in the design doc for why that
    ablation is the first thing to run once real data exists.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    lum = to_luminance(rgb)
    resid = weave_notch(lum, **(notch_kwargs or {})) if notch else np.zeros_like(lum)
    hl = highlight_map(rgb)
    out = np.concatenate(
        [(rgb - 0.5).transpose(2, 0, 1), resid[None], hl[None]], axis=0
    )
    return np.ascontiguousarray(out, dtype=np.float32)
