"""
calibration.py  (vision branch)
===============================

Post-hoc calibration of the Dirichlet concentration, and the interval definition
that makes it meaningful.

This file is the record of three corrections, because each one was wrong in a way
worth not repeating.

**Correction 1 — the head is structurally overconfident, and the loss weight is
not the knob.** `coverage_90` came back at 0.70 against a nominal 0.90. Raising
the NLL weight 6x was tested: MAE identical, final coverage identical, coverage
merely peaking higher mid-training before decaying back. The NLL term is fit on
*training* residuals, which by late training are much smaller than held-out ones,
so the model learns a confidence describing data it memorised.

**Correction 2 — temperature scaling is NOT monotone in coverage.** The obvious
post-hoc fix, alpha' = alpha / s, was implemented on the claim that larger `s`
means a wider interval means higher coverage. That claim is false. Once any
component of alpha drops below 1 the Beta marginal stops being unimodal and
becomes U-shaped, converging to point masses of weight b/(a+b) at 0 and a/(a+b)
at 1. For an asymmetric alpha the *central* interval then collapses onto a single
endpoint -- measured: alpha = (0.0055, ~0) gives the interval [1, 1], and observed
coverage fell from 0.77 to 0.70 as `s` was increased. The search must therefore be
constrained to the unimodal regime, min(alpha) >= 1, where widening is monotone.

**Correction 3 — a central interval cannot cover a vertex label, ever.** Real
blend corpora contain pure swatches: y = (1, 0). At the widest admissible point of
the constrained search every alpha is 1, the marginal is uniform, and the central
90% interval is [0.05, 0.95] -- which excludes 1.0 by construction. In the
measured corpus, 24 of 60 validation swatches lay outside [0.05, 0.95] and 11 sat
exactly on a vertex, so nominal coverage was *unattainable at any temperature*.
The metric was mis-specified, not just the model. The fix is the **highest-density
interval**: for a J-shaped Beta (a > 1, b -> 1) the HDI hugs the boundary and
includes the vertex, which is exactly the behaviour a pure-cotton swatch should
produce.

The invariant that survives all three: scaling alpha leaves the Dirichlet mean
exactly unchanged,

    (alpha/s) / sum(alpha/s) = alpha / sum(alpha)

so calibration can never trade accuracy for coverage, whatever else it does.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "central_interval",
    "hdi_interval",
    "empirical_coverage",
    "max_admissible_temperature",
    "fit_temperature",
]


def _marginal(alpha: np.ndarray):
    a = np.asarray(alpha, dtype=np.float64)
    return a[:, 0], a[:, 1:].sum(1)


def central_interval(alpha: np.ndarray, level: float = 0.90):
    """
    Equal-tailed interval for the first component's Beta marginal.

    Kept for comparison and for the regression test that documents its failure on
    vertex labels. ``hdi_interval`` is what should be reported.
    """
    from scipy.stats import beta as beta_dist

    a1, a2 = _marginal(alpha)
    tail = (1.0 - level) / 2.0
    return beta_dist.ppf(tail, a1, a2), beta_dist.ppf(1.0 - tail, a1, a2)


def hdi_interval(alpha: np.ndarray, level: float = 0.90, grid: int = 201):
    """
    Highest-density interval for the first component's Beta marginal.

    Found by scanning the lower-tail mass p over [0, 1-level] and taking the
    narrowest [ppf(p), ppf(p+level)]. A grid scan rather than an optimiser
    because the objective is 1-D, bounded, cheap, and can be flat at the
    boundary -- where an optimiser is most likely to stop early and where the
    interesting case (a pure swatch) actually lives.

    For a J-shaped marginal the minimum sits at an endpoint, so the interval
    reaches 0.0 or 1.0 exactly and a vertex label is coverable. That is the whole
    reason this replaces the central interval.
    """
    from scipy.stats import beta as beta_dist

    a1, a2 = _marginal(alpha)
    ps = np.linspace(0.0, 1.0 - level, grid)[None, :]
    lo = beta_dist.ppf(ps, a1[:, None], a2[:, None])
    hi = beta_dist.ppf(ps + level, a1[:, None], a2[:, None])
    width = np.where(np.isfinite(hi - lo), hi - lo, np.inf)
    j = np.argmin(width, axis=1)
    rows = np.arange(len(a1))
    return lo[rows, j], hi[rows, j]


def empirical_coverage(
    alpha: np.ndarray, y: np.ndarray, level: float = 0.90, kind: str = "hdi"
) -> float:
    """Fraction of targets inside the model's own ``level`` interval."""
    lo, hi = (hdi_interval if kind == "hdi" else central_interval)(alpha, level)
    t = np.asarray(y, dtype=np.float64)[:, 0]
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float(((t >= lo) & (t <= hi) & ok).mean())


def max_admissible_temperature(alpha: np.ndarray) -> float:
    """
    Largest `s` keeping every concentration >= 1, i.e. inside the unimodal regime
    where coverage is monotone in `s`.

    This is the bound whose absence produced Correction 2. It is usually small
    (the head's softplus+1 floor means some alpha sit just above 1), which limits
    how much post-hoc widening is available -- a real limitation to report rather
    than route around by letting alpha go sub-1.
    """
    return float(max(np.asarray(alpha, dtype=np.float64).min(), 1.0))


#: Below this many calibration swatches, a fitted temperature is noise. Estimating
#: a 90% quantile from a handful of points is not possible, and the failure is
#: silent and confident: the --smoke run fitted s=0.145 on 2 swatches and drove
#: validation coverage from 1.00 to 0.50.
MIN_CALIB_N = 25


def fit_temperature(
    alpha: np.ndarray,
    y: np.ndarray,
    level: float = 0.90,
    kind: str = "hdi",
    iters: int = 40,
    min_n: int = MIN_CALIB_N,
) -> tuple[float, dict]:
    """
    Bisect for the `s` achieving `level` empirical coverage.

    The search is two-sided. s > 1 widens the interval and is bounded by
    min(alpha)/s >= 1, the unimodal regime; s < 1 sharpens it and needs no bound,
    since scaling alpha up only makes the marginal more concentrated. The
    sharpening direction matters: switching to the HDI turned a 0.70-coverage
    model into a 0.93/1.00 one, i.e. the failure mode flipped from over- to
    under-confident, and a widen-only tool would have reported "already
    calibrated" and left an interval that is too wide to be informative.

    Returns ``(s, info)``. ``info['attainable']`` is False when even the widest
    admissible temperature falls short -- in which case the interval *shape* or
    the model is wrong, and no amount of width will fix it. Callers must report
    that rather than shipping the bound as if it were a fit.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(alpha) < min_n:
        return 1.0, {
            "attainable": False,
            "kind": kind,
            "n": int(len(alpha)),
            "note": (
                f"calibration set of {len(alpha)} < {min_n} is too small to fit a "
                f"{level:.0%} quantile; left unscaled rather than fitting noise"
            ),
        }

    cov = lambda s: empirical_coverage(alpha / s, y, level, kind)
    s_max = max_admissible_temperature(alpha)
    s_min = 1e-3
    at_1, at_max = cov(1.0), cov(s_max)

    info = {
        "coverage_at_1": at_1,
        "coverage_at_s_max": at_max,
        "s_max": s_max,
        "attainable": bool(at_max >= level),
        "kind": kind,
    }
    if at_1 > level:
        # Under-confident: intervals are wider than they need to be. Sharpen by
        # searching s < 1, where no admissibility bound applies.
        if cov(s_min) >= level:
            info["note"] = "cannot sharpen enough; intervals remain conservative"
            return s_min, info
        lo_s, hi_s = s_min, 1.0
        for _ in range(iters):
            mid = np.sqrt(lo_s * hi_s)
            if cov(mid) < level:
                lo_s = mid
            else:
                hi_s = mid
        s = float(np.sqrt(lo_s * hi_s))
        info["coverage_at_s"] = cov(s)
        info["note"] = "sharpened an under-confident interval"
        return s, info
    if np.isclose(at_1, level):
        info["note"] = "already calibrated; left unscaled"
        return 1.0, info
    if not info["attainable"]:
        info["note"] = (
            f"{level:.0%} unreachable within the unimodal regime (max {at_max:.2f} "
            f"at s={s_max:.3f}); interval shape or model is at fault, not width"
        )
        return s_max, info

    lo_s, hi_s = 1.0, s_max
    for _ in range(iters):
        mid = np.sqrt(lo_s * hi_s)
        if cov(mid) < level:
            lo_s = mid
        else:
            hi_s = mid
    s = float(np.sqrt(lo_s * hi_s))
    info["coverage_at_s"] = cov(s)
    return s, info
