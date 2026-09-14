"""
data.py  (vision branch)
========================

Dataset, patch extraction and the two augmentations that carry most of the
data-efficiency argument in docs/vision_architecture_decision.md §5.

Three things here are non-obvious and worth reading before touching the code.

1. **Patches are cut at native sensor resolution. Nothing is ever resized.**
   The blend signal is sub-yarn micro-texture at the scale of a few pixels. The
   reflex of resizing a 2048x2048 swatch to 224x224 for an ImageNet backbone is a
   ~9x low-pass filter and it destroys the entire signal. Resolution is the one
   thing this pipeline refuses to trade.

2. **Area fraction vs mass fraction.** An image can only observe the *visible
   surface area* held by each fibre type. The label a recycler cares about, and
   the label on a NIR reference method, is *mass* fraction. The two differ by a
   per-fibre constant (density x effective visible thickness), and the map is
   linear-fractional, not linear:

       a_k = (m_k / c_k) / sum_j (m_j / c_j)

   Everything that must be *mixed* is mixed in area space; everything reported is
   reported in mass space. `c` is a calibration vector — the values in
   ``DEFAULT_C`` are placeholders and must be fitted once against a reference
   method on real swatches (a one-parameter fit for a cotton/PET pair).

3. **CompositionMix is a physically exact augmentation, and it is also the
   imbalance fix.** Real blend labels pile up at 100/0, 65/35 and 50/50. Standard
   mixup is a heuristic; here it is not. If a bag of P patches takes k patches
   from swatch A and P-k from swatch B, the bag's *area* composition is exactly
   the (k/P)-weighted mix of theirs, because area fractions are additive over
   disjoint area. So we can synthesise correctly-labelled training bags anywhere
   on the simplex, including the gaps the real label distribution leaves empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from . import preprocessing as P

__all__ = [
    "DEFAULT_C",
    "mass_to_area",
    "area_to_mass",
    "FabricSwatch",
    "SwatchBagDataset",
    "compute_lds_weights",
    "sliding_window_coords",
]

#: Placeholder mass-per-visible-area constants, [cotton, polyester], arbitrary
#: units (only their ratio matters). MUST be re-fitted on real reference data.
DEFAULT_C = np.array([1.00, 0.86], dtype=np.float64)


def mass_to_area(m: np.ndarray, c: np.ndarray = DEFAULT_C) -> np.ndarray:
    """Mass fractions on the simplex -> visible-area fractions."""
    m = np.asarray(m, dtype=np.float64)
    r = m / c
    return r / np.maximum(r.sum(-1, keepdims=True), 1e-12)


def area_to_mass(a: np.ndarray, c: np.ndarray = DEFAULT_C) -> np.ndarray:
    """Visible-area fractions -> mass fractions on the simplex."""
    a = np.asarray(a, dtype=np.float64)
    r = a * c
    return r / np.maximum(r.sum(-1, keepdims=True), 1e-12)


def sliding_window_coords(h: int, w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    """Top-left coordinates of a sliding window covering (h, w), edges included."""
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys and ys[-1] != h - patch and h >= patch:
        ys.append(h - patch)
    if xs and xs[-1] != w - patch and w >= patch:
        xs.append(w - patch)
    return [(y, x) for y in ys for x in xs]


def compute_lds_weights(
    labels: np.ndarray,
    n_bins: int = 50,
    sigma: float = 2.0,
    power: float = 0.5,
) -> np.ndarray:
    """
    Label-Distribution-Smoothing weights (Yang et al., 2021), the regression
    analogue of inverse-frequency class weighting.

    A kernel-smoothed empirical density over the label axis is inverted; ``power``
    interpolates between no reweighting (0) and full inverse-density (1). 0.5 is
    the usual compromise — full inversion hands enormous weight to the two or
    three swatches sitting in an empty region of the simplex and the variance is
    worse than the bias it fixes.

    Implemented for the K=2 case over the first coordinate. For K>2 the honest
    move is a KDE on the simplex in CLR coordinates; until there is real
    multi-fibre data to fit that on, this returns uniform weights and says so.
    """
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 2:
        return np.ones(len(labels), dtype=np.float32)
    x = np.clip(labels[:, 0], 0.0, 1.0)
    hist, edges = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    dens = gaussian_filter1d(hist.astype(np.float64), sigma, mode="nearest")
    dens = np.maximum(dens, 1e-6)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    w = (1.0 / dens[idx]) ** power
    return (w / w.mean()).astype(np.float32)


@dataclass
class FabricSwatch:
    """One physical swatch. ``mass`` is the reference composition on the simplex."""

    swatch_id: str
    mass: np.ndarray                      # (K,) mass fractions, sums to 1
    path: str | None = None               # image on disk, or ...
    array: np.ndarray | None = None       # ... an in-memory float RGB image in [0,1]
    meta: dict = field(default_factory=dict)


class SwatchBagDataset(Dataset):
    """
    Yields a *bag* of P patches per swatch, plus the swatch-level composition.

    The bag formulation is the reason this works with a few thousand swatches:
    the label is a property of the whole swatch, but every patch of a homogeneous
    swatch carries the same composition, so P patches per swatch multiply the
    effective sample count without inventing labels. Bags also let the model
    downweight patches that landed on a seam, a print or a shadow (see the
    attention aggregator in ``model.py``).

    Parameters
    ----------
    swatches
        The corpus. Split by ``swatch_id`` BEFORE constructing train/val sets —
        never by patch, or patches of one swatch leak across the split and the
        validation score becomes fiction.
    patch, n_patches
        Patch side in native pixels, and bag size P.
    train
        Random patch origins + flips/rot90 + CompositionMix when True;
        deterministic sliding-window grid when False.
    mix_prob, mix_alpha
        CompositionMix rate and its Beta(alpha, alpha) shape. alpha=0.4 is
        U-shaped (mostly near-pure mixes, occasionally aggressive); raise toward
        1.0 to synthesise more mid-simplex bags when the real labels are bimodal.
    """

    def __init__(
        self,
        swatches: Sequence[FabricSwatch],
        patch: int = 384,
        n_patches: int = 8,
        train: bool = True,
        mix_prob: float = 0.5,
        mix_alpha: float = 0.4,
        c: np.ndarray = DEFAULT_C,
        notch: bool = True,
        loader: Callable[[FabricSwatch], np.ndarray] | None = None,
        weights: np.ndarray | None = None,
        seed: int = 0,
    ):
        self.swatches = list(swatches)
        self.patch = patch
        self.n_patches = n_patches
        self.train = train
        self.mix_prob = mix_prob
        self.mix_alpha = mix_alpha
        self.c = np.asarray(c, dtype=np.float64)
        self.notch = notch
        self.loader = loader or self._default_loader
        self.weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else np.ones(len(self.swatches), dtype=np.float32)
        )
        self.seed = seed

    # -- loading ------------------------------------------------------------
    @staticmethod
    def _default_loader(sw: FabricSwatch) -> np.ndarray:
        if sw.array is not None:
            return np.asarray(sw.array, dtype=np.float32)
        from PIL import Image  # imported lazily: in-memory corpora need no PIL

        img = np.asarray(Image.open(sw.path).convert("RGB"), dtype=np.float32) / 255.0
        return img

    # -- patch extraction ---------------------------------------------------
    def _cut(self, img: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
        h, w, _ = img.shape
        p = self.patch
        if h < p or w < p:
            raise ValueError(f"swatch {h}x{w} smaller than patch {p}")
        if self.train:
            coords = [
                (int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1)))
                for _ in range(n)
            ]
        else:
            grid = sliding_window_coords(h, w, p, stride=max(p // 2, 1))
            # Deterministic, evenly spread subsample of the full grid.
            sel = np.linspace(0, len(grid) - 1, num=n).round().astype(int)
            coords = [grid[i] for i in sel]

        out = []
        for (y, x) in coords:
            tile = img[y : y + p, x : x + p]
            if self.train:
                k = int(rng.integers(0, 4))
                if k:
                    tile = np.rot90(tile, k)
                if rng.random() < 0.5:
                    tile = tile[:, ::-1]
                # Photometric jitter: the dye and the lighting are nuisances, and
                # a real waste stream will show colours this corpus never did.
                tile = np.clip(
                    tile * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0.0, 1.0
                )
            out.append(np.ascontiguousarray(tile, dtype=np.float32))
        return out

    def __len__(self) -> int:
        return len(self.swatches)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng((self.seed, idx) if self.train else (0, idx))
        sw = self.swatches[idx]
        P_ = self.n_patches

        area = mass_to_area(sw.mass, self.c)
        weight = float(self.weights[idx])

        if self.train and len(self.swatches) > 1 and rng.random() < self.mix_prob:
            # --- CompositionMix -------------------------------------------
            j = int(rng.integers(0, len(self.swatches)))
            while j == idx:
                j = int(rng.integers(0, len(self.swatches)))
            other = self.swatches[j]
            lam = float(rng.beta(self.mix_alpha, self.mix_alpha))
            k = int(round(lam * P_))                     # realised mix is exactly k/P
            k = min(max(k, 0), P_)
            lam_eff = k / P_
            tiles = self._cut(self.loader(sw), rng, k) + self._cut(
                self.loader(other), rng, P_ - k
            )
            area = lam_eff * area + (1.0 - lam_eff) * mass_to_area(other.mass, self.c)
            weight = lam_eff * weight + (1.0 - lam_eff) * float(self.weights[j])
        else:
            tiles = self._cut(self.loader(sw), rng, P_)

        chans = np.stack([P.build_channels(t, notch=self.notch) for t in tiles], 0)
        mass = area_to_mass(area, self.c)
        return {
            "patches": chans.astype(np.float32),          # (P, C, H, W)
            "mass": mass.astype(np.float32),              # (K,)
            "weight": np.float32(weight),
            "swatch_id": sw.swatch_id,
        }
