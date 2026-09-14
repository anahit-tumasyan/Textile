"""
losses.py  (vision branch)
==========================

The target is a point on the (K-1)-simplex, not a scalar and not a class. That
single fact rules out the two reflexes — MSE on a sigmoid, or cross-entropy on a
class — and dictates the compound objective below.

    L = w * [ l1 * L_dir  +  l2 * L_ait  +  l3 * L_mae ]  +  l4 * L_cons

L_dir  Dirichlet negative log-likelihood. The probabilistic backbone of the loss:
       it fits the *distribution* over compositions, so the concentration
       parameter becomes a calibrated confidence instead of a decoration.

L_ait  Squared Aitchison distance — Euclidean distance between centred log-ratio
       transforms. This is the natural metric on compositional data (Aitchison,
       1986). It matters at the extremes: 99/1 vs 100/0 is one percentage point
       of absolute error but an infinite *ratio* error, and for a recycler that
       distinction is the difference between "pure cotton, mechanically
       recyclable" and "contaminated". Absolute-error losses cannot see it.

L_mae  Plain L1 on the simplex. Kept because percentage-point MAE is the number
       the project is judged on (the NIR baseline in this repo reports 1.65 pp),
       and a model should optimise the metric it reports. It also stabilises
       early training while the Dirichlet term is still finding its scale.

L_cons Attention-weighted disagreement between patch-level predictions within a
       bag. Encodes the physical prior that a swatch is compositionally
       homogeneous, and it is a genuinely free regulariser: it needs no labels,
       so it also works on unlabelled swatches for semi-supervised extension.

Weights: l1=1.0, l2=0.3, l3=2.0, l4=0.1 is a sane start. If calibration is poor,
raise l1; if the extremes are mushy, raise l2; if MAE plateaus above the NIR
baseline, raise l3 and suspect the input resolution before the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "smooth_simplex",
    "clr",
    "dirichlet_nll",
    "aitchison_sq",
    "patch_consistency",
    "BlendCompositionLoss",
]

EPS = 1e-7


def smooth_simplex(y: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """
    Pull a composition off the boundary of the simplex: y -> (1-eps) y + eps/K.

    Not cosmetic. A 100% cotton swatch has y = (1, 0), and both log(0) in the
    Aitchison term and the Dirichlet density at a vertex are undefined. eps=0.01
    caps the achievable precision at ~0.5 pp, which is well inside the reference
    method's own uncertainty, so nothing real is lost.
    """
    k = y.shape[-1]
    return (1.0 - eps) * y + eps / k


def clr(p: torch.Tensor) -> torch.Tensor:
    """Centred log-ratio transform: the simplex -> R^K with zero-sum constraint."""
    logp = torch.log(p.clamp_min(EPS))
    return logp - logp.mean(dim=-1, keepdim=True)


def dirichlet_nll(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    -log Dir(y | alpha), per sample. alpha: (B, K) > 1, y: (B, K) on the simplex.
    """
    a0 = alpha.sum(-1)
    logB = torch.lgamma(alpha).sum(-1) - torch.lgamma(a0)
    return logB - ((alpha - 1.0) * torch.log(y.clamp_min(EPS))).sum(-1)


def aitchison_sq(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Aitchison distance between two compositions, per sample."""
    return ((clr(pred) - clr(target)) ** 2).sum(-1)


def patch_consistency(
    patch_alpha: torch.Tensor, attn: torch.Tensor
) -> torch.Tensor:
    """
    Attention-weighted variance of the per-patch compositions in each bag.

    Weighted by attention on purpose: a patch the aggregator has already decided
    to ignore (a seam, a shadow) must not be forced to agree with the rest, or
    the regulariser would fight the aggregator.
    """
    p = patch_alpha / patch_alpha.sum(-1, keepdim=True)      # (B, P, K)
    w = attn.unsqueeze(-1)                                   # (B, P, 1)
    mean = (w * p).sum(1, keepdim=True)
    return (w * (p - mean) ** 2).sum(dim=(1, 2))


class BlendCompositionLoss(nn.Module):
    """
    Compound compositional loss with per-sample (LDS) reweighting.

    ``nll_warmup`` linearly ramps the Dirichlet term over the first N steps. Cold
    Dirichlet NLL against a near-vertex target produces large gradients that can
    push alpha to its floor and stall the run; the L1 term carries the first few
    hundred steps on its own perfectly well.
    """

    def __init__(
        self,
        l_dir: float = 1.0,
        l_ait: float = 0.3,
        l_mae: float = 2.0,
        l_cons: float = 0.1,
        label_smooth: float = 0.01,
        nll_warmup: int = 500,
    ):
        super().__init__()
        self.l_dir, self.l_ait, self.l_mae, self.l_cons = l_dir, l_ait, l_mae, l_cons
        self.label_smooth = label_smooth
        self.nll_warmup = nll_warmup
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, out: dict, target: torch.Tensor, weight: torch.Tensor | None = None):
        alpha = out["alpha"].float()
        pred = alpha / alpha.sum(-1, keepdim=True)
        y = smooth_simplex(target.float(), self.label_smooth)

        l_dir = dirichlet_nll(alpha, y)
        l_ait = aitchison_sq(pred, y)
        l_mae = (pred - target.float()).abs().sum(-1) * 0.5      # = |cotton error| for K=2
        l_cons = patch_consistency(out["patch_alpha"].float(), out["attn"].float())

        if self.training:
            self._step += 1
        ramp = 1.0
        if self.nll_warmup > 0:
            ramp = min(1.0, float(self._step.item()) / self.nll_warmup)

        per_sample = self.l_dir * ramp * l_dir + self.l_ait * l_ait + self.l_mae * l_mae
        if weight is not None:
            per_sample = per_sample * weight.float()
        total = per_sample.mean() + self.l_cons * l_cons.mean()

        return total, {
            "loss": float(total.detach()),
            "nll": float(l_dir.mean().detach()),
            "aitchison": float(l_ait.mean().detach()),
            "mae_pp": float(l_mae.mean().detach()) * 100.0,
            "consistency": float(l_cons.mean().detach()),
        }
