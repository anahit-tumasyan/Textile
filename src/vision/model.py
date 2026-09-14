"""
model.py  (vision branch)
=========================

    ConvNeXt-T backbone  ->  second-order (covariance) pooling neck
                         ->  gated-attention MIL aggregation over patches
                         ->  Dirichlet simplex head

The selection argument is in docs/vision_architecture_decision.md §1-2; the three
load-bearing claims, in short:

* **Backbone: hierarchical CNN, not ViT/Swin/Mamba.** ViT-B/16's patchify stem is
  a stride-16 low-pass filter applied *before any learning happens*, and the blend
  signal is exactly the high-frequency micro-texture it discards. The long-range
  modelling that justifies transformers and SSMs buys nothing here: fabric is a
  stationary texture, so the statistics in one 384px window are the statistics of
  the whole swatch. We pay for locality, which is what we want, and we get mature
  pretrained weights and a plain-PyTorch graph that runs anywhere.

* **Neck: covariance, not global average pooling.** GAP is a first-order statistic
  and it is provably blind to the thing being measured. Two swatches with the same
  mean filter response but different *co-occurrence* of matte and specular
  responses — i.e. different blends — have identical GAP vectors. Texture
  discrimination and proportion estimation both live in second-order statistics,
  and a covariance matrix is an orderless summary, which matches a material whose
  composition is invariant to how the swatch happened to be laid out.

* **Head: Dirichlet, not a 2-way softmax with MSE.** The target is a point on the
  simplex, and a Dirichlet gives the composition *and* a concentration parameter
  that is a usable confidence. For a sorting line, "62% cotton, and I am not sure"
  must be routable differently from "62% cotton, confident" — the whole economic
  case for the sensor is not misrouting bales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

__all__ = [
    "isqrt_newton_schulz",
    "SecondOrderPool",
    "GatedAttentionMIL",
    "FibreBlendNet",
    "composition_from_alpha",
    "uncertainty_from_alpha",
]


# --------------------------------------------------------------------------- #
# Matrix square root
# --------------------------------------------------------------------------- #
def isqrt_newton_schulz(A: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Differentiable matrix square root by coupled Newton-Schulz iteration
    (iSQRT-COV, Li et al. 2018).

    Why not ``torch.linalg.eigh``/SVD: eigendecomposition backward is numerically
    unstable when eigenvalues cluster — which is precisely the regime of a
    covariance matrix built from correlated CNN channels — and on GPU it is slow
    and largely serial. Newton-Schulz is five batched matmuls and is stable.

    A: (B, d, d), symmetric positive semi-definite, pre-normalised so ||A||_F <= 1
    is achieved internally (the iteration only converges inside that ball).

    NOTE: call this in fp32. Under fp16 autocast the iteration drifts and can
    diverge; ``SecondOrderPool`` disables autocast around it for that reason.
    """
    d = A.shape[-1]
    norm = A.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    Y = A / norm
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand_as(A).contiguous()
    Z = I.clone()
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * norm.sqrt()


class SecondOrderPool(nn.Module):
    """
    Orderless second-order pooling: (B, d, H, W) -> (B, d(d+1)/2).

    Steps: centre the token set, form the covariance, trace-normalise it (this is
    what keeps the Newton-Schulz iteration in its convergence ball), take the
    matrix square root — which compresses the covariance's very long eigenvalue
    tail and is worth several points on its own — then vectorise the upper
    triangle with off-diagonal terms scaled by sqrt(2) so the Frobenius geometry
    survives the flattening.

    Dimension note: the covariance is estimated from N = H*W tokens. Keep
    N >> d or the estimate is rank-deficient and the square root is garbage.
    At 384px input, stage-3 grid is 24x24 = 576 tokens, so d = 128 is a
    comfortable 4.5:1.
    """

    def __init__(self, dim: int, num_iters: int = 5, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.num_iters = num_iters
        self.eps = eps
        idx = torch.triu_indices(dim, dim)
        self.register_buffer("triu_i", idx[0], persistent=False)
        self.register_buffer("triu_j", idx[1], persistent=False)
        self.out_dim = dim * (dim + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        n = h * w
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().reshape(b, d, n)
            x = x - x.mean(dim=2, keepdim=True)
            cov = (x @ x.transpose(1, 2)) / max(n - 1, 1)
            eye = torch.eye(d, device=x.device).expand_as(cov)
            cov = cov + self.eps * eye
            trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1).view(b, 1, 1).clamp_min(1e-8)
            cov = cov / trace
            root = isqrt_newton_schulz(cov, self.num_iters)
            vec = root[:, self.triu_i, self.triu_j]
            off = (self.triu_i != self.triu_j).to(vec.dtype)
            vec = vec * (1.0 + (2.0 ** 0.5 - 1.0) * off)
        return vec


class GatedAttentionMIL(nn.Module):
    """
    Gated attention pooling over a bag of patch embeddings (Ilse et al., 2018).

    Why attention rather than a mean over patches: a real swatch is not perfectly
    homogeneous. Some windows land on a seam, a print, a label, a fold shadow or
    a blown highlight, and those patches carry no composition information but do
    carry confident nonsense. A learned, permutation-invariant weighting lets the
    model abstain on them, and the weights double as a free explanation of which
    part of the swatch drove the answer.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        """h: (B, P, D) -> pooled (B, D), attention (B, P)."""
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        logits = self.w(self.drop(a)).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        return torch.einsum("bp,bpd->bd", attn, h), attn


class FibreBlendNet(nn.Module):
    """
    Parameters
    ----------
    n_fibres : K, the number of fibre classes on the simplex (2 for cotton/PET).
    in_ch    : input channels from ``preprocessing.build_channels`` (5).
    proj_dim : d for the covariance. 128 -> an 8256-dim second-order descriptor.
    embed_dim: patch embedding width after the descriptor projection.
    pretrained: ImageNet weights. Keep this on. With a few thousand swatches the
        low-level filters are most of what makes the model work, and they are
        exactly the part ImageNet transfers well.
    freeze_stages: how many leading backbone blocks to freeze. 2 (stem + stage 1)
        is a good default on a small corpus and saves activation memory.
    """

    def __init__(
        self,
        n_fibres: int = 2,
        in_ch: int = 5,
        proj_dim: int = 128,
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze_stages: int = 0,
        dropout: float = 0.2,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_fibres = n_fibres
        self.grad_checkpoint = grad_checkpoint

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = convnext_tiny(weights=weights)
        feats = net.features
        self.stage_early = nn.Sequential(*feats[0:6])   # -> 384ch, stride 16
        self.stage_last = nn.Sequential(*feats[6:8])    # -> 768ch, stride 32

        self._inflate_stem(in_ch)
        if freeze_stages > 0:
            for i, block in enumerate(self.stage_early):
                if i < freeze_stages:
                    for p in block.parameters():
                        p.requires_grad_(False)

        self.fuse = nn.Sequential(
            nn.Conv2d(384 + 768, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )
        self.pool = SecondOrderPool(proj_dim)
        self.embed = nn.Sequential(
            nn.LayerNorm(self.pool.out_dim),
            nn.Linear(self.pool.out_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mil = GatedAttentionMIL(embed_dim)
        # One head, applied to both the per-patch and the bag embedding. Sharing
        # it is what makes the patch-consistency term in the loss meaningful:
        # the two predictions live in the same output space by construction.
        self.head = nn.Linear(embed_dim, n_fibres)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    # -- stem surgery -------------------------------------------------------
    def _inflate_stem(self, in_ch: int) -> None:
        """
        Widen the 3-channel pretrained stem to ``in_ch`` inputs.

        The extra channels are **zero-initialised**, not mean-inflated: at step 0
        the network is then numerically identical to the pretrained model, so the
        frequency and highlight channels can only ever help, and a bad front-end
        cannot wreck the transfer before training has a chance to start.
        """
        stem_conv = self.stage_early[0][0]
        if in_ch == stem_conv.in_channels:
            return
        new = nn.Conv2d(
            in_ch,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )
        with torch.no_grad():
            new.weight.zero_()
            k = min(3, in_ch)
            new.weight[:, :k] = stem_conv.weight[:, :k]
            if stem_conv.bias is not None:
                new.bias.copy_(stem_conv.bias)
        self.stage_early[0][0] = new

    # -- forward ------------------------------------------------------------
    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        (N, C, H, W) -> (N, embed_dim).

        channels_last is applied here rather than in the training loop: a bag
        tensor arrives 5-D (B, P, C, H, W), and ``to(channels_last)`` is a silent
        no-op on anything that is not 4-D, so converting before the flatten
        quietly buys nothing on tensor-core cards.
        """
        x = x.contiguous(memory_format=torch.channels_last)
        if self.grad_checkpoint and self.training:
            f3 = torch.utils.checkpoint.checkpoint(self.stage_early, x, use_reentrant=False)
            f4 = torch.utils.checkpoint.checkpoint(self.stage_last, f3, use_reentrant=False)
        else:
            f3 = self.stage_early(x)
            f4 = self.stage_last(f3)
        f4up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([f3, f4up], dim=1))
        return self.embed(self.pool(fused))

    def forward(self, patches: torch.Tensor) -> dict:
        """
        patches: (B, P, C, H, W)

        Returns alpha (B, K), patch_alpha (B, P, K) and attention (B, P).
        """
        b, p = patches.shape[:2]
        h = self.encode_patches(patches.flatten(0, 1)).view(b, p, -1)
        bag, attn = self.mil(h)
        alpha = F.softplus(self.head(bag)) + 1.0
        patch_alpha = F.softplus(self.head(h)) + 1.0
        return {"alpha": alpha, "patch_alpha": patch_alpha, "attn": attn}


def composition_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Dirichlet posterior mean — the reported composition."""
    return alpha / alpha.sum(dim=-1, keepdim=True)


def uncertainty_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Total evidential uncertainty in [0, 1]: K / sum(alpha).

    Near 1 the model is saying "I have seen nothing like this" — a dark or
    carbon-black swatch, a coating, a fibre outside the training set. On a sorting
    line that is the signal to divert to manual/NIR check rather than to guess.
    """
    return alpha.shape[-1] / alpha.sum(dim=-1)
