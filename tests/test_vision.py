"""
Tests for the vision branch.

These target the four things most likely to be silently wrong: the
area<->mass conversion, the exactness claim made for CompositionMix, the
numerical behaviour of the covariance pooling under autocast, and whether the
frequency front end actually removes a weave carrier.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.vision import preprocessing as P
from src.vision.data import (
    DEFAULT_C,
    FabricSwatch,
    SwatchBagDataset,
    area_to_mass,
    compute_lds_weights,
    mass_to_area,
    sliding_window_coords,
)
from src.vision.fabric_simulator import render_swatch
from src.vision.losses import BlendCompositionLoss, aitchison_sq, dirichlet_nll
from src.vision.model import FibreBlendNet, SecondOrderPool, isqrt_newton_schulz


# --------------------------------------------------------------------------- #
# Compositional algebra
# --------------------------------------------------------------------------- #
def test_area_mass_roundtrip():
    m = np.array([[0.62, 0.38], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    back = area_to_mass(mass_to_area(m))
    assert np.allclose(back, m, atol=1e-12)
    assert np.allclose(back.sum(1), 1.0)


def test_area_mass_is_not_identity():
    """If c is not uniform the two coordinate systems must actually differ,
    otherwise the whole calibration story is vacuous."""
    m = np.array([0.5, 0.5])
    assert not np.allclose(mass_to_area(m, DEFAULT_C), m)
    # ...but a uniform c must collapse to the identity.
    assert np.allclose(mass_to_area(m, np.array([1.0, 1.0])), m)


def test_compositionmix_label_is_exact():
    """
    The central claim: a bag of k patches from A and P-k from B has area
    composition exactly (k/P) a_A + (1-k/P) a_B. Verified by reconstructing the
    label the dataset produced from the mix fraction it must have used.
    """
    rng = np.random.default_rng(0)
    a_img = np.zeros((64, 64, 3), np.float32)
    b_img = np.ones((64, 64, 3), np.float32)
    sw = [
        FabricSwatch("a", np.array([1.0, 0.0]), array=a_img),
        FabricSwatch("b", np.array([0.0, 1.0]), array=b_img),
    ]
    P_ = 8
    ds = SwatchBagDataset(sw, patch=32, n_patches=P_, train=True, mix_prob=1.0, notch=False)
    for i in range(len(ds)):
        item = ds[i]
        area = mass_to_area(item["mass"].astype(np.float64))
        # The realised mix must land on the k/P grid, exactly.
        k = area[0] * P_
        assert np.isclose(k, round(k), atol=1e-9), f"mix off the k/P grid: {k}"


def test_lds_upweights_the_sparse_region():
    labels = np.zeros((100, 2))
    labels[:95, 0] = 0.5            # dense mode
    labels[95:, 0] = 0.9            # sparse tail
    labels[:, 1] = 1 - labels[:, 0]
    w = compute_lds_weights(labels)
    assert w[95:].mean() > w[:95].mean() * 3
    assert np.isclose(w.mean(), 1.0, atol=1e-5)


def test_sliding_window_covers_edges():
    coords = sliding_window_coords(100, 100, patch=40, stride=30)
    ys = {y for y, _ in coords}
    assert 0 in ys and 60 in ys           # 60 = h - patch, the flush-right window


# --------------------------------------------------------------------------- #
# Frequency front end
# --------------------------------------------------------------------------- #
def test_weave_notch_suppresses_a_periodic_carrier():
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    carrier = 0.5 * np.sin(2 * np.pi * xx * 12 / 256) * np.sin(2 * np.pi * yy * 12 / 256)
    noise = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32) * 0.05
    img = 0.5 + carrier + noise

    resid = P.weave_notch(img)
    # Energy at the carrier frequency, relative to total, must drop hard.
    def carrier_ratio(a):
        F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
        peak = F[128 - 12, 128 - 12] + F[128 + 12, 128 + 12]
        return peak / F.sum()

    assert carrier_ratio(resid) < 0.05 * carrier_ratio(img)


def test_notch_is_a_noop_without_a_lattice():
    """A nonwoven has no lattice; the peak test must not invent one."""
    img = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    img = img / img.std()
    resid = P.weave_notch(img)
    assert np.corrcoef(img.ravel(), resid.ravel())[0, 1] > 0.9


def test_build_channels_shape_and_finiteness():
    rgb = render_swatch(0.6, size=128, rng=np.random.default_rng(0))
    ch = P.build_channels(rgb)
    assert ch.shape == (P.N_CHANNELS, 128, 128)
    assert np.isfinite(ch).all()


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def test_newton_schulz_matches_true_matrix_sqrt():
    torch.manual_seed(0)
    a = torch.randn(3, 32, 32)
    spd = a @ a.transpose(1, 2) / 32 + torch.eye(32) * 1e-3
    spd = spd / spd.diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
    root = isqrt_newton_schulz(spd, num_iters=12)
    assert torch.allclose(root @ root, spd, atol=1e-3)


def test_second_order_pool_survives_autocast():
    """
    The pool must return finite fp32 output even when called inside an autocast
    region -- this is the regression test for the fp16/bf16 divergence that the
    internal autocast-disable exists to prevent.
    """
    pool = SecondOrderPool(16)
    x = torch.randn(2, 16, 12, 12)
    ref = pool(x)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = pool(x.to(torch.bfloat16))
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    assert got.shape == ref.shape == (2, 16 * 17 // 2)


def test_pool_is_permutation_invariant():
    """Orderless by construction: shuffling the spatial tokens must not matter."""
    pool = SecondOrderPool(8)
    x = torch.randn(1, 8, 6, 6)
    flat = x.reshape(1, 8, 36)
    perm = flat[:, :, torch.randperm(36)].reshape(1, 8, 6, 6)
    assert torch.allclose(pool(x), pool(perm), atol=1e-4)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_aitchison_separates_what_l1_cannot():
    """99/1 vs 100/0 is 1pp of L1 but a large ratio error -- the reason this term exists."""
    near = torch.tensor([[0.99, 0.01]])
    pure = torch.tensor([[0.995, 0.005]])
    mid_a = torch.tensor([[0.50, 0.50]])
    mid_b = torch.tensor([[0.505, 0.495]])
    assert aitchison_sq(near, pure) > aitchison_sq(mid_a, mid_b) * 10


def test_dirichlet_nll_rewards_the_truth():
    y = torch.tensor([[0.7, 0.3]])
    good = torch.tensor([[7.0, 3.0]])       # mean 0.7, concentrated
    bad = torch.tensor([[3.0, 7.0]])        # mean 0.3
    assert dirichlet_nll(good, y) < dirichlet_nll(bad, y)


def test_loss_is_finite_at_the_simplex_vertices():
    """y = (1, 0) is a real label (100% cotton) and must not produce inf/nan."""
    loss_fn = BlendCompositionLoss()
    out = {
        "alpha": torch.tensor([[4.0, 1.5]]),
        "patch_alpha": torch.tensor([[[4.0, 1.5], [3.0, 2.0]]]),
        "attn": torch.tensor([[0.5, 0.5]]),
    }
    for target in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])):
        total, stats = loss_fn(out, target)
        assert torch.isfinite(total), stats


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_simplex_output():
    model = FibreBlendNet(pretrained=False).eval()
    bags = torch.randn(2, 3, P.N_CHANNELS, 96, 96)
    with torch.no_grad():
        out = model(bags)
    assert out["alpha"].shape == (2, 2)
    assert out["patch_alpha"].shape == (2, 3, 2)
    assert torch.allclose(out["attn"].sum(1), torch.ones(2), atol=1e-5)
    assert (out["alpha"] > 1.0).all()
    comp = out["alpha"] / out["alpha"].sum(-1, keepdim=True)
    assert torch.allclose(comp.sum(-1), torch.ones(2), atol=1e-6)


def test_bag_output_is_permutation_invariant():
    """Patch order is an artefact of sampling; the answer must not depend on it."""
    model = FibreBlendNet(pretrained=False).eval()
    bags = torch.randn(1, 4, P.N_CHANNELS, 96, 96)
    with torch.no_grad():
        a = model(bags)["alpha"]
        b = model(bags[:, torch.tensor([2, 0, 3, 1])])["alpha"]
    assert torch.allclose(a, b, atol=1e-4)


def test_stem_inflation_preserves_the_pretrained_response():
    """
    Zero-init of the extra input channels means the widened stem must reproduce
    the 3-channel pretrained response exactly on RGB-only input.
    """
    model = FibreBlendNet(pretrained=False, in_ch=5)
    stem = model.stage_early[0][0]
    assert stem.in_channels == 5
    assert torch.count_nonzero(stem.weight[:, 3:]) == 0


# --------------------------------------------------------------------------- #
# Post-hoc calibration
# --------------------------------------------------------------------------- #
from src.vision.calibration import (
    central_interval,
    empirical_coverage,
    fit_temperature,
    hdi_interval,
    max_admissible_temperature,
)


def test_temperature_scaling_cannot_change_the_prediction():
    """
    The one invariant that survives: scaling alpha leaves the Dirichlet mean
    exactly unchanged, so calibration can never trade accuracy for coverage.
    """
    rng = np.random.default_rng(0)
    alpha = rng.uniform(1.0, 50.0, size=(64, 2))
    for s in (0.1, 1.0, 7.3, 250.0):
        a = alpha / s
        assert np.allclose(
            a / a.sum(1, keepdims=True), alpha / alpha.sum(1, keepdims=True), atol=1e-12
        )


def test_coverage_is_not_monotone_below_alpha_one():
    """
    Regression test for the bug this module was rewritten around. Once alpha
    drops below 1 the Beta marginal is U-shaped and the CENTRAL interval of an
    asymmetric alpha collapses onto one endpoint, so coverage can fall as the
    temperature rises. The constrained search exists because of this.
    """
    # The collapse needs a strongly asymmetric alpha: as s grows the marginal
    # tends to point masses of weight b/(a+b) at 0 and a/(a+b) at 1, so the
    # central interval degenerates only once one of those weights exceeds 1-level/2.
    # A near-vertex prediction -- exactly what a pure swatch produces -- is that case.
    lopsided = np.array([[40.0, 0.5]])          # mass at 1 is 0.988 > 0.95
    lo, hi = central_interval(lopsided / 10000.0)
    assert hi[0] - lo[0] < 1e-6, f"expected collapse, got [{lo[0]}, {hi[0]}]"
    assert lo[0] > 0.99, "expected collapse onto the upper endpoint"

    # A balanced alpha does NOT collapse -- it widens to [0, 1] as expected. The
    # bug was never universal, which is why it survived the first review.
    balanced = np.array([[40.0, 15.0]])
    lo2, hi2 = central_interval(balanced / 10000.0)
    assert hi2[0] - lo2[0] > 0.99


def test_central_interval_cannot_cover_a_vertex_but_hdi_can():
    """
    A pure-cotton swatch is y = (1, 0). At the widest admissible concentration
    (all alpha = 1, i.e. uniform) the central interval is [0.05, 0.95] and
    excludes it -- nominal coverage is then unattainable at any temperature.
    The HDI of a J-shaped marginal reaches the boundary and covers it.
    """
    y = np.array([[1.0, 0.0]])
    uniform = np.array([[1.0, 1.0]])
    assert empirical_coverage(uniform, y, kind="central") == 0.0

    confident_pure = np.array([[50.0, 1.05]])
    lo, hi = hdi_interval(confident_pure)
    assert hi[0] >= 1.0 - 1e-9, f"HDI should reach the vertex, got {hi[0]}"
    assert empirical_coverage(confident_pure, y, kind="hdi") == 1.0


def test_max_admissible_temperature_keeps_alpha_unimodal():
    alpha = np.array([[40.0, 3.0], [12.0, 1.5]])
    s = max_admissible_temperature(alpha)
    assert np.isclose(s, 1.5)
    assert (alpha / s).min() >= 1.0 - 1e-12


def test_fit_temperature_repairs_an_overconfident_model():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.1, 0.9, size=400)
    pred = np.clip(truth + rng.normal(0, 0.04, size=400), 0.02, 0.98)
    # Overconfident by ~two orders of magnitude, but left with headroom to widen.
    alpha = np.stack([pred, 1 - pred], 1) * 2000.0
    y = np.stack([truth, 1 - truth], 1)

    before = empirical_coverage(alpha, y)
    s, info = fit_temperature(alpha, y, level=0.90)
    after = empirical_coverage(alpha / s, y)

    assert before < 0.5, f"test setup is not overconfident: {before}"
    assert info["attainable"] and s > 1.0
    assert abs(after - 0.90) < 0.05, f"coverage not repaired: {after}"


def test_fit_temperature_reports_when_width_cannot_fix_it():
    """
    The failure this module must not paper over: if even the widest admissible
    temperature falls short, `attainable` is False and the caller has to say so
    rather than ship the bound as a fit.
    """
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.0, 1.0, size=200)
    pred = np.clip(1.0 - truth, 0.02, 0.98)      # systematically inverted
    alpha = np.stack([pred, 1 - pred], 1) * 3.0  # little headroom above 1
    y = np.stack([truth, 1 - truth], 1)
    s, info = fit_temperature(alpha, y, level=0.90)
    assert info["attainable"] is False
    assert "unreachable" in info["note"]


def test_fit_temperature_leaves_a_calibrated_model_alone():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.2, 0.8, size=300)
    alpha = np.stack([truth, 1 - truth], 1) * 30.0
    y = np.stack([truth, 1 - truth], 1)
    s, info = fit_temperature(alpha, y, level=0.90)
    assert empirical_coverage(alpha / s, y) >= 0.88


def test_fit_temperature_sharpens_an_underconfident_model():
    """
    The mirror of the overconfident case, and the one the first implementation
    silently ignored: intervals wider than they need to be are also miscalibrated,
    and s < 1 is always admissible because scaling alpha up keeps it above 1.
    """
    rng = np.random.default_rng(5)
    truth = rng.uniform(0.15, 0.85, size=400)
    pred = np.clip(truth + rng.normal(0, 0.01, size=400), 0.02, 0.98)
    alpha = np.stack([pred, 1 - pred], 1) * 6.0     # far too diffuse for a 1pp error
    y = np.stack([truth, 1 - truth], 1)

    before = empirical_coverage(alpha, y)
    s, info = fit_temperature(alpha, y, level=0.90)
    after = empirical_coverage(alpha / s, y)

    assert before > 0.97, f"test setup is not under-confident: {before}"
    assert s < 1.0, f"expected sharpening, got s={s}"
    assert abs(after - 0.90) < 0.05, f"not repaired: {after}"


def test_fit_temperature_refuses_a_tiny_calibration_set():
    """
    A handful of swatches cannot estimate a 90% quantile. Fitting anyway is worse
    than not fitting: on the smoke run a 2-swatch calibration set produced
    s=0.145 and halved validation coverage.
    """
    rng = np.random.default_rng(7)
    truth = rng.uniform(0.2, 0.8, size=4)
    alpha = np.stack([truth, 1 - truth], 1) * 500.0
    y = np.stack([truth, 1 - truth], 1)
    s, info = fit_temperature(alpha, y, level=0.90)
    assert s == 1.0
    assert info["attainable"] is False and "too small" in info["note"]
