"""
test_pipeline.py — scientific correctness tests
===============================================

These assert the things a reviewer would want guaranteed: the physics is right,
the preprocessing does what it claims, the model learns, and outputs are always
physically valid.

Run:
    python tests/test_pipeline.py          (no pytest needed)
    pytest tests/                          (if pytest is installed)
"""

from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from spectral_simulator import (make_dataset, simulate_blend, pure_endmembers,
                                SimConfig, WAVELENGTHS)
from preprocessing import snv, preprocess
from model import BlendRegressor
from sklearn.model_selection import train_test_split


def test_endmembers_peak_at_published_bands():
    """Cotton's strongest band ~1940nm (O-H); polyester's distinctive band ~1660nm."""
    wl = WAVELENGTHS
    cotton, poly = pure_endmembers(wl)
    assert 1900 < wl[cotton.argmax()] < 1980, "cotton peak not at O-H band"
    # Polyester's largest peak should sit in an aromatic/C-H region, not cotton's O-H band
    assert abs(wl[poly.argmax()] - 1940) > 50, "polyester peak collides with cotton O-H"
    print("  [OK] endmember peaks match published absorption bands")


def test_blend_is_monotonic_between_endmembers():
    """A noise-free blend must lie between the two pure spectra."""
    cfg = SimConfig(noise_sd=0, scatter_sd=0, baseline_slope_sd=0,
                    baseline_offset_sd=0, wl_shift_sd=0, nonlinear=0,
                    band_jitter_sd=0, dye_tilt_sd=0, seed=1)
    wl = WAVELENGTHS
    ends = pure_endmembers(wl)
    pure_c = simulate_blend(1.0, cfg, wl, ends)
    pure_p = simulate_blend(0.0, cfg, wl, ends)
    half = simulate_blend(0.5, cfg, wl, ends)
    lo, hi = np.minimum(pure_c, pure_p), np.maximum(pure_c, pure_p)
    assert np.all(half >= lo - 1e-9) and np.all(half <= hi + 1e-9), \
        "50/50 blend falls outside the pure-fibre envelope"
    print("  [OK] blends interpolate correctly between pure fibres (Beer-Lambert)")


def test_snv_normalises():
    """SNV must give each spectrum zero mean and unit standard deviation."""
    X, _, _ = make_dataset(20, cfg=SimConfig(seed=3))
    Z = snv(X)
    assert np.allclose(Z.mean(axis=1), 0, atol=1e-9), "SNV rows not zero-mean"
    assert np.allclose(Z.std(axis=1), 1, atol=1e-6), "SNV rows not unit-variance"
    print("  [OK] SNV normalisation is mathematically correct")


def test_predictions_always_valid_fractions():
    """Composition must never be negative or exceed 100%."""
    X, y, _ = make_dataset(300, cfg=SimConfig(seed=4))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4)
    reg = BlendRegressor("pls").fit(preprocess(Xtr, "snv+sg1"), ytr)
    pred = reg.predict(preprocess(Xte, "snv+sg1"))
    assert (pred >= 0).all() and (pred <= 1).all(), "invalid fibre fraction produced"
    print("  [OK] all predictions are physically valid fractions")


def test_model_beats_naive_mean_baseline():
    """The model must substantially beat 'always guess the average'."""
    X, y, _ = make_dataset(400, cfg=SimConfig(seed=5))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=5)
    reg = BlendRegressor("pls").fit(preprocess(Xtr, "snv+sg1"), ytr)
    model_mae = np.mean(np.abs(reg.predict(preprocess(Xte, "snv+sg1")) - yte))
    naive_mae = np.mean(np.abs(ytr.mean() - yte))
    assert model_mae < naive_mae / 3, \
        f"model MAE {model_mae:.3f} not clearly better than naive {naive_mae:.3f}"
    print(f"  [OK] model MAE {model_mae*100:.2f}pp << naive baseline {naive_mae*100:.2f}pp")


def test_preprocessing_helps_under_scatter():
    """SNV+derivative should beat raw spectra when scatter/baseline noise is present."""
    X, y, _ = make_dataset(400, cfg=SimConfig(seed=6))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=6)
    mae = {}
    for pre in ("none", "snv+sg1"):
        reg = BlendRegressor("pls").fit(preprocess(Xtr, pre), ytr)
        mae[pre] = np.mean(np.abs(reg.predict(preprocess(Xte, pre)) - yte))
    assert mae["snv+sg1"] < mae["none"], "preprocessing did not improve accuracy"
    print(f"  [OK] preprocessing improves MAE "
          f"{mae['none']*100:.2f}pp -> {mae['snv+sg1']*100:.2f}pp")


def test_reproducible():
    """Same seed must give identical data."""
    a, ya, _ = make_dataset(50, cfg=SimConfig(seed=11))
    b, yb, _ = make_dataset(50, cfg=SimConfig(seed=11))
    assert np.array_equal(a, b) and np.array_equal(ya, yb), "not reproducible"
    print("  [OK] results are reproducible with a fixed seed")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    print("\nFibreBlend — pipeline tests")
    print("=" * 52)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 52)
    if failed:
        print(f"{failed} of {len(TESTS)} tests FAILED\n")
        sys.exit(1)
    print(f"All {len(TESTS)} tests passed.\n")
