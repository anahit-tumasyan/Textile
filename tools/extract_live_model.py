"""Extract an exactly-equivalent affine model so the browser can run the real
prediction live, rather than replaying recorded outputs.

The pipeline is  raw -> SNV -> Savitzky-Golay(1st deriv) -> PLS.
SNV is per-sample and non-linear, so it stays in JS. Everything AFTER it --
the SG convolution and PLS itself -- is affine, so the composition of the two
is a single vector A and scalar B:  y = A . snv(x) + B.
Extracted numerically (not from .coef_) so it is immune to sklearn version
differences in how PLS stores its internals. Verified below to 1e-9.
"""
import sys, json, numpy as np
sys.path.insert(0, "src")
from spectral_simulator import make_dataset, SimConfig
from preprocessing import preprocess, snv
from model import BlendRegressor
from sklearn.model_selection import train_test_split

SEED = 42
X, y, wl = make_dataset(1500, cfg=SimConfig(seed=SEED), fraction_dist="uniform")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED)

pls = BlendRegressor("pls").fit(preprocess(Xtr, "snv+sg1"), ytr)

def after_snv(Z):
    """The affine tail: SG1 then PLS, on already-SNV'd input."""
    from preprocessing import savgol_derivative
    return np.asarray(pls.model.predict(savgol_derivative(Z))).ravel()

n = X.shape[1]
B = float(after_snv(np.zeros((1, n)))[0])
I = np.eye(n)
A = after_snv(I) - B                       # one pass, all 601 basis vectors

# --- verification against the real pipeline, on real test spectra ----------
Zte = snv(Xte)
live = np.clip(Zte @ A + B, 0, 1)
ref = pls.predict(preprocess(Xte, "snv+sg1"))
err = float(np.abs(live - ref).max())
print(f"max |JS-equivalent - sklearn| = {err:.3e}", file=sys.stderr)
assert err < 1e-9, "affine extraction does not reproduce the model"
print(f"live model MAE = {np.abs(live-yte).mean()*100:.2f} pp", file=sys.stderr)

# --- the raw baseline, for a live side-by-side ---------------------------
# PLS on raw spectra has no SNV, so the whole chain is affine in the raw input.
pls_raw = BlendRegressor("pls").fit(Xtr, ytr)
Braw = float(np.asarray(pls_raw.model.predict(np.zeros((1, n)))).ravel()[0])
Araw = np.asarray(pls_raw.model.predict(I)).ravel() - Braw
live_raw = np.clip(Xte @ Araw + Braw, 0, 1)
ref_raw = pls_raw.predict(Xte)
err_raw = float(np.abs(live_raw - ref_raw).max())
print(f"max |JS-equivalent - sklearn| (baseline) = {err_raw:.3e}", file=sys.stderr)
assert err_raw < 1e-9
print(f"baseline live MAE = {np.abs(live_raw-yte).mean()*100:.2f} pp", file=sys.stderr)

json.dump({
    "improved": {"A": [float(v) for v in A], "B": B, "snv": True,
                 "mae": round(float(np.abs(live - yte).mean() * 100), 2)},
    "baseline": {"A": [float(v) for v in Araw], "B": Braw, "snv": False,
                 "mae": round(float(np.abs(live_raw - yte).mean() * 100), 2)},
}, sys.stdout)
