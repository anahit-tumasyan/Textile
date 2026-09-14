"""
predict.py — demo the trained blend-composition model
=====================================================

Loads the saved model and predicts cotton/polyester percentages for a batch of
freshly simulated "unknown" garments, printing a small report. This stands in
for the real inference step: hand it a measured NIR spectrum, get back a
quantitative composition.

Usage:
    python predict.py --model results/best_model.joblib --n 8
"""

from __future__ import annotations
import argparse, os, sys

if sys.version_info < (3, 9):
    sys.exit(f"ERROR: Python 3.9+ required (found {sys.version_info.major}."
             f"{sys.version_info.minor}). Install a newer Python, then retry.")

try:
    import numpy as np
    import joblib
except ImportError as e:
    sys.exit(f"ERROR: missing dependency ({e.name}).\n"
             f"Fix:   pip install -r requirements.txt\n"
             f"Then:  python check_setup.py")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from spectral_simulator import make_dataset, SimConfig  # noqa
from preprocessing import preprocess  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/best_model.joblib")
    ap.add_argument("--n", type=int, default=8, help="number of unknown garments")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"ERROR: no trained model at '{args.model}'.\n"
                 f"Fix:   python train.py     (this trains and saves it)\n"
                 f"Then:  python predict.py")

    bundle = joblib.load(args.model)
    model, pre, wl = bundle["model"], bundle["preprocessing"], bundle["wavelengths"]

    # Fresh, unseen "garments" (different seed => new noise realisations).
    cfg = SimConfig(seed=args.seed)
    X, y_true, _ = make_dataset(args.n, cfg=cfg, wl=wl, fraction_dist="realistic")
    y_pred = model.predict(preprocess(X, pre))

    print(f"\nTrained model: {model.kind.upper()}  |  preprocessing: {pre}")
    print("=" * 62)
    print(f"{'garment':>8} | {'true cotton/poly':>20} | {'predicted cotton/poly':>22} | err")
    print("-" * 62)
    for i, (t, p) in enumerate(zip(y_true, y_pred), 1):
        tc, tp = t * 100, (1 - t) * 100
        pc, pp = p * 100, (1 - p) * 100
        err = abs(pc - tc)
        print(f"{i:>8} | {tc:5.1f}% / {tp:5.1f}%     | "
              f"{pc:6.1f}% / {pp:6.1f}%       | {err:4.1f} pp")
    print("-" * 62)
    print(f"mean absolute error on this batch: "
          f"{np.mean(np.abs(y_pred - y_true))*100:.2f} percentage points\n")


if __name__ == "__main__":
    main()
