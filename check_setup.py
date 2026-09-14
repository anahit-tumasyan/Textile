#!/usr/bin/env python3
"""
check_setup.py — run this FIRST
================================

Verifies that this machine can run the FibreBlend prototype, then executes a
fast end-to-end smoke test. Designed so a reviewer can confirm everything works
in under a minute, with actionable messages if anything is missing.

Usage:
    python check_setup.py
"""

from __future__ import annotations
import sys
import os

GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
if os.name == "nt" or not sys.stdout.isatty():   # no colour on Windows/pipes
    GREEN = RED = YELLOW = BOLD = RESET = ""

MIN_PYTHON = (3, 9)
REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "joblib": "joblib",
}

ok = lambda m: print(f"  {GREEN}[OK]{RESET}   {m}")
bad = lambda m: print(f"  {RED}[FAIL]{RESET} {m}")
warn = lambda m: print(f"  {YELLOW}[WARN]{RESET} {m}")


def check_python() -> bool:
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        bad(f"Python {v.major}.{v.minor} found — need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
        print(f"\n  Fix: install a newer Python from https://www.python.org/downloads/")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_deps() -> bool:
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{pkg:<14} {ver}")
        except ImportError:
            bad(f"{pkg:<14} NOT INSTALLED")
            missing.append(pkg)
    if missing:
        print(f"\n  Fix: pip install {' '.join(missing)}")
        print(f"  (or: pip install -r requirements.txt)")
        return False
    return True


def smoke_test() -> bool:
    """Tiny end-to-end run: simulate -> preprocess -> fit -> predict."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    try:
        import numpy as np
        from spectral_simulator import make_dataset, SimConfig
        from preprocessing import preprocess
        from model import BlendRegressor
        from sklearn.model_selection import train_test_split

        X, y, wl = make_dataset(200, cfg=SimConfig(seed=0))
        ok(f"simulated {X.shape[0]} spectra x {X.shape[1]} wavelengths")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        Z, Zt = preprocess(Xtr, "snv+sg1"), preprocess(Xte, "snv+sg1")
        ok("preprocessing (SNV + Savitzky-Golay) applied")

        reg = BlendRegressor("pls").fit(Z, ytr)
        pred = reg.predict(Zt)
        mae = float(np.mean(np.abs(pred - yte)) * 100)
        ok(f"model trained and predicted — MAE {mae:.2f} percentage points")

        if not ((pred >= 0).all() and (pred <= 1).all()):
            bad("predictions outside valid 0-100% range")
            return False
        ok("all predictions are valid fibre fractions (0-100%)")

        if mae > 15:
            warn(f"MAE {mae:.1f}pp is high for a smoke test (expected <15pp)")
        return True
    except Exception as e:
        bad(f"smoke test crashed: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False


def main() -> int:
    print(f"\n{BOLD}FibreBlend — setup check{RESET}")
    print("=" * 52)

    print(f"\n{BOLD}1. Python version{RESET}")
    if not check_python():
        return 1

    print(f"\n{BOLD}2. Dependencies{RESET}")
    if not check_deps():
        return 1

    print(f"\n{BOLD}3. End-to-end smoke test{RESET}")
    if not smoke_test():
        return 1

    print("\n" + "=" * 52)
    print(f"{GREEN}{BOLD}ALL CHECKS PASSED{RESET} — this machine can run the prototype.\n")
    print("Next steps:")
    print("  python train.py     # full study: benchmark, model, 5 figures")
    print("  python predict.py   # demo predictions on unseen samples\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
