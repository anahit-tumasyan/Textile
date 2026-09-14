"""
train.py — end-to-end Phase-0 experiment
=========================================

Runs the whole quantitative blend-composition study:

  1. Simulate a labelled dataset of cotton/polyester blend NIR spectra.
  2. Split into train / test (held-out).
  3. Benchmark preprocessing x model combinations.
  4. Report MAE / RMSE / R2 for cotton-fraction prediction (percentage points).
  5. Persist the best model, a metrics table (JSON + CSV), and all figures.

Usage:
    python train.py --n 1500 --outdir results
"""

from __future__ import annotations
import argparse, json, os, sys

if sys.version_info < (3, 9):
    sys.exit(f"ERROR: Python 3.9+ required (found {sys.version_info.major}."
             f"{sys.version_info.minor}). Install a newer Python, then retry.")

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
except ImportError as e:
    sys.exit(f"ERROR: missing dependency ({e.name}).\n"
             f"Fix:   pip install -r requirements.txt\n"
             f"Then:  python check_setup.py")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from spectral_simulator import make_dataset, SimConfig, pure_endmembers, WAVELENGTHS  # noqa
from preprocessing import preprocess  # noqa
from model import BlendRegressor  # noqa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- house style -------------------------------------------------------------
COT = "#2a7f62"    # cotton green
POLY = "#3b6ea5"   # polyester blue
ACC = "#d9822b"    # accent orange
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def metrics(y_true, y_pred) -> dict:
    """Errors reported in percentage points (fractions x 100)."""
    yt, yp = y_true * 100, y_pred * 100
    return {
        "MAE_pp": float(mean_absolute_error(yt, yp)),
        "RMSE_pp": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "max_abs_err_pp": float(np.max(np.abs(yt - yp))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="number of blend samples")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # ---------------------------------------------------------------- data ----
    cfg = SimConfig(seed=args.seed)
    X, y, wl = make_dataset(args.n, cfg=cfg, fraction_dist="uniform")
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=args.seed)
    print(f"[data] {X.shape[0]} spectra x {X.shape[1]} wavelengths "
          f"({wl.min():.0f}-{wl.max():.0f} nm) | train={len(ytr)} test={len(yte)}")

    # ----------------------------------------------- benchmark grid -----------
    # PLS (fast) is evaluated across every preprocessing to find the best
    # signal conditioning; the non-linear RF comparator is then run on the
    # winning preprocessing (RF is ~1000x slower to fit, so we don't grid it).
    pre_methods = ["none", "snv", "sg1", "snv+sg1"]
    rows = []
    best = None  # (mae, name, model, method, preds)

    def evaluate(kind, pre):
        Xtr = preprocess(Xtr_raw, pre)
        Xte = preprocess(Xte_raw, pre)
        reg = BlendRegressor(kind).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        m = metrics(yte, pred)
        name = f"{kind.upper()} · {pre}"
        rows.append({"model": kind.upper(), "preprocessing": pre, **m})
        print(f"[bench] {name:20s}  MAE={m['MAE_pp']:5.2f}pp  "
              f"RMSE={m['RMSE_pp']:5.2f}pp  R2={m['R2']:.3f}", flush=True)
        nonlocal best
        if best is None or m["MAE_pp"] < best[0]:
            best = (m["MAE_pp"], name, reg, pre, pred)
        return m

    pls_scores = {pre: evaluate("pls", pre)["MAE_pp"] for pre in pre_methods}
    best_pls_pre = min(pls_scores, key=pls_scores.get)
    evaluate("rf", best_pls_pre)

    bench = pd.DataFrame(rows).sort_values("MAE_pp").reset_index(drop=True)
    bench.to_csv(os.path.join(args.outdir, "benchmark.csv"), index=False)

    best_mae, best_name, best_model, best_pre, best_pred = best
    # baseline = PLS on raw spectra (the naive published-baseline stand-in)
    base_row = bench[(bench.model == "PLS") & (bench.preprocessing == "none")].iloc[0]

    summary = {
        "n_samples": int(args.n),
        "n_wavelengths": int(X.shape[1]),
        "wavelength_range_nm": [float(wl.min()), float(wl.max())],
        "baseline_PLS_raw": {"MAE_pp": float(base_row.MAE_pp),
                             "RMSE_pp": float(base_row.RMSE_pp),
                             "R2": float(base_row.R2)},
        "best_model": best_name,
        "best_metrics": metrics(yte, best_pred),
        "improvement_MAE_pp_vs_baseline": float(base_row.MAE_pp - best_mae),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump({"model": best_model, "preprocessing": best_pre,
                 "wavelengths": wl}, os.path.join(args.outdir, "best_model.joblib"))

    print(f"\n[best] {best_name}: MAE={best_mae:.2f}pp  "
          f"R2={summary['best_metrics']['R2']:.3f}  "
          f"(baseline PLS/raw MAE={base_row.MAE_pp:.2f}pp -> "
          f"improvement {summary['improvement_MAE_pp_vs_baseline']:.2f}pp)")

    # =====================================================================
    #  FIGURES
    # =====================================================================
    cotton, poly = pure_endmembers(wl)

    # Fig 1 — pure endmember spectra with labelled bands
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(wl, cotton, color=COT, lw=2, label="Cotton (cellulose)")
    ax.plot(wl, poly, color=POLY, lw=2, label="Polyester (PET)")
    for c, txt in [(1480, "O–H\n(cotton)"), (1660, "aromatic\nC=O (PET)"),
                   (1940, "O–H comb.\n(cotton)"), (2140, "aromatic\n(PET)")]:
        ax.axvline(c, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Pure-fibre NIR signatures — built from published absorption bands")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "1_endmembers.png")); plt.close(fig)

    # Fig 2 — simulated blend spectra across the composition range
    fig, ax = plt.subplots(figsize=(8, 4.2))
    demo_cfg = SimConfig(seed=7)
    ends = pure_endmembers(wl)
    from spectral_simulator import simulate_blend
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = simulate_blend(frac, demo_cfg, wl, ends)
        ax.plot(wl, s, lw=1.4, label=f"{int(frac*100)}% cotton", alpha=0.9)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Simulated measured spectra with sensor noise, scatter & baseline")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "2_blend_spectra.png")); plt.close(fig)

    # Fig 3 — parity plot (predicted vs true cotton %) for the best model
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(yte * 100, best_pred * 100, s=14, alpha=0.5, color=COT,
               edgecolor="none")
    ax.plot([0, 100], [0, 100], color=ACC, lw=1.5, ls="--", label="perfect")
    bm = summary["best_metrics"]
    ax.set_xlabel("True cotton content (%)"); ax.set_ylabel("Predicted cotton content (%)")
    ax.set_title(f"{best_name}\nMAE = {bm['MAE_pp']:.2f} pp   R² = {bm['R2']:.3f}")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "3_parity.png")); plt.close(fig)

    # Fig 4 — error distribution
    err = (best_pred - yte) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(err, bins=30, color=POLY, alpha=0.85)
    ax.axvline(0, color=ACC, lw=1.5)
    ax.set_xlabel("Prediction error (percentage points)"); ax.set_ylabel("Count")
    ax.set_title(f"Held-out error distribution (n={len(yte)})  |  "
                 f"MAE={bm['MAE_pp']:.2f}pp")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "4_error_hist.png")); plt.close(fig)

    # Fig 5 — benchmark bar chart (MAE per preprocessing x model)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    piv = bench.pivot(index="preprocessing", columns="model", values="MAE_pp")
    piv = piv.reindex(pre_methods)
    piv.plot(kind="bar", ax=ax, color={"PLS": POLY, "RF": COT}, width=0.75)
    ax.set_ylabel("MAE (percentage points)"); ax.set_xlabel("Preprocessing")
    ax.set_title("Preprocessing × model — lower is better")
    ax.tick_params(axis="x", rotation=0); ax.legend(title="")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "5_benchmark.png")); plt.close(fig)

    print(f"[done] wrote metrics, model and 5 figures to '{args.outdir}/'")
    return summary


if __name__ == "__main__":
    main()
