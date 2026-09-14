"""
Dark-fabric stress test.
========================

Dark, especially carbon-black, fabric is the known hard case for NIR fibre
identification, and it is the gap this project claims to target. This script
quantifies how badly the pipeline degrades as fabric gets darker, and — the
question that actually matters for planning — whether simply *training on dark
samples* fixes it.

    If training on dark data recovers accuracy  -> the problem is DATA.
    If it does not                              -> a novel METHOD is required.

Two models are compared at every darkness level:

  * "light-trained"  — trained only on light fabric. This is the realistic
    failure mode: a model built from easy samples, then deployed on a waste
    stream that contains black garments.
  * "dark-aware"     — trained on fabric spanning the full darkness range.

READ THIS BEFORE QUOTING ANY NUMBER BELOW
-----------------------------------------
The darkness model is physically motivated (carbon black is a broadband NIR
absorber, so band contrast is attenuated while detector noise stays at its
absolute level) but it is NOT calibrated against measured dark fabric. The
*shape* of the degradation is meaningful; the absolute percentage points are
indicative only. Real carbon-black spectra may be worse, and will certainly be
messier. This sizes the problem — it does not measure it.

Usage:  python tools/dark_fabric_stress.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spectral_simulator import make_dataset, SimConfig          # noqa
from preprocessing import preprocess                             # noqa
from model import BlendRegressor                                 # noqa

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_TRAIN, N_TEST, PRE = 1200, 400, "snv+sg1"


def build(n, darkness, seed):
    cfg = SimConfig(seed=seed)
    cfg.darkness = darkness
    X, y, wl = make_dataset(n, cfg=cfg, fraction_dist="uniform")
    return X, y


def mae(model, X, y, pre):
    return float(np.abs(model.predict(preprocess(X, pre)) - y).mean() * 100)


def main() -> int:
    # --- training sets -----------------------------------------------------
    Xl, yl = build(N_TRAIN, 0.0, seed=42)            # light only
    Xd, yd = build(N_TRAIN, (0.0, 1.0), seed=43)     # full darkness range

    models = {}
    for tag, (Xt, yt) in {"light-trained": (Xl, yl), "dark-aware": (Xd, yd)}.items():
        models[(tag, "RF")] = BlendRegressor("rf").fit(preprocess(Xt, PRE), yt)
        models[(tag, "PLS")] = BlendRegressor("pls").fit(preprocess(Xt, PRE), yt)
    # the untouched industry baseline: PLS on raw spectra, light-trained
    models[("light-trained", "PLS-raw")] = BlendRegressor("pls").fit(Xl, yl)

    rows = []
    for dk in LEVELS:
        Xte, yte = build(N_TEST, dk, seed=100 + int(dk * 10))
        for (tag, kind), m in models.items():
            pre = "none" if kind == "PLS-raw" else PRE
            rows.append({"darkness": dk, "training": tag, "model": kind,
                         "MAE_pp": round(mae(m, Xte, yte, pre), 2)})

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "dark_fabric_stress.csv")
    df.to_csv(out, index=False)

    piv = df.pivot_table(index="darkness", columns=["training", "model"], values="MAE_pp")
    print("\nMean absolute error (percentage points) vs fabric darkness\n")
    print(piv.to_string())

    light = piv[("light-trained", "RF")]
    dark = piv[("dark-aware", "RF")]
    print(f"\nlight-trained RF:  {light.iloc[0]:.2f} pp (light)  ->  "
          f"{light.iloc[-1]:.2f} pp (black)   [{light.iloc[-1]/light.iloc[0]:.1f}x worse]")
    print(f"dark-aware  RF:  {dark.iloc[0]:.2f} pp (light)  ->  "
          f"{dark.iloc[-1]:.2f} pp (black)   [{dark.iloc[-1]/dark.iloc[0]:.1f}x worse]")
    recovered = 1 - (dark.iloc[-1] - light.iloc[0]) / max(light.iloc[-1] - light.iloc[0], 1e-9)
    print(f"\nTraining on dark data recovers {recovered*100:.0f}% of the loss.")
    print(f"\nwrote {os.path.normpath(out)}")

    # --- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for (tag, kind), style in {("light-trained", "RF"): ("-o", "#b5483f"),
                                   ("dark-aware", "RF"): ("-o", "#2a7f62"),
                                   ("light-trained", "PLS-raw"): ("--s", "#9a9a9a")}.items():
            ax.plot(piv.index, piv[(tag, kind)], style[0], color=style[1],
                    label=f"{kind}, {tag}")
        ax.set_xlabel("fabric darkness  (0 = light, 1 = carbon black)")
        ax.set_ylabel("mean error (percentage points)")
        ax.set_title("How dark fabric breaks the pipeline — simulated")
        ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout()
        figp = os.path.join(os.path.dirname(__file__), "..", "results", "figures",
                            "6_dark_fabric.png")
        fig.savefig(figp, dpi=150)
        print(f"wrote {os.path.normpath(figp)}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
