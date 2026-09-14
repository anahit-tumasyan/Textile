"""Generate the walkthrough's figures using train.py's exact dataset and split,
so no number on the page can contradict results/benchmark.csv."""
import sys, json, numpy as np
sys.path.insert(0, "src")
from spectral_simulator import make_dataset, SimConfig, simulate_blend, pure_endmembers
from preprocessing import preprocess
from model import BlendRegressor
from sklearn.model_selection import train_test_split

SEED = 42                                     # train.py --seed default
cfg = SimConfig(seed=SEED)
X, y, wl = make_dataset(1500, cfg=cfg, fraction_dist="uniform")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED)

SPECS = [
    ("baseline", "Standard method, raw signal",  "pls", "none"),
    ("cleaned",  "Standard method + cleanup",    "pls", "snv+sg1"),
    ("rf",       "Random forest",                "rf",  "snv+sg1"),
    ("cnn",      "Neural network",               "cnn", "snv+sg1"),
]

fitted = {}
for key, label, kind, pre in SPECS:
    m = BlendRegressor(kind).fit(preprocess(Xtr, pre), ytr)
    p = m.predict(preprocess(Xte, pre))
    fitted[key] = {
        "model": m, "pre": pre, "label": label,
        "mae": float(np.abs(p - yte).mean() * 100),
        "parity": [[round(float(t) * 100, 1), round(float(q) * 100, 1)]
                   for t, q in zip(yte[:300], p[:300])],
        "err": [round(float(e) * 100, 2) for e in (p - yte)],
    }
    print(f"{key:9s} {fitted[key]['mae']:.2f} pp", file=sys.stderr)

keep = np.arange(0, len(wl), 5)               # 601 -> 121 points for the web
demo_cfg = SimConfig(seed=7)
rows = []
for pct in range(0, 101, 2):
    spec = simulate_blend(pct / 100.0, cfg=demo_cfg, wl=wl)
    if isinstance(spec, tuple):
        spec = spec[0]
    spec = np.asarray(spec, float).reshape(1, -1)
    rows.append({
        "true": pct,
        # FULL resolution: the browser runs the real model on this, so it must
        # be the same 601 points the model was fitted on. The chart downsamples
        # it for display.
        "raw": [round(float(v), 4) for v in spec[0]],
        "cleaned": [round(float(v), 4) for v in preprocess(spec, "snv+sg1")[0][keep]],
        "preds": {k: round(float(f["model"].predict(preprocess(spec, f["pre"]))[0]) * 100, 2)
                  for k, f in fitted.items()},
    })

cot, pet = pure_endmembers(wl)
json.dump({
    "wavelengths": [float(w) for w in wl[keep]],
    "wavelengths_full": [float(w) for w in wl],
    "keep": [int(i) for i in keep],
    "rows": rows,
    "endmembers": {"cotton": [round(float(v), 4) for v in np.asarray(cot)[keep]],
                   "polyester": [round(float(v), 4) for v in np.asarray(pet)[keep]]},
    "models": [{"key": k, "label": l, "mae": round(fitted[k]["mae"], 2),
                "parity": fitted[k]["parity"]} for k, l, _, _ in SPECS],
    "errors": fitted["cnn"]["err"],
}, sys.stdout)
