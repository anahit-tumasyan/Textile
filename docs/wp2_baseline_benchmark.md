# WP2 — Baseline Benchmark Report

**Deliverable for:** SkillVentory TRL 4 task document, WP2 ("Baseline Reproduction").
**Data:** 1,500 simulated cotton/polyester NIR spectra, 601 wavelengths, 1000–2200 nm.
**Split:** held-out test set, never seen during fitting.

> **Caveat, stated once and applying to every number below:** these are *simulated*
> spectra, generated from published absorption bands of cellulose and PET. No physical
> fabric has been measured. These results demonstrate that the pipeline and the
> comparison are sound; they are not laboratory validation and do not constitute TRL 4.

---

## 1. A discrepancy in the task document, and how it was resolved

WP2 asks for **accuracy, precision, recall and F1**. Those are *classification* metrics —
they score putting things into categories. WP3 asks for **MAE, RMSE and R²**, which score
*estimating a quantity*.

This project predicts a continuous percentage ("72% cotton"), so the classification metrics
do not apply to it. Reporting them would require redefining the task as e.g. "does this
contain polyester, yes/no", which is a different and much easier problem — and the
"measurable improvement over baseline" claim would then compare two incomparable things.

**Every model below is therefore scored with MAE / RMSE / R².** This should be confirmed
with the task author before WP2 is signed off.

Two of the four named baselines also needed interpreting:

| Asked for | Benchmarked as | Why |
|---|---|---|
| PCA | **PCR** (PCA + linear regression) | PCA is a decomposition, not a predictor. PCR is the standard way to benchmark it on a regression task. |
| SVM | **SVR** (support vector regression) | An SVM classifier cannot output a continuous percentage. |
| Random Forest | Random Forest | — |
| CNN | 1-D CNN over the wavelength axis | The only model here that exploits neighbouring wavelengths being related. |

An **MLP** (plain neural network) was added as a control, to separate "neural networks help"
from "*this* neural architecture helps".

---

## 2. Results

Sorted best to worst. "pp" = percentage points of cotton content.

| Model | Preprocessing | MAE ↓ | RMSE | R² | Worst single error |
|---|---|---:|---:|---:|---:|
| **CNN** | snv+sg1 | **1.30 pp** | 1.66 | 0.997 | 6.1 pp |
| Random Forest | snv+sg1 | 1.65 pp | 2.05 | 0.995 | 6.7 pp |
| SVR | snv+sg1 | 1.95 pp | 2.76 | 0.991 | **12.9 pp** |
| PLS | snv+sg1 | 1.97 pp | 2.50 | 0.993 | 8.8 pp |
| PLS | snv | 2.12 pp | 2.67 | 0.992 | 10.7 pp |
| PCR | snv+sg1 | 2.41 pp | 3.04 | 0.989 | 8.8 pp |
| **PLS (baseline)** | none | **3.38 pp** | 4.46 | 0.977 | 14.7 pp |
| PLS | sg1 | 3.49 pp | 4.55 | 0.976 | 14.1 pp |
| MLP | snv+sg1 | 13.41 pp | 17.36 | 0.648 | 58.1 pp |

**Headline:** best model 1.30 pp vs 3.38 pp baseline — a **2.6× reduction in error**.

---

## 3. What the numbers actually say

**Preprocessing matters as much as the model.** PLS alone goes from 3.38 pp to 1.97 pp
purely from signal conditioning — a larger gain than most model swaps produce. Any claim
about a model's superiority must hold preprocessing constant, which is why every non-PLS
model here runs on the same `snv+sg1`.

**Rank on MAE alone and you will pick the wrong model.** SVR beats PLS on average
(1.95 vs 1.97 pp) while being **47% worse in the worst case** (12.9 vs 8.8 pp). On a sorting
line a single 13-point error misroutes a bale; the average error never sees that event.
Worst-case error belongs in the acceptance criteria alongside MAE.

**The MLP result is real, and it is informative.** 13.41 pp is not a bug — a plain neural
network with ~1,200 training samples of 601 dimensions has too little structure to exploit
and too many parameters. It is evidence that *architecture* carries the CNN's advantage, not
"deep learning" in general. It should stay in the table; deleting an unflattering baseline is
how benchmarks become marketing.

**The CNN's advantage is modest and unproven.** 1.30 vs 1.65 pp is a real gap on this data,
but the CNN has by far the most capacity to exploit quirks of the *simulator*. Until real
spectra exist, treat the CNN's lead as provisional — it is exactly the kind of result that
shrinks or reverses on real material.

---

## 4. Reproducing this

```bash
pip install -r requirements.txt
python train.py --n 1500 --outdir results     # everything except the CNN
```

The CNN additionally needs PyTorch (`pip install -r requirements-vision.txt`) on a Python
version PyTorch supports — 3.12 at the time of writing, **not 3.14**. Without it the CNN row
is skipped with a clear message and the rest of the benchmark runs normally.

**Two saved models are written:**

- `results/best_model.joblib` — best overall. When the CNN wins, **this file requires torch
  to load.**
- `results/best_model_portable.joblib` — best model that loads with the base requirements
  only. `predict.py` and any torch-free tooling should use this one.

**Before submission, re-run the whole benchmark in a single environment.** The results above
were produced across two Python environments (3.14 for the classical models, 3.12 for the
CNN), which leaves a harmless scikit-learn version warning on unpickling. One environment
removes it.

---

## 5. Status against the WP2 deliverable

| WP2 requirement | Status |
|---|---|
| Implement PCA baseline | Done (as PCR) |
| Implement SVM baseline | Done (as SVR) |
| Implement Random Forest baseline | Done |
| Implement CNN baseline | Done |
| Produce benchmark metrics | Done — as MAE/RMSE/R², see §1 |
| Baseline Benchmark Report | This document |

**Not addressed by WP2, and still the binding constraint:** every figure here is from
simulated data. The success criteria "verified laboratory dataset" and "independent
validation" remain open, and no amount of additional modelling closes them.
