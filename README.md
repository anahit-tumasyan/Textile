# FibreBlend — Quantitative NIR Blend-Composition Prototype

**A Phase-0 feasibility prototype for automated, *quantitative* textile fibre-composition identification.**
Predicts the **percentage of each fibre in a cotton/polyester blend** (e.g. 62 % cotton / 38 % polyester) from a near-infrared (NIR) spectrum — not just a pure-fibre class label.

Built as the concrete technical basis for a research funding application under the SkillVentory textile-recycling research direction.

---

## 1. Why this exists (the problem, in one paragraph)

Over **92 million tonnes** of textile waste are generated globally each year and **less than 1 %** is recycled into new garments (Ellen MacArthur Foundation). In the EU (~12.6 Mt/yr), new Extended-Producer-Responsibility rules are forcing the industry toward circularity. The core technical blocker to recycling is **automated fibre-composition identification** — knowing what a scrap is actually made of so it can be routed to the right process. Labels are unreliable, so identification must be sensor-based. Pure fibres are largely solved; the **open frontier is *blended* fabrics** (polyester–cotton and beyond), where spectral features overlap and the *quantitative* composition — the actual percentages — is still an unsolved problem. **That open gap is what this prototype targets.**

## 2. What this prototype demonstrates

This is the **Phase 0** step defined in the research brief: *reproduce a credible baseline on data we control, quantify the failure mode, and establish the SOTA bar a novel method must beat — before investing in months of sample collection.*

It shows, end-to-end, that:

1. A physically-realistic NIR blend spectrum can be generated from the **published absorption bands** of cellulose (cotton) and PET (polyester), including the overlapping regions (~1120 nm, ~1660 nm) that make blends hard.
2. Standard chemometric preprocessing (**SNV + Savitzky–Golay derivative**) recovers the composition signal from realistic sensor nuisances (scatter, sloping baselines, wavelength drift, detector noise, fabric-to-fabric band jitter, colour/dye tilt).
3. A regression model predicts the **continuous cotton fraction** across the full 0–100 % range, and a non-linear model **measurably beats** the linear PLS baseline.

### Headline result (held-out test set, simulated data)

| Model | Preprocessing | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| **PLS (baseline, raw spectra)** | none | **3.38 pp** | 4.46 pp | 0.977 |
| PLS | SNV + SG-1 | 1.97 pp | 2.50 pp | 0.993 |
| **Random Forest (best)** | SNV + SG-1 | **1.65 pp** | 2.05 pp | 0.995 |

*pp = percentage points of cotton content. Preprocessing + a non-linear model cut baseline error by roughly half.* These absolute numbers are on **simulated** spectra and will differ on real fabric — the point of Phase 0 is the working pipeline and the **relative** baseline-vs-method gap, not the exact figure.

## 3. Honesty box (read this before showing reviewers)

- **The data here is simulated, not measured.** Labelled NIR datasets of *quantitative blends* are scarce and fragmented — that scarcity is itself the Phase-1 bottleneck. The simulator is grounded in the real, published absorption-band positions of cotton and polyester, but it is a **stand-in** to de-risk the method, not a substitute for real spectra.
- **This is not yet a novel contribution.** Running a regressor on NIR spectra to separate cotton from polyester is established. The genuine research (Phase 2 of the brief) is the *method* that closes a specific open gap — e.g. quantitative accuracy on **dark / carbon-black** blends, or blend-level accuracy from a **cheap RGB + few-band sensor**. This prototype gives that research a running baseline to beat.
- **What makes it grant-credible:** it proves the team can build and honestly evaluate the full pipeline, it names the exact SOTA bar, and it maps cleanly onto the staged TRL plan below.

## 4. How it maps to the funded plan

| Brief phase | This repo | Next |
|---|---|---|
| **Phase 0 — Baseline & gap** | ✅ **Done here.** Baseline + quantified difficulty. | — |
| **Phase 1 — Data (the moat)** | Simulator + references to real datasets (below). | Collect real blend + dark-fabric spectra with verified composition. |
| **Phase 2 — Method** | Swap the simulator for real data; the regressor is the baseline to beat. | Develop the novel method against one chosen gap. |
| **Phase 3 — Validation (TRL 4)** | Same evaluation harness (`train.py`), real held-out data. | Demonstrate measured improvement over SOTA. |
| **Phase 4 — Relevant env (TRL 5–6)** | — | Test on a real waste stream with a recycler / sensor partner. |

**Real datasets to seed Phase 1** (cite these in the application):
- **NIST NIR-SORT** — Near-Infrared Spectra of Origin-defined and Real-world Textiles (custom blends included).
- **OpenTextile-NIR** — NIR hyperspectral + RGB dataset of post-industrial textiles (Finland), >6 M annotated spectra.
- **University of Tartu** spectroscopy group — natural local collaborator for the sensing side (existing connection noted in the brief).

## 5. Repository layout

```
fibreblend/
├── README.md                 ← this file
├── requirements.txt
├── check_setup.py            ← RUN FIRST: verifies environment + smoke test
├── build_report.py           ← generates the plain-language HTML report
├── FibreBlend_Report.html    ← self-contained visual summary for non-technical readers
├── train.py                  ← end-to-end experiment: data → benchmark → figures → model
├── predict.py                ← demo: load model, predict composition of "unknown" garments
├── tests/
│   └── test_pipeline.py      ← scientific correctness tests
├── src/
│   ├── spectral_simulator.py ← physically-grounded NIR blend generator
│   ├── preprocessing.py      ← SNV, Savitzky–Golay derivative
│   └── model.py              ← PLS baseline + Random Forest, clipped to valid fractions
└── results/
    ├── metrics.json          ← headline metrics
    ├── benchmark.csv         ← full preprocessing × model grid
    ├── best_model.joblib     ← saved trained model
    └── figures/              ← 5 publication-ready figures
```

## 6. Run it

Requires **Python 3.9+**. From inside the `fibreblend/` folder:

```bash
pip install -r requirements.txt

# 1. verify this machine can run everything (do this first — takes ~10s)
python check_setup.py

# 2. reproduce the whole study (data, benchmark, model, figures)
python train.py --n 1500 --outdir results

# 3. demo predictions on fresh unseen "garments"
python predict.py --model results/best_model.joblib --n 8

# 4. optional: scientific correctness tests
python tests/test_pipeline.py

# 5. build the plain-language HTML report (for non-technical readers)
python build_report.py     # -> FibreBlend_Report.html, open by double-clicking
```

If `python` is not found, use `python3`. Fully reproducible (fixed seeds);
total runtime is a few seconds. `check_setup.py` reports exactly what is
missing and how to fix it, so no step should fail silently.

### Test suite

`tests/test_pipeline.py` asserts the claims this prototype rests on:

| Test | What it guarantees |
|---|---|
| endmember peak positions | simulated spectra match *published* cotton/PET absorption bands |
| blend interpolation | blends obey Beer–Lambert mixing between pure fibres |
| SNV normalisation | preprocessing maths is correct (zero mean, unit variance) |
| valid fractions | predictions are never <0 % or >100 % |
| beats naive baseline | model is far better than guessing the mean |
| preprocessing helps | SNV+derivative measurably improves accuracy under scatter |
| reproducibility | fixed seed ⇒ identical results |

## 7. Figures (in `results/figures/`)

1. **`1_endmembers.png`** — pure cotton vs polyester NIR signatures, from published bands.
2. **`2_blend_spectra.png`** — simulated measured spectra across 0–100 % cotton, with realistic sensor noise.
3. **`3_parity.png`** — predicted vs true cotton % on held-out data (the money shot).
4. **`4_error_hist.png`** — error distribution in percentage points.
5. **`5_benchmark.png`** — preprocessing × model comparison.

---

*Prototype prepared for a research project competition. Scientific grounding: published NIR absorption bands of cellulose and PET; baseline/methodology follows the standard chemometric approach (PLS + spectral preprocessing) used across the textile-NIR literature.*
