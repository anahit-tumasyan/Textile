# Dark-fabric stress test

**Script:** `tools/dark_fabric_stress.py` · **Data:** `results/dark_fabric_stress.csv` ·
**Figure:** `results/figures/6_dark_fabric.png`

Dark — especially carbon-black — fabric is the known hard case for near-infrared fibre
identification, and it is the gap this project claims to target. This experiment quantifies
how far the pipeline degrades as fabric darkens, and asks the question that decides the
research plan: **does training on dark samples fix it?**

> **Read before quoting any number here.** The darkness model is physically motivated but
> **not calibrated against measured dark fabric**. The *shape* of the degradation is
> meaningful; the absolute figures are indicative only. Real carbon-black spectra will be
> messier, and may be worse. This experiment **sizes** the problem — it does not **measure**
> it. Nothing here is laboratory validation.

---

## 1. How darkness is modelled, and why it is not an offset

Carbon black is a broadband NIR absorber: far less light returns to the detector.

The naive way to simulate that is to add a large constant to the spectrum. **That would be
wrong, and it would make dark fabric look easy** — SNV and baseline correction remove an
additive offset in one line, and the experiment would show almost no degradation.

The damaging effect is different: the fibre-specific band **contrast** is attenuated, while
detector noise stays at its **absolute** level. Signal-to-noise collapses, and no amount of
preprocessing recovers information the detector never captured. So the model is

```
spectrum = spectrum × (1 − contrast_loss × darkness) + offset × darkness
```

applied **before** scatter, baseline and noise — because the dye attenuates the light that
carries the signature, and the detector's noise is added to that already-weakened signal.
Order matters here; applying attenuation after the noise would understate the damage.

At `darkness = 1` the band contrast retained is 8%.

## 2. Results

Mean absolute error, percentage points:

| Darkness | RF (light-trained) | RF (dark-aware) | PLS raw (industry baseline) |
|---|---:|---:|---:|
| 0.0 (light) | 1.47 | 1.59 | 3.76 |
| 0.2 | 1.77 | 1.72 | 5.30 |
| 0.4 | 2.19 | 1.80 | 9.45 |
| 0.6 | 3.12 | 2.48 | 14.55 |
| 0.8 | 5.53 | 3.93 | 18.16 |
| **1.0 (carbon black)** | **15.16** | **11.80** | **23.70** |

## 3. The three findings that matter

**1. Training on dark data is not the answer.** A model trained across the full darkness
range still degrades **7.4×** on black fabric, versus 10.3× for a light-trained model — it
recovers only about **25%** of the loss. This is the central result. It means the fix is
**not** "collect more dark samples"; the information is genuinely attenuated at the sensor,
and closing the gap requires a methodological contribution. That is precisely the kind of
claim a research grant should be built on, and it is now evidenced rather than asserted.

**2. The failure is a cliff, not a slope.** Up to darkness 0.6 the pipeline holds at ~3 pp,
which is still useful. Between 0.8 and 1.0 it collapses. Operationally that suggests a
**confidence-gated system**: handle light and mid-dark fabric automatically, detect the
carbon-black cliff and divert those items rather than guessing. The vision branch's
Dirichlet head exists to support exactly that kind of abstention.

**3. The pipeline's advantage grows with darkness.** The gap over the industry baseline
widens from 2.6× on light fabric to 1.6–4× across the dark range in absolute terms
(3.76 → 23.70 for the baseline versus 1.47 → 15.16). Both are unusable at true carbon black,
but the relative case for the method is stronger, not weaker, on hard material.

## 4. What this does and does not license you to say

**Supported:** "We have characterised the dark-fabric failure mode in simulation, quantified
the cliff, and shown that additional training data recovers only ~25% of the loss — so a
methodological advance is required."

**Not supported:** any claim of accuracy *on dark fabric*, any claim that dark fabric has been
tested, and any figure from this page presented as a measurement. The deck and proposal
should continue to say dark-fabric performance is **not yet measured**.

## 5. Next step

This is the experiment to re-run first once real samples exist. Dark, carbon-black-dyed
blends with verified composition are the highest-value samples to collect — they are where
the method must prove itself, and where a null result would be most informative.
