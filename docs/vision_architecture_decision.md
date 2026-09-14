# Architecture decision — RGB fibre-blend composition regression

**Status:** design fixed, reference implementation in `src/vision/`, validated only on
the procedural simulator. No real swatch has been through this yet.

**Task.** Given an RGB macro image of a fabric swatch, predict the continuous
composition vector **y** on the (K−1)-simplex (K = 2: cotton / polyester).
This is the imaging counterpart of the NIR pipeline in `src/`, and it targets the
Phase-2 gap the README names: *blend-level accuracy from a cheap RGB sensor.*

**Constraints as given.** No dataset yet — designed against a spec, assumptions
stated below. Training and inference on **a single Kaggle GPU** (T4 16 GB / P100
16 GB, occasionally L4 or A100), 12-hour session ceiling, ~30 GPU-h/week.

### Assumptions (each one is a thing to check before trusting a number)

| # | Assumption | If it is wrong |
|---|---|---|
| A1 | Swatches are imaged at **macro scale** — a fibre is several pixels across (≈20–50 µm/px). | The signal is not in the image at all. This is the assumption to verify first, with a ruler and one test shot, before any modelling. |
| A2 | Composition is approximately **uniform across a swatch**, so every patch shares the swatch label. | The bag/MIL formulation and the consistency loss both break; you would need patch-level labels. |
| A3 | The reference label is **mass fraction**; the image observes **area fraction**. They differ by a per-fibre constant (§4). | Systematic bias, roughly linear in composition — visible as a tilted parity plot, fixable by re-fitting `c`. |
| A4 | Dye colour, weave construction and illumination are **uncorrelated with composition** in the corpus. | The model learns the confound and validates beautifully, then fails in the field. This is the single most likely way this project produces a fake result. |

---

## 1. Trade-off matrix

Scores are 1–5 (5 = best) for **this** task. The weights matter more than the
scores, and the weight column is where most of the real argument lives.

| Dimension | Weight | Why this weight |
|---|---|---|
| High-frequency / resolution preservation | **0.30** | The discriminative signal *is* sub-yarn micro-texture. Anything that low-passes early destroys it. |
| Data efficiency (small, imbalanced corpus) | **0.25** | A few thousand swatches at best, labels piled up at 100/0, 65/35, 50/50. Pretrained-weight quality dominates. |
| Orderless / stationary-texture fit | **0.20** | Fabric is a stationary texture and composition is permutation-invariant. Architectures that model *position* spend capacity on a nuisance. |
| Long-range receptive field | **0.05** | **Deliberately near-zero.** Composition is a local statistic — a 384 px window already contains hundreds of fibres. The headline selling point of ViTs and SSMs is close to worthless here. |
| Throughput / FLOPs on one Kaggle GPU | **0.10** | No hard latency target, but a 12 h ceiling and 30 h/week is a real budget. |
| Ecosystem maturity (weights, no custom CUDA) | **0.10** | Kaggle images change under you. A custom CUDA kernel that needs compiling is a recurring tax. |

| Candidate | HF res. | Data eff. | Orderless | Long-range | Speed | Maturity | **Weighted** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet-50 + GAP | 3 | 4 | 2 | 2 | 4 | 5 | 3.25 |
| **ConvNeXt-T + GAP** | 4 | 5 | 2 | 3 | 4 | 5 | **3.90** |
| Swin-T | 3 | 4 | 2 | 4 | 3 | 4 | 3.20 |
| SegFormer/MiT-B2 | 3 | 3 | 2 | 4 | 3 | 4 | 2.95 |
| ViT-B/16 (DINOv2) | **1** | 4 | 3 | 5 | 2 | 5 | 2.85 |
| VMamba-T | 3 | 2 | 2 | 5 | 3 | **2** | 2.60 |
| **ConvNeXt-T + 2nd-order pool + MIL** | 4 | 5 | **5** | 3 | 3 | 5 | **4.30** |

**Reading the matrix.** Three candidates lose on structural grounds, not on points:

- **ViT-B/16** scores 1 on resolution preservation because its patchify stem is a
  learned stride-16 downsample applied *before any feature extraction*. It throws
  away the micro-texture as its first operation. DINOv2 features are excellent for
  semantics and near-useless for a signal that lives below its patch grid.
- **VMamba / SSMs** win the dimension that carries 0.05 weight, and pay for it with
  immature weights and a scan order imposed on a material that has no canonical
  order. A selective scan is a sequence model; fabric is not a sequence.
- **Swin** is engineered for exactly the long-range interaction this task does not
  have, and its shifted windows cost throughput to deliver it.

The winner is not "CNN"; it is **CNN backbone + orderless second-order pooling**.
The last row beats the fourth row by 0.40 almost entirely on the orderless
dimension, and that gap is the actual design decision.

---

## 2. Selection and justification

**Chosen: ConvNeXt-T (ImageNet-pretrained) → multi-scale fusion → iSQRT covariance
pooling → gated-attention MIL over patches → Dirichlet head.** (`src/vision/model.py`)

Why each piece is load-bearing, specifically for textile:

1. **Native-resolution patches, never resized.** The reflex of resizing a 2048²
   swatch to 224² is a ~9× low-pass filter over exactly the band that carries the
   answer. Patches are cut at sensor resolution and the swatch is covered by a bag
   of them. If one sentence of this document survives, this is it.

2. **Second-order pooling instead of GAP.** GAP is a first-order statistic, and it
   is *provably blind to the quantity being measured*: two swatches with identical
   mean filter responses but different co-occurrence between matte-response and
   specular-response channels — i.e. different blends — produce the same GAP
   vector. Proportion estimation is a second-moment problem. The covariance of the
   fused feature map, matrix-square-rooted and vectorised, is the natural
   descriptor, and it is the same object that makes bilinear CNNs strong on
   texture benchmarks.

3. **Attention-MIL over patches, not a mean.** Real swatches have seams, prints,
   labels, folds and blown highlights. Those patches carry no composition and
   confident nonsense. A permutation-invariant learned weighting lets the model
   abstain on them, and the weights are a free per-swatch explanation map.

4. **Dirichlet head.** The output is a point on a simplex with a usable
   concentration parameter. On a sorting line, *"62 % cotton and I am not sure"*
   must route differently from *"62 % cotton, confident"* — an unrecognised
   coating, a carbon-black dye, a fibre outside the training set. The economic case
   for the sensor is avoiding misrouted bales, so a calibrated abstention is worth
   more than a fraction of a point of MAE.

**What would change the decision.** If A1 fails and imaging is at garment scale
rather than fibre scale, no amount of architecture recovers the signal and the
answer becomes a multispectral sensor, not a better network. If the corpus turns
out to be 100k+ swatches, DINOv2 features on *unresized crops* become a serious
competitor and should be re-benchmarked.

---

## 3. Frequency-domain pre-processing

`src/vision/preprocessing.py`. Two channels are appended to RGB:

**Weave-lattice notch.** A woven fabric's dominant image energy is a near-periodic
carrier set by yarn count, twist and weave — a property of the *construction*, not
the *blend*, and near-constant within a swatch, which makes it an ideal handle for
memorising swatch identity. It is also cleanly separable: it is a set of discrete
peaks in the 2-D Fourier magnitude. So: take the log-magnitude spectrum,
subtract the radial mean profile (removing the 1/f^a envelope so the lattice peaks
become the maxima rather than the DC blob), find peaks above a MAD-robust
threshold, suppress them and their conjugates with Gaussian stop-bands, and invert.
On a knit or nonwoven with no sharp lattice the threshold correctly fires on
nothing and the transform is a no-op.

**Specular-highlight map.** Cotton is a flat, convoluted, matte ribbon; PET staple
is a smooth round filament with a specular lobe. The highlight population is
therefore a direct proxy for PET area. Isolated as *local brightness excess ×
desaturation* — a specular reflection carries the illuminant's spectrum rather
than the dye's, so it is less saturated than the diffuse body colour. The
desaturation term is what makes this channel **dye-colour invariant**, which is the
whole point given A4.

Both are cheap (~2 ms/patch), run in dataloader workers, and are ablatable with one
flag. Extra stem channels are **zero-initialised**, so at step 0 the network is
numerically identical to the pretrained model and a bad front end cannot damage
the transfer before training starts.

---

## 4. Mathematical formulation

### Area vs mass — the calibration nobody mentions

An image observes visible **area** fraction; the reference method and the customer
both speak **mass** fraction. The two are related by a per-fibre constant `c_k`
(density × effective visible thickness), and the map is linear-fractional:

    a_k = (m_k / c_k) / Σ_j (m_j / c_j)          m_k = (a_k c_k) / Σ_j (a_j c_j)

For K = 2 this is a one-parameter fit against the NIR reference on real swatches.
Consequence for the code: **everything that is mixed is mixed in area space,
everything reported is reported in mass space** (`src/vision/data.py`). Skipping
this produces a composition-dependent bias that shows up as a tilted parity plot
and gets misdiagnosed as model underfitting.

### CompositionMix — a mixup that is exact, not heuristic

Area fractions are additive over disjoint area. So a bag that takes *k* of its *P*
patches from swatch A and *P−k* from swatch B has area composition **exactly**
(k/P)·a_A + (1−k/P)·a_B. Unlike ordinary mixup this is not a smoothness
heuristic — it is a correctly-labelled synthetic sample. It is also **the
imbalance fix**: real labels cluster at 100/0, 65/35, 50/50, and this synthesises
correctly-labelled bags anywhere on the simplex, including the empty regions.
(Pixel-wise alpha-blending would *not* be valid — it produces an image of no
physical fabric. The mixing has to be spatial.)

Label-Distribution Smoothing (kernel-smoothed inverse label density, `power=0.5`)
handles the residual skew.

### Loss

    L = w · [ λ₁ L_dir + λ₂ L_ait + λ₃ L_mae ] + λ₄ L_cons
        λ = (1.0, 0.3, 2.0, 0.1),  w = LDS sample weight

with ỹ = (1−ε)y + ε/K, ε = 0.01 (the vertex y = (1,0) makes both log-ratio and
Dirichlet density undefined; ε caps precision at ~0.5 pp, well inside the
reference method's own uncertainty).

- **L_dir** = −log Dir(ỹ | α), α = softplus(Wh)+1. The probabilistic backbone;
  makes the concentration a calibrated confidence rather than a decoration.
  Ramped over the first 500 steps — cold Dirichlet NLL against a near-vertex
  target pushes α to its floor and stalls the run.
- **L_ait** = ‖clr(ŷ) − clr(ỹ)‖². Squared Aitchison distance, the natural metric
  on compositional data. It is what distinguishes 99/1 from 100/0: one percentage
  point of absolute error, but the difference between "mechanically recyclable"
  and "contaminated". Absolute-error losses cannot see this.
- **L_mae** = ½‖ŷ − y‖₁. The metric the project reports (the NIR baseline is
  1.65 pp), optimised directly.
- **L_cons** = attention-weighted variance of per-patch compositions within a bag.
  Encodes A2, needs no labels, and therefore extends to semi-supervised use on
  unlabelled swatches.

---

## 5. Implementation

| File | Contents |
|---|---|
| `src/vision/preprocessing.py` | Weave notch, highlight map, channel assembly |
| `src/vision/fabric_simulator.py` | Procedural swatch renderer (testing only) |
| `src/vision/data.py` | Area↔mass, `SwatchBagDataset`, CompositionMix, LDS, sliding window |
| `src/vision/model.py` | Backbone, iSQRT covariance pooling, attention-MIL, Dirichlet head |
| `src/vision/losses.py` | The compound loss above |
| `src/vision/calibration.py` | Post-hoc Dirichlet temperature (see §7) |
| `src/vision/train.py` | AMP training loop, EMA, cosine schedule, resume, metrics |

```bash
python -m src.vision.train --smoke          # full pipeline on the simulator, no data
```

### Kaggle-specific engineering

- **AMP dtype chosen from the card**: bf16 where available (no loss scaler), fp16 +
  `GradScaler` on T4/P100 which lack bf16. Note `torch.cuda.amp.autocast` is
  deprecated since torch 2.4; the code uses `torch.amp.autocast(device_type=…)`.
- **The Newton–Schulz square root is forced to fp32** inside `SecondOrderPool`.
  It is the one op in this graph that fp16 genuinely breaks — the iteration drifts
  and diverges, and the failure surfaces much later as an unexplained NaN.
- **Checkpoint every epoch, resume from anywhere.** A 12 h ceiling means every
  serious run is a resumed run.
- **Bags are the batch**: effective batch = `batch_size × n_patches`. batch 4 × P 8
  at 384 px fits a T4. Raise P before raising batch — the covariance estimate
  improves with more patches per bag, a bigger bag batch only smooths gradients.
- **Covariance conditioning**: the covariance is estimated from N = H·W tokens at
  d = 128. At 384 px the stage-3 grid is 24×24 = 576, a comfortable 4.5:1. Drop the
  patch size and this ratio, not the loss, is what breaks first.

### Metrics

MAE / RMSE in percentage points and R² — directly comparable with the NIR table in
the README — plus **`coverage_90`**, the fraction of swatches whose true value falls
in the model's own 90 % credible interval. If that is far below 0.90 the Dirichlet
is overconfident and its uncertainty cannot gate a sorting decision, which was most
of its value.

---

## 6. First experiments, in order

1. **Verify A1 with a ruler and one photograph.** Everything else is conditional on it.
2. **Resolution ablation** — 384 px native vs the same patch downsampled 2×, 4×.
   This measures how much signal is where the design claims it is, and it is the
   cheapest possible falsification of the whole architecture.
3. **GAP vs covariance pooling**, same backbone. Isolates the main design decision.
4. **Front-end ablation** — RGB only vs +notch vs +highlight vs both.
5. **CompositionMix on/off**, measured on the *sparse* region of the label simplex
   rather than on the aggregate, which the dense modes dominate.
6. **Leave-one-fabric-family-out** validation. Random splits over-report badly here
   (A4); the honest question is whether it generalises to constructions and dyes it
   has never seen.

---

## 7. What has actually been measured (simulator only)

240 procedurally-rendered swatches, split by swatch id, 14 epochs on CPU.
**These numbers describe the simulator, not fabric.** They are reported because
two of them changed the design.

| Run | Split | MAE | RMSE | R² | coverage₉₀ |
|---|---|---:|---:|---:|---:|
| two-way split, central interval | 180 / 60 | **2.58 pp** | 3.16 pp | 0.990 | 0.70 |
| three-way split, central interval | 144 / 36 / 60 | 3.05 pp | 3.68 pp | 0.987 | 0.70 |
| three-way split, **HDI** | 144 / 36 / 60 | 3.05 pp | 3.68 pp | 0.987 | **0.93** |

Three things to read honestly out of that table:

* The MAE difference between rows 1 and 2 is **not** a regression — it is the
  price of the calibration split, 36 swatches taken out of training. On a corpus
  this small that costs ~0.5 pp. On a real corpus of a few thousand it should be
  negligible, but it is a real cost and it is why the split is a flag.
* Rows 2 and 3 are **the same trained model**. Only the interval definition
  changed. Nothing about the network improved; a mis-specified metric was
  replaced with one that can represent the answer.
* The fitted temperature came back **s = 1.0** — no scaling. Switching to the HDI
  fixed the coverage problem entirely on its own, and the whole temperature
  apparatus turned out not to be needed here. It is kept because it is one scalar,
  it provably cannot hurt accuracy, and it will be needed on real data where the
  residual structure is different.

Coverage of 0.93 against a nominal 0.90, with 1.00 on the calibration split, means
the intervals are now mildly **conservative** rather than calibrated. `s_max` was
1.002 — essentially no headroom to widen — so on this corpus the useful direction
is sharpening, which is why the search was subsequently made two-sided.

Do **not** compare 3.05 pp to the NIR baseline's 1.65 pp. The simulator was
written to contain a recoverable signal; the number mostly says the pipeline is
wired correctly.

### Finding 1 — the EMA was doing real work, and was nearly wasted

Raw weights oscillate between epochs (2.7, 7.7, 3.0, 6.0 pp) while the EMA
descends smoothly and stays. In the first version of the loop the EMA was
accumulated and checkpointed but never evaluated or shipped — the reported metric
described weights that were not the deliverable. Validation now reports both, and
selection and `best.pt` both use the EMA.

### Finding 2 — the Dirichlet is overconfident, and three fixes in a row were wrong

`coverage_90` came back at 0.70 against a nominal 0.90. What followed is worth
recording in full, because each attempted fix failed for a different reason.

**Attempt 1 — raise the NLL weight. Tested, does not work.** At λ_dir = 6.0 (6×),
MAE was identical (2.57 vs 2.58 pp) and coverage converged to the *same* 0.70,
peaking at 0.82 mid-training before decaying back. That decay is the diagnosis:
the NLL term is fit on *training* residuals, which by late training are far
smaller than held-out residuals, so the model learns a confidence describing data
it has memorised. Sharpening that fit cannot repair it. The mid-training peak is
just the model passing through the point where in-sample and out-of-sample error
briefly coincide.

**Attempt 2 — post-hoc temperature α′ = α/s, bisected on coverage. Wrong, on a
false premise.** The implementation assumed coverage is monotone increasing in
`s`. It is not. Once any component of α drops below 1 the Beta marginal stops
being unimodal and tends to point masses of weight b/(a+b) at 0 and a/(a+b) at 1.
When one of those weights exceeds 0.95 the *central* interval degenerates onto a
single endpoint: measured, α = (40, 0.5)/10⁴ gives the interval [1, 1]. Near-vertex
predictions — precisely what pure swatches produce — are exactly that case. The
unconstrained bisection ran to its 10⁴ bound and val coverage **fell**, 0.77 → 0.70.
Note the bug was not universal: a balanced α = (40, 15) widens to [0, 1] as
expected, which is why the premise looked fine until it was measured. The search is
now constrained to min(α) ≥ 1, the unimodal regime where widening really is monotone.

**Attempt 3 — the metric itself was mis-specified.** With the search correctly
constrained, the widest admissible point is α = 1 everywhere, i.e. a uniform
marginal, whose central 90% interval is [0.05, 0.95]. A pure swatch is y = (1, 0),
so **no central interval can ever cover it**. In the measured corpus 11 of 60
validation swatches sat exactly on a vertex and 24 fell outside [0.05, 0.95]:
nominal coverage was unattainable at any temperature, by construction. The fix is
the **highest-density interval** — for a J-shaped marginal the HDI reaches the
boundary and covers the vertex, which is the behaviour a confident pure-cotton
prediction should have. `evaluate` now reports HDI coverage.

The invariant that survived all three attempts, and the reason this is the right
family of fix at all: the Dirichlet mean is exactly invariant to scaling,

    (α/s) / Σ(α/s) = α / Σα

so calibration can never trade accuracy for coverage. `fit_temperature` returns an
`attainable` flag and the caller **must report a bound as a bound, not as a fit** —
if nominal coverage is unreachable inside the unimodal regime, the interval shape
or the model is at fault and no width will fix it.

`best.pt` stores `alpha_scale` beside the weights. Inference that ignores it gets
the right composition with a dishonest interval — the exact failure that matters
when the uncertainty is meant to gate a sorting decision.

**The transferable lesson:** an evidential head hands you a confidence-shaped
number for free, but not a calibrated one, and not one whose interval shape suits
labels that live on the boundary of the simplex. Coverage must be measured against
a held-out split, with an interval definition that can represent the answer, and
re-fitted on real swatches — a temperature fitted on the simulator says nothing
about fabric.
