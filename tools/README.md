# tools/

Scripts that regenerate the data embedded in `FibreBlend_Analyser.html`.

The analyser page carries its data inline so it can be opened from disk with no
server. That means **the page does not update when you retrain** — if you re-run
`train.py`, re-run these too or the page will quietly show stale figures.

Both need the PyTorch environment (`requirements-vision.txt`) because the
benchmark includes the CNN. Run from the repository root:

```bash
python tools/build_demo_data.py      > demo_data.json     # charts + recorded results
python tools/extract_live_model.py   > live_model.json    # the in-browser model
```

Then paste each into the matching `<script type="application/json">` block in
`FibreBlend_Analyser.html`.

## What `extract_live_model.py` does, and why it is trustworthy

The page runs the real prediction in JavaScript rather than replaying recorded
answers. It can do this because the pipeline is
`raw -> SNV -> Savitzky-Golay -> PLS`, and everything after SNV is affine, so the
whole tail collapses to one vector `A` and one scalar `B`:

    prediction = A . snv(spectrum) + B

`A` and `B` are extracted **numerically** — by evaluating the fitted sklearn model
on the zero vector and on each basis vector — rather than read out of `.coef_`.
That makes the extraction immune to how a given scikit-learn version stores PLS
internals.

The script asserts the reconstruction matches sklearn to better than `1e-9` on
real test spectra and refuses to emit a model otherwise. Measured agreement at
the time of writing: `2e-15`, i.e. floating-point noise. What runs in the browser
is not an approximation of the model — it is the model.
