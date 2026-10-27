# Patient data

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Architecture](architecture.md) · [Models](models/index.md) · [References](references.md)

The analytical forward models were fitted to what implant users reported.
pulse2percept ships that psychophysics. This page lists it, says what each
set can and cannot constrain, and fits the axon map to the phosphene drawings
of three Argus II users.

Source: `src/sentionaut/patients.py` · CLI: `sentionaut-fit-drawings`

## The datasets

| Dataset | Implant, subjects | What was measured | Size | Constrains |
| --- | --- | --- | --- | --- |
| `beyeler2019` (Beyeler et al. 2019) | Argus I (S1), Argus II (S2–S4) | Drawings of the phosphene from single-electrode stimulation at 2× threshold, 20 Hz, 0.45 ms; array position and rotation per subject | 400 drawings, 80 electrodes (69 MB, OSF) | axon map `rho`, `axlambda` |
| `horsager2009` (Horsager et al. 2009) | Argus I, S05, S06 | Detection thresholds (500 rows) and brightness matches (108) over pulse width, frequency and train type | 608 rows | temporal model |
| `nanduri2012` (Nanduri et al. 2012) | Argus I, S06 | Brightness ratings (88) and phosphene size (40) against amplitude (× threshold) and frequency | 128 rows | brightness and size effects |
| `greenwald2009` (Greenwald et al. 2009) | Argus I, S05, S06 | Brightness ratings against amplitude in µA, with each electrode's threshold | 83 rows | brightness growth |
| `perezfornos2012` (Pérez Fornos et al. 2012) | Argus II, 9 subjects | Brightness reported over a 75 s pulse train (perceptual fading) | 45 traces | fading time course |
| `han2021` (Han et al. 2021) | none | 20 outdoor videos used with sighted participants viewing simulated prosthetic vision | 318 MB | encoder inputs only; no percept data |

```python
from sentionaut.patients import load_beyeler2019

df = load_beyeler2019("data/p2p")   # downloads from OSF once, checks SHA-256
```

`download()` keeps TLS certificate checking on. pulse2percept's own
`fetch_url` switches it off; the files are the same and the checksums match.

**What this data cannot do.** It is too small to train a world model: 400
single-electrode drawings from four people, one stimulus setting. It can fit
the handful of physical parameters a forward model exposes, and it can test
whether a forward model predicts held-out electrodes. That is what the fit
below does. Everything multi-electrode, every pulse train beyond the fading
traces, and every cortical model is still unconstrained by in-tree data.

## Fitting the axon map to drawings

Beyeler et al. fitted `rho` (phosphene spread) and `axlambda` (streak length
along the nerve fibre bundles) by matching phosphene **shape**: area,
orientation and elongation. `fit_drawings` does the same by gradient descent
through `BiphasicAxonMapTorch`, which the grid search in `calibrate.py`
cannot:

1. Build the axon map over the subject's drawing field (1° grid) with the
   subject's array position and rotation; the electrode positions are checked
   against pulse2percept's `ArgusII`.
2. For each electrode, average the drawings' shape descriptors over trials
   (image moments; orientation is averaged on the doubled angle).
3. Predict each electrode's percept at 2× threshold, 20 Hz, 0.45 ms,
   normalise it by its peak and soft-threshold it at a fitted level, the
   level a patient would trace.
4. Minimise `(Δ log area)² + (Δ elongation)² + elongation · (1 − cos 2Δθ)` over
   `rho`, `axlambda` and the threshold with Adam, on half the electrodes.
5. Report the same loss on the **other half**, and the loss with the default
   `rho = 200 µm`, `axlambda = 500 µm` and only the threshold fitted.

Drawings are compared by shape, not position, because the recorded drawing
positions are not reliable enough to fit (pulse2percept's own plotting
aligns each drawing on its electrode). The optic disc is at the default
location for all subjects; Beyeler et al. measured it per subject, so the
streak orientation is only as good as that default.

### Fit results (Mila, L40S, commit `d25327f`, 150 Adam iterations)

The fit did not work: `rho` and `axlambda` came back NaN for all three
subjects. Only the threshold was fitted to a finite value, so the "fitted"
test losses below come from a model with NaN physics parameters and are not
a valid comparison with the defaults. Raw output:
`docs/assets/patient-data/drawing_fits.json`.

| Subject | Fitted `rho` (µm) | Fitted `axlambda` (µm) | Threshold | Test loss | Default test loss | Default terms (log area / elongation / orientation) | Train / test electrodes |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| S2 | NaN | NaN | 0.269 | 0.555 | 0.580 | 0.162 / 0.057 / 0.360 | 11 / 11 |
| S3 | NaN | NaN | 0.343 | 1.678 | 1.309 | 0.449 / 0.221 / 0.638 | 9 / 9 |
| S4 | NaN | NaN | 0.246 | 1.440 | 1.477 | 0.561 / 0.148 / 0.768 | 14 / 14 |
