# 04 — Calibration

> **If you read one page in these docs, read this one.** Every number this repo
> prints is meaningless without the step described here, and the failure is
> silent — you get plausible scores that encode nothing.

Calibration happens in two places, and they fix different things.

---

## Stage 1 — whitening (feature space)

`novelty.calibrate.Whitener`, fitted by `fit_whitener(signatures)`.

Any embedding of a homogeneous corpus has a large shared component: the
"this is a factory, shot on a head camera, in daylight" direction that every
single clip contains. It contributes nothing to telling clips apart and
dominates the dot product. Subtract the corpus mean, divide by the
per-dimension standard deviation, renormalise.

Fitted per feature block, because they live in different spaces:

| block | fitted from | treatment |
|---|---|---|
| `app` | all frame vectors, concatenated | centre + scale |
| `motion_pooled` | one vector per signature | centre + scale |
| `motion_seq` | all per-frame descriptors | centre + scale |
| `rhythm` | `sqrt` of each spectrum | **centre only** |

`rhythm` is centred but not scaled because it is a probability distribution;
rescaling per band destroys the distributional meaning.

### Shrinkage

With few samples relative to dimensions, some dimensions have a near-zero sample
standard deviation purely by luck. Dividing by it explodes that dimension and
the whitened cosine becomes noise. `_shrunk_sd` shrinks toward the average
variance with weight `n / (n + 40)` — the cheap diagonal analogue of
Ledoit-Wolf. It makes the whitener safe to fit on a small index and converges to
the plain estimate as the corpus grows.

### How much it matters

Measured on the real footage in this repo:

```
                              before whitening    after whitening
env_cos  factory ↔ factory        0.92 – 0.94        0.12 – 0.26
env_cos  factory ↔ construction   0.75               −0.41 – −0.51
task_cos every pair               0.995 – 0.998      −0.19 – +0.28
```

---

## Stage 2 — the null model (score space)

`novelty.calibrate.NullModel`, fitted by `fit_null(signatures)`.

Sample up to `--max-pairs` random pairs from the corpus, compute every raw
metric on each, and store the empirical quantiles. Thereafter, report a pair's
**percentile** rather than its raw value.

```python
null.percentile("env_raw", 0.53)   # -> 98.4
null.z("env_raw", 0.53)            # -> 4.1  (sigma above chance)
```

"98.4th percentile" means *closer than 98.4% of random pairs in this dataset*.
That statement transfers across sites, backbones and resolutions. A cosine of
0.94 does not.

This is why `decision.env_percentile: 97.0` is a percentile and not a cosine.
Read it as "flag the top 3% most-similar pairs as same-environment".

### Two ways to get this wrong

**Letting windows of one recording into the null.** Two 30-second windows of the
same 10-minute file are not a "random pair". Including them inflates the null,
which then makes genuinely similar pairs look ordinary. `exclude_same_video=True`
is the default; leave it on.

**Fitting the null on a corpus that isn't representative.** This one bit during
development of this very repo. With 12 signatures of which 9 were factory
footage, the null distribution *was* the positive distribution — factory-vs-
factory pairs scored at the 50th percentile, because they were the median pair.
The percentile transform is only meaningful if the null describes "an arbitrary
pair from the data you will actually see".

`novelty calibrate` prints a warning below 50 signatures. Take it seriously:
aim for several hundred. `--window` is how you get there from a handful of long
files.

---

## Stage 3 — supervised weights (optional but worth it)

The fusion weights in `metrics/fuse.py` (`ENV_WEIGHTS`, `TASK_WEIGHTS`) are the
only hand-set numbers in the pipeline. Give `calibrate` some labelled pairs and
it replaces them with a fit:

```bash
novelty calibrate --index .novelty --labels eval/pairs.yaml
```

```yaml
pairs:
  - {a: clipA.mp4, b: clipB.mp4, same_env: true,  same_task: true}
  - {a: clipA.mp4, b: clipD.mp4, same_env: false, same_task: false}
```

A label is attached to a **file**, but the index holds one signature per
**window**, so each labelled file pair is expanded into every cross-window pair.
That is both more data and more honest: it measures the property the label
actually asserts, rather than whichever single window got looked up first.
(Before this expansion the label evaluation ran on one window per file and
reported a task AUC of 0.22 — worse than chance — purely from sampling noise.)

Logistic regression per axis, coefficients clipped at zero and renormalised to
sum to 1. Negative coefficients are dropped rather than kept: a component that
anti-correlates with the label on a handful of pairs is almost always noise, and
letting it subtract makes the fused score unstable on unlabelled data. The fit
is refused outright when there are fewer than `2 × n_components` labelled pairs.

After fitting, **the null is automatically refitted** under the new weights —
otherwise the stored percentiles refer to a scale that no longer exists.

### What the output tells you

```
environment: fitted weights env_cos=0.18 env_chamfer=0.45 env_bhat_sim=0.37
environment: AUC=0.997  best-F1=0.979 @ percentile>=52.6  separation=3.39 sigma  (n+=48, n-=48)
task       : fitted weights task_cos=0.92 task_dtw_sim=0.08 rhythm_cos=0.00 period_agree=0.00
task       : AUC=0.627  best-F1=0.681 @ percentile>=6.3   separation=0.45 sigma  (n+=48, n-=48)
```

- **AUC** is the probability a random positive pair outranks a random negative
  pair. 0.5 is chance.
- **separation** is Cohen's d between the two label groups. Below ~1σ, no
  threshold will work well no matter where you put it.
- **fitted weights of 0.00** mean that component contributed nothing on your
  labels. Here, rhythm and period contributed nothing — because the windows
  (30 s) were too short to resolve the ~6 s cycles. That is a diagnosis, not a
  verdict on the feature.

Run this **before** you trust the pipeline on a million clips. It is the only
step that tells you whether your features can do the job at all.

### Confounded labels give you a number that means nothing

The shipped `eval/pairs.yaml` covers only two quadrants — every same-task pair
in it is also a same-environment pair. A model that ignores motion entirely
would still score well on the task axis, by leaking environment. Until you add
`same_env: true, same_task: false` pairs, treat every task-axis number as
unvalidated. Record ten minutes of two different jobs in the same corner of the
same room; that one recording session is worth more than any amount of tuning.

→ Next: [05 — Coverage selection](05-coverage-selection.md)
