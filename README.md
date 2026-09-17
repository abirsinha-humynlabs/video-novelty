# novelty

**Does this video tell my model anything it doesn't already know?**

Built for the physical-AI data problem: you have hundreds of hours of
egocentric footage of people doing repetitive work, and past some point more
of the same footage stops improving the model. You need to know which clips
are genuinely new — not which files are byte-identical.

```
novelty index      data/            # build two-tier signatures
novelty calibrate  --labels eval/pairs.yaml
novelty compare    a.mp4 b.mp4      # explain one pair
novelty select     --budget 200     # the 200 most-informative clips
novelty gate       new.mp4          # accept/reject at ingest
novelty report                      # self-contained HTML
```

---

## The short version of why this isn't just a vector DB

The obvious design — embed frames with CLIP, average them, cosine, threshold,
done — fails on this data in four specific ways. Each one has a fix in here.

**1. The metric saturates.** Every clip shot in one factory scores >0.9 cosine
against every other clip from that factory. A threshold of 0.95 then encodes
nothing about your data, only about how the backbone compresses grey plastic
and human hands. Move to another site and the "right" threshold moves too,
silently.
→ Fixed by **corpus whitening** (remove the shared "this is a factory" direction
before the dot product) and **percentile calibration** (report position in a
null distribution of random pairs from *your* corpus, not a raw cosine).
See [docs/04](docs/04-calibration.md).

**2. One embedding can't answer two questions.** An appearance embedding is
nearly invariant to what the hands are doing — that's what makes it good at
recognising a place and useless at recognising a task.
→ Fixed by **two independent axes**, environment and task, and a verdict that is
a quadrant rather than a number. See [docs/03](docs/03-signatures.md).

**3. "Same file" and "same content" are different questions.** Cut a 10-minute
recording in half and the two halves share zero frames but show identical work.
Re-encode a file and it shares every frame.
→ Fixed by a separate non-semantic **tier 0** (perceptual hashing with offset
voting) that answers only "did these share literal footage", kept apart from
the semantic tiers so the two claims never get blended.

**4. Dedup is the wrong objective.** Threshold dedup is order-dependent,
deletes the rare clip that happens to sit near a common one, and can't answer
the question you'll actually be asked: *which 200 of my 2000 hours should I
label first?*
→ Fixed by **facility-location coverage maximisation**, a monotone submodular
objective with a (1−1/e) greedy guarantee, whose diminishing-returns curve tells
you where to stop collecting. Dedup falls out for free. See [docs/05](docs/05-coverage-selection.md).

---

## The verdict is a quadrant, not a score

|                    | **task similar**        | **task different**            |
|--------------------|-------------------------|-------------------------------|
| **env similar**    | `REDUNDANT` — drop      | `SAME_PLACE_NEW_TASK` — *keep* |
| **env different**  | `SAME_TASK_NEW_PLACE` — *keep* | `NOVEL` — *keep*      |

plus `DUPLICATE_SOURCE` from tier 0, which overrides everything.

Collapsing this to one scalar destroys the two "keep" quadrants — precisely the
samples a physical-AI dataset is short of. That is the concrete cost of
"one embedding, one cosine, one threshold".

---

## Install

```bash
git clone <this repo> && cd video-novelty
make install          # CPU: numpy, scipy, opencv, pyyaml. No weights, no network.
make install-gpu      # adds torch + transformers for DINOv2 / V-JEPA 2
make test             # 44 tests on synthetic video, ~25s
```

Needs `ffmpeg` and `ffprobe` on `PATH`.

## Quickstart

```bash
# 1. cut some clips (or point at a directory you already have)
SRC=~/Downloads/long_recording.mp4 SRC2=~/Downloads/other_site.mp4 \
  bash scripts/prepare_eval_clips.sh

# 2. index -> calibrate -> look
novelty index data/clips --index .novelty --window 30 --hop 15
novelty calibrate --index .novelty --labels eval/pairs.yaml
novelty select --index .novelty
novelty report --index .novelty --out novelty-report.html
```

**Always calibrate before reading a score.** Uncalibrated, `compare` prints raw
numbers and says so in a note; those numbers are not comparable to anything.

On the GPU box, add `--config configs/gpu.yaml` to the index step.

---

## What it actually did on real footage

Measured on 60-second egocentric clips (1080p30), windowed at 30 s / 15 s hop,
**using the dependency-free `gist` encoder — no learned weights at all**:

- `clipA`, `clipB` — adjacent minutes of one worker bagging PVC fittings
- `clipC` — the same recording, 5 minutes later, same station
- `clipD` — a different worker, outdoor construction site, same camera rig
  (a deliberately hard negative: also egocentric, also manual labour, also
  hands-in-frame)

| pair type | `env_raw` (mean ± sd) | `task_raw` (mean ± sd) | n |
|---|---|---|---|
| windows within one clip | **+0.564 ± 0.179** | **+0.230 ± 0.167** | 24 |
| factory ↔ factory, different clip | +0.379 ± 0.126 | +0.116 ± 0.086 | 48 |
| factory ↔ construction site | −0.033 ± 0.071 | +0.078 ± 0.094 | 48 |

Both axes order correctly — within-clip > same-site > different-site. Note the
standard deviations: on the **environment** axis, factory and construction are
separated by ~4 pooled sd. On the **task** axis the distributions overlap
heavily; the ordering is right but individual pairs are not separable.

Against 96 labelled window pairs (`novelty calibrate --labels`):

```
environment: AUC=0.997  best-F1=0.979  separation=3.39 sigma
             fitted weights  env_cos=0.18  env_chamfer=0.45  env_bhat_sim=0.37
task       : AUC=0.630  best-F1=0.686  separation=0.46 sigma
             fitted weights  task_cos=0.92  task_dtw_sim=0.08  rhythm_cos=0.00  period_agree=0.00
```

The `rhythm_cos=0.00` and `period_agree=0.00` are a diagnosis, not a verdict on
those features: the 30 s windows are too short to resolve the ~4–6 s work cycles
(autocorrelation needs 5–6 repetitions). Widen the window to ~60 s and they
start contributing.

**Read that honestly.** The environment axis is strong and usable today. The
task axis, on hand-crafted optical-flow features, is only just above chance —
and even that number is inflated, because every same-task pair in the shipped
label file is also a same-environment pair, so the task axis can score by
leaking environment. Two things fix it, in this order:

1. **Record the missing quadrant.** Ten minutes of sorting and ten minutes of
   sweeping *in the same corner of the same room*. Until `SAME_PLACE_NEW_TASK`
   pairs exist in `eval/pairs.yaml`, every task-axis number here is unvalidated.
2. **Put a real video backbone on it.** `motion.clip_encoder: vjepa2` in
   `configs/gpu.yaml`. Flow histograms see "how much moved, where, how
   rhythmically"; they do not see *what is being manipulated*.

The synthetic test suite *does* vary the two factors independently
(`tests/test_end2end.py`) and the axes separate correctly there — so the
plumbing is right, and what's missing is real data in the missing quadrant.

---

## Docs

| | |
|---|---|
| [01 — Why not just cosine](docs/01-why-not-just-cosine.md) | the four failure modes, with the numbers |
| [02 — Architecture](docs/02-architecture.md) | file-by-file map, data flow, where to plug things in |
| [03 — Signatures](docs/03-signatures.md) | every feature, what it sees, why it earns its keep |
| [04 — Calibration](docs/04-calibration.md) | whitening, null models, percentiles, fitted weights |
| [05 — Coverage selection](docs/05-coverage-selection.md) | submodularity, the greedy guarantee, the knee |
| [06 — Running on GPU](docs/06-running-on-gpu.md) | DINOv2, V-JEPA 2, throughput, scaling the store |
| [07 — Tuning & failure modes](docs/07-tuning.md) | every knob, what breaks, how you'll know |

## Licence

MIT.
