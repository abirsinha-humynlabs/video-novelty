# WORK.md — handoff for the next agent

**Read this whole file before touching anything.** It is the state of the project,
what was measured, what is broken, and what to do next. The `docs/` directory
explains *how the code works*; this file explains *where the project is*.

Last updated: 2026-09-17 · commit `a301d95` · repo `abirsinha-humynlabs/video-novelty`

---

## 1. The 60-second version

**Goal:** given hundreds of hours of egocentric video of people doing repetitive
manual work, decide which clips add new information to a training corpus for a
physical-AI / VLA model, and which are redundant. "Redundant" must mean
*semantically* redundant (same place, same work), not byte-identical.

**What exists:** a working Python package, `novelty`, that builds three-tier
signatures per video span, calibrates similarity against the user's own corpus,
and selects a maximally-diverse subset by submodular coverage maximisation.
44 tests pass. Pushed to GitHub. Runs on CPU with zero learned weights; has a
GPU path wired but **never executed** (no torch in the build environment).

**Where it stands:**

| tier | what it answers | status |
|---|---|---|
| 0 — perceptual hash | did these share literal footage? | **works**, tested |
| 1 — appearance | same physical place? | **works**, AUC 0.997 on real data |
| 2 — motion/task | same work happening? | **barely above chance**, AUC 0.630 |

**The single most important fact for you:** the task tier is unvalidated and
probably not good enough, and *you cannot fix it by writing code*, because the
user has no data that can distinguish a working task-detector from a broken one.
See §6.

---

## 2. Who this is for, and how to talk to them

Abir Sinha (abir.sinha@humynlabs.ai, KGeN). Building physical-AI training data.
His stated user preferences are explicit and enforced: **lead with the
uncomfortable answer, never open with agreement, tag claims [Certain]/[Likely]/
[Guessing], disagree with structure, do not fold under pushback without new
information.** He responds well to being told his plan is wrong when it is, with
the reason and an alternative. Do not soften negative results — the negative
result in §5 is the most valuable thing produced so far.

His original proposal was "visual embeddings in a vector DB, match like RAG".
That was pushed back on and partially rejected; see §4 for what replaced it and
why. Do not silently revert to it.

---

## 3. The data

All on the user's Mac (`abirs-macbook-pro-2-local`), in `~/Downloads`. The
session had folder access granted to `~/Downloads` only.

### Source recordings

| file | duration | content |
|---|---|---|
| `left_rectified(5).mp4` | 598.8 s, 1920×1080, 30 fps, HEVC, no audio | **egocentric (head-mounted), PVC pipe-fitting factory.** A worker sorting grey/white plastic fittings and bagging them into white woven sacks. Hands and the wearer's own legs/torso visible in the lower third throughout. Highly repetitive. |
| `front_left_rectified.mp4` | 598.7 s, 1920×1080, 30 fps | **egocentric, outdoor construction site.** Different worker, blue jeans, sandals, tying rebar / handling concrete blocks on gravel. Same camera rig and same rectification pipeline. |
| `left_rectified(4).mp4` | 600.1 s | not inspected |
| `left_rectified(3).mp4` | ~? | not inspected |
| `left_rectified(2).mp4` | 42.8 s | not inspected |
| `left_rectified(1).mp4` | ~? | not inspected |
| `left_rectified.mp4` | 99.0 s | **mostly black frames** at t=30 s. Probably lens-cap / junk footage. Do not use without checking. |

`front_left_rectified.mp4` is the **hard negative** and it is genuinely good:
same rig, same modality, same "manual labour with hands in frame", completely
different environment and task. Any method that cannot separate these two is
worthless. Keep using it.

### Derived eval clips — `~/Downloads/simrepo_clips/`

Cut with ffmpeg, re-encoded H.264 CRF 23, 1080p30, 60 s each:

| clip | source | span | role |
|---|---|---|---|
| `clipA_040-100.mp4` | `left_rectified(5)` | 0:40–1:40 | the user's requested first half |
| `clipB_100-160.mp4` | `left_rectified(5)` | 1:40–2:40 | the user's requested second half |
| `clipC_v5_400-460.mp4` | `left_rectified(5)` | 6:40–7:40 | **added.** Same place, 5 min later, slightly different phase of the work |
| `clipD_front_040-100.mp4` | `front_left_rectified` | 0:40–1:40 | **added.** The hard negative |

Thumbnails at `~/Downloads/simrepo_clips/thumbs/`.

> The user originally asked only for A and B. **Two halves of one continuous
> recording is the easiest possible positive** — mean pixel colour passes it. It
> proves nothing about a threshold. C and D were added so the eval set has
> something falsifiable in it. Regenerate all four with
> `scripts/prepare_eval_clips.sh`.

---

## 4. What was built, and the reasoning behind each choice

Package `novelty`. Full design rationale is in `docs/01`–`docs/07`; this section
is the *why we chose this over the obvious thing*, so you do not relitigate it.

### 4.1 Three tiers, deliberately not blended

```
tier 0  encoders/hashing.py     64-bit DCT pHash @2fps + Hough vote over time offsets
tier 1  encoders/appearance.py  gist (no weights) | dinov2 | dinov2-large | siglip
tier 2  encoders/motion.py      dense flow -> ego/object split -> histogram + rhythm
        encoders/video.py       (GPU, optional) V-JEPA 2 / VideoMAE clip embeddings
```

**Why tier 0 is separate:** "same file" and "same content" are different
questions and an embedding gives the same answer to both.
`video.mp4[0:300]` vs `video.mp4[300:600]` share zero frames but show identical
work; `video.mp4` vs `video_720p.mp4` share every frame. Tier 0 answers only the
second. There is a test asserting tier 0 does **not** fire on disjoint halves —
that non-firing is the feature.

**Why two semantic axes and not one:** an appearance embedding is near-invariant
to what the hands are doing. That invariance is why DINOv2 is good at place
recognition and why no appearance threshold will ever separate "bagging fittings"
from "sweeping the floor" in the same room. So a comparison returns a quadrant:

|                   | task similar              | task different                   |
|-------------------|---------------------------|----------------------------------|
| **env similar**   | `REDUNDANT` → drop        | `SAME_PLACE_NEW_TASK` → **keep** |
| **env different** | `SAME_TASK_NEW_PLACE` → **keep** | `NOVEL` → **keep**        |

The two "keep" quadrants are the samples a physical-AI dataset is short of. A
single fused scalar deletes both. **Do not collapse this to one number.**

### 4.2 Calibration — the part that makes numbers mean anything

Two stages, both in `calibrate.py`:

1. **Whitening** (feature space). Subtract corpus mean, divide by per-dimension
   sd, per feature block, with Ledoit-Wolf-style shrinkage `n/(n+40)` so it is
   safe on small corpora. `rhythm` is centred but **not** scaled (it is a
   probability distribution; scaling per band destroys the meaning).
2. **Null model** (score space). Sample random corpus pairs, store empirical
   quantiles, report percentiles thereafter. `decision.env_percentile: 97` means
   "top 3% most similar pairs in *your* corpus", not "cosine > 0.97".

`exclude_same_video=True` by default when fitting the null: two windows of one
recording are not a random pair, and letting them in inflates the null.

3. **Optional supervised weights.** `calibrate --labels eval/pairs.yaml` fits a
   logistic regression per axis, clips negative coefficients to zero,
   renormalises, then **refits the null** under the new weights (otherwise the
   stored percentiles refer to a scale that no longer exists).

### 4.3 Coverage selection instead of dedup

`select.py` implements lazy-greedy facility location:
`f(S) = Σ_v max_{s∈S} sim(v,s)`. Monotone submodular, so greedy is within
(1−1/e) of optimum.

Threshold dedup was rejected because it is order-dependent, it deletes the rare
clip that happens to sit near a common one, and it cannot answer the question
the user will actually be asked ("which 200 of my 2000 hours do I label first").
The `knee()` of the marginal-gain curve is the direct, data-derived answer to
"monotonous task data stops helping — when?".

### 4.4 Egocentric body mask

`appearance.static_weight_map`. On a head camera the wearer's torso/legs occupy a
large **constant** region of every frame they will ever shoot. Two unrelated
tasks by the same person share it exactly, so part of any similarity score is
trouser recognition. The mask estimates the region from per-pixel temporal sd
across the sampled frames and down-weights the bottom 20th percentile to 0.15.

**This is an egocentric fix. On a tripod it will suppress the whole scene —
set `static_mask: false` for fixed-camera footage.**

---

## 5. Measured results (real footage, reproducible)

Setup: the four clips above, `--window 30 --hop 15` → 16 signatures.
Appearance encoder: **`gist`, the dependency-free one. No learned weights.**

### Raw fused scores, default weights

| pair type | `env_raw` mean ± sd | `task_raw` mean ± sd | n |
|---|---|---|---|
| windows within one clip | **+0.564 ± 0.179** | **+0.230 ± 0.167** | 24 |
| factory ↔ factory, different clip | +0.379 ± 0.126 | +0.116 ± 0.086 | 48 |
| factory ↔ construction | −0.033 ± 0.071 | +0.078 ± 0.094 | 48 |

Both axes order correctly (within-clip > same-site > different-site). On the
environment axis factory and construction are ~4 pooled sd apart. On the task
axis the distributions **overlap heavily** — the ordering is right, individual
pairs are not separable.

### Supervised evaluation, 96 labelled window pairs

```
environment: AUC=0.997  best-F1=0.979 @ pct>=52.6  separation=3.39 sigma
             fitted weights  env_cos=0.18  env_chamfer=0.45  env_bhat_sim=0.37
task       : AUC=0.630  best-F1=0.686 @ pct>=7.1   separation=0.46 sigma
             fitted weights  task_cos=0.92  task_dtw_sim=0.08  rhythm_cos=0.00  period_agree=0.00
```

### Coverage selection

On the 16 windows, greedy picked one window from **each of the four distinct
clips** before taking a second from any, then saturated: picks 10–16 added
exactly zero. 44% of that footage carries no information the rest doesn't.

### How to read these honestly

- The environment tier is **usable today**, with no GPU and no downloads.
- The task tier is **not**. AUC 0.630 with 0.46σ separation is not a detector.
- **And even 0.630 is inflated.** Every same-task pair in `eval/pairs.yaml` is
  also a same-environment pair, so the task axis can score by leaking
  environment. The true task-only number is unknown and could be ~0.5.
- `rhythm_cos=0.00` / `period_agree=0.00` in the fitted weights is a **diagnosis,
  not a verdict on those features**: 30 s windows cannot resolve a 4–6 s cycle
  (autocorrelation needs 5–6 repetitions). Widen the window to ≥60 s and retest
  before concluding rhythm is useless.

---

## 6. The blocker. Read this twice.

**[Certain] There is currently no way to tell whether the task tier works,
because no pair in the dataset differs on task while holding environment fixed.**

The eval set covers two of four quadrants:

```
                 same_task    diff_task
  same_env    |  YES (3)    |  MISSING   <-- the one that matters
  diff_env    |  MISSING    |  YES (3)
```

A model that ignores motion entirely would score well on the current labels.
No amount of feature engineering, backbone swapping or weight tuning can be
validated against a confounded label set.

**The fix is a recording session, not a commit.** Ask the user for:

> Ten minutes of task A and ten minutes of task B, **in the same corner of the
> same room, same camera, same lighting, same person**. E.g. bagging fittings vs
> sweeping the floor. Then label them `same_env: true, same_task: false` in
> `eval/pairs.yaml`.

Ideally also the mirror: the *same* task in two different rooms
(`same_env: false, same_task: true`).

Until that exists, every task-axis number in this repo is unvalidated, and you
should say so plainly rather than reporting improvements.

The synthetic suite (`tests/test_end2end.py`) *does* vary scene and cadence
independently and the axes separate correctly there — so the **plumbing is
verified correct**; what is missing is real data in the missing quadrant.

---

## 7. Bugs found and fixed. Do not reintroduce these.

All four produced confident, plausible, entirely meaningless output. None raised
an error. They are documented in code comments at the fix sites.

**1. Block-scale domination — `encoders/motion.py`, `SCALAR_W`.**
The per-frame descriptor concatenated a 128-bin L1-normalised histogram (L2 norm
≈ 0.1) with three raw scalars (≈ 2.4, 2.1, 0.98). Cosine is scale-sensitive
across blocks, so the 128 numbers describing the *task* contributed ~4% of the
vector and the three numbers saying "this is handheld video" contributed the
rest. **Every pair scored `task_cos` 0.995–0.998, including different
worksites.** Fix: L2-normalise each block independently, `log1p` the unbounded
scalars, down-weight the tiny block. Spread went to [−0.19, +0.28].
→ *If you add any feature to any descriptor, normalise its block.*

**2. No whitening — `calibrate.Whitener`.**
Before: `env_cos` was 0.92–0.94 factory↔factory and 0.75 factory↔construction.
Everything crushed at the top of the scale. After: 0.12–0.26 vs −0.41. Same
features, usable metric. **This is why the repo refuses to print a calibrated
score until `novelty calibrate` has run**, and why `compare` emits a loud note
when uncalibrated.

**3. Autocorrelation `argmax` instead of peak-picking — `motion._dominant_period`.**
A genuine cycle is a *local maximum* of the autocorrelation. A plain argmax
picks the monotone shoulder of the zero-lag peak, so **every clip reported a
"period" exactly equal to the search floor (0.40 s)**. Compounded by not
smoothing the motion-energy signal (head jitter dominates the high frequencies).
Fix: 0.5 s Hann smoothing + explicit local-maximum detection + a minimum
strength of 0.05. Periods became 2.1–5.7 s, which is plausible manual work.

**4. `drop_last_shorter_than` applied to every window — `signature._windows`.**
The guard meant for a short *trailing* window was applied to all of them, so any
`segment.seconds` below the threshold (default 5 s) silently reduced every file
to a **single** signature. It looked like "windowing isn't helping". Caught by
`tests/test_signature.py::test_windowing_produces_multiple_segments`.

**5. (Evaluation, not code) File-level labels resolved to one window.**
`calibrate --labels` originally looked up one signature per labelled file, so 6
labelled pairs became 6 comparisons and the task AUC came out at **0.222 — worse
than chance — purely from sampling noise.** Now each labelled file pair expands
to every cross-window pair (6 → 96). Same data, honest number.

**6. (Test fixture) Perfectly periodic synthetic video trips tier 0.**
Frames one period apart in a noiseless, driftless synthetic loop genuinely *are*
the same pixels, so pHash correctly reported shared footage between disjoint
halves. Fixed by giving the fixture camera drift and sensor noise, like a real
camera. Worth knowing if you ever run this on rendered or looped content.

---

## 8. What to do next, ranked

### P0 — unblock validation (no code)
Get the `SAME_PLACE_NEW_TASK` recordings described in §6 and add them to
`eval/pairs.yaml`. Everything below is unmeasurable until this exists.

### P1 — run the GPU path for the first time
`configs/gpu.yaml` is written and wired but **has never executed** — the build
container has no torch. Expect to debug:
- `encoders/appearance._TorchVisionEncoder` — the processor/dtype handling for
  DINOv2 is written from the API, not verified.
- `encoders/video._HFVideoEncoder` — `AutoVideoProcessor` fallback to
  `AutoImageProcessor`, and the `_pool` over `last_hidden_state`, are both
  unverified. V-JEPA 2's output shape in particular should be checked, not
  assumed.
- The model IDs (`facebook/dinov2-base`, `facebook/vjepa2-vitl-fpc64-256`,
  `MCG-NJU/videomae-base`) should be confirmed against the current hub.

Then re-measure §5 with DINOv2 + V-JEPA 2 and compare. **Report the delta
honestly; do not assume the learned backbone wins.**

### P2 — widen windows and retest the rhythm features
`--window 60 --hop 30`. The 4–6 s cycles in this footage need ≥ ~35 s of window.
If `rhythm_cos` and `period_agree` still fit to 0.00, they can be dropped.

### P3 — index at real scale
Everything so far used 16 signatures. The null model and whitener are fitted on
the corpus and are noise below ~50 signatures (the CLI warns). Index the full
`left_rectified*.mp4` set windowed — roughly 40 min of video → ~150 signatures →
a null distribution that means something. Budget ~28 s per 60 s of video per
core (see §9).

### P4 — throughput
Currently **three separate ffmpeg decode passes per signature** (appearance,
flow, pHash). Merging into one `filter_complex` with three outputs should take
~40% off indexing. Also: `-hwaccel cuda` in `io/decode.py`, and parallelising
across files (embarrassingly parallel — run N workers on disjoint file lists and
merge the manifests).

### P5 — scale the store past ~1e5 signatures
`store/numpy_store.py` is brute force; 1e5 signatures is a 40 GB pairwise matrix.
The pattern is **two-stage**, not "swap numpy for FAISS": ANN over whitened
`app_mean` → top ~100 candidates → full `raw_scores` on those. Stage 2 is not
optional, because chamfer, DTW and rhythm **are not inner products** and no ANN
index can serve them. An ANN-only system silently degrades to "cosine on the
mean", which is the design this repo exists to argue against. For selection at
that scale, build a sparse k-NN graph (k≈50) and run lazy greedy on it.

### Explicitly NOT done, and why
- **No vector database.** At 1e4 signatures a dense float32 matmul is
  milliseconds with zero operational surface. Adding Qdrant now would be
  infrastructure without a problem. Revisit at P5.
- **No RAFT flow backend.** `motion.backend` accepts `"raft"` but only
  `"farneback"` is implemented. Farnebäck at 160×90 is fast and adequate; RAFT
  only matters if flow quality is shown to be the bottleneck.
- **No audio.** These recordings have no audio stream. For footage that does,
  audio is a strong and very cheap task signal — worth adding as a tier 2
  component.
- **No object/hand detection.** A hand-pose or object-detector track would
  likely beat everything in tier 2, but it is a much larger dependency. Consider
  only after P1 shows V-JEPA 2 is insufficient.

---

## 9. Environment notes (these cost time to rediscover)

### Two machines
- **Cloud container** (`Bash` tool): Python 3.11, numpy 2.4, scipy, OpenCV 4.13,
  scikit-learn, pandas, PyYAML, ffmpeg. **No torch. 2 cores, 7 GB RAM.**
- **User's Mac** (`mcp__remote-devices__device_bash`): a Linux VM with
  `~/Downloads` mounted at `$HOME/mnt/Downloads`. Python 3.10, ffmpeg. Files in
  the mounted folder **cannot be deleted** without an explicit permission grant
  — `rm` fails with "Operation not permitted", and `git` emits
  `unable to unlink ... tmp_obj_*` warnings as a result (harmless, but do git
  work in `$HOME` scratch and copy across if it bothers you).

### GitHub push must happen from the Mac
**[Certain]** The cloud container's egress proxy intercepts `api.github.com` and
rejects bring-your-own tokens with
`{"message":"No linked GitHub account. Connect your GitHub account and retry."}`.
The same token works fine from `device_bash`. So: build in the container, tar it,
`device_commit_files` it across, `git push` from `device_bash`.

GitHub account: `abirsinha-humynlabs`. The user pasted a PAT in chat; **he was
told to rotate it and you should assume it is dead.** Ask for a fresh one. Never
write a token into a file, a git remote, or a commit.

### Timing
Full signature (appearance + flow + pHash) on 60 s of 1080p30, 2 cores: **~28 s**,
i.e. ~2× realtime. A 30 s window costs ~13 s. The three decode passes dominate.

### Scratch state (ephemeral — will not survive)
- Container index used for all §5 numbers: `/tmp/nvx` (gone).
- Generated report: `/tmp/nvx-report.html`, copied to the user's chat.
- Device copy of the repo: `~/Downloads/video-novelty` (origin already set,
  no token in `.git/config`), plus `$HOME/vn` in the device VM scratch.
- Tarball: `~/Downloads/video-novelty.tar.gz`.

---

## 10. Commands

```bash
# setup
make install          # CPU: numpy scipy opencv pyyaml sklearn. No weights, no network.
make install-gpu      # + torch, transformers
make test             # 44 tests, synthetic video fixtures, ~25 s

# reproduce the §5 numbers
SRC=~/Downloads/'left_rectified(5).mp4' \
SRC2=~/Downloads/front_left_rectified.mp4 \
  bash scripts/prepare_eval_clips.sh
novelty index data/clips --index .novelty --window 30 --hop 15
novelty calibrate --index .novelty --labels eval/pairs.yaml
novelty select --index .novelty
novelty report --index .novelty --out novelty-report.html

# the rest
novelty compare a.mp4 b.mp4 --index .novelty   # explain one pair
novelty search  a.mp4 --index .novelty -k 10   # nearest neighbours
novelty gate    new.mp4 --index .novelty       # exit 1 = reject
novelty encoders                               # list backbones
```

**Always calibrate before reading a score.** Uncalibrated, `compare` prints raw
numbers and says so in a note; those numbers are not comparable to anything.

---

## 11. Where the reasoning lives

| file | what it explains |
|---|---|
| `docs/01-why-not-just-cosine.md` | the four failure modes of the naive design, with numbers |
| `docs/02-architecture.md` | data flow, file-by-file map, the Signature, windowing |
| `docs/03-signatures.md` | every feature, what it sees, why it earns its keep |
| `docs/04-calibration.md` | whitening, null models, percentiles, supervised weights |
| `docs/05-coverage-selection.md` | submodularity, the greedy guarantee, reading the knee |
| `docs/06-running-on-gpu.md` | DINOv2, V-JEPA 2, throughput, scaling the store |
| `docs/07-tuning.md` | every knob, every failure mode and how to recognise it |
| `eval/pairs.yaml` | the label file — its header documents its own inadequacy |

Code comments at every non-obvious decision explain the *why*, especially at the
four bug sites in §7. If you change something there, update the comment; the
next agent after you will rely on it the way you are relying on this file.
