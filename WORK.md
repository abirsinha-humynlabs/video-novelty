# WORK.md — handoff for the next agent

**Read this whole file before touching anything.** It is the state of the project,
what was measured, what is broken, and what to do next. The `docs/` directory
explains *how the code works*; this file explains *where the project is*.

Last updated: 2026-09-18 · commit `c1bd9ae` + uncommitted work · repo `abirsinha-humynlabs/video-novelty`

> **2026-09-18 session changed the picture substantially. Read §0 first.**

---

## 0. What changed on 2026-09-18

Four things, in order of how much they should change your plans:

1. **The environment tier is now validated against a real hard negative.**
   AUC **0.9983**, 4.40 pooled sd, best-F1 0.982, over 630 pairs from two
   Pipe_Factory sessions. §6's "you cannot validate this without new
   recordings" is **resolved for the environment axis**. It is still fully open
   for the task axis — see §6.
2. **The GPU path ran for the first time.** P1 is done. DINOv2, V-JEPA 2,
   VideoMAE and NVIDIA Cosmos-Embed1 all execute on an A10G. Three missing
   dependencies and three silent bugs were found doing it (§7).
3. **`decision.env_percentile` moved 97 → 52**, because it was measured rather
   than guessed. This is the single most consequential config change in the
   repo's history and it is corpus-dependent — read §5.3 before trusting it
   anywhere else.
4. **The environment is no longer the 2-core container described in old §9.**
   It is a Linux box with an A10G. §9 is rewritten.

---

## 1. The 60-second version

**Goal:** given hundreds of hours of egocentric video of people doing repetitive
manual work, decide which clips add new information to a training corpus for a
physical-AI / VLA model, and which are redundant. "Redundant" must mean
*semantically* redundant (same place, same work), not byte-identical.

**What exists:** a working Python package, `novelty`, that builds three-tier
signatures per video span, calibrates similarity against the user's own corpus,
and selects a maximally-diverse subset by submodular coverage maximisation.
44 tests pass. Runs on CPU with zero learned weights; the GPU path is now
**executed and verified** on an A10G.

**Where it stands:**

| tier | what it answers | status |
|---|---|---|
| 0 — perceptual hash | did these share literal footage? | **works**, tested |
| 1 — appearance | same physical place? | **validated**, AUC 0.998 vs a real hard negative |
| 2 — motion/task | same work happening? | **still unvalidated** — every number is confounded |

**The single most important fact for you:** the task tier is unvalidated, and
*you still cannot fix it by writing code*. Every labelled pair that exists —
old and new — varies environment and task **together**, so a detector that
ignores motion entirely scores ~0.99. See §6. The environment tier no longer
has this problem; the task tier is unchanged since day one.

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

### 3.0 Current corpus (2026-09-18) — this is what to use

Pulled from S3 presigned URLs the user supplies in chat, laid out mirroring the
bucket path. **Never commit a presigned URL — they carry AWS credentials.**

```
data/normalized/bitrobot/Pipe_Factory/2026-08-27/
  PIP-246/20260827_123109_session36/000/left_rectified.mp4   598.8s 1080p30 HEVC
                                        chunks/              18 x 30s (20s->560s)
  PIP-250/20210105_153825_session9/000/left_rectified.mp4    598.8s 1080p30 HEVC
                                        chunks/              18 x 30s (20s->560s)
```

| | PIP-246 / session36 | PIP-250 / session9 |
|---|---|---|
| product | **grey** PVC fittings | **white** PVC fittings/tees |
| ground | tiled factory floor, woven sacks | dark tarp + white sheeting |
| worker | brown trousers, red bracelet | black trousers, yellow watch |
| scene | open workshop, co-workers visible | enclosed, dim |

**These two sessions are the hard negative this project was blocked on.** Same
factory domain, same rig, same egocentric hands-in-frame PVC work — differing in
worker, product colour, lighting and ground surface. Session membership is
therefore free ground truth for the environment axis with no hand labelling.
That is what produced the AUC 0.998 in §5.

Three practical notes that cost time to rediscover:

* **Chunks are stream-copied (`-c copy`), not re-encoded.** Source keyframes sit
  exactly every 1.0 s, so 30 s cuts are frame-exact and lossless, and cutting
  36 chunks takes seconds instead of ~25 min of libx264. The cost is that the
  chunks stay HEVC, which decodes slower than H.264 — indexing runs ~22 s per
  30 s chunk rather than ~15 s. Worth it.
* **`left_rectified.mp4` sits *inside* `data/`, next to `chunks/`.** `novelty
  index <dir>` walks recursively and will happily index the 10-minute source
  alongside its own chunks, duplicating everything. **Always point `index` at
  the `chunks/` directories explicitly.**
* **Check uploaded files actually decode.** The first upload of session36 was
  truncated at 58% (261 MB of 450 MB). `ffprobe` still reported the full 598.8 s
  from metadata; only `ffmpeg -f null -` revealed it. Chunks past 347 s came out
  empty. Verify with a full decode pass, not a duration probe.

### 3.1 Older corpus (2026-09-17, on the user's Mac)

Historical — kept because §5.1 and §7 refer to it. All on the user's Mac
(`abirs-macbook-pro-2-local`), in `~/Downloads`. That session had folder access
granted to `~/Downloads` only. **The previous run was done on a local Apple M3;
the project has since migrated to the Linux GPU box (§9).**

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

## 5. Measured results

### 5.0 Current headline (2026-09-18, GPU, two sessions)

Setup: 36 chunks (18 per session) × 30 s, `--window 30 --hop 30` → 36
signatures, one per chunk. `configs/gpu.yaml`: **DINOv2** appearance +
**V-JEPA 2** clip encoder. Null fitted over 630 pairs.

Positives = pairs **within** one session (same environment).
Negatives = pairs **across** sessions (different environment).
No hand labelling — session membership is the label.

```
environment: AUC=0.9983  best-F1=0.982 @ raw>=+0.2690  separation=4.40 sigma
             within-session +0.5010 ± 0.111   cross-session +0.0754 ± 0.080
task       : AUC=0.9907  best-F1=0.954 @ raw>=+0.0922  separation=3.57 sigma
             within-session +0.3469 ± 0.119   cross-session -0.0376 ± 0.095
```

**All 36 chunks have their nearest neighbour inside their own session. Zero
cross-session confusions.**

> **[Certain] Do not quote the task AUC as validation.** Session membership
> determines `same_env` and `same_task` identically here, so the task axis can
> score 0.99 purely by leaking environment. It is the §6 confound reproduced in
> new data, not progress. See §6.

> Weaker version of the same caveat applies to the environment number:
> within-session pairs also share worker, lighting and product colour, so
> "environment" here means *session identity*, not *place* in isolation. For the
> practical question ("is this more of what I already have?") that is the right
> quantity. For a claim about place recognition specifically, it is not isolated.

### 5.1 Threshold: why `env_percentile` moved 97 → 52

The best-F1 operating point above sits at the **52nd percentile** of the corpus
null. The repo shipped 97.

A percentile threshold asks *"is this pair in the top (100−p)% of my corpus"*.
It only means "same environment" if same-environment pairs are about that rare.
This corpus is **49% same-environment pairs**, so 97 can only ever fire on
near-duplicates — which is exactly why two adjacent chunks of one continuous
recording kept returning `NOVEL` and looked like a broken detector.

**This value is corpus-dependent and will be wrong for your next corpus.** Add
twenty sessions, the same-environment base rate collapses, and the correct
percentile rises. `calibrate --labels` prints the best-F1 percentile; re-fit it
rather than inheriting 52. The number is now in `configs/*.yaml` and
`config.DecisionConfig`, with that reasoning in a comment at each site.

`task_percentile` was deliberately **left at 95**, unmeasured. Its best-F1 point
(48.5) comes from confounded labels, so shipping it would be laundering a
guess into a measurement. 95 makes `REDUNDANT` hard to reach, which is the safe
direction to be wrong in. Consequence you will see: two adjacent chunks of
identical repetitive work currently label `SAME_PLACE_NEW_TASK`, not
`REDUNDANT`. That is the task tier being untrusted, working as intended.

### 5.2 Throughput measured on the A10G

| step | cost |
|---|---|
| index, 30 s HEVC chunk, DINOv2 + V-JEPA2 + flow + pHash | **~22 s** |
| index, 60 s H.264 chunk → 4 windows | ~62 s |
| 36-chunk corpus, end to end | ~13 min |
| Cosmos-Embed1 first call (incl. 2.4 GB download) | ~42 s |

Still three separate ffmpeg decode passes per signature (P4 below).

### 5.3 Historical: single-session run (2026-09-18, earlier)

Before session9 existed, 10 chunks of session36 alone were indexed. Useful only
as a record of what a single-environment corpus looks like: the environment axis
showed a clean monotone decay with time separation (env_raw +0.609 at 0–30 s
apart → +0.271 at 120–240 s, correlation −0.683; adjacent chunks 73.9th
percentile vs 44.3rd for ≥5 min apart, ~90% correct ordering) — the axis
demonstrably worked, but **no threshold was validatable**, because a corpus with
one environment contains no negative. That is the trap; do not repeat it.

### 5.4 Historical: 2026-09-17 CPU run (`gist`, four clips)

Setup: the four clips in §3.1, `--window 30 --hop 15` → 16 signatures.
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

**Environment axis: RESOLVED 2026-09-18.** PIP-250/session9 supplied the hard
negative. AUC 0.998 over 630 pairs, ground truth from session membership. No
further recordings needed for this axis.

**Task axis: UNCHANGED. Still blocked. Still the most important thing.**

**[Certain] There is still no way to tell whether the task tier works, because
no pair in the dataset differs on task while holding environment fixed.** Adding
session9 did not help: it differs on environment *and* task simultaneously, so
it filled in the same diagonal the old eval set already had.

The eval set still covers two of four quadrants:

```
                 same_task    diff_task
  same_env    |  YES        |  MISSING   <-- the one that matters
  diff_env    |  MISSING    |  YES
```

The task AUC went 0.630 → 0.991 between the two corpora. **That is not an
improvement in the task tier.** It is the environment signal getting stronger
(DINOv2 instead of gist) and leaking through a confounded label set. A detector
wired to ignore motion entirely would post a similar number. Treat 0.991 as a
measurement of the confound, not of the model.

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

All of them produced confident, plausible, entirely meaningless output. None
raised an error. They are documented in code comments at the fix sites.

### Found 2026-09-18

**7. `compare` judged two files on ONE window pair — `cli._sig_for`.**
`_sig_for` returned the *first* signature matching a path and silently dropped
the rest, so comparing two 60 s files windowed at 30 s/15 s used 1 of 16
available window pairs — whichever sorted first. On real footage that read the
61st percentile as the 49th, a 21-point swing, easily enough to flip a verdict.
This is **bug 5 all over again** in a different function: bug 5 was fixed for
`calibrate --labels` and nobody checked `compare`.
Fix: `metrics.fuse.compare_windows()` scores every cross-window pair, averages
the percentiles, re-derives the label, and reports the spread as a note
(`env percentile sd=20.7` on that pair — that is how much the window lottery was
worth). `DUPLICATE_SOURCE` wins if any window pair fires it.
→ *Any file-level claim must aggregate over windows. Check every place that
resolves a path to a signature.*

**8. Labels resolved by basename, so sessions collided — `cli.cmd_calibrate`.**
The label lookup keyed on `os.path.basename`. The corpus layout repeats chunk
names under every session (`.../PIP-246/.../chunks/chunk01_020-050.mp4` and
`.../PIP-250/.../chunks/chunk01_020-050.mp4`), so both collapsed into one
bucket and a `same_env: true` label silently expanded to include cross-session
pairs — **a wrong-label generator that raises nothing**. Fix: resolve by path
suffix, and refuse an ambiguous label loudly instead of picking one.
→ *Caught only because the directory layout changed. It was latent before.*

**9. The `gpu` extra was missing three hard dependencies — `pyproject.toml`.**
None are pulled in by `torch`/`transformers`: **Pillow** (HF
`AutoImageProcessor` requires it), **torchvision** (`AutoVideoProcessor`, i.e.
V-JEPA 2, requires it), **einops** (Cosmos-Embed1's `trust_remote_code`
modeling file imports it). Each surfaced only as an ImportError at first model
load, which is why "the GPU path is wired" was never the same claim as "the GPU
path runs".

### Found 2026-09-17

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

### P0 — unblock the TASK axis (no code, still)
Unchanged and still first. Get the `SAME_PLACE_NEW_TASK` recordings described in
§6 and label them. Two different jobs **in the same station, same worker, same
lighting**. Everything about the task tier is unmeasurable until this exists,
and the new session9 data did *not* supply it.

### P1 — run the GPU path — **DONE 2026-09-18**
All four backbones execute on the A10G, fp16/bf16, L2-normed, no NaNs:

| encoder | tier | dim | notes |
|---|---|---|---|
| `dinov2` | 1 | 768 | verified; the §5.0 numbers use it |
| `vjepa2` | 2 | 1024 | needed torchvision (bug 9) |
| `videomae` | 2 | 768 | verified |
| `cosmos-embed1` | 2 | 768 | added this session; needed einops (bug 9) |

Model IDs confirmed against the live hub. `_TorchVisionEncoder`'s CLS pooling
and `_HFVideoEncoder`'s processor fallback both work as written.

**Still not done: the A/B.** `configs/gpu-cosmos.yaml` exists and Cosmos-Embed1
is verified in isolation, but no head-to-head against V-JEPA 2 on the task axis
has been run. Note that such an A/B is **currently unreadable anyway** — with
confounded labels (§6) both will score ~0.99 by leaking environment. Do P0
first, or the comparison measures nothing.

### P1.5 — the one V-JEPA 2 finding worth acting on
`task_cos` on raw V-JEPA 2 embeddings is **0.9921 vs 0.9927** across every pair
in the corpus — a 0.0006 spread, within- and cross-session alike. That is the
transformer anisotropy / cone effect, and it is the same *class* of problem as
bug 1. Whitening rescues it into a usable range, which is the entire reason the
repo refuses to print uncalibrated scores. If you add any new clip encoder,
check its raw spread before trusting it.

### P2 — widen windows and retest the rhythm features
`--window 60 --hop 30`. The 4–6 s cycles in this footage need ≥ ~35 s of window.
If `rhythm_cos` and `period_agree` still fit to 0.00, they can be dropped.

### P3 — index at real scale
The largest run so far is **36 signatures** (two sessions × 18 chunks). The null
model and whitener are fitted on the corpus and are noise below ~50 signatures —
the CLI warns on every run so far, including the one behind the AUC 0.998. Those
numbers are strong enough that the warning is unlikely to be hiding a reversal,
but **they are still fitted on 36 points and should be re-run at scale.**
More sessions also fix the other half of §5.1: the right `env_percentile` is a
function of how redundant the corpus is, and two sessions is not a corpus.
Budget ~22 s per 30 s HEVC chunk on the A10G (§5.2).

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
- **Nothing adopted from NVIDIA Cosmos Curator**, after researching it on
  request. It is a Ray/Cosmos-Xenna distributed pipeline needing NVCF or Slurm,
  with per-stage GPU fractions tuned for 48 GB cards — wildly disproportionate
  for a 36-signature corpus on one A10G. More importantly its semantic dedup is
  *k*-means + within-cluster cosine + a magic `eps=0.01` threshold, which is
  exactly the cluster-and-threshold pattern `docs/01` measured and rejected.
  Adopting it would be a regression dressed as an upgrade. Its *orchestration*
  may matter at P3–P5 scale; its similarity logic never will.
  **Cosmos-Embed1 (the model) was adopted — that part is worth having.** Caveat
  recorded in its docstring: it is contrastively text-aligned, so it carries the
  caption-collapse risk `docs/01` cites against CLIP. It is wired to the task
  axis only, never appearance.
- **DINOv3 not added.** Available and commercially licensed, but **gated** —
  needs the user's own HF account to accept terms. Shipping a registry entry
  that cannot be executed is worse than not shipping it. One caution for
  whoever does add it: DINOv3 introduces register tokens. CLS stays at index 0
  so `_pool`'s `h[:, 0]` should still be right, but **verify that against real
  weights rather than assuming it.**
- **V-JEPA 2.1 deferred.** Not officially on HF (torch.hub only; transformers
  issue #45496 open). Only a community conversion exists. Revisit when official
  checkpoints land.
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

### The machine (rewritten 2026-09-18 — the old description is obsolete)

Work has **migrated off the user's Apple M3 and off the 2-core cloud container**
onto a Linux GPU box. Anything in an older note claiming "no torch" or "2 cores"
is stale.

```
Amazon Linux 2023 · 4 cores · 15 GB RAM
NVIDIA A10G, 23 GB VRAM, compute 8.6 (Ampere -> native bf16)
driver 595.91.07 · CUDA 13.2 toolkit at /usr/local/cuda-13.2 (nvcc present)
torch 2.8.0+cu128 (cu12 wheels run fine on the 13.2 driver)
```

Setup that is **not** reproducible from `make install` alone:

* **Use `uv`.** `uv venv .venv && uv pip install -e ".[gpu,fit,dev]"`. Installs
  in seconds where pip takes minutes.
* **ffmpeg is not in the AL2023 dnf repos.** `dnf install ffmpeg` fails with
  "no match". Fetch the static build
  (`johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz`) and
  drop `ffmpeg`/`ffprobe` into `~/.local/bin`, which is already on PATH.
  Without them **22 of the 44 tests error out** — they need real video.
* **No NVENC** in that static build, so chunk re-encoding is CPU-bound. Prefer
  `-c copy` (see §3.0).
* Hugging Face is reachable; model downloads work directly.

### Data arrives as S3 presigned URLs
The user pastes them in chat. They carry AWS credentials in the query string and
expire (~7 days). **Use them inline in `curl` only — never write one to a file,
a git remote, or a commit.**

### S3 is where chunks and results live (local disk is staging only)

The box runs at ~84% disk with 1.6 GB of video per two sessions, so **nothing
large stays local**. The instance profile (`SSM-Role`) grants S3 access with no
keys to manage — `aws s3 ...` just works.

```
s3://stage-humyn-egocentric-stereo-data/labelling_results/
    novelty_data/<mirrored source path>/chunks/*.mp4   30 s chunks
    novelty_result/<mirrored source path>/*.csv        chunks.csv, pairs.csv
    novelty_result/summary.csv
```

> ## [Certain] WRITE ONLY TO THOSE TWO PREFIXES.
> `labelling_results/` holds ~48 sibling prefixes belonging to other pipelines
> (`hand_pose_*`, `6dof_head_pose_*`, `depth_*`, `delivery/`, `gt/`, …). A
> mistyped `--out` into one of those is not a recoverable mistake. The rule is
> enforced in code — `scripts/report_csv.py` has an `S3_WRITE_ALLOWLIST` and
> refuses anything else — because a runbook cannot stop a typo.
> Never pass `--delete` to `aws s3 sync` against a shared prefix.

### Video never stays on local disk. Results do.

The rule, set 2026-09-18: **`data/` is not kept locally.** Video is downloaded,
chunked, uploaded and deleted inside one script. `output/` is the exception —
CSVs are ~220 KB and are what people actually read, so they live **both**
locally and on S3.

```bash
# ingest one recording: download -> verify -> chunk -> upload -> index -> delete
python scripts/ingest_session.py --url "<presigned url>" --index .novelty-2sess

# results: local output/ AND the S3 mirror, in one command
python scripts/report_csv.py --index .novelty-2sess
```

`ingest_session.py` derives the whole mirrored directory layout from the URL's
own key, so nobody types a path. It uses a temp dir that is removed in a
`finally`, deletes the 430 MB source as soon as chunking finishes, and verified
at **zero net disk growth** on a full run. `--keep-local` exists for debugging
decode behaviour and defeats the point of the script.

**Why this is safe:** nothing downstream needs the video. `compare`, `select`
and `report_csv` all read *signatures* from the index (~210 KB each; a 36-chunk
corpus is ~8 MB). Verified after deleting `data/` entirely — all three still
run. Only indexing a **new** file needs bytes on disk, which is what
`ingest_session.py` is for.

The 10-minute `left_rectified.mp4` sources are **not** uploaded to
`novelty_data` — they already exist in `prod-egc-stereo-v2-data`, and
duplicating 430 MB per session buys nothing. `novelty_data` holds chunks only.

**Aside worth following up:** those sibling prefixes show the org already
produces hand-pose, 6-DoF head-pose and SLAM outputs. A tier-3 "physical
variation" axis built on those is far more feasible than it looks from inside
this repo, which ingests RGB only. Revisit after §6's task blocker is cleared.

### GitHub
GitHub account: `abirsinha-humynlabs`. The user pasted a PAT in chat once; **he
was told to rotate it and you should assume it is dead.** Ask for a fresh one.
Never write a token into a file, a git remote, or a commit.

*(The old note here said pushes must happen from the Mac because the cloud
container's egress proxy intercepted `api.github.com`. That applied to the
retired container — re-test from the GPU box before assuming it still holds.)*

### Timing
See §5.2 for measured A10G numbers. Historical, 2 cores, no GPU: a full
signature on 60 s of 1080p30 took **~28 s**, ~2× realtime.

### State on the GPU box
- `.novelty-2sess/` — the 36-signature two-session index behind every §5.0
  number. Calibrated. **This is the one to use.**
- `.novelty-gpu/` — earlier single-session index (§5.3). Superseded.
- `output/` — per-session CSVs, mirroring the S3 tree (see §10).
- `.venv/` — uv-managed, `[gpu,fit,dev]` + pillow/torchvision/einops.
- Both indices and `data/` are gitignored (`data/`, `*.mp4`, `.novelty*`).

---

## 10. Commands

```bash
# setup on the GPU box (see §9 -- ffmpeg is NOT in the AL2023 repos)
uv venv .venv && uv pip install -e ".[gpu,fit,dev]" --python .venv/bin/python
.venv/bin/python -m pytest -q      # 44 tests, ~16 s, needs ffmpeg on PATH

# reproduce the §5.0 numbers (two sessions, 30 s chunks)
# one command per recording -- download, verify, chunk, upload, index, delete
python scripts/ingest_session.py --url "<session36 presigned url>" --index .novelty-2sess
python scripts/ingest_session.py --url "<session9  presigned url>" --index .novelty-2sess
novelty calibrate --index .novelty-2sess
python scripts/report_csv.py --index .novelty-2sess     # local output/ + S3 mirror
novelty select --index .novelty-2sess
novelty report --index .novelty-2sess --out novelty-report.html

# the rest
novelty compare a.mp4 b.mp4 --index .novelty   # explain one pair
novelty search  a.mp4 --index .novelty -k 10   # nearest neighbours
novelty gate    new.mp4 --index .novelty       # exit 1 = reject
novelty encoders                               # list backbones
```

**Always calibrate before reading a score.** Uncalibrated, `compare` prints raw
numbers and says so in a note; those numbers are not comparable to anything.

**An index stores its own config.** `compare` prefers the index's stored config
over `configs/*.yaml`, so changing a threshold on disk does *not* change the
verdicts of an existing index. Either pass `--config` explicitly or edit
`<index>/config.yaml`. This wastes ten minutes if you do not know it.

### Output CSVs (`scripts/report_csv.py`)

Writes into `output/`, mirroring the source tree:

```
output/normalized/bitrobot/Pipe_Factory/2026-08-27/<PIP>/<session>/000/
    chunks.csv   one row per chunk: span, detected period + strength, nearest
                 neighbour and which session it is in, env/task percentiles,
                 verdict, within- vs cross-session similarity, coverage
                 selection rank and marginal gain
    pairs.csv    every comparison behind those numbers
output/summary.csv   per-session within/cross means
```

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
| `scripts/report_csv.py` | the `output/` CSVs: what each column means |
| `configs/gpu-cosmos.yaml` | why Cosmos-Embed1 is task-axis-only, and the fps/clip_len reasoning |

**Docs not yet updated for 2026-09-18.** `docs/04-calibration.md` and
`docs/07-tuning.md` still describe `env_percentile: 97` as the operating point,
and `docs/06-running-on-gpu.md` still speaks of the GPU path as untried. The
code and this file are correct; those three docs lag. Fix them when you touch
that area.

Code comments at every non-obvious decision explain the *why*, especially at the
nine bug sites in §7. If you change something there, update the comment; the
next agent after you will rely on it the way you are relying on this file.
