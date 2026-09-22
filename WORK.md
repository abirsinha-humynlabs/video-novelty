# WORK.md — handoff for the next agent

**Read this whole file before touching anything.** It is the state of the project,
what was measured, what is broken, and what to do next. The `docs/` directory
explains *how the code works*; this file explains *where the project is*.

Last updated: 2026-09-22 · repo `abirsinha-humynlabs/video-novelty`

> **Read §000 first — task1 is delivered and it changed several conclusions in
> this file. Then §00 (task2, parked), then §0. Everything older is still
> valid unless §000 says otherwise.**
>
> The project splits into two tasks, and conflating them caused real confusion:
>
> | | question | status |
> |---|---|---|
> | **task1** | similarity between two *different* videos | **delivered 2026-09-22** (§000) |
> | **task2** | repetition *within* one video, across its chunks | parked on 3 fps NPZ (§00) |

---

## 000. task1 — video-to-video similarity. DELIVERED 2026-09-22.

Read this before anything else. It supersedes §6's "task axis is blocked" and
§8's P0.

### What shipped

Both axes, never fused, over all 13,964 same-site pairs of the 435 episodes:

```
DISTINCT              10206   73.1%
SAME_PLACE_NEW_TASK    2088   15.0%   keep
REDUNDANT              1064    7.6%   <- the actionable set
BORDERLINE_ENV          401    2.9%   env axis cannot decide; see below
SAME_TASK_NEW_PLACE     134    1.0%   keep
NO_TASK_DATA             71    0.5%
```

`s3://stage-humyn-egocentric-stereo-data/labelling_results/novelty_result_v2/`
— `env_pairs.csv` (site-blocked), `env_pairs_by_industry.csv` (industry-blocked),
`eyeball/` (every human label, the contact sheets, the negative-result scores).

Columns: `env_cos env_chamfer env_bhat_sim env_raw env_percentile_vs_negatives
verdict verdict_calibrated env_threshold task_sim task_threshold
same_job_family same_canon_task job_a job_b verdict_quadrant`. A row carries the
provenance of its own decision — the original `verdict` sits beside the
calibrated one.

**1064 pairs are same place AND same job**, against the 2994 the environment
axis alone called same-place. Most same-place pairs are the worker doing a
*different* job in the same room, which is not redundant at all. Keeping the
axes separate cut the redundancy candidate set by two thirds — that is the
whole argument for the quadrant, now with a number attached.

### The env threshold was calibrated twice, and the first one was wrong

Worth reading as a method lesson, not just a changelog.

**Round 1** — 40 pairs sampled across the whole score range, judged by the user
in an artifact. Perfect separation, AUC 1.0000, so the threshold went to the
midpoint of the gap: `env_raw >= 0.4673`.

**That gap was an artifact of the sampling.** Almost nothing in round 1 sat
inside it. **Round 2** sampled 48 pairs *only* from inside the band
(0.4400–0.4947), 96 distinct clips with none reused, shown **blind** (scores
hidden by default, so the model could not anchor the judgement) in randomised
order:

```
same        3
different  33
unsure     12
```

The region round 1 could not see is overwhelmingly **not** the same place. On
the combined 88 labels the 0.4673 cut scores **precision 0.485** — a coin flip.

```
AUC combined              0.9796   (was 1.0000 on round 1 alone)
AUC inside the band       0.8182   still ordered, no longer separable
separation                OVERLAPPING: max different 0.4919 > min same 0.4666
best F1        0.9412 at >= 0.4901   precision 0.889  recall 0.941
zero-FP cut          at >= 0.4920    recall 15/17 = 0.882
```

Operating point: **SAME_ENV at `env_raw >= 0.4920`**, BORDERLINE down to
0.4666 (the overlap region, bounded below by the lowest human "same"), else
DIFFERENT_ENV. Zero-FP was chosen over best-F1 because the costs are not
symmetric: a false positive discards genuinely novel footage, a false negative
only leaves a redundant clip in the set.

**The lesson: calibrate on the region where the decision is hard, not on the
range where it is easy.** A threshold fitted to easy pairs will look perfect
and be wrong.

**Round 3 confirmed it and closed the question.** 50 more pairs, sampled only
from the refit band, 100 clips, none reused, blind. Same shape as round 2 —
4 same, 37 different, 9 unsure — and over all 138 labels:

```
ALL LABELS         n=138   same 21   different 89   unsure 28
AUC combined                       0.9056
AUC inside the band (7 v 70)       0.6398   <- barely above chance
best F1  0.8333 at env_raw >= 0.4919   precision 1.000  recall 0.714
zero-FP cut          >= 0.4920         catches 15/21
```

The best-F1 cut over 138 labels lands at **0.4919**, i.e. the 0.4920 already
deployed. **The threshold needed no change** — three independent rounds of
labelling converged on it, and the second and third were drawn specifically to
break it.

The band AUC of **0.6398** is the important number. The env axis carries
almost no information between 0.4666 and 0.4920 — it is not mis-scaled there,
it is blind there. Combined with the local-matching negative result below,
that is two independent lines of evidence that the band is genuinely
ambiguous.

**So BORDERLINE is collapsed rather than left unresolved.** Measured inside the
band: 7 same against 70 different, so calling the whole band DIFFERENT_ENV is
right **90.9%** of the time. The band is 2.9% of all pairs, so the cost is
~0.26% of the corpus mislabelled — cheaper than shipping rows a consumer
cannot act on. Both CSVs therefore carry BOTH:

| column | BORDERLINE rows |
|---|---|
| `verdict_calibrated` / `verdict_quadrant` | kept as `BORDERLINE` / `BORDERLINE_ENV` — the honest flag |
| `verdict_resolved` / `verdict_quadrant_resolved` | forced to `DIFFERENT_ENV` and a real quadrant — the actionable column |

Resolved quadrant over the 13964 same-site pairs:

```
DISTINCT              10592   75.9%
SAME_PLACE_NEW_TASK    2087   14.9%   keep
REDUNDANT              1065    7.6%   the actionable set
SAME_TASK_NEW_PLACE     149    1.1%   keep
NO_TASK_DATA             71    0.5%
```

**No unresolved rows.** Use `verdict_quadrant_resolved` downstream and
`verdict_quadrant` when you need to know where the model was unsure.

### Held out, and still holding

No round of labelling touched these strata. Sensitivity and specificity both
survive the refit:

```
same_session_adjacent     n=   32   96.9% SAME_ENV
same_session_distant      n=  180   95.6% SAME_ENV
same_industry_diff_site   n= 9563    1.1% SAME_ENV
```

Pairs from one recording come out ~96% same-place; pairs from different sites
in the same industry come out 98.9% not-same. Under a threshold refit entirely
from same-site labels.

### The task axis was never actually blocked

§6 and §8-P0 say the task axis cannot be measured without new recordings, and
that it waits on VLM captioning of the 766. **Both were wrong, and the data to
settle it was already in the repo.**

`rejected_repetitive_shorter_segment.csv` carries, per episode, a
`task_description (our)` sentence of exactly the shape the VLM would produce —
one clause naming the action, the object and the fixture it is done on — plus a
12-way `job_family` and a `canon_task×site_h`. Coverage on the pair set is
**13893/13964 — 99.5%**. (No description is quoted here: they are customer
data, and this file is committed.)

`scripts/task_axis.py` builds IDF-weighted stem profiles from it (IDF is not
optional — see `novelty.captions.fit_idf`; the generic stems here are
`plastic`, `metal`, `floor`, `pick`). Measured against `canon_task×site_h`:

```
same canonical task       n=  979   task_sim mean 0.5462
different canonical task  n=12914   task_sim mean 0.0952
AUC                                 0.9768
best-F1 cut               0.3798    precision 0.676  recall 0.838
```

979 positives, against the env threshold's 17 human labels — this is now the
**better-grounded of the two axes.** (Precision 0.676 is a floor, not an error
rate: `canon_task×site_h` is task AND site, so the same job at two sites counts
as a miss while being a true `SAME_TASK_NEW_PLACE`.)

**It passes the independence test V-JEPA 2 failed.** That test has to hold the
environment fixed, or it measures how the world correlates rather than how the
axis behaves:

| | corr with env axis |
|---|---|
| V-JEPA 2 task score, within one session | **+0.55** |
| this text axis, within one session | **+0.243** |
| this text axis, all same-site pairs | +0.607 |

The +0.607 is not a defect — a pipe factory really does pipe work, so task and
place genuinely co-vary in this corpus. The within-session number is the one
that says whether the axis measures the job or the room.

**Why text and not pixels, restated because it keeps getting re-proposed:**
DINOv2 cannot do task. Its global token *is* scene appearance, which is
precisely why it works as the environment axis. Two different jobs at the same
bench are near-identical to it. Every pixel-based task attempt here failed the
same way — flow rhythm at 0.11 σ, V-JEPA 2 at +0.55 within a session,
VideoMAE likewise.

### The 401 BORDERLINE pairs are genuinely ambiguous, not badly represented

The obvious theory: the env axis pools each frame to one global DINOv2 vector,
which is right for "same kind of scene" and wrong for "same physical place",
where the evidence is a specific pillar or bench recurring. Tested it —
`eval/exp_local_place_matching.py`, 126 clips, 8 frames each, three scorers:

```
                    band 3v33   session24 v band33   easy 6v6
global                 0.5051            0.9533       1.0000
patch                  0.7626            0.7607       0.5833
patch_nobody           0.7626            0.7336       0.5000
sift                   0.4040            0.6622       0.5417
sift_nobody            0.4545            0.7077       0.6528

production env_raw on band 3v33 = 0.8182
```

Patch matching's 0.7626 is not a result: three positives, and the same scorer
gets **0.5833 on the easy pairs the global axis separates perfectly** while
losing the well-powered session-vs-band test 0.7607 to 0.9533. SIFT+RANSAC is
below chance on the band. Masking the wearer's body changed nothing.

The match rates say why — fraction of candidate matches surviving RANSAC:

```
patch   same_session 0.00429   band same 0.00429   band different 0.00386
sift    same_session 0.09500   band same 0.06667   band different 0.07333
```

Positives and hard negatives are indistinguishable, and the absolute values
are a noise floor — 0.4% of patches surviving is RANSAC fitting an affine to
four or five coincidental matches. Real correspondence appears in exactly one
place: same-session SIFT (max 1.14, >100 inliers), two segments of **one
recording**, where viewpoints genuinely overlap.

Classical place recognition assumes overlapping views of rigid structure. Two
visits to the same bench on different days, on a head-mounted camera, share
almost no viewpoint. **So BORDERLINE is the correct verdict for those pairs,
not a placeholder**, and only human judgement moves them.

### One finding that cuts against the above, and matters more than the rest

The `global` row there scored **0.5051** — chance — where production `env_raw`
scores **0.8182** on the same pairs. The difference is that `global` is plain
CLS chamfer: no corpus whitening, no static body mask, none of the cos/bhat
terms.

**Most of the env axis's power on hard pairs is the whitening, not raw DINOv2
similarity.** Consistent with the 23× whitening effect in `docs/08`, but larger
than anyone here had appreciated. Two consequences: the comparison above was
*generous* to patch matching and it still lost; and if the env axis ever needs
improving, whitening and calibration are where the leverage is, not the
backbone.

### task1 reproduction

```bash
# environment axis (embed then compare; presigned HTTP range reads, no bulk download)
.venv/bin/python scripts/match_env.py embed   --manifest /tmp/manifest_435.json --out env_sigs
.venv/bin/python scripts/match_env.py compare --sigs env_sigs --out env_pairs.csv

# task axis + quadrant, in place on the pair CSVs
.venv/bin/python scripts/task_axis.py env_pairs.csv env_pairs_by_industry.csv

# the negative result, if anyone proposes local matching again
.venv/bin/python eval/exp_local_place_frames.py && .venv/bin/python eval/exp_local_place_matching.py
```

`eval/human_labels_all.csv` holds all 138 judgements across the three rounds,
so both operating points can be re-derived from raw labels rather than trusted.

### What is left on task1

- ~~401 BORDERLINE pairs~~ **CLOSED.** Round 3 labelled 50 of them; band AUC
  0.6398 says the axis is blind in there, so they are collapsed to
  DIFFERENT_ENV at a measured 90.9% hit rate. Do not spend more labelling
  effort on this band — three rounds converged and the last two added no
  threshold movement at all.
- **71 pairs have no task description** (`NO_TASK_DATA`) — 2 episodes missing
  from the QA export.
- One clip never embedded: `c9aaec51-2cd8-5145-8f22-46a31a35e733` (ffmpeg
  timeout). 434 of 435.

---

## 00. task2 — cycle-aware chunking (2026-09-21, PARKED)

> **Parked on frame rate, checked 2026-09-22.** `model_output/` now holds 522
> episodes, newest write 2026-09-22 00:19 UTC — and `step`/`fps` read out of
> the NPZ (not inferred from file size) are `step: 10, fps 30` on 22 of 24
> sampled across the whole timeline, i.e. **3 fps effective, all of them**.
> Files grew from 20 KB to ~1 MB, which looks like a rate change and is not:
> longer episodes, more detected hands.
>
> A separate `model_output_fullrate/` prefix holds **8 episodes at `step: 1`**
> (true 30 fps, 11–17 MB NPZ, written 2026-09-21 18:10). Only 3 of the 8 are in
> the 435. So someone is already regenerating at full rate into a different
> prefix — pointing that job at the 435 is the unblock; 432 to go.
>
> At 3 fps a 1-second grab-and-release is 3 samples, so the period estimate has
> nothing to lock onto. **Do not run the cycle pipeline on step-10 data.**
> Poll BOTH prefixes.

**The project now has a commercial target with 435 labelled examples.** Customer
QA rejected 766 episodes. 435 (57%) say *"Repetitive motion or repeated simple
work; Needs a shorter usable segment"* (median 366 s, hands active 95%, on
objects 90%); 312 (41%) say *"Idle or stalled task progress"*. The deliverable:
given a long repetitive episode, output the shorter segment a customer would
accept. Full reasoning and every failed approach is in
**`docs/08-cycle-segmentation.md`**. `rejected_episodes.csv` is the label file
(gitignored — it carries per-episode customer data).

**What changed mechanically:** v1 cut a strict 30 s grid, slicing through work
cycles at a random phase. v2 derives cut points from **hand-track periodicity**
so each chunk is a whole number of repetitions starting at the same phase. New
module `novelty/cycles.py`, new runner `scripts/run_v2.py`.

**The most important result is a negative one:**

> **[Certain] Customer-"repetitive" and signal-"periodic" are different
> properties, and only the second yields cut points.** Of 5 episodes
> ground-truth-rejected as repetitive, **1** had detectable cadence. Of 8
> Pipe_Factory segments, **0**. Of 8 prod episodes, **2 (25%)**. Best
> periodicity strength: 0.89 for the one machine-paced case, 0.24–0.51 for the
> rest; widening the period search 8 s → 40 s changed nothing. Cycle
> segmentation serves the machine-paced **minority (~15–25%)** — it is not a
> replacement for fixed chunking, and most of the 766 will yield empty CSVs.

`trackable_fraction` and `cadence_fraction` are reported separately so "we could
not see the hands" is never mistaken for "this work is not repetitive".

| phase | what | measured | 766 projection |
|---|---|---|---|
| A | segment from NPZ, no video | **56 ms/episode** | ~45 s |
| B | download → cut → DINOv2 | **6.8 s/chunk + 5.2 s/episode** | ~2.25 h @25% yield |
| C | global whitener + null → per-episode CSV | seconds | < 1 min |

Outputs: CSVs → `labelling_results/novelty_result_v2/`, chunks →
`novelty_data_v2/`. Phase B is idempotent (per-episode `done` marker), so the
runner is safe to kill and restart at any point.

**`run_v2.py drain`** polls prod for new NPZs every 5 min and processes what has
landed. **Poll BOTH `model_output/` and `model_output_fullrate/`** — both are
written concurrently and each holds episodes the other does not, so polling one
silently loses episodes (verified by set difference).

**The binding constraint is not this pipeline.** Their hand-detection model
produced 11 episodes in ~35 min ≈ **36/hour**; 766 needs ~60/hour to hit the
deadline. Levers: more workers on their side (`_launch.sh` runs 2 per host), or
process only the 435 repetitive-rejected episodes, which makes it at 36/hour.

**Not carried over from v1:** the calibrated decision layer. AUC 0.998 and
`env_percentile: 60` were measured on DINOv2 *appearance* features from video;
pose-only similarity is **uncalibrated** until there is ground truth for "same
action". Tier 0 cannot fire on the NPZ-only route (no frames → no pHash).

**No `head.npz` in the prod output.** Ablation showed head pose prevents a
specific failure: without it, one episode's period read 2.55 s instead of
1.30 s (harmonic doubling), which halves or doubles every boundary. Camera
frame works, but ask for head pose if it is cheap to emit.

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
3. **`decision.env_percentile` moved 97 → 52 → 60**, because it is measured
   rather than guessed, and it moved again the moment a third environment
   arrived. It is corpus-dependent — read §5.1 before trusting it anywhere else.
4. **The quadrant model is switched off. The verdict is now environment-only**
   (`decision.use_task_axis: false`). This is a retreat from the repo's founding
   design and it was forced by measurement, not preference — see §4.1a. The task
   axis ranks an automobile plant as more task-similar to pipe-factory A than
   pipe-factory B is. Read that before you turn it back on.
5. **The environment is no longer the 2-core container described in old §9.**
   It is a Linux box with an A10G, video is never kept on local disk, and
   results go to S3. §9 is rewritten.

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
| 0 — perceptual hash | did these share literal footage? | **works**; false-positive on repetitive static-view work fixed 2026-09-20 (§7 bug 10) |
| 1 — appearance | same physical place? | **the product.** Decides the verdict alone. 0.973 accuracy over 741 pairs, 3 environments |
| 2 — motion/task | same work happening? | **switched off** (§4.1a). Reported as a diagnostic, does not gate anything |

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

### 4.1a …and yet it IS collapsed right now. Read why before restoring it.

**Status 2026-09-20: `decision.use_task_axis` is `false`. The verdict is
environment-only — `REDUNDANT` or `NOVEL`, plus tier 0.** §4.1's argument above
is still correct in principle, which is why it is left standing. What changed is
that we measured the task axis against three environments and it is not fit to
name a quadrant:

| | environment axis | task axis |
|---|---|---|
| best F1 (741 pairs, 3 sessions) | **0.969** | 0.905 |
| zero-false-positive threshold | 62.5, keeps **90%** of true redundants | 80.5, keeps **47%** |
| correlation with the other axis | — | **+0.82** |
| orders domains correctly? | yes | **no** — see below |

The disqualifying result: on `task_raw`, pipe-factory-A vs pipe-factory-B scores
**−0.032**, while pipe-factory-A vs an **automobile plant** scores **+0.121**.
The task axis says a different industry is *more* task-similar than a different
station in the same industry. A detector that inverts across whole domains
cannot be allowed to decide between `REDUNDANT` and `SAME_PLACE_NEW_TASK` — it
produced exactly that failure in practice, labelling automobile-vs-pipe-factory
pairs `SAME_PLACE_NEW_TASK`.

**Why the task axis is hard here, beyond the label confound (§6):** this is
cluttered egocentric factory footage. Other workers move through frame, machines
run in the background, and the wearer's head turns constantly. Dense flow
integrates *all* of that, so the descriptor is dominated by scene activity
rather than by what the wearer's hands are doing. Meanwhile `task_cos` comes
from V-JEPA 2, which reads RGB and so re-imports appearance (§5.5). Between
them, very little of the task axis is actually about the task.

**What we knowingly gave up:** two different jobs filmed in the same room now
both read `REDUNDANT`, and one gets dropped. That is the single most valuable
sample type for a physical-AI dataset. We accept that cost because we cannot
currently *detect* that case — being honestly coarse beats being confidently
wrong. The switch is one config field; flip it back when §6's eval set exists
and the task axis clears a measured bar on it.

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

### 5.0a Three environments (2026-09-20) — the test that changed the design

A third session was added: `Automobile_Manufacturing / RCB-13`, 3 × 30 s chunks
(black metal automotive parts, grinding at a machine, metal floor plate).
Against the two Pipe_Factory sessions this gives two *tiers* of negative:

* **hard** — S36 vs S9: same industry, different station
* **easy** — either vs AUTO: different industry entirely

A metric that measures environment should separate the easy pair further than
the hard one. **It does not:**

| pairing | `env_raw` |
|---|---|
| within S36 | +0.559 |
| S36 vs S9 (hard negative) | **+0.096** |
| S36 vs AUTO (easy negative) | **+0.090** |

**[Certain] The environment axis saturates.** Beyond "not the same place" it
carries no information about *how* different two places are. It is a good
detector and a bad distance. Do not build anything that needs graded
environmental distance on top of `env_raw` without re-checking this.

Caveat on the strength of this test: the corpus is 36 pipe-factory signatures
against 3 automobile ones, so the whitener is ~92% pipe factory and the AUTO
chunks sit far from a mean they barely influenced. **Ingest ~18 automobile
chunks before treating the saturation result as final.**

End-to-end accuracy of the shipped rule over all 741 pairs, ground truth =
session identity:

```
correctly dropped (redundant) 293    correctly kept 428
FALSE DROP (different env)      4    missed redundancy 16
precision 0.987   recall 0.948   accuracy 0.973
```

### 5.1 Threshold: why `env_percentile` moved 97 → 52 → 60

The best-F1 operating point above sits at the **52nd percentile** of the corpus
null. The repo shipped 97.

A percentile threshold asks *"is this pair in the top (100−p)% of my corpus"*.
It only means "same environment" if same-environment pairs are about that rare.
This corpus is **49% same-environment pairs**, so 97 can only ever fire on
near-duplicates — which is exactly why two adjacent chunks of one continuous
recording kept returning `NOVEL` and looked like a broken detector.

**This value is corpus-dependent** — and that is not a theoretical warning, it
has already happened once. Adding the third session dropped the
same-environment base rate from 49% to 41.7% and moved best-F1 from 52 to
**60**. Left at 52, **11.3% of genuinely different pairs (49 of 432) were
called same-place**, including 13 automobile-vs-pipe-factory pairs. Threshold
sweep over the three-session corpus:

| threshold | false positives | missed | precision | recall | F1 |
|---|---|---|---|---|---|
| 50 | 63 | 1 | 0.830 | 0.997 | 0.906 |
| 55 | 28 | 4 | 0.916 | 0.987 | 0.950 |
| **60** | **4** | **16** | **0.987** | **0.948** | **0.967** |
| 62.5 | **0** | 32 | **1.000** | 0.897 | 0.946 |
| 65 | 0 | 49 | 1.000 | 0.841 | 0.914 |

**The two errors are not symmetric.** A false `REDUNDANT` *drops* footage and is
unrecoverable; a false `NOVEL` merely keeps something you did not need. That
asymmetry argues for 62.5 over 60 — 62.5 is zero-false-positive on this corpus
and still catches 90% of true redundants. We ship 60 (best F1) because zero-FP
on 741 pairs is thin evidence, but **raise it if a dropped clip is expensive to
you.** `calibrate --labels` prints the best-F1 point; re-fit rather than
inheriting any of these numbers.

`task_percentile` stays at 95 and **no longer affects the verdict at all**
(§4.1a). It is still computed and still written to `pairs.csv`, as a diagnostic.

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

**Task axis: NO LONGER BLOCKED for task1 — see §000.** The `task_description
(our)` column in the QA export supplies a whole-video task descriptor for
13893/13964 pairs, AUC 0.9768 against `canon_task×site_h`, within-session
confound +0.243. Everything below is still true of the *pixel* task tier and
of task2's within-video comparison, where the missing quadrant still bites.

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

### Found 2026-09-20

**10. Tier 0 counted votes instead of measuring a run — `hashing.source_overlap`.**
`chunk02_030-060` vs `chunk03_060-090` of the automobile session — **disjoint**
segments — were labelled `DUPLICATE_SOURCE` off **4 scattered** matching pHash
frames that happened to land in one offset bin: 4 × 0.5 s = exactly the 2.0 s
`min_overlap_seconds`, at 7% of each clip.

They share no footage. On repetitive manual work with a near-static head pose,
individual frames seconds apart genuinely *are* near-identical pixels — that is
pHash working correctly. What two disjoint segments never contain is a
continuous **run** of identical frames; only actually shared footage does.
Fix: `overlap_seconds` is now the longest *contiguous* run of aligned frames
(`max_gap_frames=2` tolerates a dropped match), not the total vote count, and
the run must itself clear `min_votes`. Re-encode detection is unaffected —
those produce long runs. Regression test:
`test_hashing.py::test_scattered_matches_are_not_shared_footage`.

**This one mattered disproportionately:** `DUPLICATE_SOURCE` overrides every
semantic axis, so a false positive silently discards novel footage with no
appeal. → *When a rule overrides all others, its false-positive behaviour is
the only behaviour that matters.*

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

### P0 — ~~unblock the TASK axis~~ **RETIRED 2026-09-22 (§000)**
This asked for new `SAME_PLACE_NEW_TASK` recordings before the task tier could
be measured. It was answered without them: the QA export's task descriptions
gave a text task axis at AUC 0.9768, and the 138 human labels supplied the
positives the eval set lacked. **The lesson is worth keeping — the data needed
to unblock this had been sitting in a CSV in the repo the whole time.** Look
there before asking for a new capture.

The request still stands for the *pixel* task tier and for task2: a paired
capture of two different jobs in the same station, same worker, same lighting
is the only way to measure those. It is no longer on the critical path.

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
output/<mirrored source key>/
    chunks.csv   one row per chunk: span, detected period + strength, nearest
                 neighbour and which session it is in, env/task percentiles,
                 verdict, within- vs cross-session similarity, coverage
                 selection rank and marginal gain
    pairs.csv    every comparison behind those numbers
output/videos.csv    VIDEO-level similarity, one row per pair of recordings
output/summary.csv   per-session within/cross means
```

### Video-level matching (`videos.csv`)

A 10-minute recording is **not** turned into one vector — averaging its chunks
deletes precisely the brief exception that makes a recording worth keeping. A
video is its *set* of chunk signatures, and two videos are compared set-to-set,
the same operation `env_chamfer` already performs one level down over frames.

```
coverage(A|B) = fraction of A's chunks whose best match anywhere in B
                clears decision.env_percentile
```

**Directional on purpose.** A short recording can sit entirely inside a longer
one while the longer one still holds material the short one never saw, and one
symmetric number cannot express that. Measured on the current corpus:

| pair | cov(A\|B) | cov(B\|A) | verdict |
|---|---|---|---|
| RCB-13 vs PIP-246 | 0.00 | 0.00 | `DISTINCT` |
| RCB-13 vs PIP-250 | 0.00 | 0.00 | `DISTINCT` |
| PIP-246 vs PIP-250 | **1.00** | **0.72** | `A_CONTAINED_IN_B` |

Read that last row: every PIP-246 chunk has a match in PIP-250, but 5 of
PIP-250's 18 do not have one in PIP-246 — so PIP-250 is the superset and
PIP-246 adds nothing over it. That is the "which 200 of my 2000 hours" question
answered at video level.

> **[Certain] Do not over-trust the 0.72.** Those 5 "novel" chunks score 56.0,
> 57.5, 58.7, 59.5 and 59.8 against a threshold of 60 — every one of them is
> within 4 points of the cutoff. At a threshold of 55 the coverage would read
> 1.00 and the conclusion would invert. The distribution is dense exactly where
> the threshold sits, so coverage is fragile on this corpus. Report the
> `mean_best_env_pct` columns alongside it, which are threshold-free.

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
| `docs/08-cycle-segmentation.md` | **v2**: hand-track cycle cutting, the four traps, measured yield |
| `eval/pairs.yaml` | the label file — its header documents its own inadequacy |
| `scripts/report_csv.py` | the `output/` CSVs: what each column means |
| `scripts/run_v2.py` | the v2 three-phase runner and the drain loop |
| `configs/gpu-cosmos.yaml` | why Cosmos-Embed1 is task-axis-only, and the fps/clip_len reasoning |

**Docs not yet updated for 2026-09-18.** `docs/04-calibration.md` and
`docs/07-tuning.md` still describe `env_percentile: 97` as the operating point,
and `docs/06-running-on-gpu.md` still speaks of the GPU path as untried. The
code and this file are correct; those three docs lag. Fix them when you touch
that area.

Code comments at every non-obvious decision explain the *why*, especially at the
ten bug sites in §7. If you change something there, update the comment; the
next agent after you will rely on it the way you are relying on this file.
