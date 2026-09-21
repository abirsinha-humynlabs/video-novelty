# 08 — Cycle-aware segmentation from hand tracks

v1 cut every recording on a strict 30 s grid. That grid is blind: a 30 s chunk
of repetitive manual work contains ~23 work cycles starting at an arbitrary
phase, so two chunks of the *same* job are misaligned by a random fraction of a
cycle and every comparison pays for it.

v2 cuts where the hand tracks say a work cycle begins, so each chunk is a whole
number of repetitions starting at the same phase. Comparison becomes
apples-to-apples.

This document is mostly a list of things that did **not** work, because each one
looks plausible and costs a day to rediscover.

---

## Why this exists commercially

766 episodes were rejected by customer QA. The reasons:

| reason | n | median dur | hand_active% | ml_hoi_ratio |
|---|---|---|---|---|
| Repetitive motion / repeated simple work | 435 (57%) | **366 s** | 95.5% | 0.90 |
| Idle or stalled task progress | 312 (41%) | 68 s | 87.5% | 0.73 |
| Working hands not visible enough | 26 (3%) | 51 s | 83.6% | 0.61 |

Nearly all add *"Needs a shorter usable segment."* The repetitive ones are not
idle — hands are busy 95% of the time and on objects 90% of the time. The
customer is saying *the same thing happens for six minutes, give us the
informative slice*. Cycle segmentation finds the repetition unit; coverage
selection (docs/05) picks which units to keep.

**The two buckets are opposite failures and only the first is served by this
document.** Cycle-chunking will produce nothing useful for "idle/stalled".

---

## Input

Output of the hand-stabilisation pipeline
(https://github.com/Maiemdiab/egocentric-hand-stabilisation):

```
kp3d_cam   (N, 21, 3)  camera-relative metres, OpenPose-21:
                       0=wrist, 1-4 thumb, 5-8 index, 9-12 middle,
                       13-16 ring, 17-20 pinky
frame_idx  (N,)        hand (N,)  0=LEFT 1=RIGHT 2=bystander -1=dropped
is_wearer  (N,)        kept (N,)  fps  K  width  height
```

`hand == 2` and `is_wearer == False` are dropped at load. A second person
working in frame has their own cadence, and mixing it in is indistinguishable
from the wearer changing rhythm.

Optionally a head `.npz` with `T` (F, 4, 4). **`T` is camera-to-world** — not
documented anywhere, resolved empirically: cam→world decorrelates wrist speed
from head angular speed (r = 0.06) better than its inverse (r = 0.23).

---

## Four traps

### 1. Never use speed. Use a signed projection.

Wrist *speed* is rectified, so a reach-and-return traces two peaks per cycle:
the period comes out halved and the phase is destroyed. Measured periodicity
strength on the same footage:

| signal | strength |
|---|---|
| wrist speed | 0.21 |
| **wrist position projected on its own principal axis (signed)** | **0.88 / 0.92** |
| inter-hand distance | 0.54 |
| grasp aperture (thumb-tip↔index-tip) | 0.45 |

That one change is the difference between this working and not.

### 2. Remove a LOCAL baseline before looking for zero-crossings.

The worker walks around the workspace, so the projection drifts further than it
oscillates and stops crossing a global threshold for long stretches — 50
detected boundaries where ~230 were expected. A running median over ~3 cycles
fixes it: 272 boundaries, interval sd 11.25 s → 0.64 s.

### 3. Never interpolate across long tracking gaps.

Hand tracking drops out for up to 11 s at a time (hands into a sack, bystanders
stealing the track). A global linear fill replaces the oscillation with a
straight line, periodicity reads zero, and the result *looks* like "this work is
not repetitive" when the truth is "we could not see the hands". Those are very
different conclusions, so `trackable_fraction` is reported separately from
`cadence_fraction`.

### 4. The window must be ~10x the period, not 5x.

Autocorrelation needs ~5 repetitions (`MIN_REPS`), so the search ceiling is
`window / MIN_REPS`. A window of exactly `MIN_REPS * P` puts the true peak at
lag P *on the boundary*, where it cannot be a local maximum and is missed
entirely. `WINDOW_REPS = 10` puts the ceiling at 2P and the peak mid-range.

The original code searched to 8 s inside a 6 s window — internally impossible,
and it made every task with a cycle above ~1.2 s structurally invisible.

**Consequence worth internalising: detecting a cycle of length P requires
`MIN_REPS * P` seconds of *continuously tracked* hand.** A 20 s assembly cycle
needs ~100 s of unbroken tracking. That is a data requirement, not a knob.

---

## Nothing may assume a period

The cycle length is a property of the task: a pick-and-drop repeats in ~1 s, a
gather-and-bag or assembly cycle takes tens of seconds. So `estimate_period`
runs first over the longest tracked runs with a wide search, and the detrending
baseline (3P), analysis window (10P) and local search range (P/2.5 … 2.5P) are
all derived from the result.

---

## Measured yield — the honest part

| corpus | trackable | cadenced | outcome |
|---|---|---|---|
| Polymer_Bags (machine-paced packing) | 89% | 32%, P=1.30 s | **11 chunks**, cv 0.14 |
| 8 Pipe_Factory segments | 13–74% | 0–1% | **0 chunks** |
| 5 rejected-as-repetitive episodes | 0–96% | 0–16% | **1 of 5** produced chunks |
| 8 prod episodes | — | — | **2 of 8** (25%) |

**[Certain] Customer-"repetitive" and signal-"periodic" are different
properties, and only the second yields cut points.** "Picking up cuttings and
inserting them into trays" is repetitive work — same action, low information —
but each repetition takes a variable time with pauses and occasional fumbles.
The task repeats; the kinematics never lock to a metronome. Best achievable
strength was 0.89 for the one machine-paced episode and 0.24–0.51 for the rest,
and widening the period search from 8 s to 40 s changed nothing.

So **cycle segmentation is a conditional method serving the machine-paced
minority (~15–25%), not a replacement for fixed chunking.** The condition is
measurable before committing any GPU time: `trackable_fraction` and
`cadence_fraction` separate the cases cleanly (89%/32% vs 74%/0%).

Also: even on Polymer_Bags the cadenced chunks cover only **25%** of the
recording. The other 75% is transitions, walking, fetching stock —
non-repetitive and therefore *more* novel per second than the repetitive work.
`transition_segments()` returns them labelled rather than discarding them.

---

## Choosing the chunk unit

Two schemes, measured on the same episode:

| scheme | n | mean | sd | **cv** | cycles/chunk |
|---|---|---|---|---|---|
| **N whole cycles (N=6)** | 9 | 5.38 s | 0.69 s | **0.13** | 6–6 |
| 10 s target, cycle-aligned | 14 | 5.43 s | 2.59 s | 0.48 | 1–11 |

**Fixed cycle count wins.** The duration-target scheme degenerates because
cadence runs are usually shorter than the target, so it hits the end of a run
and emits fragments — including 1-cycle, 1.33 s chunks. Fixed count gives
uniform, directly comparable chunks.

Note a 1.30 s cycle means one chunk = one cycle is **far** too short to embed.
N is the knob: N=6 ≈ 8 s, N=15 ≈ 20 s.

---

## Comparing the chunks

Same three steps as docs/04, and **whitening matters more here than anywhere
else**: within one episode every chunk shares the same hands, person, bench and
lighting, so the common component is maximal. Measured over 9 chunks of one
episode:

| pairwise cosine | mean | sd | range | **spread** |
|---|---|---|---|---|
| raw | +0.9843 | 0.0105 | [0.953, 0.996] | **0.043** |
| whitened | −0.1220 | 0.2650 | [−0.608, +0.412] | **1.019** |

A 23× expansion of usable spread. Raw, every chunk sits in a 0.04-wide band at
0.98 and the number is decorative.

**Fit the whitener and null GLOBALLY, across episodes — never per episode.**
Mean-centring *n* vectors mechanically forces the average pairwise cosine to
≈ −1/(n−1); with 9 chunks that predicts −0.125 and we measured −0.1220. So a
per-episode fit gives rankings you can use but numbers you cannot interpret.

---

## What is NOT carried over from v1

The validated decision layer. The AUC 0.998, 4.40σ separation and
`env_percentile: 60` threshold were all measured on **DINOv2 appearance
features from video** (docs/04, WORK.md §5). Pose descriptors say nothing about
the environment, so pose-only similarity is **uncalibrated** until there is
ground truth for "these two cycles are the same action". Tier 0 also disappears
on the NPZ-only route: no frames, no pHash, so `DUPLICATE_SOURCE` cannot fire.
