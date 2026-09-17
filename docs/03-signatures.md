# 03 — Signatures: every feature and why it earns its keep

## Tier 0 — perceptual hash

**What it sees:** the low-frequency DCT structure of each sampled frame,
reduced to 64 bits.

**Algorithm.** Decode at 2 fps into 32×32 grayscale. 2-D DCT. Take the top-left
8×8 block (the lowest spatial frequencies — the coarse layout of the image,
which is what survives compression and rescaling). Drop the DC term, because it
only carries overall brightness. Threshold the remaining 63 coefficients at
their median. That is the hash.

**Matching.** For every near-matching frame pair `(i, j)` — Hamming ≤ 8 — cast a
vote for the time offset `t_b[j] − t_a[i]`. Shared footage puts a tall spike at
one offset; unrelated videos give a flat histogram. The spike height times the
sampling interval is the overlap in seconds. This is a 1-D Hough transform and
it is robust to the occasional spurious hash collision in a way that
"count matching frames" is not.

**Why it belongs in a separate tier:** see [01 §3](01-why-not-just-cosine.md).
It answers a question the semantic tiers cannot, and it must not be blended
with them.

**Cost:** ~1 s per minute of video. Always leave it on.

---

## Tier 1 — appearance

### Frame sampling

48 frames evenly spaced across the span, decoded at 384×216 with letterboxing.
Evenly-spaced-by-resampling, not 48 independent seeks — one sequential decode is
an order of magnitude cheaper.

Letterboxing (`appearance.letterbox: true`) preserves aspect and pads to the
box, so geometry is comparable across sources with different aspect ratios. The
alternative (stretch) is fine if every video in your corpus has the same aspect,
and slightly cheaper.

### The egocentric body mask

This is the feature most specific to head-mounted footage, and the one most
likely to be quietly costing you accuracy if you skip it.

On a head camera the wearer's own torso, forearms and trouser legs occupy a
large, **constant** region at the bottom of every frame they will ever record.
Two clips of completely unrelated work, shot by the same person on the same day,
share that region exactly. A chunk of whatever similarity you measure is
therefore *trouser recognition*, not environment recognition.

`static_weight_map` estimates the region from the data rather than hard-coding a
crop: compute the per-pixel temporal standard deviation across the sampled
frames, blur it, and down-weight pixels below the 20th percentile to 0.15.
Things that never change get suppressed; things that move do not.

```yaml
appearance:
  static_mask: true
  static_percentile: 20.0   # raise to suppress more
  static_floor: 0.15        # 0.0 = erase entirely; 1.0 = disable
  bottom_crop: 0.0          # optional extra hard crop, e.g. 0.2
```

For the learned backbones the mask is applied as a soft blend **toward grey**,
not toward black — a hard black rectangle is itself a strong, spurious feature
for a ViT.

> Caveat worth knowing: on a *tripod* the whole scene is static, so this mask
> will suppress the scene itself. Turn it off (`static_mask: false`) for fixed-
> camera footage. It is a fix for egocentric video specifically.

### The encoders

| name | dim | needs | use it when |
|---|---|---|---|
| `gist` | 564 | nothing | CI, laptops, plumbing, first look at a new corpus |
| `dinov2` | 768 | torch | **the default on a GPU box** |
| `dinov2-large` | 1024 | torch | when 768 is measurably not enough |
| `siglip` | 768 | torch | when you also want to query the index with text |

`gist` is 3 scales × 4×4 cells × 8 gradient orientations (384) plus 3×3 cells ×
(12 hue + 4 sat + 4 val) (180), each block L2-normalised before the whole vector
is. It is a respectable classical scene descriptor and it is genuinely useful —
on the real footage in this repo it separated factory from construction site
with AUC 0.997 — but it is not robust to lighting change the way a learned
backbone is. Treat it as the thing that proves your pipeline runs, not the thing
you ship.

**DINOv2 over CLIP, deliberately.** CLIP is text-aligned, which means it
collapses visually distinct scenes that share a caption — every room that is "a
factory" lands in the same place. That is the exact failure mode we are trying
to avoid. DINOv2 is self-supervised on images and keeps instance-level
appearance.

### The three environment metrics

`env_cos` — cosine between mean vectors. Coarse, cheap, and the first thing to
saturate. Weighted lowest.

`env_chamfer` — symmetric mean-of-max similarity between the two *sets* of frame
vectors. Asks "does every moment of A have a counterpart somewhere in B", rather
than "do these average out to the same thing". This is what makes the split-a-
long-video case work: the mean vectors drift apart as the worker moves down the
line, but every frame still has a near neighbour. There is a unit test
(`test_chamfer_beats_mean_cosine_on_partial_overlap`) that pins this property.

`env_bhat_sim` — `exp(−Bhattacharyya)` between diagonal Gaussians fitted to the
frame vectors. Compares **spread**, not just centre. Two clips can have nearly
identical mean vectors while one roams the whole building and the other stares at
a single bench; this term separates them. Averaged per-dimension so the magnitude
doesn't scale with embedding width.

---

## Tier 2 — motion

### The ego/object split

The median optical-flow vector over the frame is, to first order, the camera's
own movement. Subtract it and what remains is flow caused by things moving in
the world — hands, parts, the conveyor.

On a head-mounted camera this split is not optional. Without it you are mostly
measuring how much the wearer turned their head, which is a property of the
person and the day, not of the task.

### The flow histogram

Residual flow, binned into 2×2 spatial cells × 8 orientations × 4 magnitude
bands = 128 bins, L1-normalised, plus three scalars: mean residual magnitude,
ego-motion magnitude, and the fraction of pixels moving.

**The scale bug, because you will hit the same one somewhere else.** Those three
scalars have raw values around 2.4, 2.1 and 0.98. The whole 128-bin histogram
has L2 norm around 0.1. Cosine similarity is scale-sensitive *across blocks*, so
in the first version of this code the 128 numbers describing the task
contributed about 4% of the vector and the three numbers saying "this is
handheld video" contributed the rest. Every pair scored 0.995+, including pairs
from completely different worksites. The metric was a constant.

The fix is three lines: L2-normalise each block independently, compress the
unbounded scalars with `log1p`, and down-weight the tiny scalar block
(`SCALAR_W`, `SCALAR_BLOCK_W` in `motion.py`). Afterwards `task_cos` spread from
[0.995, 0.998] to [−0.19, +0.28].

### The rhythm spectrum

The power spectrum of per-frame motion energy, log-binned into 24 bands and
L1-normalised.

This is the most useful feature for repetitive manual work and it is almost
never used. A worker on a 4-second pick-and-place cycle puts a peak at 0.25 Hz.
Two clips of the same repetitive task match here **even when they are
phase-shifted** — which is precisely the "I cut a 10-minute video into two
5-minute halves" case. Two different tasks in the same room do not match, even
though their appearance embeddings are nearly identical.

Compared in Hellinger space (`sqrt` of the distribution) and **centred**, not
scaled, against the corpus mean spectrum: the question that discriminates tasks
inside one factory is "how does this clip's rhythm differ from the average
rhythm here", and rescaling per band would destroy the distributional meaning.

### Cycle period and strength

Autocorrelation of the same (smoothed) motion-energy signal → dominant period in
seconds, and the normalised autocorrelation at that lag as a confidence.

Two bugs are worth knowing about because both produce plausible-looking output:

1. **Unsmoothed energy.** Raw per-frame motion energy on a head camera is
   dominated by high-frequency jitter. The autocorrelation then peaks at the
   shortest lag it is allowed to, and every clip reports a "period" exactly
   equal to the search floor. It looks like a result. Smoothing at ~0.5 s fixes
   it (`motion.smooth_s`).
2. **argmax instead of peak-picking.** A genuine cycle is a *local maximum* of
   the autocorrelation. A plain argmax picks up the monotone shoulder of the
   zero-lag peak instead. Same symptom.

`period_agreement` accepts 1×, 2× and 3× harmonics (a detector that locks onto a
half-cycle is still describing the same work) with a deliberately tight
tolerance — at a looser one, a 1.58× ratio scored 0.49 as a "near-2× harmonic"
and genuinely different cadences agreed.

**Window length constraint:** autocorrelation needs roughly five or six
repetitions before the estimate stops being noise. A 6-second cycle needs a
window of ≥ ~35 s. If your windows are shorter than that, expect `rhythm_cos`
and `period_agree` to contribute nothing, and set their fusion weights to zero
rather than letting them add variance.

### The learned alternative (GPU)

`motion.clip_encoder: vjepa2` replaces `motion_pooled` with a sliding-window
V-JEPA 2 embedding. The flow encoder still runs alongside, because rhythm and
cycle period are cheap, interpretable, and **not recoverable** from a pooled
clip embedding — a 16-frame window is about half a second and cannot see a
6-second work cycle.

### DTW

`task_dtw_sim` is a band-constrained DTW alignment cost between the two
per-frame descriptor sequences, normalised by path length.

Why DTW and not just pooled cosine: two runs of the same repetitive task are the
same *sequence of sub-actions* at slightly different speeds and phases. Pooling
throws the ordering away; DTW keeps it while tolerating the speed difference.
The Sakoe-Chiba band keeps it O(n·band) and also prevents degenerate alignments
that map everything to one frame.

→ Next: [04 — Calibration](04-calibration.md)
