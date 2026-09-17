# 07 — Tuning and failure modes

## Every knob that matters

| key | default | raise it when | lower it when |
|---|---|---|---|
| `segment.seconds` | 30 | your clips contain several distinct activities | you need finer-grained selection |
| `segment.hop` | 15 | indexing is too slow (set = `seconds`) | boundaries are cutting through work cycles |
| `appearance.n_frames` | 48 | the scene changes a lot within a window | indexing is too slow |
| `appearance.static_percentile` | 20 | the wearer's body fills more of frame | on a tripod — or just disable the mask |
| `appearance.static_floor` | 0.15 | you want the mask gentler (1.0 = off) | you want static regions erased harder |
| `motion.fps` | 10 | the work is fast and fine-grained | indexing is too slow (6 is usually fine) |
| `motion.smooth_s` | 0.5 | the camera is very shaky | the work cycle is shorter than ~2 s |
| `hashing.max_hamming` | 8 | your re-encodes are aggressive and tier 0 misses | tier 0 fires on unrelated clips |
| `decision.env_percentile` | 97 | too many false "same place" | you're missing real duplicates |
| `decision.task_percentile` | 95 | too many false "same task" | you're missing real duplicates |

## Failure modes, and how you will recognise each one

### Every pair scores ~0.95 and nothing separates

**You skipped calibration.** `compare` will have told you so in a note. Run
`novelty calibrate`. If you already did and it's still flat, your corpus is too
small for the whitener to estimate anything — the warning at <50 signatures is
literal.

### Everything is at the 50th percentile

Your null distribution is your positive distribution. This happens when the
corpus is dominated by one site, so "a random pair" *is* "two clips of the same
place". Add material from other sites, or accept that within-site percentiles
are all you can measure and set thresholds much higher (99+).

### `period_s` is identical across every clip and equals ~0.4 s

The motion-energy signal is not being smoothed, so the autocorrelation is
peaking at the search floor. Check `motion.smooth_s > 0`. This one looks like a
result, which is why it's worth checking explicitly.

### Task scores are flat but environment scores are fine

Most likely one of:

1. **Windows too short for the cycle.** Autocorrelation needs 5–6 repetitions.
   A 6 s cycle needs ≥ ~35 s windows. Check whether `calibrate --labels` gave
   `rhythm_cos` and `period_agree` weights of 0.00 — that is this diagnosis.
2. **A block-scale problem** like the one documented in
   [03](03-signatures.md#the-flow-histogram). If you add features to the motion
   descriptor, normalise each block independently or the largest-magnitude block
   will eat the cosine.
3. **Flow features aren't enough for your task.** Move to
   `motion.clip_encoder: vjepa2`.

### Unrelated clips flagged `DUPLICATE_SOURCE`

Lower `hashing.max_hamming` to 6. If the content is genuinely periodic (fixed
camera on a machine with an exact cycle), tier 0 is answering honestly — frames
one period apart *are* the same pixels — and you should raise
`min_overlap_seconds` well above the cycle length or disable tier 0 for that
source.

### Two clips of obviously the same place score as different environments

Check whether `static_mask` is eating your scene. On a **tripod** everything is
static, so the mask suppresses the whole frame. Set `static_mask: false` for
fixed-camera footage — the mask is a fix for egocentric video specifically.

### Indexing is too slow

In order of payoff: `--no-motion` for a first pass → drop `motion.fps` to 6 →
run N processes over disjoint file lists → merge the three ffmpeg decodes into
one `filter_complex`. See [06](06-running-on-gpu.md).

## Choosing thresholds

Don't guess them. Run `novelty calibrate --labels eval/pairs.yaml` and read
`best_threshold` off the output — it's the F1-optimal operating point on your
labels, already expressed as a percentile.

If you have no labels and cannot get any, start at
`env_percentile: 97, task_percentile: 95` and adjust by looking at what
`novelty search` returns for a clip you know well. But understand that you are
tuning against your intuition rather than against data, and the numbers will not
survive a change of site.

## Adding a new appearance encoder

```python
from novelty.encoders.base import register, l2norm

@register("myencoder")
class MyEncoder:
    name = "myencoder"
    dim = 512
    supports_weights = False          # True if you honour the spatial mask

    def encode(self, frames, weights=None):
        # frames: (T, H, W, 3) uint8 RGB
        # -> (T, 512) float32, L2-normalised rows
        return l2norm(my_model(frames), axis=1)
```

Then `appearance.encoder: myencoder`. That is the whole contract. **Re-index and
re-calibrate after changing it** — signatures from different encoders are not
comparable, and a stale null model is worse than none.

## Adding a new metric

1. Add the function to `metrics/distance.py`.
2. Compute it in `metrics/fuse.raw_scores` and add its key to `RAW_METRICS`.
3. Add it to `ENV_COMPONENTS` or `TASK_COMPONENTS` in `calibrate.py`.

It then gets a null distribution, a percentile, and a fitted weight
automatically. Verify it earns its place by checking the fitted weight after
`calibrate --labels`: a weight of 0.00 means it contributed nothing.

## Design notes

The report's colours come from a validated categorical/sequential palette
(single-hue blue ramp for the heatmap, blue + orange for the two chart accents),
chosen for colour-vision-deficiency separation and checked against both light
and dark surfaces. Coverage and marginal gain are drawn as **two separate
charts** rather than one dual-axis chart: their scales are unrelated, and
overlaying them invents a crossing point that means nothing.

← Back to the [README](../README.md)
