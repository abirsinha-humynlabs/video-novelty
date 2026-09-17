# 01 — Why not just cosine

The design this repo argues against is the obvious one, and it is obvious for
good reasons: embed frames with a strong backbone, mean-pool, put the vectors
in a vector DB, cosine, threshold. It is the RAG pattern, it works well for
documents, and the first 80% of it is correct here too.

The last 20% is where video-for-robotics breaks it. Four ways.

---

## 1. Saturation: the metric runs out of resolution exactly where you need it

Take any modern image backbone and embed a thousand clips shot in one factory.
The pairwise cosine distribution will be crushed into roughly `[0.88, 0.99]`.
Not because the clips are all the same, but because every one of them contains
the same building, the same light, the same palette, the same camera, the same
human body in the lower third of frame. That shared content is a large vector
that every embedding has in common, and the dot product is dominated by it.

Retrieval hides this. If you only ever read the top-5, a squashed scale is
fine — ranking survives. A **redundancy gate does not read the top-5**; it reads
the *value* and compares it to a threshold, and a threshold on a squashed scale
is noise amplification. Worse, the squashing factor depends on how homogeneous
your corpus is, so the threshold you tuned on the factory data is silently wrong
on the warehouse data.

**Two fixes, applied in order.**

*Whitening* (`calibrate.Whitener`). Subtract the corpus mean and divide by the
per-dimension standard deviation before any dot product. This removes exactly
the shared component and rescales each dimension in proportion to how much it
actually varies in your data. Observed effect on the real footage in this repo:

| metric | before whitening | after whitening |
|---|---|---|
| `env_cos`, factory↔factory | 0.92 – 0.94 | 0.12 – 0.26 |
| `env_cos`, factory↔construction | 0.75 | −0.41 – −0.51 |
| `task_cos`, **every** pair | 0.995 – 0.998 | −0.19 – +0.28 |

That last row is the whole argument in one line. Before whitening the task
metric had no resolution at all; it was a constant with rounding error on top.

*Percentile calibration* (`calibrate.NullModel`). Even whitened, a raw number
is not interpretable. Sample random pairs from the corpus, build the empirical
distribution of every metric, and thereafter report **where a pair falls in it**.
"99.4th percentile" means "closer than 99.4% of random pairs in this dataset"
and that statement transfers across sites, backbones and resolutions. A cosine
of 0.94 does not.

---

## 2. One vector cannot answer two questions

You are asking two things at once:

- *Is this the same environment?* — same room, same rig, same lighting, same
  station.
- *Is this the same work?* — same actions, same cadence, same objects handled.

An appearance embedding answers the first and is close to **invariant** to the
second. That invariance is not a defect; it is precisely the property that makes
DINOv2 good at place recognition. But it means no threshold on an appearance
cosine will ever separate "bagging fittings" from "sweeping the floor" in the
same corner of the same room.

So there are two axes, computed from disjoint evidence:

```
environment  <- appearance embeddings of sampled frames  (tier 1)
task         <- dense optical flow + motion rhythm       (tier 2)
                or a learned video backbone on the GPU path
```

and a comparison returns a point in that plane. The four quadrants map to four
different curation actions, two of which say *keep* — and a single fused scalar
would have thrown both of them away:

|                    | task similar               | task different                  |
|--------------------|----------------------------|---------------------------------|
| **env similar**    | `REDUNDANT` → drop         | `SAME_PLACE_NEW_TASK` → **keep** |
| **env different**  | `SAME_TASK_NEW_PLACE` → **keep** | `NOVEL` → **keep**        |

`SAME_PLACE_NEW_TASK` is the single most valuable bucket for a physical-AI
dataset — it is new behaviour with the confound of scene change removed — and
it is exactly what a one-number pipeline deletes.

---

## 3. "Same file" is a different question from "same content"

Two cases that a semantic embedding gets exactly backwards:

```
A = recording.mp4[0:300]   B = recording.mp4[300:600]
   share zero frames. Same room, same work, same everything that matters.

A = recording.mp4          B = recording_reencoded_720p.mp4
   share every frame. One of them should not be in your dataset at all.
```

These need different answers, and an embedding gives the same answer to both
("very similar"). So tier 0 is a separate, non-semantic, cheap
(**milliseconds, no GPU**) layer: a 64-bit DCT perceptual hash per sampled
frame, then a Hough-style vote over time offsets to find the alignment between
two hash streams. Shared footage produces a tall spike at one offset; unrelated
videos produce a flat histogram.

It survives re-encoding, rescaling, fps change, trimming and concatenation. It
deliberately does *not* fire on the disjoint-halves case — and there is a test
asserting that (`tests/test_hashing.py::test_disjoint_halves_are_not_source_duplicates`),
because that non-firing is the feature.

> One real caveat found while writing the tests: on *perfectly periodic*
> content — synthetic loops, a static camera on a machine with an exact cycle —
> frames one period apart genuinely are the same pixels, and tier 0 will
> correctly report shared footage between disjoint spans. Real cameras drift
> and real sensors add noise, so this does not occur in practice; but if you
> ever run this on rendered or looped content, know that tier 0 is answering
> "are these the same pixels", which on that data is a different question from
> "is this the same file".

---

## 4. Dedup is a proxy for the thing you actually want

The stated goal is *dataset variety*: monotonous data stops helping the model,
so keep the informative stuff. Threshold dedup approximates that badly.

- **It is order-dependent.** Whichever near-duplicate you ingested first
  survives; the better-lit, longer, better-framed twin is deleted.
- **It optimises the wrong local quantity.** Deleting everything within ε of a
  kept clip removes the rare clip that merely *sits near* a common one, while
  ten mediocre clips scattered across empty space all survive.
- **It cannot answer the real question**, which is not "is this a duplicate" but
  "which 200 of my 2000 hours do I label first".

The right objective is **facility location**:

```
f(S) = Σ_{v ∈ corpus}  max_{s ∈ S}  sim(v, s)
```

"how well is every clip in my corpus represented by something I selected". It is
monotone and submodular, so greedy maximisation is within (1 − 1/e) ≈ 63% of the
optimum — a guarantee thresholding does not have — and the marginal gains fall
monotonically, giving you the diminishing-returns curve directly from data.

Dedup falls out for free: a clip whose marginal gain is ≈0 is redundant, and
now you know *how* redundant rather than just which side of a line it fell on.

On the demo index (16 windows from 4 clips) the greedy selector picks one window
from each of the four distinct clips before it picks a second window from any of
them, and coverage saturates at 9 of 16. That ordering is the guarantee doing
its job; threshold dedup would have given you whichever 9 happened to arrive
first.

→ Next: [02 — Architecture](02-architecture.md)
