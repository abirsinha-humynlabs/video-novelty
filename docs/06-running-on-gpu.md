# 06 — Running on a GPU box

```bash
git clone <repo> && cd video-novelty
make install-gpu                       # torch, transformers, scikit-learn
novelty index data/ --config configs/gpu.yaml --index .novelty
novelty calibrate --index .novelty --labels eval/pairs.yaml
novelty select --index .novelty --budget 500
```

## What changes, and why

```yaml
appearance:
  encoder: dinov2          # was: gist
motion:
  clip_encoder: vjepa2     # was: null (flow only)
```

**`dinov2`** replaces hand-crafted gradient/colour statistics with a real scene
representation. The practical difference is robustness rather than raw
separation: `gist` separated a factory from a construction site at AUC 0.997,
but it does that by matching texture and colour, so it will fold the moment your
morning and afternoon footage differ in lighting. DINOv2 is invariant to that
because it was trained to be.

DINOv2 rather than CLIP on purpose: CLIP is text-aligned and collapses visually
distinct scenes sharing a caption — every room that is "a factory" lands in one
place. That is the failure mode we are avoiding. Use `siglip` only if you also
want to query the index with words.

**`vjepa2`** is the bigger change. The flow descriptor sees "how much moved,
where, in which direction, how rhythmically". It cannot see *what is being
manipulated*, so it cannot separate two different jobs at the same bench with
similar gross motion. A self-supervised video backbone can.

The flow encoder keeps running alongside. Rhythm and cycle period are cheap,
interpretable, and **not recoverable** from a pooled clip embedding — a 16-frame
window is about half a second and structurally cannot see a 6-second work cycle.

## Throughput

Measured on this repo's CPU path, per 60 s of 1080p30 video: **~28 s** — three
ffmpeg decode passes plus dense flow. That is ~2× realtime on two cores.

Where the time goes and what to do about it:

| cost | fix |
|---|---|
| three separate ffmpeg decodes per signature | one `filter_complex` with three outputs; ~40% off indexing |
| dense Farnebäck flow on CPU | `backend: raft` on GPU, or drop `motion.fps` to 6 |
| decoding on CPU | `-hwaccel cuda` in `io/decode.py`; large win on 1080p+ |
| one file at a time | index is embarrassingly parallel across files — run N workers over disjoint file lists, merge the manifests |

For a first pass over a large archive, `--no-motion` gives you tier 0 + tier 1
at roughly 3× the speed. The environment axis alone already finds most of the
obvious redundancy; add tier 2 on the survivors.

## Sizing the store

`numpy_store.Index` is brute force — a dense float32 matmul over all pairs.

| signatures | pairwise matrix | verdict |
|---|---|---|
| 1e3 | 4 MB | instant |
| 1e4 | 400 MB | fine, seconds |
| 1e5 | 40 GB | **stop**; you need ANN |

Past ~1e5, the pattern is **two-stage retrieval**, not "swap numpy for FAISS":

1. ANN (FAISS `IndexHNSWFlat`, or Qdrant if you want a server) over the whitened
   `app_mean` vectors → top ~100 candidates.
2. Full `raw_scores` on those 100 only.

Stage 2 is not optional, because **chamfer, DTW and rhythm are not inner
products** and no ANN index can serve them. An ANN-only system silently degrades
to "cosine on the mean", which is the design this repo exists to argue against.

For the selection step at that scale, facility location on a 1e5 × 1e5 matrix is
not tractable either. Use the standard approximation: build a sparse k-NN graph
(k ≈ 50) from stage 1, treat missing entries as zero similarity, and run the
lazy greedy on the sparse matrix. The guarantee degrades gracefully and the
picks barely change, because far-apart pairs contribute nothing to the `max` in
any case.

## Model download notes

Weights come from the HuggingFace hub on first use and are cached in
`~/.cache/huggingface`. On an air-gapped box, pre-download and set
`HF_HOME`/`TRANSFORMERS_OFFLINE=1`. Model ids are overridable:

```yaml
appearance:
  encoder: dinov2
  encoder_kwargs:
    model_id: /models/dinov2-base    # local path works
    batch_size: 64
    dtype: float16
```

→ Next: [07 — Tuning & failure modes](07-tuning.md)
