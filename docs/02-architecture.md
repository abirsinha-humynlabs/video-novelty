# 02 — Architecture

## Data flow

```
video file
   │
   ├── ffmpeg ──► 48 frames @ 384×216 ──► appearance encoder ──┐
   │              (io/decode.py)          (encoders/appearance) │
   │                                                            ├──► Signature
   ├── ffmpeg ──► gray @ 10 fps 160×90 ─► dense optical flow ───┤    (signature.py)
   │              streaming, chunked      (encoders/motion.py)   │
   │                                                            │
   └── ffmpeg ──► gray @ 2 fps 32×32 ───► 64-bit pHash ─────────┘
                                          (encoders/hashing.py)
                                                  │
                                                  ▼
                              Index (store/numpy_store.py): .npz + manifest.json
                                                  │
                      ┌───────────────────────────┼───────────────────────────┐
                      ▼                           ▼                           ▼
              calibrate.py                  metrics/fuse.py               select.py
        whitener + null model          raw scores → 2 axes → verdict   facility location
```

Three ffmpeg passes per signature, not one. That is a deliberate trade: each
tier wants a different resolution and frame rate, and three sequential decodes
are simpler and less memory-hungry than one `filter_complex` with three
outputs. It is also the first thing to optimise if indexing throughput ever
becomes your bottleneck — see [07](07-tuning.md).

## File map

| path | what lives there |
|---|---|
| `novelty/io/decode.py` | ffmpeg-over-a-pipe frame access. Streams in chunks, so a 10-minute 4K file costs the same RAM as a 10-second one. Everything else in the repo gets its pixels from here. |
| `novelty/encoders/hashing.py` | **Tier 0.** Perceptual hash + offset voting. Answers "did these share literal footage". |
| `novelty/encoders/base.py` | Appearance-encoder registry. Swapping backbones is the highest-leverage knob here, so it is a registry and not an import. |
| `novelty/encoders/appearance.py` | **Tier 1.** `gist` (no weights, runs anywhere), `dinov2`, `dinov2-large`, `siglip`. Also `static_weight_map`, the egocentric body suppressor. |
| `novelty/encoders/motion.py` | **Tier 2, CPU.** Dense flow → ego/object split → spatial-orientation-magnitude histogram → rhythm spectrum + cycle period. |
| `novelty/encoders/video.py` | **Tier 2, GPU.** V-JEPA 2 / VideoMAE sliding-window clip embeddings. Optional; lazily imported. |
| `novelty/signature.py` | The `Signature` dataclass and its construction. One signature = one span of one file. Also the windowing logic. |
| `novelty/metrics/distance.py` | The similarity primitives: cosine, chamfer, diagonal Bhattacharyya, banded DTW, spectrum cosine, period agreement. |
| `novelty/metrics/fuse.py` | Raw scores → two axes → a `Verdict`. The only place fusion weights are applied. |
| `novelty/calibrate.py` | `Whitener` (feature-space) and `NullModel` (score-space), plus supervised weight fitting. |
| `novelty/select.py` | Lazy-greedy facility location, marginal gain, the ingest gate. |
| `novelty/store/numpy_store.py` | The on-disk index. Small interface on purpose. |
| `novelty/report.py` | Self-contained HTML report. No JS libraries, no network. |
| `novelty/cli.py` | Subcommands. |

## The Signature

A `Signature` is deliberately **not** a single vector:

```python
Signature(
    segment_id, video_id, path, t0, t1,

    app_frames,      # (N, D) L2-normed rows  ─┐ tier 1: where is this?
    app_mean,        # (D,)                    │
    app_var,         # (D,)                   ─┘

    motion_pooled,   # (262,) block-normalised ─┐ tier 2: what work is this?
    motion_seq,      # (T', 131) for DTW        │
    rhythm,          # (24,) power spectrum     │
    period_s,        # dominant cycle, seconds  │
    period_strength, #                         ─┘

    phash,           # (K,) uint64             ─┐ tier 0: same footage?
    phash_ts,        # (K,) seconds            ─┘
)
```

Keeping `app_frames` and not just `app_mean` is what makes the chamfer metric
possible, and chamfer is what makes "one long video cut into halves" work. It
costs ~73 KB per signature at `dinov2` width — 73 MB for a thousand signatures.
Cheap.

## Windowing

`segment.seconds` chops long files before signing. **Use it.** A 10-minute
recording is not one homogeneous thing, and a single pooled vector for it is a
lie you will later mistake for a measurement. 30 s windows with a 15 s hop is a
sane default; the constraint from the rhythm feature is that a window should
contain at least five or six repetitions of the cycle you care about.

Windows carry the same `video_id`, which matters for calibration: two windows of
one recording are not a "random pair", and letting them into the null
distribution inflates it. `fit_null(exclude_same_video=True)` is the default.

## Swapping the store

`numpy_store.Index` is brute force: a dense float32 matmul over every pair. At
1e4–1e5 signatures that is milliseconds and zero operational surface. Past
roughly 1e5, swap in FAISS or Qdrant — the interface you need to implement is
five methods (`add`, `signatures`, `null`, `save_null`, `similarity_matrix`).

The thing to be careful about when you do: **ANN indexes the appearance vector
only**. The chamfer, DTW and rhythm terms are not inner products and cannot be
served by an ANN index. The right pattern is two-stage — ANN on `app_mean` to
get ~100 candidates, then full `raw_scores` on those. See [06](06-running-on-gpu.md).

→ Next: [03 — Signatures](03-signatures.md)
