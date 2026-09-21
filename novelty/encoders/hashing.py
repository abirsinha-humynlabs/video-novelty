"""Tier 0 -- perceptual hashing: "do these two files literally contain the same footage?"

This tier is not semantic and is not supposed to be. It answers a narrow,
cheap, unambiguous question: was some span of frames in A produced by the same
camera exposure as some span of frames in B? Re-encoding, changing container,
changing resolution, changing fps, trimming and concatenating all preserve the
answer. Reshooting the same scene does not.

Keeping this separate from the embedding tiers matters, because the two
questions have different right answers:

    A = video.mp4[0:300]        B = video.mp4[300:600]
    -> tier 0 says NOT duplicates (disjoint spans, no shared frames)
    -> tier 1/2 say near-identical (same room, same task)

and you want both facts, not a blend of them.

Algorithm: 64-bit DCT hash per sampled frame, then a Hough-style vote over
time offsets to find the dominant alignment between the two hash streams.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy.fft import dctn

from ..io.decode import iter_frames

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_frames(gray: np.ndarray) -> np.ndarray:
    """``gray``: (T, 32, 32) uint8 -> (T,) uint64 perceptual hashes."""
    if gray.ndim == 4:
        gray = gray[..., 0]
    x = gray.astype(np.float32) / 255.0
    d = dctn(x, axes=(1, 2), type=2, norm="ortho")
    low = d[:, :8, :8].reshape(len(x), 64)
    med = np.median(low[:, 1:], axis=1, keepdims=True)
    bits = (low > med).astype(np.uint64)
    bits[:, 0] = 0  # DC carries only brightness; drop it
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    return (bits * weights).sum(axis=1, dtype=np.uint64)


def video_phashes(path: str, *, fps: float = 2.0, t0=None, t1=None) -> Tuple[np.ndarray, np.ndarray]:
    """Hash a whole file at ``fps`` samples/second. Streams; RAM is O(chunk)."""
    hs, ts = [], []
    for frames, t in iter_frames(path, fps=fps, width=32, height=32, t0=t0, t1=t1, gray=True, chunk=256):
        hs.append(phash_frames(frames))
        ts.append(t)
    if not hs:
        return np.zeros(0, np.uint64), np.zeros(0, np.float64)
    return np.concatenate(hs), np.concatenate(ts)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between uint64 hash vectors -> (len(a), len(b)) uint8."""
    x = np.bitwise_xor(a[:, None].astype(np.uint64), b[None, :].astype(np.uint64))
    return _POPCOUNT8[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1).astype(np.uint8)


@dataclass(frozen=True)
class SourceOverlap:
    overlap_seconds: float
    offset_seconds: float
    matched_frames: int
    fraction_of_a: float
    fraction_of_b: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def source_overlap(
    ha: np.ndarray, ta: np.ndarray,
    hb: np.ndarray, tb: np.ndarray,
    *, max_hamming: int = 8, offset_bin: float = 0.5, min_votes: int = 3,
    max_gap_frames: int = 2,
) -> SourceOverlap:
    """Find the dominant time offset that aligns two hash streams.

    For every near-matching frame pair (i, j) we cast a vote for the offset
    ``tb[j] - ta[i]``. Shared footage produces a tall spike at one offset; two
    unrelated videos produce a flat, low histogram. Returns the spike.

    ``overlap_seconds`` measures the longest **contiguous** run of aligned
    frames, not the total number of them. That distinction is the difference
    between a true and a false positive on repetitive manual work: two disjoint
    segments of one recording, filmed at the same bench with a near-static head
    pose, genuinely contain individual frames that are near-identical pixels,
    and enough of them land in one offset bin to clear a vote count. What they
    never contain is a continuous *stretch* of the same frames -- only actually
    shared footage does. Counting votes flagged two disjoint automobile-plant
    chunks as DUPLICATE_SOURCE off four scattered frames; measuring the run
    does not. ``max_gap_frames`` tolerates a dropped match inside a real run.
    """
    empty = SourceOverlap(0.0, 0.0, 0, 0.0, 0.0)
    if len(ha) == 0 or len(hb) == 0:
        return empty
    d = hamming(ha, hb)
    ii, jj = np.nonzero(d <= max_hamming)
    if len(ii) < min_votes:
        return empty
    offsets = tb[jj] - ta[ii]
    bins = np.round(offsets / offset_bin).astype(np.int64)
    uniq, counts = np.unique(bins, return_counts=True)
    k = int(np.argmax(counts))
    if counts[k] < min_votes:
        return empty
    best_bin = uniq[k]
    sel = bins == best_bin
    # distinct source frames that participate in the winning alignment
    idx_a = np.unique(ii[sel])
    idx_b = np.unique(jj[sel])

    def longest_run(idx: np.ndarray) -> int:
        if len(idx) == 0:
            return 0
        splits = np.where(np.diff(idx) > max_gap_frames)[0] + 1
        return max(len(r) for r in np.split(idx, splits))

    run_a, run_b = longest_run(idx_a), longest_run(idx_b)
    if min(run_a, run_b) < min_votes:
        return empty
    step_a = float(np.median(np.diff(ta))) if len(ta) > 1 else 0.5
    return SourceOverlap(
        overlap_seconds=float(run_a * step_a),
        offset_seconds=float(best_bin * offset_bin),
        matched_frames=int(sel.sum()),
        fraction_of_a=float(run_a / len(ha)),
        fraction_of_b=float(run_b / len(hb)),
    )
