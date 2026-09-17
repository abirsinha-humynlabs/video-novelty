"""The similarity primitives.

Read ``docs/01-why-not-just-cosine.md`` before changing anything here; each of
these exists because cosine-on-the-mean fails in a specific, reproducible way.
"""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Plain cosine similarity between two vectors."""
    na = float(np.linalg.norm(a)) + eps
    nb = float(np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / (na * nb))


def chamfer(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric mean-of-max-similarity between two SETS of frame vectors.

    ``cosine(mean(A), mean(B))`` asks "do these clips average out to the same
    thing". Chamfer asks "does every moment of A have a counterpart somewhere
    in B, and vice versa". Those differ exactly when you need them to:

    * a 10-min clip cut into two halves -> mean vectors can drift apart as the
      worker moves down the line, but every frame still has a near neighbour,
      so chamfer stays high (this is the user's stated case), and
    * a clip that spends 80% of its time on a wall and 20% on the workbench vs
      one that does the reverse -> means look similar, chamfer correctly does
      not reward the mismatch in coverage.

    Both inputs must have L2-normalised rows. Returns roughly [-1, 1].
    """
    if len(A) == 0 or len(B) == 0:
        return 0.0
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    S = A @ B.T
    return float(0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean()))


def gaussian_bhattacharyya(mu1, var1, mu2, var2, eps: float = 1e-6) -> float:
    """Per-dimension mean Bhattacharyya distance between two diagonal Gaussians.

    Compares *spread*, not just centre. Two clips can have near-identical mean
    appearance vectors while one roams the whole building and the other stares
    at a single bench; this term separates them. Averaged over dimensions so
    the magnitude does not scale with embedding width (a 768-d and a 564-d
    backbone give comparable numbers).
    """
    var1 = np.maximum(var1, eps)
    var2 = np.maximum(var2, eps)
    vm = 0.5 * (var1 + var2)
    d = mu1 - mu2
    term1 = 0.125 * (d * d) / vm
    term2 = 0.5 * np.log(vm / np.sqrt(var1 * var2))
    return float(np.mean(term1 + term2))


def dtw_cost(A: np.ndarray, B: np.ndarray, band_frac: float = 0.25, eps: float = 1e-8) -> float:
    """Band-constrained DTW alignment cost between two descriptor SEQUENCES.

    Normalised by path length, so it is comparable across clip durations.
    Cost is built from ``1 - cosine`` per step, so 0 == perfectly alignable and
    ~1 == unrelated.

    Why DTW and not just pooled cosine: two runs of the same repetitive task
    are the same *sequence of sub-actions* at slightly different speeds and
    phases. Pooling throws the ordering away; DTW keeps it while tolerating
    the speed difference. The Sakoe-Chiba band keeps it O(n * band) and also
    stops degenerate alignments that map everything to one frame.
    """
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return 1.0
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    C = 1.0 - (A @ B.T)
    band = max(int(band_frac * max(n, m)), 1)
    INF = np.inf
    prev = np.full(m + 1, INF, np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF, np.float64)
        j_lo = max(1, int(i * m / n) - band)
        j_hi = min(m, int(i * m / n) + band)
        for j in range(j_lo, j_hi + 1):
            best = min(prev[j], prev[j - 1], cur[j - 1])
            if best == INF:
                continue
            cur[j] = C[i - 1, j - 1] + best
        prev = cur
    total = prev[m]
    if not np.isfinite(total):
        return 1.0
    return float(total / (n + m))


def spectrum_cosine(r1: np.ndarray, r2: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine between two rhythm spectra (already L1-normalised power bands)."""
    if r1.sum() <= 0 or r2.sum() <= 0:
        return 0.0
    return cosine(np.sqrt(r1), np.sqrt(r2), eps)   # Hellinger-style: sqrt makes it a proper
                                                   # similarity between distributions


def period_agreement(p1: float, s1: float, p2: float, s2: float, tol: float = 0.15) -> float:
    """How well two estimated repetition periods agree, weighted by confidence.

    Returns 0 when either clip has no detectable cycle -- absence of rhythm is
    not evidence of a shared rhythm.
    """
    if p1 <= 0 or p2 <= 0:
        return 0.0
    ratio = max(p1, p2) / max(min(p1, p2), 1e-6)
    # Accept 1x, and allow 2x/3x harmonics: a detector that locks onto a
    # half-cycle is still describing the same work. The tolerance is RELATIVE
    # to the harmonic (|ratio - k| / k) but deliberately tight -- at tol=0.25
    # a 1.58x ratio scored 0.49 as a "near-2x harmonic", which let genuinely
    # different cadences agree.
    best = min(abs(ratio - k) / k for k in (1.0, 2.0, 3.0))
    agree = float(np.exp(-(best / tol) ** 2))
    return agree * float(np.sqrt(max(s1, 0.0) * max(s2, 0.0)))
