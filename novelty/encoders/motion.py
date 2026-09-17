"""Tier 2 -- motion / "is the same WORK happening?"

This is the tier that the naive "embed frames, cosine, done" plan is missing,
and it is the one that actually answers the user's question. An appearance
embedding is close to invariant to what the hands are doing -- that is what
makes it good at recognising a place and useless at recognising a task.

What we extract, and why each piece earns its keep:

1. **Ego-motion vs object-motion split.** The median optical-flow vector over
   the frame is (to first order) the camera's own movement. Subtracting it
   leaves the flow caused by things moving in the world -- hands, parts,
   conveyor. On a head-mounted camera this split is essential: otherwise you
   are mostly measuring how much the wearer turned their head.

2. **Residual flow histogram** (2x2 spatial x 8 orientation x 4 magnitude).
   A coarse "what kind of motion, where in frame" descriptor. Bagging parts
   into a sack in the lower-centre looks different from tying rebar at arm's
   length, even in the same building.

3. **Rhythm spectrum.** The power spectrum of per-frame motion energy. This is
   the single most useful feature for repetitive manual work and almost nobody
   uses it. A worker doing a 4-second pick-place cycle puts a peak at 0.25 Hz
   in this spectrum. Two clips of the same repetitive task match here *even
   when they are phase-shifted*, which is exactly the "I cut a 10-minute video
   into two 5-minute halves" case. Two different tasks in the same room do
   not match, even though their appearance embeddings are nearly identical.

4. **Cycle period + strength** from the autocorrelation of the same signal:
   an interpretable number ("this clip is a 3.8 s loop, repeated strongly")
   that you can eyeball and sanity-check against the footage.

Flow is Farneback by default -- dense, CPU-cheap, no weights. Set
``backend="raft"`` to use torchvision's RAFT on a GPU when accuracy matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..io.decode import iter_frames

N_ORI = 8
N_MAG = 4
N_CELL = 2                      # 2x2 spatial grid
HIST_DIM = N_CELL * N_CELL * N_ORI * N_MAG          # 128
FRAME_DIM = HIST_DIM + 3                             # + [mean|res|, |ego|, frac_moving]
RHYTHM_BANDS = 24

#: Relative weight of the 3 scalar channels against the 128-bin histogram.
#:
#: This constant exists because of a real bug, and the bug is worth knowing
#: about because every "concatenate some features and take a cosine" pipeline
#: has it somewhere. The raw scalars (mean residual flow ~2.4 px, ego motion
#: ~2.1 px, moving fraction ~0.98) are numerically ~24x larger than the whole
#: L1-normalised histogram block. Cosine similarity is scale-sensitive across
#: blocks, so the 128 numbers that actually describe the task contributed ~4%
#: of the vector and the three numbers that just say "this is handheld video"
#: contributed the rest. Every pair scored 0.995+, including pairs from
#: completely different worksites.
#:
#: Fix: L2-normalise each block independently, compress the scalars with
#: log1p (they are unbounded), then down-weight the tiny scalar block so it
#: informs rather than dominates.
SCALAR_W = 0.3
SCALAR_BLOCK_W = 0.5


@dataclass
class MotionFeatures:
    """Everything tier 2 knows about one clip."""
    pooled: np.ndarray        # (2*HIST_DIM + 6,) block-normalised clip descriptor
    rhythm: np.ndarray        # (RHYTHM_BANDS,) log-binned power spectrum of motion energy
    seq: np.ndarray           # (T', FRAME_DIM) float32, subsampled, for DTW alignment
    energy: np.ndarray        # (T,) raw per-frame motion energy (kept for plots)
    period_s: float           # dominant repetition period, seconds (0 = none found)
    period_strength: float    # normalised autocorrelation at that lag, in [0, 1]
    ego_magnitude: float      # mean camera motion, px/frame at the analysis resolution
    fps: float


def _flow_descriptor(flow: np.ndarray, mag_edges: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """One (H, W, 2) flow field -> (HIST_DIM,) histogram + 3 scalars."""
    ego = np.median(flow.reshape(-1, 2), axis=0)
    res = flow - ego
    mag = np.hypot(res[..., 0], res[..., 1])
    ang = np.mod(np.arctan2(res[..., 1], res[..., 0]), 2 * np.pi)
    ori = np.minimum((ang / (2 * np.pi) * N_ORI).astype(np.int32), N_ORI - 1)
    mbin = np.clip(np.digitize(mag, mag_edges) - 1, 0, N_MAG - 1)
    h, w = mag.shape
    hist = np.zeros((N_CELL, N_CELL, N_ORI, N_MAG), np.float32)
    moving = mag > mag_edges[0]
    for r in range(N_CELL):
        for c in range(N_CELL):
            sl = (slice(r * h // N_CELL, (r + 1) * h // N_CELL),
                  slice(c * w // N_CELL, (c + 1) * w // N_CELL))
            m = moving[sl]
            if not m.any():
                continue
            flat = (ori[sl][m] * N_MAG + mbin[sl][m]).ravel()
            hist[r, c] = np.bincount(flat, minlength=N_ORI * N_MAG).reshape(N_ORI, N_MAG)
    hist = hist.ravel()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist, float(mag.mean()), float(np.hypot(*ego)), float(moving.mean())


def smooth(x: np.ndarray, fps: float, win_s: float = 0.5) -> np.ndarray:
    """Hann-window moving average.

    Raw per-frame motion energy on a head-mounted camera is dominated by
    high-frequency jitter (micro head movements, rolling shutter, flow noise).
    Left unsmoothed, the autocorrelation peaks at the shortest lag it is
    allowed to and the "repetition period" becomes a constant equal to the
    search floor -- a bug that looks like a result. Smoothing at ~0.5 s keeps
    manual work cycles (0.5-20 s) and removes the jitter band.
    """
    w = max(int(round(win_s * fps)), 1)
    if w < 3 or len(x) < w:
        return x
    k = np.hanning(w + 2)[1:-1]
    k = k / k.sum()
    return np.convolve(x, k, mode="same").astype(np.float32)


def _rhythm_spectrum(energy: np.ndarray, fps: float, bands: int = RHYTHM_BANDS,
                     f_lo: float = 0.05) -> np.ndarray:
    """Log-binned power spectrum of the motion-energy signal, L1-normalised."""
    n = len(energy)
    if n < 8:
        return np.zeros(bands, np.float32)
    x = energy - energy.mean()
    x = x * np.hanning(n)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    f_hi = max(fps / 2.0, f_lo * 2)
    edges = np.geomspace(f_lo, f_hi, bands + 1)
    out = np.zeros(bands, np.float32)
    idx = np.digitize(freqs, edges) - 1
    valid = (idx >= 0) & (idx < bands)
    if valid.any():
        out = np.bincount(idx[valid], weights=spec[valid], minlength=bands).astype(np.float32)
    s = out.sum()
    return out / s if s > 0 else out


def _dominant_period(energy: np.ndarray, fps: float, min_period_s: float = 0.4,
                     max_period_s: float = 20.0) -> Tuple[float, float]:
    """Period and strength from the normalised autocorrelation of motion energy."""
    n = len(energy)
    if n < 16:
        return 0.0, 0.0
    x = energy - energy.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[n - 1:] / denom
    lo = max(int(min_period_s * fps), 1)
    hi = min(int(max_period_s * fps), n - 1)
    if hi <= lo + 2:
        return 0.0, 0.0
    seg = ac[lo:hi]
    # A genuine cycle is a LOCAL MAXIMUM of the autocorrelation. Taking a plain
    # argmax instead picks up the monotone shoulder of the zero-lag peak and
    # reports the search floor as "the period" for every clip.
    inner = seg[1:-1]
    is_peak = (inner > seg[:-2]) & (inner >= seg[2:])
    idx = np.nonzero(is_peak)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    k = int(idx[int(np.argmax(inner[idx]))]) + 1
    strength = float(np.clip(seg[k], 0.0, 1.0))
    if strength < 0.05:
        return 0.0, 0.0
    return float((lo + k) / fps), strength


class FlowMotionEncoder:
    """Streaming dense-optical-flow task encoder.

    ``width``/``height`` are the *analysis* resolution. 160x90 is deliberate:
    dense flow cost is quadratic in resolution, and hand/part motion at 160x90
    is still several pixels per frame. Raising it buys little and costs a lot.
    """

    def __init__(self, *, fps: float = 10.0, width: int = 160, height: int = 90,
                 mag_edges=(0.20, 0.75, 2.0, 5.0), max_seq: int = 256,
                 backend: str = "farneback", letterbox: bool = True,
                 smooth_s: float = 0.5):
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self.mag_edges = np.asarray(mag_edges, np.float32)
        self.max_seq = int(max_seq)
        self.backend = backend
        self.letterbox = letterbox
        self.smooth_s = float(smooth_s)
        self.dim = 2 * FRAME_DIM

    def _flow(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev, cur, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )

    def encode(self, path: str, *, t0: Optional[float] = None, t1: Optional[float] = None) -> MotionFeatures:
        descs, energies, egos = [], [], []
        prev = None
        for chunk, _ts in iter_frames(
            path, fps=self.fps, width=self.width, height=self.height,
            t0=t0, t1=t1, gray=True, letterbox=self.letterbox, chunk=64,
        ):
            g = chunk[..., 0]
            for i in range(len(g)):
                cur = g[i]
                if prev is not None:
                    hist, e, ego, frac = _flow_descriptor(self._flow(prev, cur), self.mag_edges)
                    descs.append(np.concatenate([hist, [e, ego, frac]]).astype(np.float32))
                    energies.append(e)
                    egos.append(ego)
                prev = cur
        if not descs:
            z = np.zeros(FRAME_DIM, np.float32)
            return MotionFeatures(np.zeros(2 * FRAME_DIM, np.float32), np.zeros(RHYTHM_BANDS, np.float32),
                                  z[None], np.zeros(0, np.float32), 0.0, 0.0, 0.0, self.fps)
        D = np.stack(descs)
        energy = np.asarray(energies, np.float32)
        energy_s = smooth(energy, self.fps, self.smooth_s)

        # --- block normalisation (see SCALAR_W above) ----------------------
        hist = D[:, :HIST_DIM]
        scal = np.log1p(np.maximum(D[:, HIST_DIM:], 0.0)) * SCALAR_W
        hist = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-8)
        seq_full = np.concatenate([hist, scal], axis=1).astype(np.float32)

        def _l2(v):
            return v / (np.linalg.norm(v) + 1e-8)

        pooled = np.concatenate([
            _l2(hist.mean(0)),
            _l2(hist.std(0)),
            _l2(np.concatenate([scal.mean(0), scal.std(0)])) * SCALAR_BLOCK_W,
        ]).astype(np.float32)
        pooled = _l2(pooled)

        rhythm = _rhythm_spectrum(energy_s, self.fps)
        period, strength = _dominant_period(energy_s, self.fps)
        if len(seq_full) > self.max_seq:
            idx = np.linspace(0, len(seq_full) - 1, self.max_seq).round().astype(int)
            seq = seq_full[idx]
        else:
            seq = seq_full
        return MotionFeatures(
            pooled=pooled, rhythm=rhythm, seq=seq.astype(np.float32), energy=energy,
            period_s=period, period_strength=strength,
            ego_magnitude=float(np.mean(egos)) if egos else 0.0, fps=self.fps,
        )
