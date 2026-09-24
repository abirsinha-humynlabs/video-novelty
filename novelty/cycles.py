"""Cycle-aware segmentation from egocentric hand tracks.

Fixed-duration chunking cuts blind. A 30 s chunk of repetitive manual work
contains ~23 work cycles starting at an arbitrary phase, so two chunks of the
same job are misaligned by a random fraction of a cycle and every comparison
pays for that. This module finds where the cycles actually are, so a segment
can be a whole number of repetitions starting at a consistent phase.

Input is the output of the hand-stabilisation pipeline
(https://github.com/Maiemdiab/egocentric-hand-stabilisation): a keypoints
``.npz`` with ``kp3d_cam`` (N, 21, 3) in **camera-relative** metres plus
``frame_idx``, ``hand`` (0=left, 1=right, 2=bystander, -1=dropped),
``is_wearer``, ``kept`` and ``fps``; and optionally a head ``.npz`` with
``T`` (F, 4, 4) camera-to-world transforms.

Three findings from the Polymer_Bags and Pipe_Factory tracks drove the design,
and each one is a trap worth not re-discovering:

1. **Use a signed signal, never speed.** Wrist *speed* is rectified, so a
   reach-and-return traces two peaks per cycle: the period comes out halved and
   the phase is gone. Measured periodicity strength on the same footage was
   0.21 for speed against **0.88-0.92** for the signed projection of the wrist
   onto its own principal axis. That single change is the difference between
   this working and not.
2. **Remove a LOCAL baseline before looking for crossings.** The worker walks
   around the workspace, so the projection drifts further than it oscillates
   and never crosses a global threshold for long stretches -- 50 detected
   boundaries where ~230 were expected. A running median over ~3 cycles fixes
   it (272 boundaries, interval sd 11.25 s -> 0.64 s).
3. **Most of a recording is not cadenced.** Only ~36% of windows were strongly
   periodic; cycle-aligned segments covered 27% of the clip. The remainder is
   transitions, walking and fetching stock -- non-repetitive, and therefore
   *more* novel per second than the repetitive work, not less. So this module
   returns both kinds of segment and labels them, instead of discarding the
   gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

#: OpenPose-21 hand layout: 0=wrist, 1-4 thumb, 5-8 index, 9-12 middle,
#: 13-16 ring, 17-20 pinky.
WRIST = 0
MIDDLE_MCP = 9
LEFT, RIGHT = 0, 1


@dataclass
class HandTracks:
    """Per-frame wearer hand pose, gap-filled, optionally in world frame."""
    fps: float
    n_frames: int
    pose: Dict[int, np.ndarray]          # hand -> (n_frames, 21, 3), NaN where absent
    world: bool
    coverage: Dict[int, float]           # hand -> fraction of frames present


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str                            # "cadenced" | "transition"
    n_cycles: int = 0
    period_s: float = 0.0
    strength: float = 0.0
    #: mean cut-point recurrence of the cycles INSIDE this chunk; see
    #: ``recurrence_test``. Informational: one chunk has too few cycles to
    #: judge on its own, so acceptance is decided on the pooled episode score.
    #: NaN when not measured.
    recurrence: float = float("nan")

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0


@dataclass
class CycleAnalysis:
    fps: float
    duration_s: float
    period_s: float                      # median period where cadenced
    boundaries: np.ndarray               # cycle boundary times, seconds
    cadence_fraction: float              # share of the clip that is cadenced
    #: share of the clip where the hands were tracked continuously enough to
    #: judge cadence at all. Distinguishing this from cadence_fraction matters:
    #: low trackable_fraction means "we could not see", which is a data-quality
    #: problem, not evidence that the work is irregular.
    trackable_fraction: float = 0.0
    segments: List[Segment] = field(default_factory=list)
    #: which hand the analysis actually used. Never assume it: the working hand
    #: differs by operator and by task, and the OTHER hand may simply be better
    #: tracked. Measured across four episodes the better-tracked hand was LEFT
    #: twice and RIGHT twice, with coverage gaps as wide as 0.31 vs 0.75.
    hand: int = RIGHT

    #: which channel carried the cadence (see ``SIGNAL_NAMES``), and the gate
    #: it had to be relaxed to. Both belong in the output, not just in a log:
    #: a clip found at 0.45 on the wrist and one found at 0.25 on a single
    #: finger are not equally good evidence, and without these two fields the
    #: CSVs are indistinguishable. Downstream should weight by them rather than
    #: treat every chunk as equal.
    signal: str = "wrist"
    min_strength: float = 0.45
    #: pooled cut-point recurrence of the returned chunks and its permutation
    #: p-value against random cuts (see ``recurrence_test``). NaN / 1.0 when
    #: no candidate was tested; a returned result with p above the alpha is a
    #: best-effort near-miss, not an accepted cadence.
    recurrence: float = float("nan")
    recurrence_p: float = 1.0


# ---------------------------------------------------------------- loading
def load_tracks(keypoints_npz: str, head_npz: Optional[str] = None) -> HandTracks:
    d = np.load(keypoints_npz, allow_pickle=True)
    kp, fi = d["kp3d_cam"], d["frame_idx"]
    hand, kept, wearer = d["hand"], d["kept"], d["is_wearer"]
    fps = float(d["fps"])

    # `step` is the detector's frame stride: fps is the SOURCE rate but
    # detections exist only every `step`-th frame. Ignoring it is silently
    # fatal on subsampled runs -- frame_idx still spans the full video, so a
    # 3 fps run (step=10) looks like 90% missing data, coverage reads 4% when
    # the true detection rate on sampled frames is 83%, and every 0.33 s
    # inter-detection interval trips the long-gap guard and blanks the whole
    # signal. Work on the sampled grid instead, where the effective rate is
    # fps/step and one index is one real sample.
    step = int(d["step"]) if "step" in d.files else 1
    step = max(step, 1)
    fps = fps / step
    fi = fi // step
    n = int(fi.max()) + 1

    # bystander hands (hand==2) and non-wearer tracks are dropped here rather
    # than filtered downstream: a second person working in frame has their own
    # cadence, and mixing it in is indistinguishable from the wearer changing
    # rhythm.
    sel = kept & wearer & ((hand == LEFT) | (hand == RIGHT))
    pose = {LEFT: np.full((n, 21, 3), np.nan, np.float32),
            RIGHT: np.full((n, 21, 3), np.nan, np.float32)}
    for k in (LEFT, RIGHT):
        m = sel & (hand == k)
        pose[k][fi[m]] = kp[m]

    world = False
    if head_npz:
        h = np.load(head_npz)
        T = np.full((n, 4, 4), np.nan)
        # head frame_idx is in SOURCE frames, so it needs the same stride
        # mapping as the keypoints or the two would be misaligned by `step`.
        hf = h["frame_idx"] // step
        ok = hf < n
        T[hf[ok]] = h["T"][ok]
        for k in (LEFT, RIGHT):
            pose[k] = _to_world(pose[k], T)
        world = True

    cov = {k: float(np.isfinite(pose[k][:, WRIST, 0]).mean()) for k in (LEFT, RIGHT)}
    return HandTracks(fps=fps, n_frames=n, pose=pose, world=world, coverage=cov)


def _to_world(pose: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(n,21,3) camera -> world. ``T`` is camera-to-world.

    Direction was not documented and was resolved empirically: cam->world
    decorrelates wrist speed from head angular speed (r=0.06) better than its
    inverse (r=0.23). Getting it backwards leaves head rotation inside the
    signal, which is exactly what the segmentation must not see.
    """
    out = np.full_like(pose, np.nan)
    g = np.isfinite(pose[:, WRIST, 0]) & np.isfinite(T[:, 0, 0])
    if not g.any():
        return out
    R, t = T[g][:, :3, :3], T[g][:, :3, 3]
    out[g] = np.einsum("nij,nkj->nki", R, pose[g]) + t[:, None, :]
    return out


# ---------------------------------------------------------------- signal
#: Gap tolerance is expressed in SAMPLES as well as seconds, and the looser of
#: the two wins. A fixed 0.3 s is under one sample at 3 fps, so a single missed
#: detection would blank the run -- which silently zeroed every subsampled
#: episode. Tolerating a few missing samples is what the rule was always meant
#: to express.
MAX_GAP_SAMPLES = 3

#: Samples needed per cycle before a period is measurable at all. This replaces
#: what used to be a hardcoded 0.4 s search floor -- an absolute time is an
#: assumption about the WORK, and the cycle rate changes with every video and
#: within one. Samples-per-cycle is an assumption about the SIGNAL, which is
#: the thing that is actually fixed: below ~4 samples a cycle cannot be
#: distinguished from noise at any frame rate. The floor is therefore
#: MIN_SAMPLES_PER_CYCLE / fps -- 0.13 s at 30 fps, 1.33 s at 3 fps -- so the
#: same code adapts instead of hiding fast work.
MIN_SAMPLES_PER_CYCLE = 4.0


def period_floor(fps: float) -> float:
    """Shortest cycle this frame rate can support, in seconds."""
    return MIN_SAMPLES_PER_CYCLE / max(fps, 1e-6)

#: A peak this close to the tallest one counts as the same evidence, so the
#: SHORTEST such lag wins and the estimator returns the fundamental rather than
#: whichever harmonic happens to be tallest. See local_period for the measured
#: case this fixes. Stable anywhere in 0.60-0.90 on the three real episodes
#: tested, so it is a plateau rather than a tuned value.
HARMONIC_FRAC = 0.80


def _fill(x: np.ndarray, fps: float = 30.0, max_gap_s: float = 0.3
          ) -> Optional[np.ndarray]:
    """Interpolate only across SHORT gaps; leave long ones as NaN.

    Interpolating globally is what made this silently useless on fragmentary
    tracks. Hand tracking on Pipe_Factory footage drops out for up to 11 s at a
    time (hands into a sack, bystanders stealing the track), and a linear fill
    replaces the oscillation with a straight line. The periodicity then reads
    zero and looks like "this work is not repetitive" rather than "we could not
    see the hands", which are very different conclusions.
    """
    n = len(x)
    m = np.isfinite(x)
    if m.sum() < 16:
        return None
    out = np.interp(np.arange(n), np.arange(n)[m], x[m])
    # re-blank anything that sat inside a gap longer than max_gap_s
    max_gap = max(int(max_gap_s * fps), MAX_GAP_SAMPLES)
    edges = np.diff(np.concatenate([[1], m.astype(int), [1]]))
    starts = np.nonzero(edges == -1)[0]
    ends = np.nonzero(edges == 1)[0]
    for a, b in zip(starts, ends):
        if b - a > max_gap:
            out[a:b] = np.nan
    return out


def valid_runs(x: np.ndarray, fps: float, min_run_s: float = 6.0
               ) -> List[Tuple[int, int]]:
    """Contiguous stretches of real (non-gap) signal long enough to hold cycles.

    Autocorrelation needs five or six repetitions before a period estimate stops
    being noise, so a run shorter than ~6 s cannot support one at these cadences.
    """
    m = np.isfinite(x)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    on = np.nonzero(edges == 1)[0]
    off = np.nonzero(edges == -1)[0]
    need = int(min_run_s * fps)
    return [(int(a), int(b)) for a, b in zip(on, off) if b - a >= need]


def _smooth(x: np.ndarray, fps: float, win_s: Optional[float] = None) -> np.ndarray:
    """Hanning smooth. The window defaults to HALF the shortest detectable
    cycle, never a fixed 0.25 s: a 0.25 s window attenuates a 0.4 s cycle badly,
    so the old default quietly suppressed exactly the fast work this module is
    meant to find."""
    if win_s is None:
        win_s = period_floor(fps) / 2.0
    w = max(int(win_s * fps), 3)
    k = np.hanning(w)
    return np.convolve(x, k / k.sum(), mode="same")


def _running_median(x: np.ndarray, w: int) -> np.ndarray:
    w = int(w) | 1
    pad = np.pad(x, w // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


#: Repetitions an autocorrelation needs before a period estimate stops being
#: noise. It sets the hard detectability limit: a cycle of length P can only be
#: measured inside a continuously-tracked run of at least MIN_REPS * P seconds.
#: A 20 s assembly cycle therefore needs ~100 s of unbroken tracking, which is
#: a data requirement, not a tuning knob.
MIN_REPS = 5.0

#: Window length as a multiple of the period. It must be comfortably more than
#: MIN_REPS: the search ceiling is window/MIN_REPS, so a window of exactly
#: MIN_REPS * P puts the true peak at lag P right on the boundary, where it
#: cannot be identified as a local maximum and is missed entirely. At 10x the
#: ceiling is 2P and the peak sits mid-range.
WINDOW_REPS = 10.0


# ------------------------------------------------------- signal channels
#: Joint chains per finger, wrist first. The wrist is included so the chain
#: carries the MCP angle: a push comes mostly from flexing at the knuckle, and
#: a chain starting at the MCP cannot see it.
FINGERS = {
    "thumb":  (WRIST, 1, 2, 3, 4),
    "index":  (WRIST, 5, 6, 7, 8),
    "middle": (WRIST, 9, 10, 11, 12),
    "ring":   (WRIST, 13, 14, 15, 16),
    "pinky":  (WRIST, 17, 18, 19, 20),
}


def finger_curl(pose: np.ndarray, finger: str) -> Optional[np.ndarray]:
    """Total flexion of one finger: the summed interior angles of its chain.

    Invariant to where the hand is and which way it points, by construction --
    it is built only from angles between adjacent bones. That is the whole
    reason it exists. A worker pushing a part into a machine with the fingers,
    or repeatedly pinching and releasing, moves the wrist barely at all, so the
    wrist projection sees a nearly stationary point while the work is plainly
    periodic. Measured on the metal-bracket episode
    (DDY-160 session2/013/seg_001) the wrist channel peaked at 0.426 against
    0.553 for the articulation channels -- on either side of the 0.45 gate, so
    the episode reported "no cadence" for a motion the customer had already
    labelled repetitive.
    """
    idx = FINGERS[finger]
    J = pose[:, idx, :]                           # (n, 5, 3)
    v = np.diff(J, axis=1)                        # 4 bone vectors
    a, b = v[:, :-1, :], v[:, 1:, :]              # consecutive pairs
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    ang = np.arccos(np.clip(num / np.where(den > 1e-9, den, np.nan), -1.0, 1.0))
    return ang.sum(-1)                            # NaN propagates on missing joints


def _kabsch(pose: np.ndarray):
    """Rigid-align every frame's hand to a canonical pose.

    Returns ``(mask, residual, rotvec)``: the frames used, the shape residual
    after rotation is removed, and the rotation itself as a signed rotation
    vector.

    Removing the rotation is what makes the finger signal usable. The naive
    'fingertip relative to the wrist' does not work, and the reason is
    geometric rather than statistical: rotating the whole rigid hand moves the
    fingertip relative to the wrist just as much as flexing the finger does. On
    the metal-bracket episode that channel scored 0.030 where the wrist scored
    0.139 -- worse than the thing it was meant to improve on. Only after the
    rotation is taken out does the articulation separate (0.553).
    """
    good = np.isfinite(pose).all((1, 2))
    if good.sum() < 64:
        return None, None, None
    X = pose[good]
    C = X - X[:, WRIST:WRIST + 1, :]
    # scale-normalise: hand size varies with distance from the camera, and an
    # unnormalised residual would then encode reach rather than articulation
    scale = np.linalg.norm(C, axis=(1, 2), keepdims=True) / np.sqrt(C.shape[1])
    C = C / np.clip(scale, 1e-9, None)
    canon = np.median(C, axis=0)
    canon = canon - canon.mean(0)
    A = C - C.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("nji,jk->nik", A, canon))
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    D = np.zeros((len(A), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("nij,njk,nkl->nil", U, D, Vt)
    resid = (np.einsum("nij,njk->nik", A, R) - canon).reshape(len(A), -1)
    # The rotation must come back SIGNED, as a rotation vector -- axis times
    # angle -- not as the angle alone. The angle is a magnitude, so a hand that
    # rolls one way and back traces two peaks per cycle and the period comes out
    # halved: on the synthetic fixture the signed channel reports 2.00 s and the
    # magnitude reports 1.00 s. That is finding #1 at the top of this module,
    # rediscovered on a different channel.
    angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    axis = np.stack([R[:, 2, 1] - R[:, 1, 2],
                     R[:, 0, 2] - R[:, 2, 0],
                     R[:, 1, 0] - R[:, 0, 1]], axis=1)
    nrm = np.linalg.norm(axis, axis=1, keepdims=True)
    rotvec = np.where(nrm > 1e-9, axis / np.where(nrm > 1e-9, nrm, 1.0), 0.0) \
        * angle[:, None]
    return good, resid - resid.mean(0), rotvec


#: Every channel the adaptive search may use, in the order it tries them.
#: `wrist` is first because it is the cheapest and the one three episodes were
#: validated on; the articulation channels exist for work the wrist cannot see.
SIGNAL_NAMES = ("wrist", "articulation", "rotation",
                "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky")


def raw_signal(tracks: HandTracks, hand: int, name: str = "wrist") -> Optional[np.ndarray]:
    """The undetrended 1-D channel for one hand. See ``SIGNAL_NAMES``."""
    pose = tracks.pose[hand]
    if name == "wrist":
        W = pose[:, WRIST, :]
        g = np.isfinite(W[:, 0])
        if g.sum() < 32:
            return None
        X = W[g] - W[g].mean(0)
        pc = np.linalg.svd(X, full_matrices=False)[2][0]
        out = np.full(len(W), np.nan)
        out[g] = X @ pc
        return out
    if name.startswith("curl_"):
        return finger_curl(pose, name[len("curl_"):])
    if name in ("articulation", "rotation"):
        good, resid, rotvec = _kabsch(pose)
        if good is None:
            return None
        out = np.full(len(pose), np.nan)
        if name == "rotation":
            pc = np.linalg.svd(rotvec - rotvec.mean(0), full_matrices=False)[2][0]
            out[good] = (rotvec - rotvec.mean(0)) @ pc
        else:
            pc = np.linalg.svd(resid, full_matrices=False)[2][0]
            out[good] = resid @ pc
        return out
    raise ValueError(f"unknown signal {name!r}; have {SIGNAL_NAMES}")


def _condition(raw: Optional[np.ndarray], fps: float,
               baseline_s: Optional[float]) -> Optional[np.ndarray]:
    """Gap-fill, smooth and detrend a raw channel, run by run."""
    if raw is None:
        return None
    filled = _fill(raw, fps)
    if filled is None:
        return None
    out = np.full(len(filled), np.nan)
    for a, b in valid_runs(filled, fps, min_run_s=1.0):
        seg = _smooth(filled[a:b], fps)
        if baseline_s is None:
            out[a:b] = seg - seg.mean()
        else:
            w = min(baseline_s * fps, max(len(seg) // 2, 3))
            out[a:b] = seg - _running_median(seg, w)
    return out


def cadence_signal(tracks: HandTracks, hand: int = RIGHT,
                   baseline_s: Optional[float] = None,
                   signal: str = "wrist") -> Optional[np.ndarray]:
    """Signed, locally-detrended 1-D signal whose oscillation is one work cycle.

    The wrist path of a repetitive reach-and-place is close to a line segment
    travelled back and forth, so projecting onto its own principal axis
    recovers a signed oscillation -- one crossing per cycle, phase preserved.

    ``baseline_s`` is the window for local-baseline removal and must scale with
    the cycle: ~3 cycles. Pass ``None`` for the first, period-agnostic pass --
    it then only removes each run's mean, which is enough to estimate a period
    but not enough to survive the worker walking around.
    """
    return _condition(raw_signal(tracks, hand, signal), tracks.fps, baseline_s)


def local_period(x: np.ndarray, fps: float, centre: int, window_s: float,
                 lo_s: Optional[float] = None, hi_s: Optional[float] = None
                 ) -> Tuple[float, float]:
    """Dominant period and autocorrelation strength in a window around ``centre``.

    ``hi_s`` defaults to ``window_s / MIN_REPS`` and is clamped to it, because a
    lag longer than that is averaged over too few repetitions to mean anything.
    The previous defaults searched to 8 s inside a 6 s window, which reported
    confident nonsense at the long end and made every task with a cycle above
    ~1.2 s structurally invisible.
    """
    W = int(window_s * fps)
    lo_s = period_floor(fps) if lo_s is None else lo_s
    ceiling = window_s / MIN_REPS
    hi_s = ceiling if hi_s is None else min(hi_s, ceiling)
    seg = x[max(0, centre - W // 2): centre + W // 2]
    if len(seg) < 32:
        return 0.0, 0.0
    seg = seg - seg.mean()
    den = float((seg * seg).sum())
    if den <= 0:
        return 0.0, 0.0
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:] / den
    lo, hi = int(lo_s * fps), min(int(hi_s * fps), len(ac) - 2)
    if hi <= lo + 2:
        return 0.0, 0.0
    s = ac[lo:hi]
    inner = s[1:-1]
    # local maximum, not argmax -- argmax rides the shoulder of the zero-lag
    # peak and reports the search floor as "the period" for every window.
    pk = np.nonzero((inner > s[:-2]) & (inner >= s[2:]))[0]
    if len(pk) == 0:
        return 0.0, 0.0
    # Take the FUNDAMENTAL, not the tallest peak. A repeated motion puts peaks
    # at P, 2P, 3P ... and which one is tallest is an accident of how even the
    # repetitions are; a sub-step that recurs every third cycle (pick, polish,
    # clean, release) lifts 3P above P. Measured on a real polythene-bagging
    # episode the peaks ran 1.27 s (0.281), 2.63 s (0.263), 3.90 s (0.308) --
    # argmax returned 3.90 s for a cycle the operator performs in about a
    # second, and six of those chunked to an 11 s "cycle" in the output.
    #
    # So: among the peaks, take the SHORTEST lag that is still within
    # HARMONIC_FRAC of the tallest. On three real episodes the choice is stable
    # anywhere in 0.60-0.90, which is why 0.80 is safe rather than tuned.
    vals = inner[pk]
    best = float(vals.max())
    # Scaling by a fraction only means "nearly as tall" for a POSITIVE peak.
    # When every peak is negative -- an un-cadenced window, where the answer is
    # discarded on strength anyway -- `frac * best` is larger than `best` and
    # selects nothing, so fall back to the tallest and let the strength gate
    # reject it.
    thresh = HARMONIC_FRAC * best if best > 0 else best
    k = int(pk[vals >= thresh].min()) + 1
    return float((lo + k) / fps), float(np.clip(s[k], 0.0, 1.0))


# ---------------------------------------------------------------- analysis
def estimate_period(tracks: HandTracks, *, hand: int = RIGHT,
                    lo_s: Optional[float] = None, hi_s: float = 60.0,
                    signal: str = "wrist") -> Tuple[float, float]:
    """Coarse period over the longest tracked runs, searching a wide range.

    The cycle length is a property of the task, not a constant: a pick-and-drop
    repeats in about a second, an assembly or gather-and-bag cycle can take tens
    of seconds. So nothing downstream may assume a value -- it is measured here
    first, and every window in ``analyse`` is then derived from it.

    Each run is searched only up to ``len(run) / MIN_REPS``, so a long cycle is
    reported only where there is enough continuous tracking to support it.
    Returns ``(period_s, strength)``, or ``(0, 0)`` if nothing qualifies.
    """
    fps = tracks.fps
    lo_s = period_floor(fps) if lo_s is None else lo_s
    sig = cadence_signal(tracks, hand, baseline_s=None, signal=signal)
    if sig is None:
        return 0.0, 0.0
    runs = valid_runs(sig, fps, min_run_s=lo_s * MIN_REPS)
    cands: List[Tuple[float, float, int]] = []
    for a, b in runs:
        seg = sig[a:b]
        span = len(seg) / fps
        p, s = local_period(seg, fps, len(seg) // 2, window_s=span,
                            lo_s=lo_s, hi_s=min(hi_s, span / MIN_REPS))
        if p > 0:
            cands.append((p, s, len(seg)))
    if not cands:
        return 0.0, 0.0
    # weight each run's estimate by its strength and its length: a confident
    # estimate from 60 s of tracking outranks a marginal one from 7 s.
    w = np.array([s * n for _, s, n in cands], float)
    p = np.array([p for p, _, _ in cands], float)
    if w.sum() <= 0:
        return 0.0, 0.0
    order = np.argsort(p)
    cw = np.cumsum(w[order])
    med = float(p[order][np.searchsorted(cw, cw[-1] / 2)])
    return med, float(np.average([s for _, s, _ in cands], weights=[n for _, _, n in cands]))


def analyse(tracks: HandTracks, *, hand: Optional[int] = None, min_strength: float = 0.45,
            probe_step_s: Optional[float] = None, period_hint: Optional[float] = None,
            search_factor: float = 2.5, signal: str = "wrist") -> CycleAnalysis:
    """Locate cycle boundaries and measure how much of the clip is cadenced.

    Two passes. First a period-agnostic estimate (``estimate_period``); then the
    detrending baseline, the analysis window and the local search range are all
    set from it, so the same code handles a 1 s pick-and-drop and a 30 s
    assembly without retuning.
    """
    fps, n = tracks.fps, tracks.n_frames

    # No hand given: analyse BOTH and keep whichever yields more cadenced
    # footage. Defaulting to the right hand was an assumption about the worker,
    # and on Polymer_Bags session5/001/seg_000 it was simply wrong -- the right
    # wrist is tracked 76% of the time against the left's 97%, which after the
    # run-length rule left 55% of the clip analysable instead of 96%, and 7
    # chunks instead of 10. Running both costs one extra pass over a signal that
    # is already in memory.
    if hand is None:
        best = None
        for h in (LEFT, RIGHT):
            if tracks.coverage.get(h, 0.0) <= 0.0:
                continue
            cand = analyse(tracks, hand=h, min_strength=min_strength,
                           probe_step_s=probe_step_s, period_hint=period_hint,
                           search_factor=search_factor, signal=signal)
            key = (cand.cadence_fraction, cand.trackable_fraction)
            if best is None or key > best[0]:
                best = (key, cand)
        if best is None:
            return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                                 boundaries=np.zeros(0), cadence_fraction=0.0,
                                 trackable_fraction=0.0)
        return best[1]

    P = period_hint if period_hint else estimate_period(tracks, hand=hand,
                                                       signal=signal)[0]
    if P <= 0:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0,
                             trackable_fraction=0.0)
    window_s = WINDOW_REPS * P                   # ceiling = 2P, peak mid-range
    sig = cadence_signal(tracks, hand, baseline_s=3.0 * P, signal=signal)
    if sig is None:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0,
                             hand=hand, signal=signal, min_strength=min_strength)

    # Only stretches where the hands were ACTUALLY tracked can be analysed.
    # Everything outside them is unknown, not un-cadenced.
    runs = valid_runs(sig, fps, min_run_s=window_s)
    trackable = float(sum(b - a for a, b in runs) / n) if n else 0.0
    if not runs:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0,
                             trackable_fraction=trackable, hand=hand,
                             signal=signal, min_strength=min_strength)

    # Probe every half cycle, not every fixed 0.5 s: at a 0.2 s cycle a fixed
    # stride steps over 2.5 repetitions at a time, and at a 30 s cycle it probes
    # 60 times inside one.
    step = max(int((probe_step_s if probe_step_s else P / 2.0) * fps), 1)
    probed = _probe(sig, runs, fps, P, step, window_s, search_factor)
    return _assemble(probed, fps, n, P, min_strength, trackable, hand, signal)


def _probe(sig, runs, fps, P, step, window_s, search_factor):
    """Everything about a channel that does NOT depend on the gate.

    Separated out because the adaptive search reuses it: `local_period` is the
    expensive part and its answer is the same whatever `min_strength` is -- only
    the comparison against the gate changes. Computing it once per channel
    instead of once per (channel, gate) turns 8 channels x 6 gates from 48
    passes into 8, which is what makes trying every finger affordable.
    """
    out = []
    for a, b in runs:
        seg = sig[a:b]
        probes = np.arange(0, len(seg), step)
        if not len(probes):
            continue
        pv = np.array([local_period(seg, fps, int(i), window_s,
                                    lo_s=P / search_factor,
                                    hi_s=P * search_factor) for i in probes])
        zc = np.nonzero((seg[:-1] <= 0) & (seg[1:] > 0))[0] + 1
        out.append((a, len(seg), probes, pv[:, 0], pv[:, 1], zc))
    return out


def _assemble(probed, fps, n, P, min_strength, trackable, hand, signal):
    """Apply one gate to cached probes."""
    keep: List[int] = []
    per_all, cadenced_frames = [], 0
    for a, seglen, probes, periods, strengths, zc in probed:
        good = strengths >= min_strength
        cadenced_frames += int(good.mean() * seglen)
        per_all += periods[good].tolist()
        if len(zc) == 0:
            continue
        zs = np.interp(zc, probes, strengths)
        keep += (zc[zs >= min_strength] + a).tolist()

    per = float(np.median(per_all)) if per_all else 0.0
    # Suppress double-crossings. Noise near the zero level makes the signal
    # cross twice within a few frames, producing boundary pairs 0.16 s apart in
    # a 1.3 s cycle -- those would become degenerate one-frame "cycles" and
    # corrupt any phase alignment built on top. Keep the first of each cluster.
    kept_t: List[float] = []
    min_gap = 0.5 * (per if per > 0 else P)
    for t in sorted(keep):
        if not kept_t or (t / fps) - kept_t[-1] >= min_gap:
            kept_t.append(t / fps)
    return CycleAnalysis(
        fps=fps, duration_s=n / fps, period_s=per,
        boundaries=np.asarray(kept_t, float),
        cadence_fraction=float(cadenced_frames / n) if n else 0.0,
        trackable_fraction=trackable, hand=hand,
        signal=signal, min_strength=min_strength,
    )


# ---------------------------------------------------------------- segmenting
def _runs(boundaries: np.ndarray, period_s: float, gap_tolerance: float = 1.8
          ) -> List[np.ndarray]:
    """Split boundary times into runs of uninterrupted cadence."""
    if len(boundaries) < 2:
        return []
    gaps = np.diff(boundaries) > max(period_s, 1e-6) * gap_tolerance
    return [r for r in np.split(boundaries, np.nonzero(gaps)[0] + 1) if len(r) >= 2]


def segment_by_cycles(an: CycleAnalysis, n_cycles: int = 6) -> List[Segment]:
    """Every segment is exactly ``n_cycles`` whole cycles, phase-aligned.

    Directly comparable across segments -- same phase, same number of
    repetitions -- at the cost of a variable wall-clock duration.
    """
    out: List[Segment] = []
    for run in _runs(an.boundaries, an.period_s):
        for i in range(0, len(run) - n_cycles, n_cycles):
            t0, t1 = float(run[i]), float(run[i + n_cycles])
            # The segment's OWN period, from its own boundaries -- not the
            # clip-wide median. The boundaries already adapt to a worker
            # speeding up, slowing down or adding a sub-step, so the spans of
            # equal-cycle segments genuinely differ (measured on one clip:
            # 2.97 s to 4.60 s for the same 6 cycles, a 1.55x spread). Stamping
            # the global median on every segment threw that away and reported a
            # constant cadence for a clip that plainly did not have one.
            out.append(Segment(t0, t1, "cadenced", n_cycles,
                               (t1 - t0) / max(n_cycles, 1)))
    return out


#: A candidate is accepted only if its cuts land at the same moment of the work
#: more consistently than the same chunks cut at random times would, at this
#: significance. See ``recurrence_test``.
RECURRENCE_ALPHA = 0.05


def _smoothed(pose: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware moving average of the pose over ``2*half+1`` frames.

    One frame is too noisy to compare poses with. Precomputing the average once
    is what makes the permutation null in ``recurrence_test`` affordable: every
    random cut is then an index lookup rather than a fresh average."""
    ok = np.isfinite(pose).all((1, 2))
    X = np.where(ok[:, None, None], pose, 0.0)
    w = 2 * half + 1
    csum = np.cumsum(np.pad(X, ((w, 0), (0, 0), (0, 0))), axis=0)
    cnt = np.cumsum(np.pad(ok.astype(float), (w, 0)))
    n = len(pose)
    hi = np.minimum(np.arange(n) + half + 1, n) + w
    lo = np.maximum(np.arange(n) - half, 0) + w
    tot = csum[hi - 1] - csum[lo - 1]
    c = (cnt[hi - 1] - cnt[lo - 1])[:, None, None]
    return np.where(c > 0, tot / np.where(c > 0, c, 1.0), np.nan)


#: Phases sampled per cycle when averaging the "anywhere else in the cycle"
#: distance. Sixteen resolves a cycle well and keeps the permutation null cheap.
_PHASES = 16


def _cycle_scores(S: np.ndarray, fps: float, cuts: np.ndarray) -> np.ndarray:
    """One score per cycle: is the hand closer to where it was one cycle ago
    than to where it was at every other phase of the cycle?

        1 - d(pose(b_i), pose(b_i+1)) / mean_t d(pose(b_i), pose(t))

    with ``t`` running over one full cycle of phases CENTRED on the next cut,
    from halfway through this cycle to halfway through the next.

    The comparison is against every phase, not a chosen one, because the first
    version compared the cut against the cycle midpoint and that is wrong for
    the commonest motion there is. Cuts sit on upward zero crossings of the
    channel signal, so half a cycle later is the DOWNWARD crossing: for a hammer
    going up and down one path, the same position moving the other way. The
    cut and the midpoint then look alike and a clean cycle scores near zero --
    a synthetic 2 s cycle cut perfectly every 2.000 s scored 0.94 in one chunk
    and -0.04 in another depending only on where the crossing fell. Averaging
    over all phases makes the score independent of where in the motion the
    cuts land.

    Centring the window on the next cut keeps the mean lag at exactly one
    cycle, which is what cancels head drift: the keypoints are camera-relative,
    and drift that grows with lag inflates both sides equally. Scores are
    clipped at -1 so one wild cycle cannot dominate a pooled mean.
    """
    b = np.asarray(cuts, float)
    if len(b) < 3:
        return np.zeros(0)
    d = np.diff(b)
    u = (np.arange(_PHASES) + 0.5) / _PHASES
    # (cycles, phases): from b_i + d_i/2 to b_i+1 + d_i+1/2
    T = (b[:-2] + d[:-1] / 2)[:, None] + u[None, :] * (d[:-1] / 2 + d[1:] / 2)[:, None]
    idx = lambda t: np.clip(np.round(t * fps).astype(int), 0, len(S) - 1)  # noqa: E731
    p0, p1 = S[idx(b[:-2])], S[idx(b[1:-1])]
    ref = S[idx(T)]                                          # (c, k, 21, 3)
    same = np.sqrt(((p0 - p1) ** 2).sum((1, 2)))
    # a phase where the hand was not tracked is NaN and simply left out; a
    # cycle with no tracked phase at all comes out NaN and is dropped below
    dist = np.sqrt(((ref - p0[:, None]) ** 2).sum((2, 3)))
    cnt = np.isfinite(dist).sum(axis=1)
    spread = np.where(cnt > 0, np.nansum(dist, axis=1) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(same) & np.isfinite(spread) & (spread > 0)
    return np.clip(1.0 - same[ok] / spread[ok], -1.0, 1.0)


def recurrence_test(pose: np.ndarray, fps: float, segments: List[Segment],
                    boundaries: np.ndarray, n_null: int = 200,
                    seed: int = 0) -> Tuple[float, float, List[float]]:
    """Do the cuts land at the same moment of the work -- better than chance?

    This is the test that separates a fast real cycle from jitter at the same
    rate. Picking a screw from one tray and dropping it in the adjacent one
    takes about a second, hammering runs at 2-3 blows a second, and both put
    the hand back in the same place every cycle; keypoint jitter oscillates at
    those rates without going anywhere. Autocorrelation strength cannot tell
    them apart -- it is computed on a normalised signal, so a 2 mm tremor and a
    15 cm reach score alike, and the tremor is often the more regular of the
    two. Only a physical measure can: is the hand's pose at each cut the same
    as at the next?

    Returns ``(observed, p, per_chunk)``. ``observed`` is the mean cycle score
    pooled over every chunk; ``p`` is the share of ``n_null`` re-cuttings -- the
    SAME chunks, the same number of cuts, placed at random times -- that score
    at least as well. That null is the honest comparison: same hand, same
    tracking quality, same drift, same footage, only the phase of the cuts
    destroyed. ``per_chunk`` is each chunk's own mean, for the record.

    WHY A PERMUTATION TEST and not a fixed threshold. A chunk carries only about
    five cycles, so a single chunk's score is noisy: measured on the chunks
    already shipping, random cuts scored a median of +0.005 but a 95th
    percentile of +0.51, so any per-chunk threshold either rejects a fifth of
    the real chunks or keeps a third of the random ones. Pooled over an
    episode the noise falls, and by how much depends on how many cycles there
    are -- a two-chunk hammering clip needs a clearer signal than a twelve-chunk
    one to mean the same thing. A permutation p-value accounts for that by
    construction; a fixed number cannot.

    WHY THE CUTS and not a fixed lag. A fixed lag assumes every cycle is equally
    long. Hammering is close to metronomic; a person pushing parts into a
    machine by hand is not -- on DDY-160 session2/013/seg_001 cycles inside one
    chunk ran 2.2 s to 5.3 s, and a fixed-lag version scored that episode
    -0.094, rejecting exactly the case it was built for. Cut points adapt to
    each cycle's own length.

    Computed on the full 21-joint pose, not on whichever channel found the
    cadence, so the check is independent of the thing it checks and sees both
    kinds of work: a reach moves the whole hand, a push moves the fingers.
    """
    b = np.asarray(boundaries, float)
    per_chunk, obs, prep = [], [], []
    for g in segments:
        cuts = b[(b >= g.t0 - 1e-6) & (b <= g.t1 + 1e-6)]
        if len(cuts) < 3:
            per_chunk.append(float("nan"))
            continue
        # a fixed share of a cycle, so a 0.3 s blow is not smeared and a 4 s
        # push is not left noisy
        half = max(1, round(0.1 * float(np.median(np.diff(cuts))) * fps))
        a0, a1 = max(int(g.t0 * fps) - 2 * half, 0), int(g.t1 * fps) + 2 * half + 2
        S = np.full_like(pose, np.nan)
        S[a0:a1] = _smoothed(pose[a0:a1], half)
        sc = _cycle_scores(S, fps, cuts)
        per_chunk.append(float(sc.mean()) if len(sc) else float("nan"))
        obs.append(sc)
        prep.append((S, g.t0, g.t1, len(cuts) - 2))
    pooled = np.concatenate(obs) if obs else np.zeros(0)
    if len(pooled) == 0:
        return float("nan"), 1.0, per_chunk
    observed = float(pooled.mean())
    rng = np.random.default_rng(seed)
    beat = 0
    for _ in range(n_null):
        null = []
        for S, t0, t1, k in prep:
            cuts = np.sort(np.concatenate([[t0, t1], rng.uniform(t0, t1, k)]))
            null.append(_cycle_scores(S, fps, cuts))
        v = np.concatenate(null)
        if len(v) and v.mean() >= observed:
            beat += 1
    return observed, (beat + 1) / (n_null + 1), per_chunk


#: Gates the adaptive search walks down, strictest first. It stops at the first
#: one that yields enough chunks, so an episode with clean cadence is still
#: judged at 0.45 and only genuinely marginal work is measured loosely.
#:
#: The floor is 0.20 and it is not negotiable-by-accident: below it the
#: autocorrelation peak is not distinguishable from the peak a random walk
#: produces over the same window, so "cadence" found there would be an artifact
#: of looking. An episode that reaches the floor without enough chunks is
#: reported as having none, which is the honest answer.
GATES = (0.45, 0.40, 0.35, 0.30, 0.25, 0.20)


def analyse_adaptive(tracks: HandTracks, *, n_cycles: int = 6, min_chunks: int = 2,
                     min_chunk_s: float = 0.0,
                     recurrence_alpha: Optional[float] = RECURRENCE_ALPHA,
                     gates: Tuple[float, ...] = GATES,
                     signals: Tuple[str, ...] = SIGNAL_NAMES,
                     probe_step_s: Optional[float] = None,
                     search_factor: float = 2.5) -> CycleAnalysis:
    """Find cadence on whichever hand and channel carries it, relaxing the gate
    only as far as it takes to get ``min_chunks`` comparable chunks.

    Two things are being searched at once and they are not the same thing:

    * WHERE the cadence is -- which hand, and which of the eight channels. This
      is not a threshold decision. A worker pushing parts with the fingers and
      a worker swinging a whole arm are both repetitive; only the second shows
      up in the wrist path. Every channel is tried at the strictest gate before
      any gate is loosened, so a clean finger cadence is preferred over a
      marginal wrist one rather than the other way round.
    * HOW confident we insist on being -- the gate. Lowered only when nothing
      at all was found above it.

    ``recurrence_alpha`` is the physical test: a candidate is accepted only if
    the hand's pose at each cut matches the next cut's more consistently than
    the same chunks cut at random times would (see ``recurrence_test``). It is
    what separates a 1 s pick-and-drop between two adjacent trays, or hammering
    at three blows a second -- fast, but the hand lands in the same place every
    time -- from jitter at the same rate, which no amount of autocorrelation
    strength can do. A candidate that fails does not end the search: the next
    candidate at the same gate is tried, then the next gate. ``None`` disables
    the test.

    If nothing passes, the best near-miss is still returned so a caller can
    report WHY, and its ``segments`` are the near-miss's chunks: check
    ``recurrence_p <= RECURRENCE_ALPHA`` before shipping them.

    ``min_chunk_s`` rejects a candidate whose chunks are too short to compare,
    and it is applied DURING the search rather than as a filter afterwards: a
    channel locked onto a 0.2 s oscillation and one that found a 4 s work cycle
    are both "cadenced", and if the short one is allowed to win first the long
    one is never considered. Whether a chunk is long enough is a property of the
    appearance comparison downstream, not of the work, which is why it is a
    parameter here and not a constant.

    WHAT THIS COSTS, stated plainly. Walking the gate down until chunks appear
    means an episode nearly always produces chunks, so "this episode has chunks"
    stops being evidence that the work is repetitive. That is an acceptable
    trade only because the repetitiveness question is answered downstream by the
    similarity between chunks, not by their existence -- chunks cut out of
    non-repetitive footage are simply dissimilar. The cost is paid in two
    places and both are recorded rather than hidden: ``min_strength`` says how
    far the gate had to fall, and ``signal`` says how many channels were tried
    before one worked. Trying eight channels and keeping the best inflates the
    apparent strength of the winner, so a chunk found at 0.25 on one finger is
    weaker evidence than the number alone suggests.
    """
    best_effort: Optional[CycleAnalysis] = None
    cache = []
    for hand in (LEFT, RIGHT):
        if tracks.coverage.get(hand, 0.0) <= 0.0:
            continue
        for name in signals:
            P = estimate_period(tracks, hand=hand, signal=name)[0]
            if P <= 0:
                continue
            window_s = WINDOW_REPS * P
            sig = cadence_signal(tracks, hand, baseline_s=3.0 * P, signal=name)
            if sig is None:
                continue
            runs = valid_runs(sig, tracks.fps, min_run_s=window_s)
            if not runs:
                continue
            trackable = float(sum(b - a for a, b in runs) / tracks.n_frames)
            step = max(int((probe_step_s if probe_step_s else P / 2.0) * tracks.fps), 1)
            probed = _probe(sig, runs, tracks.fps, P, step, window_s, search_factor)
            if probed:
                cache.append((hand, name, P, trackable, probed))

    for gate in gates:
        at_gate = []
        for hand, name, P, trackable, probed in cache:
            an = _assemble(probed, tracks.fps, tracks.n_frames, P, gate,
                           trackable, hand, name)
            segs = segment_by_cycles(an, n_cycles=n_cycles)
            if min_chunk_s > 0:
                segs = [g for g in segs if g.seconds >= min_chunk_s]
            an.segments = segs
            n = len(segs)
            # keep the strongest near-miss so a failed search still explains
            # itself instead of returning an empty object
            key = (n, an.cadence_fraction, an.trackable_fraction)
            if best_effort is None or key > best_effort[0]:
                best_effort = (key, an)
            if n >= min_chunks:
                # Rank the channels that qualify by how much of the clip they
                # explain, NOT by chunk count. Chunk count rewards short
                # periods, and a channel that locks onto the second harmonic
                # produces twice as many chunks than the one that found the
                # real cycle -- so ranking on count systematically picks the
                # harmonic. The count has already done its job as the gate.
                at_gate.append(((an.cadence_fraction, an.trackable_fraction,
                                 an.period_s), an))
        # Best first, and the first one whose cuts pass the recurrence test
        # wins. Testing only the qualifiers, and only until one passes, keeps
        # the permutation test off the hot path: most candidates never reach
        # it. If none passes, the gate is loosened -- a spurious candidate at a
        # strict gate must not block a real one at a looser gate.
        for _, an in sorted(at_gate, key=lambda t: t[0], reverse=True):
            if recurrence_alpha is None:
                return an
            got = _fundamental(tracks, an, n_cycles, min_chunks, min_chunk_s,
                               recurrence_alpha)
            if got is not None:
                return got

    if best_effort is not None:
        return best_effort[1]
    return CycleAnalysis(fps=tracks.fps, duration_s=tracks.n_frames / tracks.fps,
                         period_s=0.0, boundaries=np.zeros(0), cadence_fraction=0.0,
                         trackable_fraction=0.0, min_strength=gates[-1])


#: Multiples of a candidate's cycle the recurrence test also tries, to catch a
#: cadence found at a harmonic of the real one.
HARMONICS = (1, 2, 3)


def _decimated(an: CycleAnalysis, k: int, n_cycles: int,
               min_chunk_s: float) -> CycleAnalysis:
    """The same cuts, keeping every ``k``-th one in each cadenced run: the
    candidate re-read at ``k`` times its cycle."""
    if k == 1:
        return an
    runs = _runs(an.boundaries, an.period_s)
    kept = np.concatenate([r[::k] for r in runs]) if runs else np.zeros(0)
    out = replace(an, boundaries=kept, period_s=an.period_s * k, segments=[])
    segs = segment_by_cycles(out, n_cycles=n_cycles)
    if min_chunk_s > 0:
        segs = [g for g in segs if g.seconds >= min_chunk_s]
    out.segments = segs
    return out


def _fundamental(tracks: HandTracks, an: CycleAnalysis, n_cycles: int,
                 min_chunks: int, min_chunk_s: float,
                 alpha: float) -> Optional[CycleAnalysis]:
    """Accept a candidate at the cycle length where the hand really returns.

    Autocorrelation can lock onto a harmonic of the work, and HARMONIC_FRAC
    deliberately leans toward the SHORT lag to avoid the opposite mistake, so a
    candidate's cycle may be a fraction of the real one. The recurrence test can
    tell the two apart because it is physical: cut at a third of a hammer
    stroke, the hand is in three different places at consecutive cuts; cut at
    the stroke, it is in the same place. Measured on DDY-160
    session2/009/seg_000, which production shipped cut every 0.33 s: frames at
    the cuts repeat every THIRD cut, 1.03 s apart -- the actual blow -- and the
    0.33 s reading fails the test (p = 0.09) that the 1.03 s reading is meant
    to pass.

    So each of ``HARMONICS`` is tested and the SHORTEST cycle whose recurrence
    is significant and at least ``HARMONIC_FRAC`` of the best is returned --
    the same fundamental-over-harmonic rule ``local_period`` applies to
    autocorrelation peaks, for the same reason: every multiple of a real cycle
    also recurs, and nearly as well. Returns None if no multiple passes.
    """
    tried = []
    for k in HARMONICS:
        cand = _decimated(an, k, n_cycles, min_chunk_s)
        if len(cand.segments) < min_chunks:
            continue
        obs, pval, per = recurrence_test(tracks.pose[cand.hand], tracks.fps,
                                         cand.segments, cand.boundaries)
        cand.recurrence, cand.recurrence_p = obs, pval
        for g, r in zip(cand.segments, per):
            g.recurrence = r
        tried.append((k, cand))
    good = [(k, c) for k, c in tried
            if c.recurrence_p <= alpha and np.isfinite(c.recurrence)]
    if not good:
        return None
    best = max(c.recurrence for _, c in good)
    floor = HARMONIC_FRAC * best if best > 0 else best
    return min((kc for kc in good if kc[1].recurrence >= floor), key=lambda kc: kc[0])[1]


def segment_by_duration(an: CycleAnalysis, target_s: float = 10.0) -> List[Segment]:
    """Aim for ``target_s`` but always cut on a cycle boundary.

    Keeps duration roughly even, which the appearance and video encoders prefer,
    while never slicing through the middle of a repetition. Cycle count varies.
    """
    out: List[Segment] = []
    for run in _runs(an.boundaries, an.period_s):
        i = 0
        while i < len(run) - 1:
            j = i + 1
            while j < len(run) - 1 and (run[j] - run[i]) < target_s:
                j += 1
            # take whichever of the two straddling boundaries lands closer
            if j > i + 1 and abs(run[j - 1] - run[i] - target_s) < abs(run[j] - run[i] - target_s):
                j -= 1
            t0, t1 = float(run[i]), float(run[j])
            # its own period, as in segment_by_cycles
            out.append(Segment(t0, t1, "cadenced", j - i, (t1 - t0) / max(j - i, 1)))
            i = j
    return out


def transition_segments(an: CycleAnalysis, segments: List[Segment],
                        min_seconds: float = 3.0) -> List[Segment]:
    """The stretches no cadenced segment covers.

    These are not leftovers. Non-repetitive footage is where the unusual events
    live, so it is labelled and kept rather than dropped.
    """
    covered = sorted((s.t0, s.t1) for s in segments)
    out, t = [], 0.0
    for a, b in covered:
        if a - t >= min_seconds:
            out.append(Segment(t, float(a), "transition"))
        t = max(t, b)
    if an.duration_s - t >= min_seconds:
        out.append(Segment(t, an.duration_s, "transition"))
    return out
