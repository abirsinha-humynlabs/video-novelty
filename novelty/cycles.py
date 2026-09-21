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

from dataclasses import dataclass, field
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


def _smooth(x: np.ndarray, fps: float, win_s: float = 0.25) -> np.ndarray:
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


def cadence_signal(tracks: HandTracks, hand: int = RIGHT,
                   baseline_s: Optional[float] = None) -> Optional[np.ndarray]:
    """Signed, locally-detrended 1-D signal whose oscillation is one work cycle.

    The wrist path of a repetitive reach-and-place is close to a line segment
    travelled back and forth, so projecting onto its own principal axis
    recovers a signed oscillation -- one crossing per cycle, phase preserved.

    ``baseline_s`` is the window for local-baseline removal and must scale with
    the cycle: ~3 cycles. Pass ``None`` for the first, period-agnostic pass --
    it then only removes each run's mean, which is enough to estimate a period
    but not enough to survive the worker walking around.
    """
    W = tracks.pose[hand][:, WRIST, :]
    g = np.isfinite(W[:, 0])
    if g.sum() < 32:
        return None
    X = W[g] - W[g].mean(0)
    pc = np.linalg.svd(X, full_matrices=False)[2][0]
    proj = np.full(len(W), np.nan)
    proj[g] = X @ pc
    filled = _fill(proj, tracks.fps)
    if filled is None:
        return None
    # smooth and detrend WITHIN each tracked run, so a gap cannot leak a
    # fabricated baseline into the run beside it
    out = np.full(len(filled), np.nan)
    for a, b in valid_runs(filled, tracks.fps, min_run_s=1.0):
        seg = _smooth(filled[a:b], tracks.fps)
        if baseline_s is None:
            out[a:b] = seg - seg.mean()
        else:
            w = min(baseline_s * tracks.fps, max(len(seg) // 2, 3))
            out[a:b] = seg - _running_median(seg, w)
    return out


def local_period(x: np.ndarray, fps: float, centre: int, window_s: float,
                 lo_s: float = 0.4, hi_s: Optional[float] = None
                 ) -> Tuple[float, float]:
    """Dominant period and autocorrelation strength in a window around ``centre``.

    ``hi_s`` defaults to ``window_s / MIN_REPS`` and is clamped to it, because a
    lag longer than that is averaged over too few repetitions to mean anything.
    The previous defaults searched to 8 s inside a 6 s window, which reported
    confident nonsense at the long end and made every task with a cycle above
    ~1.2 s structurally invisible.
    """
    W = int(window_s * fps)
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
    k = int(pk[int(np.argmax(inner[pk]))]) + 1
    return float((lo + k) / fps), float(np.clip(s[k], 0.0, 1.0))


# ---------------------------------------------------------------- analysis
def estimate_period(tracks: HandTracks, *, hand: int = RIGHT,
                    lo_s: float = 0.4, hi_s: float = 60.0
                    ) -> Tuple[float, float]:
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
    sig = cadence_signal(tracks, hand, baseline_s=None)
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


def analyse(tracks: HandTracks, *, hand: int = RIGHT, min_strength: float = 0.45,
            probe_step_s: float = 0.5, period_hint: Optional[float] = None,
            search_factor: float = 2.5) -> CycleAnalysis:
    """Locate cycle boundaries and measure how much of the clip is cadenced.

    Two passes. First a period-agnostic estimate (``estimate_period``); then the
    detrending baseline, the analysis window and the local search range are all
    set from it, so the same code handles a 1 s pick-and-drop and a 30 s
    assembly without retuning.
    """
    fps, n = tracks.fps, tracks.n_frames
    P = period_hint if period_hint else estimate_period(tracks, hand=hand)[0]
    if P <= 0:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0,
                             trackable_fraction=0.0)
    window_s = WINDOW_REPS * P                   # ceiling = 2P, peak mid-range
    sig = cadence_signal(tracks, hand, baseline_s=3.0 * P)
    if sig is None:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0)

    # Only stretches where the hands were ACTUALLY tracked can be analysed.
    # Everything outside them is unknown, not un-cadenced.
    runs = valid_runs(sig, fps, min_run_s=window_s)
    trackable = float(sum(b - a for a, b in runs) / n) if n else 0.0
    if not runs:
        return CycleAnalysis(fps=fps, duration_s=n / fps, period_s=0.0,
                             boundaries=np.zeros(0), cadence_fraction=0.0,
                             trackable_fraction=trackable)

    step = max(int(probe_step_s * fps), 1)
    keep: List[int] = []
    per_all, cadenced_frames = [], 0
    for a, b in runs:
        seg = sig[a:b]
        probes = np.arange(0, len(seg), step)
        pv = np.array([local_period(seg, fps, int(i), window_s,
                                    lo_s=P / search_factor,
                                    hi_s=P * search_factor) for i in probes])
        periods, strengths = pv[:, 0], pv[:, 1]
        good = strengths >= min_strength
        cadenced_frames += int(good.mean() * len(seg)) if len(probes) else 0
        per_all += periods[good].tolist()

        zc = np.nonzero((seg[:-1] <= 0) & (seg[1:] > 0))[0] + 1
        if len(zc) == 0 or not len(probes):
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
        trackable_fraction=trackable,
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
            out.append(Segment(float(run[i]), float(run[i + n_cycles]),
                               "cadenced", n_cycles, an.period_s))
    return out


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
            out.append(Segment(float(run[i]), float(run[j]), "cadenced", j - i, an.period_s))
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
