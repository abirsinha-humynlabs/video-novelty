"""The articulation channels: a cadence the wrist cannot see.

The case these exist for is a worker whose wrist is parked -- pushing a part
into a machine with the fingers, pinching, or feeding stock -- where the work
is plainly periodic and the wrist path is not. The fixtures below are built to
be exactly that: a hand whose wrist goes nowhere while the fingers flex on a
fixed cycle, wrapped in an arbitrary rigid motion so that any channel claiming
to measure articulation has to prove it is not just measuring the hand moving.
"""
import numpy as np
import pytest

from novelty import cycles as C


def _hand(n_frames=1800, fps=30.0, period_s=2.0, flex=0.8,
          wrist_amp=0.0, rotate=False, seed=0, noise=3e-4):
    """A synthetic 21-joint hand whose fingers flex periodically.

    ``wrist_amp`` moves the whole hand; ``rotate`` spins it. Neither changes the
    finger cadence, which is the point.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * t / period_s
    pose = np.zeros((n_frames, 21, 3), np.float64)
    for f, base in enumerate(C.FINGERS.values()):
        for j, joint in enumerate(base[1:]):          # skip the wrist itself
            # flex from straight to curled and back, never through zero into
            # a mirrored pose: a symmetric +/- flex makes the joint ANGLE the
            # same at both extremes, which halves the measured period. Real
            # fingers do not hyperextend, so this is also the honest shape.
            # j = 0 is the knuckle: it stays put, as a real one does, and the
            # finger bends from it outward, distal joints most
            a = flex * (1 + np.sin(phase)) / 2 * j / 3.0
            pose[:, joint, 0] = 0.02 * (f - 2)
            pose[:, joint, 1] = 0.02 * (j + 1) * np.cos(a)
            pose[:, joint, 2] = 0.02 * (j + 1) * np.sin(a)
    if wrist_amp:
        pose += (wrist_amp * np.sin(phase))[:, None, None] * np.array([1.0, 0, 0])
    if rotate:
        th = 0.9 * np.sin(2 * np.pi * t / 7.3)        # incommensurate with the cycle
        cs, sn = np.cos(th), np.sin(th)
        R = np.zeros((n_frames, 3, 3))
        R[:, 0, 0] = cs; R[:, 0, 1] = -sn
        R[:, 1, 0] = sn; R[:, 1, 1] = cs
        R[:, 2, 2] = 1.0
        pose = np.einsum("nij,nkj->nki", R, pose)
        pose += (0.3 * np.sin(2 * np.pi * t / 11.7))[:, None, None] * np.array([0, 1.0, 0])
    if noise:
        pose += rng.normal(0, noise, pose.shape)
    return C.HandTracks(fps=fps, n_frames=n_frames,
                        pose={C.LEFT: pose, C.RIGHT: np.full_like(pose, np.nan)},
                        world=False, coverage={C.LEFT: 1.0, C.RIGHT: 0.0})


def test_palm_rotation_ignores_the_fingers_curling():
    """The wrist-level rotation channel must not move when only fingers flex."""
    rot = C.raw_signal(_hand(period_s=2.0, wrist_amp=0.0, noise=0.0), C.LEFT, "rotation")
    assert np.nanstd(rot) < 1e-6


def test_finger_curl_ignores_where_the_hand_is_and_how_it_points():
    """Rotation and translation must not leak into the curl signal.

    This is the property the naive 'fingertip relative to the wrist' lacks:
    rotating a rigid hand moves the fingertip relative to the wrist just as
    much as flexing does, which is why that channel scored WORSE than the
    wrist on real footage.
    """
    # noiseless: this is a geometric identity, and measuring it through noise
    # would only test the noise level
    still = _hand(rotate=False, noise=0.0)
    moved = _hand(rotate=True, noise=0.0)
    a = C.finger_curl(still.pose[C.LEFT], "index")
    b = C.finger_curl(moved.pose[C.LEFT], "index")
    assert np.nanmax(np.abs(a - b)) < 1e-6   # float32 keypoints, not exact zero


def test_every_named_channel_produces_a_signal():
    tr = _hand()
    for name in C.SIGNAL_NAMES:
        sig = C.raw_signal(tr, C.LEFT, name)
        assert sig is not None, name
        assert np.isfinite(sig).mean() > 0.5, name


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError):
        C.raw_signal(_hand(), C.LEFT, "elbow")


def test_finger_cadence_is_invisible_to_the_wrist_and_found_by_articulation():
    """The whole reason the channels exist, as a single assertion."""
    tr = _hand(period_s=2.0, wrist_amp=0.0)
    wrist = C.analyse(tr, hand=C.LEFT, signal="wrist")
    assert len(C.segment_by_cycles(wrist, n_cycles=6)) == 0

    an = C.analyse_adaptive(tr, n_cycles=6, min_chunks=2)
    assert len(an.segments) >= 2
    assert an.signal != "wrist"
    assert an.period_s == pytest.approx(2.0, abs=0.4)


def test_a_clean_wrist_cadence_is_still_judged_at_the_strictest_gate():
    """Relaxing the gate must be a last resort, not the default path."""
    tr = _hand(period_s=2.0, wrist_amp=0.25)
    an = C.analyse_adaptive(tr, n_cycles=6, min_chunks=2)
    assert len(an.segments) >= 2
    assert an.min_strength == C.GATES[0]


def test_the_gate_never_falls_below_the_floor():
    """Noise must not be talked into a cadence, however far the search goes."""
    rng = np.random.default_rng(1)
    pose = np.cumsum(rng.normal(0, 3e-3, (1800, 21, 3)), axis=0)
    tr = C.HandTracks(fps=30.0, n_frames=1800,
                      pose={C.LEFT: pose, C.RIGHT: np.full_like(pose, np.nan)},
                      world=False, coverage={C.LEFT: 1.0, C.RIGHT: 0.0})
    an = C.analyse_adaptive(tr, n_cycles=6, min_chunks=2)
    assert an.min_strength >= C.GATES[-1]


def test_the_analysis_records_how_it_found_the_cadence():
    """`signal` and `min_strength` travel with the result, so a chunk found on
    one finger at a loosened gate is never mistaken for a clean wrist cadence."""
    an = C.analyse_adaptive(_hand(), n_cycles=6, min_chunks=2)
    assert an.signal in C.SIGNAL_NAMES
    assert an.min_strength in C.GATES


def test_default_analyse_is_unchanged():
    """Backward compatibility: the wrist path must behave exactly as before."""
    tr = _hand(period_s=2.0, wrist_amp=0.25)
    assert C.analyse(tr).signal == "wrist"
    assert C.analyse(tr).min_strength == 0.45


def _pose(track):
    """(n, 3) -> (n, 21, 3): every joint on the same path."""
    return np.repeat(track[:, None, :], 21, axis=1)


def test_cut_recurrence_ignores_head_drift_and_rewards_a_return():
    """Drift and tremor score ~0 at the cuts; a real cycle scores high.

    The cuts are placed on ZERO CROSSINGS of the motion on purpose. That is
    where the pipeline puts them, and it is the case that broke the first
    version of this score: half a cycle after an upward crossing is the
    downward one, where a back-and-forth hand is in the same place moving the
    other way, so comparing the cut to the cycle midpoint scored a clean cycle
    near zero. Averaging over every phase of the cycle fixes that.

    Drift matters because the keypoints are camera-relative with no head pose
    on most episodes; a one-sided comparison scored pure linear drift -1.0,
    counting every head movement against the work, and rejected episodes that
    were cut correctly.
    """
    rng = np.random.default_rng(0)
    n, fps = 3000, 30.0
    t = np.arange(n) / fps
    cuts = np.arange(0.5, t[-1] - 2.0, 1.0)        # every 1 s, on the crossing
    zero = 0 * t
    drift = _pose(np.stack([t * 0.02, zero, zero], 1))
    walk = _pose(np.cumsum(rng.normal(0, 3e-3, (n, 3)), 0))
    cycle = _pose(np.stack([0.1 * np.sin(2 * np.pi * (t - 0.5)), zero, zero], 1))
    tremor = _pose(rng.normal(0, 2e-3, (n, 3)))
    score = lambda P: float(np.mean(C._cycle_scores(P, fps, cuts)))  # noqa: E731

    assert abs(score(drift)) < 0.02
    assert abs(score(walk)) < 0.1
    assert abs(score(tremor)) < 0.1
    assert score(cycle) > 0.9
    assert score(cycle + walk) > 0.6                # the work survives the drift


def test_an_accepted_cadence_beats_random_cuts():
    """A clean synthetic cycle must pass the permutation test decisively, and
    the result must say so -- the p-value travels with the chunks."""
    an = C.analyse_adaptive(_hand(period_s=2.0), n_cycles=6, min_chunks=2)
    assert an.segments
    assert an.recurrence_p <= C.RECURRENCE_ALPHA
    assert an.recurrence > 0.5
    assert all(np.isfinite(g.recurrence) for g in an.segments)


def test_cuts_at_random_times_do_not_pass():
    """The null the test is built on: the same footage, cut at random times,
    must not look like a cadence."""
    tr = _hand(period_s=2.0)
    an = C.analyse_adaptive(tr, n_cycles=6, min_chunks=2)
    rng = np.random.default_rng(3)
    fake = np.sort(rng.uniform(0, tr.n_frames / tr.fps, len(an.boundaries)))
    segs = [C.Segment(float(fake[i]), float(fake[i + 6]), "cadenced", 6,
                      float(fake[i + 6] - fake[i]) / 6)
            for i in range(0, len(fake) - 6, 6)]
    _, p, _ = C.recurrence_test(tr.pose[C.LEFT], tr.fps, segs, fake)
    assert p > C.RECURRENCE_ALPHA


def test_the_recurrence_test_can_be_disabled():
    an = C.analyse_adaptive(_hand(), n_cycles=6, min_chunks=2, recurrence_alpha=None)
    assert an.segments


def test_no_cycle_shorter_than_the_floor_is_accepted():
    """A 0.4 s oscillation is a PART of a work cycle, not one. With the floor at
    a second, every accepted chunk must be made of cycles of at least a second
    -- whole multiples of the fast motion, or nothing."""
    tr = _hand(period_s=0.4, n_frames=3600)
    an = C.analyse_adaptive(tr, n_cycles=3, min_chunks=2, min_cycle_s=C.MIN_CYCLE_S)
    assert all(g.period_s >= C.MIN_CYCLE_S - 1e-9 for g in an.segments)
    free = C.analyse_adaptive(tr, n_cycles=3, min_chunks=2)
    assert free.segments and min(g.period_s for g in free.segments) < 0.6   # the floor is what changed


def test_a_real_cycle_is_still_found_with_the_floor_and_three_cycles_per_chunk():
    an = C.analyse_adaptive(_hand(period_s=2.0), n_cycles=3, min_chunks=2,
                            min_cycle_s=C.MIN_CYCLE_S)
    assert len(an.segments) >= 2
    assert an.recurrence_p <= C.RECURRENCE_ALPHA
    assert all(g.n_cycles == 3 and 1.6 <= g.period_s <= 2.4 for g in an.segments)


def test_wrist_is_trusted_before_the_fingers():
    """With a wrist cadence present, a finger channel must not win."""
    an = C.analyse_adaptive(_hand(period_s=2.0, wrist_amp=0.25), chunk_s=3.0,
                            min_chunks=2, min_cycle_s=C.MIN_CYCLE_S, tiers=C.SIGNAL_TIERS)
    assert an.signal in C.SIGNAL_TIERS[0]
    assert an.recurrence_p <= C.RECURRENCE_ALPHA


def test_fingers_are_the_fallback_when_the_wrist_is_still():
    an = C.analyse_adaptive(_hand(period_s=2.0, wrist_amp=0.0), chunk_s=3.0,
                            min_chunks=2, min_cycle_s=C.MIN_CYCLE_S, tiers=C.SIGNAL_TIERS)
    assert an.segments and an.signal in C.SIGNAL_TIERS[1]


def _cuts(every_s, total_s=30.0):
    b = np.arange(0.0, total_s + 1e-9, every_s)
    return C.CycleAnalysis(fps=30.0, duration_s=total_s, period_s=every_s, boundaries=b,
                           cadence_fraction=1.0)


def test_a_chunk_is_the_fewest_whole_cycles_reaching_the_minimum():
    one_s = C.segment_whole_cycles(_cuts(1.0), min_s=3.0)
    assert one_s and all(g.n_cycles == 3 and abs(g.seconds - 3.0) < 1e-9 for g in one_s)
    four_s = C.segment_whole_cycles(_cuts(4.0), min_s=3.0)
    assert four_s and all(g.n_cycles == 1 and abs(g.seconds - 4.0) < 1e-9 for g in four_s)
    # every chunk starts and ends on a cut
    b = set(np.round(_cuts(1.0).boundaries, 6))
    assert all(round(g.t0, 6) in b and round(g.t1, 6) in b for g in one_s)


def test_one_cycle_chunks_are_still_tested():
    """A one-cycle chunk has no cycle inside it; the test must score the run of
    cuts it belongs to, so a long real cycle is not rejected for being long."""
    an = C.analyse_adaptive(_hand(period_s=4.0, n_frames=3600), chunk_s=3.0, min_chunks=2,
                            min_cycle_s=C.MIN_CYCLE_S)
    assert an.segments and all(g.seconds >= 3.0 - 1e-9 for g in an.segments)
    # crossing jitter can make one interval a little short, so not every chunk
    # is exactly one cycle -- but nearly all are
    assert np.mean([g.n_cycles == 1 for g in an.segments]) >= 0.8
    assert an.recurrence_p <= C.RECURRENCE_ALPHA
