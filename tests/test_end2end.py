"""The test that matters: can the pipeline resolve the 2x2?

The shipped real-world eval set (eval/pairs.yaml) cannot answer this, because
every same-task pair in it is also a same-environment pair. Here we render
synthetic clips that vary the two factors independently:

              period 2.0s        period 5.0s
    scene 1 | a1, a2            | a3
    scene 9 | b1                | b2

and assert the two axes respond to the right factor. If this test passes and
your real data still confuses the axes, the problem is your features or your
corpus -- not the plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest

from novelty.calibrate import fit_null, fit_whitener
from novelty.config import Config
from novelty.metrics.fuse import Label, compare, raw_scores
from novelty.select import facility_location_greedy
from novelty.signature import build_signature
from novelty.store.numpy_store import Index


@pytest.fixture(scope="module")
def sigs(clips):
    cfg = Config()
    cfg.appearance.n_frames = 24
    return {k: build_signature(v, cfg=cfg) for k, v in clips.items()}


@pytest.fixture(scope="module")
def wh(sigs):
    return fit_whitener(list(sigs.values()))


def test_environment_axis_tracks_the_scene(sigs, wh):
    same_scene = raw_scores(sigs["a1"], sigs["a3"], whitener=wh)["env_raw"]
    diff_scene = raw_scores(sigs["a1"], sigs["b1"], whitener=wh)["env_raw"]
    assert same_scene > diff_scene, "environment score must follow the scene, not the motion"


def test_environment_axis_ignores_the_task(sigs, wh):
    """Same scene, different motion should still read as the same place."""
    same_scene_same_task = raw_scores(sigs["a1"], sigs["a2"], whitener=wh)["env_raw"]
    same_scene_diff_task = raw_scores(sigs["a1"], sigs["a3"], whitener=wh)["env_raw"]
    diff_scene = raw_scores(sigs["a1"], sigs["b2"], whitener=wh)["env_raw"]
    assert min(same_scene_same_task, same_scene_diff_task) > diff_scene


def test_task_axis_tracks_the_motion(sigs, wh):
    same_task_diff_scene = raw_scores(sigs["a1"], sigs["b1"], whitener=wh)["task_raw"]
    diff_task_same_scene = raw_scores(sigs["a1"], sigs["a3"], whitener=wh)["task_raw"]
    assert same_task_diff_scene > diff_task_same_scene, \
        "task score must follow the motion, not the scene"


def test_cycle_period_orders_and_replicates(sigs):
    """Cadence estimation must be ORDINALLY right and reproducible.

    Deliberately not asserting that the detected period equals the rendered
    one. The marker traces its path at constant angular rate, so its *speed*
    -- which is what optical flow measures -- peaks twice per positional
    cycle. Motion energy therefore carries the half-period, plus harmonics.
    Asserting an absolute value here would be testing a coincidence.

    What the pipeline actually needs is weaker and more useful: a faster cycle
    must read as faster, and two recordings of the same cadence must agree.
    """
    fast = [sigs["a1"].period_s, sigs["a2"].period_s]      # rendered 2.0 s
    slow = [sigs["a3"].period_s, sigs["b2"].period_s]      # rendered 5.0 s
    assert all(p > 0 for p in fast + slow), "no cycle detected at all"
    assert max(fast) < min(slow), f"ordering broken: fast={fast} slow={slow}"
    assert fast[0] == pytest.approx(fast[1], rel=0.5), "same cadence, different estimates"


def test_identical_conditions_are_the_closest_pair(sigs, wh):
    base = sigs["a1"]
    scores = {k: raw_scores(base, v, whitener=wh)["env_raw"] + raw_scores(base, v, whitener=wh)["task_raw"]
              for k, v in sigs.items() if k != "a1"}
    assert max(scores, key=scores.get) == "a2"


def test_index_calibrate_select_roundtrip(clips, tmp_path):
    cfg = Config()
    cfg.appearance.n_frames = 16
    idx = Index(str(tmp_path / "idx")).create(cfg)
    for p in clips.values():
        idx.add([build_signature(p, cfg=cfg)])
    assert len(idx) == len(clips)

    null = fit_null(idx.signatures(), max_pairs=200)
    idx.save_null(null)
    assert idx.null() is not None
    assert 0.0 <= idx.null().percentile("env_raw", 0.0) <= 100.0

    S = idx.similarity_matrix(kind="fused")
    assert S.shape == (len(clips), len(clips))
    assert np.allclose(S, S.T, atol=1e-5)

    res = facility_location_greedy(S, budget=3)
    assert len(res.order) == 3
    assert np.all(np.diff(res.gains) <= 1e-6)


def test_compare_emits_a_usable_verdict(sigs):
    v = compare(sigs["a1"], sigs["b2"])
    assert v.label in {Label.NOVEL, Label.REDUNDANT, Label.SAME_PLACE_NEW_TASK,
                       Label.SAME_TASK_NEW_PLACE, Label.DUPLICATE_SOURCE}
    assert not v.calibrated
    assert any("calibrate" in n for n in v.notes), "must warn loudly when uncalibrated"
    assert "env_cos" in v.raw
    assert isinstance(v.explain(), str) and len(v.explain()) > 40


def test_duplicate_source_detected_on_a_reencode(clips, tmp_path):
    """Tier 0 must fire on a re-encoded copy -- the one case that is not a judgement call."""
    import subprocess
    out = tmp_path / "copy.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clips["a1"],
                    "-vf", "scale=100:76", "-crf", "30", str(out)], check=True)
    cfg = Config()
    cfg.appearance.n_frames = 12
    a = build_signature(clips["a1"], cfg=cfg)
    b = build_signature(str(out), cfg=cfg)
    v = compare(a, b)
    assert v.label == Label.DUPLICATE_SOURCE
    assert v.duplicate["overlap_seconds"] > 2.0
