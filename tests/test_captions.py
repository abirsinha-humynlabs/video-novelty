"""VLM caption task-similarity tests.

Synthetic captions, because the real captioning output does not exist for the
target corpus yet. The point is that the *ordering* is right: the same job
repeated must score above the same objects used for a different job, which must
score above unrelated work.
"""
import numpy as np
import pytest

from novelty.captions import (CaptionSegment, VideoCaptions, caption_similarity,
                              fit_idf, load_captions, station_agreement,
                              task_similarity, _parse_time, _norm)


def vid(name, rows):
    """rows = (t0, t1, station, verb, obj, lh_act, rh_act)"""
    segs = [CaptionSegment(segment_id=i, t0=a, t1=b, station=st, verb=v,
                           object_name=o, hand_left_action=lh,
                           hand_right_action=rh, confidence=0.9,
                           caption=f"The worker {v} the {o}.")
            for i, (a, b, st, v, o, lh, rh) in enumerate(rows)]
    return VideoCaptions(video_id=name, segments=segs)


# the same worker doing the same job at the same bench, two different days
DAY1 = vid("day1", [(0, 20, "packing bench", "folding", "plastic bag", "folding", "holding"),
                    (20, 45, "packing bench", "stacking", "plastic bag", "holding", "stacking"),
                    (45, 60, "packing bench", "folding", "plastic bag", "folding", "holding")])
DAY2 = vid("day2", [(0, 25, "packing bench", "stacking", "plastic bag", "holding", "stacking"),
                    (25, 50, "packing bench", "folding", "plastic bag", "folding", "holding")])
# same bench, different job -> SAME_PLACE_NEW_TASK
SWEEP = vid("sweep", [(0, 30, "packing bench", "sweeping", "floor", "holding", "sweeping"),
                      (30, 60, "packing bench", "sweeping", "broom", "holding", "sweeping")])
# different place, unrelated work
WELD = vid("weld", [(0, 30, "welding bay", "welding", "steel frame", "holding", "welding"),
                    (30, 55, "welding bay", "grinding", "steel frame", "holding", "grinding")])


def test_parse_time_forms():
    assert _parse_time("00:21.00") == pytest.approx(21.0)
    assert _parse_time("1:02:03") == pytest.approx(3723.0)
    assert _parse_time("12.5") == pytest.approx(12.5)
    assert _parse_time("") is None
    assert _parse_time(None) is None


def test_null_tokens_are_not_treated_as_content():
    """'none' is the VLM saying nothing is in that hand, not an action called none."""
    assert _norm("none") == ""
    assert _norm("N/A") == ""
    assert _norm("  Pushing.  ") == "pushing"
    s = CaptionSegment(0, 0, 3, verb="walking", hand_left_action="",
                       hand_left_object="none", hand_right_object="none")
    assert s.actions == ("walking",)
    assert s.objects == ()


def test_same_job_repeated_scores_highest():
    """The headline case: same worker, same job, same bench, different day."""
    idf = fit_idf([DAY1, DAY2, SWEEP, WELD])
    same = task_similarity(DAY1, DAY2, idf)["task_raw"]
    new_task = task_similarity(DAY1, SWEEP, idf)["task_raw"]
    unrelated = task_similarity(DAY1, WELD, idf)["task_raw"]
    assert same > new_task > unrelated, (same, new_task, unrelated)
    assert same > 0.8, f"same job should be clearly similar, got {same}"
    assert unrelated < 0.2, f"unrelated work should be clearly dissimilar, got {unrelated}"


def test_task_axis_is_not_fooled_by_the_shared_station():
    """Same bench, different job must read as DIFFERENT on the task axis.

    This is the case a single fused score destroys, and the reason the axes are
    reported separately.
    """
    idf = fit_idf([DAY1, DAY2, SWEEP, WELD])
    t = task_similarity(DAY1, SWEEP, idf)
    st = station_agreement(DAY1, SWEEP)
    assert st["same_dominant_station"] == 1.0          # same place
    assert t["task_raw"] < 0.35                        # different work


def test_idf_demotes_ubiquitous_tokens():
    """'holding' is in every video, so IDF must weight it far below a rare verb.

    Without this, a shared generic action drives the score and the ordering of
    same-place-different-job vs unrelated-work inverted (0.292 vs 0.315).
    """
    idf = fit_idf([DAY1, DAY2, SWEEP, WELD])
    assert idf["holding"] < idf["welding"], idf
    assert idf["holding"] < idf["folding"], idf


def test_idf_fixes_the_ordering_inversion():
    idf = fit_idf([DAY1, DAY2, SWEEP, WELD])
    without = task_similarity(DAY1, SWEEP)["task_raw"] - task_similarity(DAY1, WELD)["task_raw"]
    with_idf = task_similarity(DAY1, SWEEP, idf)["task_raw"] - task_similarity(DAY1, WELD, idf)["task_raw"]
    # same-place-different-job vs unrelated: both are "different task", so the
    # gap should be small; what must NOT happen is unrelated scoring higher by
    # a wide margin because of a shared generic verb.
    assert with_idf > without, (without, with_idf)
