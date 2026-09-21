"""Video-to-video similarity: two independent axes, reported separately.

**Separate from scripts/run_v2.py by design.** run_v2 measures variety WITHIN
one video by cutting it into chunks. This script compares WHOLE videos to each
other. Keeping them apart avoids the confusion of one script answering two
questions.

    environment  DINOv2 appearance over the VLM's own segments, compared
                 set-to-set (duration-weighted coverage + optimal transport),
                 cross-checked against the VLM `station` text
    task         verb / hand-action / object token profiles from the captions

**Never fused into one number.** Collapsing them deletes the two informative
cases: same place / new job, and same job / new place. The quadrant is the
output.

    same env + same task   -> REDUNDANT             (the worker repeated the job)
    same env + diff task   -> SAME_PLACE_NEW_TASK    keep
    diff env + same task   -> SAME_TASK_NEW_PLACE    keep
    diff env + diff task   -> DISTINCT               keep

Why the axes come from different channels: appearance embeddings are the one
validated component in this project (AUC 0.998, 4.40 sigma, 0.973 accuracy over
1378 pairs across three environments), while every pixel-based *task* attempt
failed -- flow rhythm at 0.11 sigma, and V-JEPA 2 correlating +0.55 with
appearance within a single session. So environment comes from pixels and task
comes from VLM text, each from the channel that works for it.

Segment boundaries come from the VLM, which already splits on activity change
(21 s, 23 s observed) rather than a fixed grid. No chunking of our own.

Usage (environment axis needs video; task axis does not):

    python scripts/match_videos.py --captions-dir <dir> --out pairs.csv
    python scripts/match_videos.py --captions-dir <dir> --with-appearance \
        --video-root s3://... --out pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty.captions import (VideoCaptions, caption_similarity, fit_idf,  # noqa: E402
                              load_captions, station_agreement, task_similarity)


# --------------------------------------------------------- set-to-set tools
def coverage(sim: np.ndarray, w_a: np.ndarray, threshold: float) -> float:
    """Duration-weighted fraction of A covered by its best match in B.

    Directional on purpose. A short recording can sit entirely inside a longer
    one while the longer one still holds material the short one never saw, and
    a single symmetric number cannot express that. Weighting by duration means
    "83% covered" is 83% of the FOOTAGE, not of the segment count -- which
    matters because the VLM's segments vary in length.
    """
    if sim.size == 0 or w_a.sum() <= 0:
        return 0.0
    best = sim.max(axis=1)
    return float((w_a * (best >= threshold)).sum() / w_a.sum())


def emd(sim: np.ndarray, w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Duration-weighted optimal transport similarity, threshold-free.

    Best-match/coverage lets ten segments of A all match the SAME segment of B,
    which overstates similarity when B simply does not contain enough of that
    content. Optimal transport cannot: mass must be conserved, so it charges
    for the shortfall. Mass is segment duration, which handles unequal segment
    counts and unequal segment lengths in one step.

    Returns a similarity in the same units as ``sim`` (higher = closer), i.e.
    the transport-weighted mean similarity under the optimal plan.
    """
    from scipy.optimize import linprog

    n, m = sim.shape
    if n == 0 or m == 0:
        return 0.0
    a = w_a / w_a.sum() if w_a.sum() > 0 else np.full(n, 1.0 / n)
    b = w_b / w_b.sum() if w_b.sum() > 0 else np.full(m, 1.0 / m)
    # maximise total similarity  == minimise negative similarity
    c = (-sim).ravel()
    A_eq, b_eq = [], []
    for i in range(n):
        row = np.zeros(n * m); row[i * m:(i + 1) * m] = 1.0
        A_eq.append(row); b_eq.append(a[i])
    for j in range(m):
        row = np.zeros(n * m); row[j::m] = 1.0
        A_eq.append(row); b_eq.append(b[j])
    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=(0, None), method="highs")
    if not res.success:
        # degenerate LP: fall back to the symmetric best-match mean
        return float(0.5 * (sim.max(1).mean() + sim.max(0).mean()))
    return float(-res.fun)


# ------------------------------------------------------------ task matrices
def segment_task_sim(a: VideoCaptions, b: VideoCaptions) -> np.ndarray:
    """(len(a), len(b)) task similarity between individual VLM segments.

    Token overlap on actions and objects -- deliberately not caption text, for
    the paraphrase reason in novelty.captions.
    """
    S = np.zeros((len(a.segments), len(b.segments)), np.float32)
    for i, sa in enumerate(a.segments):
        aa, oa = set(sa.actions), set(sa.objects)
        for j, sb in enumerate(b.segments):
            ab, ob = set(sb.actions), set(sb.objects)
            act = len(aa & ab) / len(aa | ab) if (aa or ab) else 0.0
            obj = len(oa & ob) / len(oa | ob) if (oa or ob) else 0.0
            S[i, j] = 0.65 * act + 0.35 * obj
    return S


def durations(v: VideoCaptions) -> np.ndarray:
    return np.array([s.seconds for s in v.segments], np.float64)


# ------------------------------------------------------------------- driver
def discover(captions_dir: str) -> Dict[str, VideoCaptions]:
    out: Dict[str, VideoCaptions] = {}
    pats = ("**/captions.parquet.zst", "**/captions.parquet")
    files = sorted({p for pat in pats
                    for p in glob.glob(os.path.join(captions_dir, pat), recursive=True)})
    for f in files:
        try:
            vc = load_captions(f)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  skip {f}: {str(exc)[:110]}", file=sys.stderr)
            continue
        if vc.segments:
            out[vc.video_id] = vc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", required=True,
                    help="local tree containing captions.parquet(.zst) files")
    ap.add_argument("--out", default="video_pairs.csv")
    ap.add_argument("--task-threshold", type=float, default=0.50,
                    help="segment-level task similarity counted as a match")
    ap.add_argument("--env-threshold", type=float, default=0.50,
                    help="appearance similarity counted as a match")
    ap.add_argument("--with-appearance", action="store_true",
                    help="also compute the environment axis (needs the videos)")
    args = ap.parse_args()

    vids = discover(args.captions_dir)
    print(f"{len(vids)} videos with captions")
    if len(vids) < 2:
        print("need at least 2 videos to compare")
        return 2
    for k, v in sorted(vids.items()):
        print(f"  {k[:64]:<66} {len(v.segments):>3} segs  {v.duration:6.1f}s  "
              f"station={v.dominant_station()[:22]!r}")

    # IDF over the whole corpus: without it, ubiquitous verbs like
    # 'holding' dominate every comparison (see captions.fit_idf).
    idf = fit_idf(list(vids.values()))
    generic = sorted(idf, key=idf.get)[:6]
    print(f"  most generic tokens (down-weighted): {generic}")

    names = sorted(vids)
    rows = []
    for na, nb in itertools.combinations(names, 2):
        A, B = vids[na], vids[nb]
        S = segment_task_sim(A, B)
        wa, wb = durations(A), durations(B)
        t = task_similarity(A, B, idf)
        st = station_agreement(A, B)
        row = {
            "video_a": na, "video_b": nb,
            "segments_a": len(A.segments), "segments_b": len(B.segments),
            "duration_a": round(A.duration, 1), "duration_b": round(B.duration, 1),
            # task axis, whole-video profiles
            **{k: round(v, 4) for k, v in t.items()},
            # task axis, set-to-set over segments
            "task_coverage_a_in_b": round(coverage(S, wa, args.task_threshold), 3),
            "task_coverage_b_in_a": round(coverage(S.T, wb, args.task_threshold), 3),
            "task_emd": round(emd(S, wa, wb), 4),
            # environment cross-check from text
            **{k: round(v, 4) for k, v in st.items()},
            "station_a": A.dominant_station(), "station_b": B.dominant_station(),
            "caption_tiebreak": round(caption_similarity(A, B), 4),
            "mean_conf_a": round(A.mean_confidence(), 3),
            "mean_conf_b": round(B.mean_confidence(), 3),
        }
        rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} video pairs -> {args.out}")
    print("\nNOTE: the environment axis (DINOv2 appearance) is not included in "
          "this run.\nTask similarity alone cannot distinguish 'same job "
          "elsewhere' from 'same job here'.\nRun with --with-appearance once "
          "the videos are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
