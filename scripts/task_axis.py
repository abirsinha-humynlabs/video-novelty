"""Task1's task axis, from the episode task descriptions already in hand.

Task1 is video-to-video similarity; task2 (scripts/run_v2.py) is repetition
WITHIN one video. This script is task1 only, and it supplies the axis that
was blocked on VLM captioning for the 766 -- which has not run, and does not
need to for this.

`rejected_repetitive_shorter_segment.csv` carries, per episode, a
`task_description (our)` sentence of exactly the shape the VLM would have
produced ("The person is selecting plant cuttings from a pile and carefully
inserting them into the soil plugs of a seedling tray"), plus a 12-way
`job_family` and a `canon_task x site_h`. Coverage on the pair set is
13893/13964 -- 99.5%.

WHY TEXT AND NOT PIXELS. Every pixel-based task attempt in this project
failed, and failed the same way: flow rhythm at 0.11 sigma, and V-JEPA 2
correlating +0.55 with the appearance score *within a single session*, where
the environment is fixed and only the job varies -- so it was re-encoding the
scene, not the activity. DINOv2 is worse by construction: it is an image
encoder whose global token IS scene appearance, which is precisely why it
works as the environment axis. The same-session confound for this text axis
is +0.243, less than half V-JEPA 2's.

MEASURED, against `canon_task x site_h` as the label (979 positives -- a far
better-grounded fit than the env threshold's 17):

    same canonical task       n=  979   task_sim mean 0.5462
    different canonical task  n=12914   task_sim mean 0.0952
    AUC                                 0.9768
    best-F1 cut               0.3798    F1 0.748, precision 0.676, recall 0.838

Precision 0.676 against `canon_task x site_h` is a floor, not the error rate:
that label is canonical task AND site, so a pair doing the same job at two
sites counts as a miss here while being a true SAME_TASK_NEW_PLACE. The AUC is
the honest summary of the ordering.

The two axes are never fused. Collapsing them deletes the two informative
cases -- same place / new job, and same job / new place -- and the quadrant
is the output:

    same env + same task   -> REDUNDANT              the worker repeated the job here
    same env + new task    -> SAME_PLACE_NEW_TASK    keep
    new env + same task    -> SAME_TASK_NEW_PLACE    keep
    new env + new task     -> DISTINCT               keep

A BORDERLINE env verdict stays BORDERLINE: task similarity cannot rescue it
(AUC 0.6566 inside the band against the env axis's 0.8182 on the same 36
human-labelled pairs), because same job is not the same room.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_CSV = os.path.join(ROOT, "rejected_repetitive_shorter_segment.csv")
CANON = "canon_task×site_h"

# Words that describe every egocentric factory clip and so separate nothing.
# IDF down-weights them anyway; dropping the worst outright keeps a profile
# small enough to read when debugging a surprising score.
STOP = set("""a an the is are was were be been being this that these those there
of in on at to from with without into onto for by and or but then than as it its
his her their them they he she person worker man woman using use used uses while
during after before over under across around near next other another some each
both all more most very much many few several one two three both hand hands left
right side front back top bottom part parts thing things item items area place
appears seems looks like seen visible camera view wearing""".split())


def stems(text: str) -> List[str]:
    """Tokenise to crude stems.

    The corpus is present-participle heavy ("inserting", "cutting", "stacking"),
    so an unstemmed profile splits one act across two tokens and the overlap
    score loses it.
    """
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) < 3 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class TaskProfiles:
    """IDF-weighted token profiles over the episode task descriptions.

    IDF is not optional here, for the reason recorded in
    novelty.captions.fit_idf: without it the ubiquitous tokens dominate and a
    same-job pair can score BELOW an unrelated one. The most generic stems in
    this corpus are 'plastic', 'metal', 'floor', 'pick' -- present nearly
    everywhere and worth nothing for telling jobs apart.
    """

    def __init__(self, qa_csv: str = QA_CSV, floor: float = 0.05):
        rows = list(csv.DictReader(open(qa_csv)))
        self.meta: Dict[str, Dict[str, str]] = {}
        toks: Dict[str, Counter] = {}
        for r in rows:
            u = r["episode_uuid"].strip()
            if not u:
                continue
            toks[u] = Counter(stems(r.get("task_description (our)", "")))
            self.meta[u] = {
                "job": r.get("job_family", "").strip(),
                "canon": r.get(CANON, "").strip(),
                "site": r.get("site", "").strip(),
                "env1": r.get("environment_l1", "").strip(),
            }
        n = max(len(toks), 1)
        df: Counter = Counter()
        for c in toks.values():
            df.update(set(c))
        self.idf = {t: math.log(n / d) for t, d in df.items()}
        self.default_idf = math.log(n)
        self.prof: Dict[str, Dict[str, float]] = {}
        for u, c in toks.items():
            p = {t: k * max(self.idf.get(t, self.default_idf), floor)
                 for t, k in c.items()}
            norm = math.sqrt(sum(x * x for x in p.values())) or 1.0
            self.prof[u] = {t: x / norm for t, x in p.items()}
        self.n_docs = n

    def __contains__(self, u: str) -> bool:
        return u in self.prof

    def similarity(self, a: str, b: str) -> float:
        """Cosine over IDF-weighted stems; iterate the shorter profile."""
        pa, pb = self.prof[a], self.prof[b]
        if len(pa) > len(pb):
            pa, pb = pb, pa
        return float(sum(w * pb.get(t, 0.0) for t, w in pa.items()))

    def agree(self, a: str, b: str, field: str) -> bool:
        va, vb = self.meta[a][field], self.meta[b][field]
        return bool(va) and va == vb


def fit_threshold(sims: np.ndarray, same: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Best-F1 cut for 'same canonical task'.

    Fitted against a real label rather than guessed or matched to a base rate:
    `canon_task x site_h` gives 979 positives over the pair set, so unlike the
    env threshold (17 human positives) this cut rests on enough data that the
    operating point will not move much with more.
    """
    order = np.argsort(-sims)
    s, y = sims[order], same[order].astype(int)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(same.sum()), 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(s[i]), {"f1": float(f1[i]), "precision": float(prec[i]),
                         "recall": float(rec[i]), "n_pos": n_pos}


def auc(pos: np.ndarray, neg: np.ndarray, cap: int = 1500, seed: int = 0) -> float:
    """Mann-Whitney AUC, subsampled when the product would be huge."""
    rng = np.random.default_rng(seed)
    p = pos if len(pos) <= cap else rng.choice(pos, cap, replace=False)
    n = neg if len(neg) <= cap else rng.choice(neg, cap, replace=False)
    if not len(p) or not len(n):
        return float("nan")
    return float(((p[:, None] > n[None, :]).sum()
                  + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


QUADRANT = {
    (True, True): "REDUNDANT",
    (True, False): "SAME_PLACE_NEW_TASK",
    (False, True): "SAME_TASK_NEW_PLACE",
    (False, False): "DISTINCT",
}


def annotate(path: str, tp: TaskProfiles, threshold: float | None) -> Tuple[float, Dict]:
    rows = list(csv.DictReader(open(path)))
    have = [r for r in rows if r["video_a"] in tp and r["video_b"] in tp]
    sims = np.array([tp.similarity(r["video_a"], r["video_b"]) for r in have])
    canon = np.array([tp.agree(r["video_a"], r["video_b"], "canon") for r in have])

    if threshold is None:
        threshold, fit = fit_threshold(sims, canon)
    else:
        fit = {}
    a = auc(sims[canon], sims[~canon])

    stats = Counter()
    sim_of = {id(r): s for r, s in zip(have, sims)}
    for r in rows:
        ok = r["video_a"] in tp and r["video_b"] in tp
        s = sim_of.get(id(r)) if ok else None
        r["task_sim"] = f"{s:.4f}" if s is not None else ""
        r["task_threshold"] = f"{threshold:.4f}"
        r["same_job_family"] = str(tp.agree(r["video_a"], r["video_b"], "job")).lower() if ok else ""
        r["same_canon_task"] = str(tp.agree(r["video_a"], r["video_b"], "canon")).lower() if ok else ""
        r["job_a"] = tp.meta[r["video_a"]]["job"] if r["video_a"] in tp else ""
        r["job_b"] = tp.meta[r["video_b"]]["job"] if r["video_b"] in tp else ""
        env = r.get("verdict_calibrated", "")
        if not ok or not env:
            r["verdict_quadrant"] = "NO_TASK_DATA" if not ok else ""
        elif env == "BORDERLINE":
            # the env axis could not decide; the task axis cannot decide it
            # for it, so say so instead of inventing a quadrant
            r["verdict_quadrant"] = "BORDERLINE_ENV"
        else:
            r["verdict_quadrant"] = QUADRANT[(env == "SAME_ENV", s >= threshold)]
        stats[r["verdict_quadrant"]] += 1

    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return threshold, {"auc_canon": a, "fit": fit, "n": len(rows),
                       "n_covered": len(have), "quadrant": stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", help="pair CSVs to annotate in place")
    ap.add_argument("--threshold", type=float, default=None,
                    help="task_sim cut; default fits best-F1 on canon_task")
    args = ap.parse_args()

    tp = TaskProfiles()
    print(f"{tp.n_docs} task descriptions, {len(tp.idf)} distinct stems")
    print("  most generic (down-weighted): "
          + str([t for t, _ in sorted(tp.idf.items(), key=lambda kv: kv[1])[:8]]))

    thr = args.threshold
    for path in args.csvs:
        thr, st = annotate(path, tp, thr)
        print(f"\n{os.path.basename(path)}  n={st['n']}  "
              f"task data on both sides: {st['n_covered']}")
        print(f"  AUC of task_sim for same canonical task = {st['auc_canon']:.4f}")
        if st["fit"]:
            f = st["fit"]
            print(f"  threshold {thr:.4f} fitted best-F1 on {f['n_pos']} canonical-task "
                  f"positives: F1={f['f1']:.3f} precision={f['precision']:.3f} "
                  f"recall={f['recall']:.3f}")
        for k, n in sorted(st["quadrant"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<22} {n:>6}  ({n / st['n']:5.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
