"""Pick the most task-varied N hours out of the delivered corpus.

Task 2 finds repetition WITHIN one video, and the thing that breaks a cycle
detector is not volume, it is *cycle length*: a pick-and-place cycle runs under
a second, a long assembly cycle runs tens of seconds, and code that quietly
assumes one of those fails on the other. So a smoke test wants many DIFFERENT
tasks in many different scenes, not many hours of the same work -- 100 hours of
one factory doing one job exercises a single cycle length and proves almost
nothing.

The delivered list already carries the LLM judgement metadata, so no S3 read is
needed to do this:

    25,762 chunks, 1,010.8 hours, mean chunk 141 s
    11,421 distinct task_id / task_description
     1,529 skill_group, 136 environment_l3, 14 job_family, 19 businesses

SELECTION

1. **Drop chunks too short to contain repetition.** Below ~90 s there is no
   room for several cycles plus the transitions between them, so a "no cadence
   found" result would say nothing about the code. p50 of the corpus is 94 s;
   the >=90 s pool still holds 828 hours over 9,153 distinct tasks, so this
   costs variety nothing.
2. **One chunk per task_id**, the longest available. Distinct tasks are the
   axis being maximised, and within a task the longest chunk shows the most
   cycles.
3. **Round-robin across scene, then business.** Taking tasks in corpus order
   would let the largest site dominate: 438 of 1,010 hours are one
   `environment_l1`. Round-robin over `environment_l3` (123 values in the pool)
   and then `business_name` spreads the pick without needing a hand-tuned quota.
4. Stop at the hour target.

The result is the maximum number of distinct tasks that fit in the budget,
spread across scenes by construction.

WHAT IT IS FOR: almost none of these episodes have full-rate hand tracking yet,
so this list is the *request* -- the set hand detection should re-run at
`step: 1`. `--require-npz` filters to what already exists, using the census
written by the report.json survey.

Usage:
    python scripts/pick_variety.py --hours 100 --out pick_100h.csv
    python scripts/pick_variety.py --hours 100 --require-npz census.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "delivered_1000h_master_list.csv")

#: Facets reported on, in the order a reader cares about them. The first is
#: what the selection maximises; the rest are what it must not collapse.
FACETS = ("task_id", "skill_group", "environment_l3", "environment_l2",
          "job_family", "business_name", "operator_id")


def load(path: str) -> List[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def hours(row: dict) -> float:
    try:
        return float(row.get("chunk_duration_hrs") or 0.0)
    except ValueError:
        return 0.0


def seconds(row: dict) -> float:
    try:
        return float(row.get("chunk_duration_s") or 0.0)
    except ValueError:
        return 0.0


def coverage(rows: List[dict]) -> Dict[str, int]:
    return {f: len({r.get(f, "") for r in rows if r.get(f)}) for f in FACETS}


def select(rows: List[dict], target_h: float, min_s: float,
           max_per_business_frac: float) -> List[dict]:
    pool = [r for r in rows if seconds(r) >= min_s]

    # one chunk per task: the longest, so the most cycles are visible
    best: Dict[str, dict] = {}
    for r in pool:
        t = r.get("task_id") or ""
        if not t:
            continue
        if t not in best or seconds(r) > seconds(best[t]):
            best[t] = r
    cands = list(best.values())

    # scene -> business -> queue of tasks, each queue longest-first so an
    # early stop still takes the most informative chunk of that bucket
    tree: Dict[str, Dict[str, List[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in cands:
        tree[r.get("environment_l3") or "?"][r.get("business_name") or "?"].append(r)
    for scene in tree.values():
        for q in scene.values():
            q.sort(key=seconds, reverse=True)

    cap = target_h * max_per_business_frac
    picked: List[dict] = []
    by_business: Dict[str, float] = collections.Counter()
    total = 0.0
    scenes = sorted(tree)
    # round-robin: one task per (scene, business) per lap, so breadth accrues
    # before depth and stopping early still leaves a spread set
    while total < target_h:
        took = False
        for scene in scenes:
            for biz in sorted(tree[scene]):
                q = tree[scene][biz]
                if not q:
                    continue
                if by_business[biz] >= cap:
                    continue
                r = q.pop(0)
                h = hours(r)
                if total + h > target_h * 1.02:      # do not overshoot badly
                    continue
                picked.append(r)
                by_business[biz] += h
                total += h
                took = True
                if total >= target_h:
                    break
            if total >= target_h:
                break
        if not took:
            break                                     # pool or caps exhausted
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--hours", type=float, default=100.0)
    ap.add_argument("--min-seconds", type=float, default=90.0,
                    help="below this a chunk cannot show several cycles")
    ap.add_argument("--max-per-business", type=float, default=0.20,
                    help="fraction of the budget any one site may take")
    ap.add_argument("--require-npz", default=None,
                    help="census.json from the report.json survey; keep only "
                         "episodes that already have full-rate hand tracking")
    ap.add_argument("--out", default="pick_100h.csv")
    args = ap.parse_args()

    rows = load(args.master)
    total_h = sum(hours(r) for r in rows)
    print(f"corpus: {len(rows)} chunks, {total_h:.1f} h")

    if args.require_npz:
        census = json.load(open(args.require_npz))
        ok = {c["ep"] for c in census
              if not c.get("err") and (c.get("eff_fps") or 0) >= 15}
        before = len(rows)
        rows = [r for r in rows if (r.get("segment_take") or "").replace("/", "__") in ok]
        print(f"  --require-npz: {before} -> {len(rows)} chunks with full-rate tracking")
        if not rows:
            print("  nothing has full-rate hand tracking yet; run without "
                  "--require-npz to produce the request list")
            return 1

    picked = select(rows, args.hours, args.min_seconds, args.max_per_business)
    got = sum(hours(r) for r in picked)

    cov_all, cov_pick = coverage(rows), coverage(picked)
    print(f"\npicked: {len(picked)} chunks, {got:.1f} h "
          f"({got / max(total_h, 1e-9):.1%} of the corpus)")
    print(f"\n{'facet':<18}{'in pick':>9}{'in corpus':>11}{'captured':>10}")
    for f in FACETS:
        a, b = cov_pick[f], cov_all[f]
        print(f"{f:<18}{a:>9}{b:>11}{a / max(b, 1):>9.1%}")

    print("\nhours by environment_l1:")
    h1: Dict[str, float] = collections.Counter()
    for r in picked:
        h1[r.get("environment_l1") or "?"] += hours(r)
    for k, v in h1.most_common():
        print(f"  {v:7.1f} h  {k}")

    print("\nlargest businesses in the pick:")
    hb: Dict[str, float] = collections.Counter()
    for r in picked:
        hb[r.get("business_name") or "?"] += hours(r)
    for k, v in hb.most_common(5):
        print(f"  {v:7.1f} h  ({v / max(got, 1e-9):4.1%})  {k}")

    d = sorted(seconds(r) for r in picked)
    if d:
        print(f"\nchunk seconds: min {d[0]:.0f}  p50 {d[len(d)//2]:.0f}  max {d[-1]:.0f}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(picked)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
