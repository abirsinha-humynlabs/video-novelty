"""Per-episode status for the task 2 run: what happened to every episode, and why.

The pipeline only writes a CSV for an episode that yields >= 2 cycle chunks. On
the 433 customer-marked-repetitive episodes that is 6 of them, so the output
prefix says nothing at all about the other 427 -- whether they lack keypoints,
arrived at the wrong frame rate, had untrackable hands, or were tracked fine and
simply had no detectable cadence. Those are four different problems with four
different owners, and silence conflates them.

This writes one row per episode with the outcome and the reason it stopped,
so the run is auditable without re-deriving anything:

    no_keypoints      upstream has not produced them yet
    coarse_rate       keypoints exist but at step > 1 (3 fps); unusable
    untrackable       hands located too rarely to judge cadence
    no_cadence        tracked well enough, but no periodic signal found
    too_few_chunks    cadence found, but < 2 chunks so nothing to compare
    ok                >= 2 chunks, a CSV exists

`untrackable` vs `no_cadence` is the distinction that matters most: the first
is a data-quality problem to raise upstream, the second is a statement about
the work itself (or about our detector). Reporting them as one number is what
made this look like a keypoint problem for two days when the measured split is
roughly half and half.

Usage:
    python scripts/task2_status.py --state <dir> --out status.csv [--s3 <prefix>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.report_csv import check_s3_destination  # noqa: E402

#: Below this share of the clip the hands are located too rarely for the
#: run-length rule to leave anything analysable. Measured, not chosen: every
#: episode that has ever produced a chunk sits above it.
TRACKABLE_FLOOR = 0.50


def episode_key(path_or_name: str) -> str:
    """The one key that identifies an episode across every source.

    NOT `segment_take`. That column is the recording take, and a take is cut
    into several chunks which each carry their own `episode_uuid` -- 25,762
    episodes share only 11,691 takes, so joining on it silently collapses
    ~8,029 groups onto one arbitrary member and under-reports coverage. The
    full path, including the trailing `seg_NNN`, is unique (25,762 of 25,762).

    Accepts either a master-list `left_stream_s3` URI or a hand-detection
    episode directory name, and normalises both to
    `<vendor>/<site>/<date>/.../<take>/seg_NNN`.
    """
    v = path_or_name.strip().rstrip("/")
    if v.startswith("s3://"):
        v = re.sub(r"^s3://[^/]+/(chunking|normalized)/", "", v)
        return re.sub(r"/left_rectified\.mp4$", "", v)
    return v.replace("__", "/")


def classify(rec: dict) -> str:
    if rec is None:
        return "no_keypoints"
    if (rec.get("step") or 99) > 1:
        return "coarse_rate"
    n = len(rec.get("segments") or [])
    if n >= 2:
        return "ok"
    if n == 1:
        return "too_few_chunks"
    if (rec.get("trackable") or 0.0) < TRACKABLE_FLOOR:
        return "untrackable"
    return "no_cadence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True, help="dir holding segments.json")
    ap.add_argument("--cohort", default=None,
                    help="JSON list of {episode_uuid} to report on; default is "
                         "every episode in segments.json")
    ap.add_argument("--master", default=os.path.join(ROOT, "delivered_1000h_master_list.csv"))
    ap.add_argument("--out", default="task2_status.csv")
    ap.add_argument("--s3", default=None, help="also upload the report here")
    ap.add_argument("--profile", default=os.environ.get("NOVELTY_STAGE_PROFILE", ""))
    args = ap.parse_args()

    seg = json.load(open(os.path.join(args.state, "segments.json")))
    by_key = {episode_key(name): rec for name, rec in seg.items()}

    master = {}
    if os.path.exists(args.master):
        for r in csv.DictReader(open(args.master)):
            master[episode_key(r.get("left_stream_s3") or "")] = r

    if args.cohort:
        want = {c["episode_uuid"] for c in json.load(open(args.cohort))}
        takes = [k for k, r in master.items() if r["episode_uuid"] in want]
    else:
        takes = sorted(by_key)

    rows = []
    for t in sorted(takes):
        rec = by_key.get(t)
        m = master.get(t, {})
        rows.append({
            "episode_key": t,
            "episode_uuid": m.get("episode_uuid", ""),
            "status": classify(rec),
            "step": (rec or {}).get("step", ""),
            "duration_s": round(float((rec or {}).get("duration_s") or 0), 1),
            "trackable": round(float((rec or {}).get("trackable") or 0), 3),
            "cadence": round(float((rec or {}).get("cadence") or 0), 4),
            "period_s": round(float((rec or {}).get("period_s") or 0), 3),
            "n_chunks": len((rec or {}).get("segments") or []),
            "job_family": m.get("job_family", ""),
            "business_name": m.get("business_name", ""),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tally: dict = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"{len(rows)} episodes -> {args.out}")
    for k in ("ok", "too_few_chunks", "no_cadence", "untrackable",
              "coarse_rate", "no_keypoints"):
        if k in tally:
            print(f"  {k:<16} {tally[k]:>5}  ({tally[k] / len(rows):5.1%})")

    if args.s3:
        dest = args.s3.rstrip("/") + "/" + os.path.basename(args.out)
        check_s3_destination(dest)
        cmd = ["aws", "s3", "cp", args.out, dest, "--only-show-errors"]
        if args.profile:
            cmd += ["--profile", args.profile]
        subprocess.run(cmd, check=True)
        print(f"  uploaded -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
