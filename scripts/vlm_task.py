"""VLM task descriptions: fetch, join to the 435, and score against the axis in use.

The VLM judgment arrived on 2026-09-22 at
`s3://prod-egc-stereo-v2-data/vlm_judgment/task_description/` -- 15,828
episodes, three files each (`task_description.json`, `verdict.json`,
`status.json`). It was expected to replace the task axis built from the QA
export's `task_description (our)` column. **It does not, and this script is
kept to show why rather than as a pipeline stage.**

WHAT IT CONTAINS, per episode: a `video_level` block (`l1_environment`,
`l2_venue`, `l3_scene`, `task_name`, `task_description`, `skill_groups`,
`difficulty`, confidences) and a `windows` list -- typically 3 windows of
240 s, each with its own environment labels and task description. Richer than
the QA column in every respect.

THE JOIN is by path, not id: no `episode_uuid` appears in the JSON, and
`source_raw_key` is null. The key under `task_description/` mirrors the QA
export's `chunk_s3_path` with the bucket, the `chunking/` or `normalized/`
prefix and the trailing `seg_NNN/` component removed. That matches **433 of
433** of our episodes.

MEASURED, all on the same 979 canonical-task positives, and all at once
because the two columns trade against each other:

    variant                          AUC(canon)   within-session corr
    QA description (in use)              0.9768              +0.243
    VLM task_name only                   0.9404              +0.206
    VLM task_name + skill_groups         0.9535              +0.217
    VLM full description                 0.9813              +0.439

The second column is the one that decides it. It is the correlation between
the task score and the ENVIRONMENT score measured within a single session,
where the environment is fixed by construction -- so any correlation left is
the task axis re-describing the room. V-JEPA 2 was rejected for scoring +0.55
on exactly this test.

The VLM full description buys **+0.0045 AUC for +0.20 of confound**. That is
the wrong trade for a quadrant whose entire value is that the two axes are
independent: SAME_PLACE_NEW_TASK is precisely the case a place-entangled task
score gets wrong. Dropping place-bound words does not fix it (tested by
ranking stems on their entropy across `l2_venue` and dropping the most
place-bound: +0.4393 -> +0.4323, AUC unchanged). The confound is diffuse --
a longer, more specific description of a clip simply correlates more with how
that clip looks.

THE VLM ENVIRONMENT LABELS do not help either. Against the 138 human
same-place judgements:

    env_raw (pixels)            AUC 0.9061
    VLM l1_environment          AUC 0.5501
    VLM l2_venue                AUC 0.5560
    VLM l3_scene                AUC 0.5108
    any-window l3 overlap       AUC 0.6172

32 distinct venue labels over 433 episodes, and pairs are already blocked by
site -- so within one site nearly every pair shares a venue label. The label
cannot separate two rooms of one factory, which is the only question being
asked. Blending it in makes the axis worse (0.9061 -> 0.8350).

WHERE IT IS GENUINELY VALUABLE, and why this is still good news:

1. **Coverage.** 15,828 episodes against the QA export's 433. The QA column
   exists only for REJECTED episodes, so it cannot score anything outside the
   766 -- the VLM can. Running task1 over the wider corpus depends on this
   file and nothing else.
2. **Use `task_name` there, not the full description** -- AUC 0.9404 at the
   BEST independence of any variant measured (+0.206, better even than the QA
   column). Accuracy costs 0.036 AUC; the quadrant keeps its meaning.
3. **It retires a circularity worry.** `canon_task x site_h` might have been
   derived from the same QA text it was validating, which would have flattered
   the 0.9768. The VLM is an independent source and scores 0.9813 against the
   same label, so the label stands on its own.
4. **Windowed descriptions** (3 per episode, timestamped) are the input
   `scripts/match_videos.py` was written for and has never had -- set-to-set
   task comparison instead of one profile per video.

So: the axis in use does not change, and both `vlm_task_sim` and
`vlm_task_name_sim` are written to the pair CSVs as cross-check columns.

Usage:
    python scripts/vlm_task.py fetch --out <dir>          # download the 433
    python scripts/vlm_task.py score <dir> env_pairs.csv  # add the columns
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROD = "prod-egc-stereo-v2-data"
PREFIX = "vlm_judgment/task_description"
QA_CSV = os.path.join(ROOT, "rejected_repetitive_shorter_segment.csv")

STOP = set("""a an the is are was were be been being this that these those there of in on at to
from with without into onto for by and or but then than as it its his her their them they he she
person worker man woman using use used uses while during after before over under across around
near next other another some each both all more most very much many few several one two three
hand hands left right side front back top bottom part parts thing things item items area place
appears seems looks like seen visible camera view wearing""".split())


def stems(text: str):
    out = []
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if len(w) < 3 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def episode_path(chunk_s3_path: str) -> str:
    """QA `chunk_s3_path` -> the VLM key's episode component.

    Both `chunking/` and `normalized/` appear as the prefix; 18 of the 433 use
    the latter, and missing that costs exactly those 18 rows.
    """
    p = chunk_s3_path.strip().rstrip("/")
    p = re.sub(r"^s3://[^/]+/(chunking|normalized)/", "", p)
    return re.sub(r"/seg_\d+$", "", p)


def cmd_fetch(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    keys = subprocess.run(
        ["aws", "s3api", "list-objects-v2", "--bucket", PROD, "--prefix", PREFIX + "/",
         "--query", "Contents[].Key", "--output", "text", "--profile", args.profile],
        capture_output=True, text=True, check=True).stdout.split()
    index = {k[len(PREFIX) + 1: -len("/task_description.json")]: k
             for k in keys if k.endswith("task_description.json")}
    print(f"{len(index)} episodes in the VLM prefix")

    want = {}
    for r in csv.DictReader(open(QA_CSV)):
        key = index.get(episode_path(r["chunk_s3_path"]))
        if key:
            want[r["episode_uuid"].strip()] = key
    print(f"joined to our episodes: {len(want)}")

    def get(item):
        uuid, key = item
        dst = os.path.join(args.out, uuid + ".json")
        if os.path.exists(dst) and os.path.getsize(dst) > 100:
            return True
        p = subprocess.run(["aws", "s3", "cp", f"s3://{PROD}/{key}", dst,
                            "--only-show-errors", "--profile", args.profile],
                           capture_output=True)
        return p.returncode == 0

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        got = sum(ex.map(get, want.items()))
    print(f"downloaded {got}/{len(want)} -> {args.out}")
    return 0 if got == len(want) else 1


def load(dirname: str):
    """uuid -> {full description text, task_name text, env labels, windows}."""
    out = {}
    for f in glob.glob(os.path.join(dirname, "*.json")):
        uuid = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))
        except Exception:                                          # noqa: BLE001
            continue
        vl = d.get("video_level") or {}
        tk = vl.get("task") or {}
        wins = d.get("windows") or []
        wtext = [(w.get("task") or {}).get("task_description")
                 or (w.get("task") or {}).get("description") or "" for w in wins]
        out[uuid] = {
            "full": " ".join([tk.get("task_description") or ""] + wtext),
            "name": (tk.get("task_name") or "").strip(),
            "l1": vl.get("l1_environment") or "",
            "l2": vl.get("l2_venue") or "",
            "l3": vl.get("l3_scene") or "",
            "n_windows": len(wins),
        }
    return out


class Profiles:
    """IDF-weighted stem profiles over one text field."""

    def __init__(self, docs, field: str):
        tok = {u: collections.Counter(stems(v[field])) for u, v in docs.items()}
        n = max(len(tok), 1)
        df: collections.Counter = collections.Counter()
        for c in tok.values():
            df.update(set(c))
        idf = {t: math.log(n / d) for t, d in df.items()}
        self.p = {}
        for u, c in tok.items():
            w = {t: k * max(idf.get(t, math.log(n)), 0.05) for t, k in c.items()}
            norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
            self.p[u] = {t: x / norm for t, x in w.items()}

    def __contains__(self, u):
        return u in self.p

    def sim(self, a, b) -> float:
        pa, pb = self.p[a], self.p[b]
        if len(pa) > len(pb):
            pa, pb = pb, pa
        return float(sum(w * pb.get(t, 0.0) for t, w in pa.items()))


def auc(pos, neg, cap: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    p = pos if len(pos) <= cap else rng.choice(pos, cap, replace=False)
    n = neg if len(neg) <= cap else rng.choice(neg, cap, replace=False)
    return float(((p[:, None] > n[None, :]).sum()
                  + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def cmd_score(args) -> int:
    docs = load(args.dir)
    print(f"{len(docs)} VLM descriptions, median windows "
          f"{int(np.median([d['n_windows'] for d in docs.values()]))}")
    full, name = Profiles(docs, "full"), Profiles(docs, "name")

    for path in args.csvs:
        rows = list(csv.DictReader(open(path)))
        have = [r for r in rows if r["video_a"] in docs and r["video_b"] in docs]
        vf = np.array([full.sim(r["video_a"], r["video_b"]) for r in have])
        vn = np.array([name.sim(r["video_a"], r["video_b"]) for r in have])
        canon = np.array([r["same_canon_task"] == "true" for r in have])
        env = np.array([float(r["env_raw"]) for r in have])
        qa = np.array([float(r["task_sim"]) for r in have if r["task_sim"]] or [0.0])
        sess = [i for i, r in enumerate(have) if r["stratum"].startswith("same_session")]

        print(f"\n{os.path.basename(path)}  covered {len(have)}/{len(rows)}")
        print(f"  {'variant':<30}{'AUC(canon)':>11}{'within-session corr':>21}")
        series = [("QA description (in use)", qa)] if len(qa) == len(have) else []
        series += [("VLM task_name only", vn), ("VLM full description", vf)]
        for nm, v in series:
            c = np.corrcoef(env[sess], v[sess])[0, 1] if len(sess) > 8 else float("nan")
            print(f"  {nm:<30}{auc(v[canon], v[~canon]):>11.4f}{c:>+21.4f}")

        by = {id(r): (f, n) for r, f, n in zip(have, vf, vn)}
        for r in rows:
            f, n = by.get(id(r), (None, None))
            r["vlm_task_sim"] = f"{f:.4f}" if f is not None else ""
            r["vlm_task_name_sim"] = f"{n:.4f}" if n is not None else ""
            r["vlm_l2_venue_a"] = docs.get(r["video_a"], {}).get("l2", "")
            r["vlm_l2_venue_b"] = docs.get(r["video_b"], {}).get("l2", "")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  + vlm_task_sim, vlm_task_name_sim, vlm_l2_venue_a/b "
              f"(cross-check only -- no verdict changes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(required=True)
    f = sub.add_parser("fetch", help="download the VLM descriptions for our episodes")
    f.add_argument("--out", default="vlm_td")
    f.add_argument("--profile", default="prod")
    f.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("score", help="add cross-check columns and print the comparison")
    s.add_argument("dir")
    s.add_argument("csvs", nargs="+")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
