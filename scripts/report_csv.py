"""Write per-session CSV results into output/, mirroring the source tree.

    output/normalized/bitrobot/<...>/<session>/000/chunks.csv
    output/normalized/bitrobot/<...>/<session>/000/pairs.csv
    output/summary.csv

`chunks.csv` is one row per chunk: what it is, what it most resembles, and
whether coverage selection wanted it. `pairs.csv` is the evidence behind that --
every comparison the chunk took part in. Both carry calibrated percentiles, not
raw cosines, because a raw cosine is not comparable across corpora.

Row semantics, because the two counts differ and the difference matters:

* Comparisons are computed once per **unordered** pair -- C(n,2), symmetric.
* `pairs.csv` emits both **directions** of each pair, so you can select every
  row touching one chunk with a single filter. With 36 chunks that is 18x35 =
  630 rows per session file, 1260 overall, for 630 distinct comparisons. Do not
  read a row count as a pair count, and de-duplicate before computing any
  statistic over it.

Chunk names are session-qualified (`PIP-246/chunk01_020-050.mp4`). Basenames
repeat across sessions, so a bare basename identifies two different clips.

Usage:  python scripts/report_csv.py --index .novelty-2sess --out output
        python scripts/report_csv.py --index .novelty-2sess \
            --out s3://stage-humyn-egocentric-stereo-data/labelling_results/novelty_result
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

#: The ONLY S3 prefixes this project may write to. Everything else in
#: stage-humyn-egocentric-stereo-data belongs to other pipelines, so an
#: accidental sync into the wrong prefix is not a recoverable mistake. The
#: check is here rather than in a runbook because a runbook cannot stop a
#: mistyped --out.
_STAGE = "s3://stage-humyn-egocentric-stereo-data/labelling_results"
#: v2 = the cycle-aware (hand-track driven) run. v1 was strict 30 s chunking and
#: is kept so its results stay comparable rather than overwritten.
S3_RESULT_PREFIX = f"{_STAGE}/novelty_result_v2"
S3_DATA_PREFIX = f"{_STAGE}/novelty_data_v2"
S3_WRITE_ALLOWLIST = (
    S3_RESULT_PREFIX, S3_DATA_PREFIX,
    f"{_STAGE}/novelty_result", f"{_STAGE}/novelty_data",   # v1, still writable
)


def check_s3_destination(uri: str) -> None:
    clean = uri.rstrip("/")
    if not any(clean == p or clean.startswith(p + "/") for p in S3_WRITE_ALLOWLIST):
        raise SystemExit(
            f"refusing to write to {uri!r}.\n"
            "This project may only write under:\n  "
            + "\n  ".join(S3_WRITE_ALLOWLIST)
        )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty.metrics.fuse import compare as compare_sigs, raw_scores  # noqa: E402
from novelty.select import facility_location_greedy  # noqa: E402
from novelty.store.numpy_store import Index  # noqa: E402


def session_dir(path: str) -> str:
    """.../data/<mirrored key>/chunks/chunk01.mp4 -> <mirrored key>

    Anchored on the staging root (`data/`) rather than on a bucket prefix name.
    Sources arrive under different top-level prefixes -- `normalized/` for the
    Pipe_Factory pulls, `chunking/` for Automobile_Manufacturing -- so matching
    a literal prefix silently collapses a whole session to its last path
    component and buries unrelated recordings in one output directory.
    """
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(d) == "chunks":
        d = os.path.dirname(d)
    parts = d.replace(os.sep, "/").split("/")
    for anchor in ("data", "normalized", "chunking"):
        if anchor in parts:
            i = parts.index(anchor)
            return "/".join(parts[i + 1:] if anchor == "data" else parts[i:])
    return parts[-1]


def chunk_span(path: str):
    m = re.search(r"_(\d+)-(\d+)\.mp4$", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def session_tag(path: str) -> str:
    """Short unique label, e.g. 'PIP-246'.

    Every session repeats the same chunk basenames, so a bare basename in a CSV
    is ambiguous exactly the way it was ambiguous in label resolution. Qualify
    it.
    """
    # project codes look like PIP-246, RCB-13 -- not just the PIP- ones
    m = re.search(r"/([A-Z]{2,5}-\d+)/", os.path.abspath(path).replace(os.sep, "/"))
    return m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(path)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=".novelty-2sess")
    ap.add_argument("--out", default="output", help="local directory (kept)")
    ap.add_argument("--s3", default=S3_RESULT_PREFIX,
                    help="also mirror to this S3 prefix; '' to skip")
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()

    idx = Index(args.index)
    sigs = idx.signatures()
    null = idx.null()
    cfg = idx.config()
    if null is None:
        print("no null model -- run `novelty calibrate` first", file=sys.stderr)
        return 2

    n = len(sigs)
    sess = [session_dir(s.path) for s in sigs]
    span = [chunk_span(s.path) for s in sigs]
    tag = [session_tag(s.path) for s in sigs]
    # session-qualified, because chunk basenames repeat across sessions
    name = [f"{t}/{os.path.basename(s.path)}" for t, s in zip(tag, sigs)]

    # pairwise verdicts
    env = np.zeros((n, n), np.float32)
    task = np.zeros((n, n), np.float32)
    label = [[""] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        v = compare_sigs(sigs[i], sigs[j], null=null,
                         decision=cfg.decision, hashing=cfg.hashing)
        env[i, j] = env[j, i] = v.env_score
        task[i, j] = task[j, i] = v.task_score
        label[i][j] = label[j][i] = v.label

    # coverage selection over the fused percentile matrix
    S = 0.5 * (env + task) / 100.0
    np.fill_diagonal(S, 1.0)
    res = facility_location_greedy(S, args.budget)
    rank = {k: r for r, k in enumerate(res.order, 1)}
    gain = {k: g for k, g in zip(res.order, res.gains)}

    # CSVs are small (~220 KB per corpus) and are the thing anyone actually
    # reads, so unlike the video they stay on local disk AND go to S3.
    if args.out.startswith("s3://"):
        raise SystemExit("--out is a local directory; use --s3 for the S3 mirror")
    out_dir = args.out
    if args.s3:
        check_s3_destination(args.s3)
    os.makedirs(out_dir, exist_ok=True)
    rows_by_sess = collections.defaultdict(list)
    pairs_by_sess = collections.defaultdict(list)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        nn = max(others, key=lambda j: env[i, j])
        same = [j for j in others if sess[j] == sess[i]]
        cross = [j for j in others if sess[j] != sess[i]]
        rows_by_sess[sess[i]].append({
            "chunk": name[i],
            "t_start_s": span[i][0],
            "t_end_s": span[i][1],
            "duration_s": round(sigs[i].t1 - sigs[i].t0, 2),
            "detected_period_s": round(sigs[i].period_s, 2),
            "period_strength": round(sigs[i].period_strength, 4),
            "nearest_chunk": name[nn],
            "nearest_session": sess[nn],
            "nearest_env_pct": round(float(env[i, nn]), 2),
            "nearest_task_pct": round(float(task[i, nn]), 2),
            "nearest_verdict": label[i][nn],
            "max_env_pct_same_session": round(float(max(env[i, j] for j in same)), 2) if same else "",
            "max_env_pct_cross_session": round(float(max(env[i, j] for j in cross)), 2) if cross else "",
            "mean_env_pct_same_session": round(float(np.mean([env[i, j] for j in same])), 2) if same else "",
            "mean_env_pct_cross_session": round(float(np.mean([env[i, j] for j in cross])), 2) if cross else "",
            "selection_rank": rank.get(i, ""),
            "marginal_gain": round(float(gain.get(i, 0.0)), 5),
        })
        for j in others:
            pairs_by_sess[sess[i]].append({
                "chunk": name[i],
                "other_chunk": name[j],
                "other_session": sess[j],
                "same_session": sess[i] == sess[j],
                "time_gap_s": abs(span[i][0] - span[j][0]) if sess[i] == sess[j] else "",
                "env_pct": round(float(env[i, j]), 2),
                "task_pct": round(float(task[i, j]), 2),
                "verdict": label[i][j],
            })

    written = []
    for s, rows in rows_by_sess.items():
        d = os.path.join(out_dir, s)
        os.makedirs(d, exist_ok=True)
        rows.sort(key=lambda r: r["t_start_s"])
        for fname, data in (("chunks.csv", rows), ("pairs.csv", pairs_by_sess[s])):
            p = os.path.join(d, fname)
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            written.append((p, len(data)))

    # ---- video-level similarity -----------------------------------------
    # A 10-minute recording cannot be one vector: averaging its chunks deletes
    # exactly the brief exception that makes a recording worth keeping. So a
    # video is its SET of chunk signatures, and two videos are compared
    # set-to-set, the same operation env_chamfer already performs one level
    # down over frames.
    #
    # coverage(A|B) = fraction of A's chunks whose best match anywhere in B
    # clears the environment threshold. It is DIRECTIONAL on purpose: a short
    # recording can sit entirely inside a longer one while the longer one still
    # holds material the short one never saw. A single number cannot say that.
    thr = cfg.decision.env_percentile
    sess_names = sorted(rows_by_sess)
    vid = os.path.join(out_dir, "videos.csv")
    vrows = []
    for sa, sb in itertools.combinations(sess_names, 2):
        ia = [i for i in range(n) if sess[i] == sa]
        ib = [i for i in range(n) if sess[i] == sb]
        best_a = [max(env[i, j] for j in ib) for i in ia]   # each A chunk -> best B
        best_b = [max(env[j, i] for i in ia) for j in ib]
        cov_ab = sum(1 for v in best_a if v >= thr) / len(ia)
        cov_ba = sum(1 for v in best_b if v >= thr) / len(ib)
        cross = [env[i, j] for i in ia for j in ib]
        if cov_ab >= 0.8 and cov_ba >= 0.8:
            verdict = "DUPLICATE_COVERAGE"
        elif cov_ab >= 0.8:
            verdict = "A_CONTAINED_IN_B"
        elif cov_ba >= 0.8:
            verdict = "B_CONTAINED_IN_A"
        elif max(cov_ab, cov_ba) >= 0.3:
            verdict = "PARTIAL_OVERLAP"
        else:
            verdict = "DISTINCT"
        vrows.append({
            "video_a": sa, "video_b": sb,
            "chunks_a": len(ia), "chunks_b": len(ib),
            "coverage_a_in_b": round(cov_ab, 3),
            "coverage_b_in_a": round(cov_ba, 3),
            "mean_best_env_pct_a": round(float(np.mean(best_a)), 2),
            "mean_best_env_pct_b": round(float(np.mean(best_b)), 2),
            "mean_env_pct": round(float(np.mean(cross)), 2),
            "max_env_pct": round(float(np.max(cross)), 2),
            "env_threshold": thr,
            "verdict": verdict,
        })
    with open(vid, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vrows[0].keys()))
        w.writeheader()
        w.writerows(vrows)
    written.append((vid, len(vrows)))

    # corpus-level summary
    summ = os.path.join(out_dir, "summary.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "chunks", "mean_env_pct_within", "mean_env_pct_cross",
                    "mean_task_pct_within", "mean_task_pct_cross", "selected_in_top_half"])
        for s in sorted(rows_by_sess):
            ii = [i for i in range(n) if sess[i] == s]
            within = [env[i, j] for i, j in itertools.combinations(ii, 2)]
            crossi = [env[i, j] for i in ii for j in range(n) if sess[j] != s]
            twithin = [task[i, j] for i, j in itertools.combinations(ii, 2)]
            tcross = [task[i, j] for i in ii for j in range(n) if sess[j] != s]
            top = sum(1 for i in ii if rank.get(i, n) <= n // 2)
            w.writerow([s, len(ii), round(float(np.mean(within)), 2), round(float(np.mean(crossi)), 2),
                        round(float(np.mean(twithin)), 2), round(float(np.mean(tcross)), 2), top])
    written.append((summ, len(rows_by_sess)))

    for p, k in written:
        print(f"{p}  ({k} rows)")

    if args.s3:
        dest = args.s3.rstrip("/")
        # no --delete: this prefix is shared, and nothing here is authoritative
        # enough to justify removing an object somebody else put there.
        proc = subprocess.run(["aws", "s3", "sync", out_dir, dest, "--only-show-errors"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"aws s3 sync failed:\n{proc.stderr.strip()}")
        print(f"mirrored -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
