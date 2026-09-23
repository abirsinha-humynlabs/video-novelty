"""Census of hand-detection outputs: which episodes are actually full-rate.

RESULT, 2026-09-22, all 775 episodes probed (not sampled):

    prefix                    step  episodes   eff fps
    model_output                 9         2       3.1
    model_output                10       764       3.0
    model_output_fullrate        1         8      30.0

    USABLE (>= 15 effective fps): 8 episodes, and all 8 are in the 1000 h list

So task2 has ~0.3 hours of usable input against a 100 hour smoke target. The
gap is the blocker; `scripts/pick_variety.py` produces the request list.

WHY report.json AND NOT THE NPZ: `step` and `fps` live inside the NPZ, but
those are 20 KB - 17 MB across ~775 episodes. The per-episode
`*_hand21_report.json` is ~16 KB and carries both (`run.step`,
`input.probe.fps`), so the whole corpus surveys for ~12 MB instead of several
GB.

WHY NOT FILE SIZE: it misleads. The files in `model_output/` grew from 20 KB to
1 MB over August, which reads as a frame-rate increase and is not -- they are
longer episodes with more detected hands, at the same `step: 10`. Sampling 24
of them by size would have given the same wrong answer as sampling none.

A handful of episodes report odd source rates (29.97, 28.39, 26.68 fps) and two
use `step: 9`; both still land at ~3 effective fps, so the rule is to divide and
threshold, never to match on `step == 10`.

Usage:
    python scripts/census_hand_tracking.py
    CENSUS_PREFIXES=model_output_30fps python scripts/census_hand_tracking.py
"""
import concurrent.futures as cf, json, os, subprocess, sys
T = os.path.dirname(os.path.abspath(__file__))
P = "s3://prod-egc-stereo-v2-data/work_items/hand_detection"
PROF = ["--profile", "prod"]
#: All hand-detection output prefixes. `model_output_30fps` was added
#: 2026-09-22 and is still filling, so a census of it is a snapshot.
PREFIXES = tuple(os.environ.get("CENSUS_PREFIXES",
    "model_output,model_output_fullrate,model_output_30fps").split(","))

def episodes(prefix):
    out = subprocess.run(["aws", "s3", "ls", f"{P}/{prefix}/"] + PROF,
                         capture_output=True, text=True, timeout=300).stdout
    return [l.split()[-1].rstrip("/") for l in out.splitlines()
            if l.strip().endswith("/") and not l.strip().endswith("..")]

def probe(item):
    prefix, ep = item
    dst = f"{T}/rep/{prefix}__{ep}.json"
    if not os.path.exists(dst):
        r = subprocess.run(["aws", "s3", "cp",
                            f"{P}/{prefix}/{ep}/{ep}_hand21_report.json", dst,
                            "--only-show-errors"] + PROF, capture_output=True)
        if r.returncode != 0:
            return {"prefix": prefix, "ep": ep, "err": r.stderr.decode()[:60]}
    try:
        d = json.load(open(dst))
    except Exception as e:                                          # noqa: BLE001
        return {"prefix": prefix, "ep": ep, "err": str(e)[:60]}
    run = d.get("run") or {}
    probe_ = ((d.get("input") or {}).get("probe") or {})
    step = run.get("step")
    fps = probe_.get("fps")
    return {"prefix": prefix, "ep": ep, "step": step, "fps": fps,
            "eff_fps": (fps / step) if (fps and step) else None,
            "dur_s": probe_.get("duration") or probe_.get("duration_s"),
            "n_frames": (d.get("outputs") or {}).get("n_frames")}

os.makedirs(f"{T}/rep", exist_ok=True)
work = []
for pref in PREFIXES:
    eps = episodes(pref)
    print(f"{pref}: {len(eps)} episodes", flush=True)
    work += [(pref, e) for e in eps]

rows, done = [], 0
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    for r in ex.map(probe, work):
        rows.append(r); done += 1
        if done % 100 == 0: print(f"  {done}/{len(work)}", flush=True)
json.dump(rows, open(f"{T}/census.json", "w"))

import collections
print(f"\nprobed {len(rows)}, errors {sum(1 for r in rows if r.get('err'))}")
c = collections.Counter((r.get("prefix"), r.get("step"), r.get("fps")) for r in rows if not r.get("err"))
print(f"\n{'prefix':<22}{'step':>6}{'fps':>7}{'eff fps':>9}   episodes")
for (pref, step, fps), n in sorted(c.items(), key=lambda kv: (-kv[1])):
    eff = f"{fps/step:.1f}" if (fps and step) else "?"
    print(f"{str(pref):<22}{str(step):>6}{str(fps):>7}{eff:>9}   {n}")
usable = [r for r in rows if not r.get("err") and r.get("eff_fps") and r["eff_fps"] >= 15]
print(f"\nUSABLE (>=15 effective fps): {len(usable)} episodes")
