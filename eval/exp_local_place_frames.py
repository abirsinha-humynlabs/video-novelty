"""Frames for the place-recognition experiment, at a resolution that can
carry local detail.

The env axis currently pools each frame to ONE global DINOv2 vector. That is
the right shape for "is this the same kind of scene" and the wrong shape for
"is this the same physical place", where the evidence is a specific pillar or
bench recurring -- which global pooling averages away. Testing the local
alternatives (patch-token matching, SIFT+RANSAC) needs frames big enough for
local structure to survive, so 8 frames per clip at 700x394 rather than the
20 at 448x252 the env pipeline uses.
"""
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)
W, H, N = 700, 394, 8
PROF = ["--profile", "prod"]
man = json.load(open("/tmp/manifest_435.json"))

# ---------------------------------------------------------------- pair sets
pairs = []

# (1) the hard case: round-2 band pairs with a confident human label
r2 = {p["slot"]: p for p in json.load(open(
    "/tmp/claude-1000/-home-ec2-user-projects-abir-video-novelty/"
    "65787d15-db72-45ee-9584-48d002d3ef19/scratchpad/border/pairs_shuffled.json"))}
for slot, ans, score in json.load(open(
        "/tmp/claude-1000/-home-ec2-user-projects-abir-video-novelty/"
        "65787d15-db72-45ee-9584-48d002d3ef19/scratchpad/border/labels_r2.json")):
    k = int(slot[1:])
    if ans in ("same", "different"):
        pairs.append({"set": "band", "a": r2[k]["a"], "b": r2[k]["b"],
                      "label": ans, "env_raw": r2[k]["env_raw"]})

# (2) sanity anchors: a few round-1 pairs, where the global axis already works
r1lab = {int(r["pair_index"]): r["human_label"]
         for r in csv.DictReader(open("/tmp/human_labels.csv"))}
MAP = {"yes": "same", "no": "different", "maybe": "unsure"}
r1 = [r for r in csv.DictReader(open("/tmp/eyeball/labels.csv"))]
for grp in ("yes", "no"):
    got = [r for r in r1 if r1lab.get(int(r["pair_index"])) == grp]
    idx = np.linspace(0, len(got) - 1, min(6, len(got))).round().astype(int)
    for i in sorted(set(idx)):
        r = got[i]
        pairs.append({"set": "easy", "a": r["video_a"], "b": r["video_b"],
                      "label": MAP[grp], "env_raw": float(r["env_raw"])})

# (3) a positive class the band cannot supply: only 3 of 48 band pairs are
#     "same", far too thin to measure a positive rate on. Same-session pairs
#     are two segments of one continuous recording, so same-place is close to
#     certain without anyone labelling them -- easier positives than the band
#     holds, but real ones, and spread across the score range so the set is
#     not only the obvious ones.
ss = [r for r in csv.DictReader(open("/home/ec2-user/projects_abir/video-novelty/env_pairs.csv"))
      if r["stratum"] == "same_session_distant"]
ss.sort(key=lambda r: float(r["env_raw"]))
idx = np.linspace(0, len(ss) - 1, 24).round().astype(int)
for i in sorted(set(idx)):
    r = ss[i]
    pairs.append({"set": "same_session", "a": r["video_a"], "b": r["video_b"],
                  "label": "same", "env_raw": float(r["env_raw"])})

pairs = [p for p in pairs if p["a"] in man and p["b"] in man]
vids = sorted({v for p in pairs for v in (p["a"], p["b"])})
json.dump(pairs, open(os.path.join(OUT, "pairs.json"), "w"), indent=1)
import collections
print(f"{len(pairs)} pairs, {len(vids)} distinct clips", flush=True)
for s, c in sorted(collections.Counter((p["set"], p["label"]) for p in pairs).items()):
    print(f"   {s[0]:<14} {s[1]:<10} {c}", flush=True)


def grab(uuid):
    dst = os.path.join(FRAMES, uuid + ".npy")
    if os.path.exists(dst):
        return uuid, "cached"
    url = subprocess.run(["aws", "s3", "presign", man[uuid], "--expires-in", "43200"] + PROF,
                         capture_output=True, text=True).stdout.strip()
    try:
        d = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", url], capture_output=True, text=True, timeout=180).stdout.strip())
    except Exception:
        d = 200.0
    out = np.zeros((N, H, W, 3), np.uint8)
    n = W * H * 3
    for j, f in enumerate(np.linspace(0.10, 0.90, N)):
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(d * f, 0.5):.1f}", "-i", url,
             "-frames:v", "1",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=300)
        if len(p.stdout) >= n:
            out[j] = np.frombuffer(p.stdout[:n], np.uint8).reshape(H, W, 3)
    np.save(dst, out)
    return uuid, "ok"


done = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for uuid, st in ex.map(grab, vids):
        done += 1
        if done % 10 == 0 or done == len(vids):
            print(f"  {done}/{len(vids)} clips", flush=True)
print("frames ->", FRAMES)
