"""Whole-video ENVIRONMENT matching. No chunking, no hand tracks, no VLM.

The third and simplest of the three pipelines, kept separate on purpose:

    scripts/run_v2.py      variety WITHIN one video, cut into cycle chunks
    scripts/match_videos.py  video-to-video TASK similarity from VLM captions
    scripts/match_env.py     video-to-video ENVIRONMENT similarity  <- this one

Answers one question: *is this the same place?* So the worker doing the same job
at the same bench today and tomorrow should match, regardless of what their
hands were doing.

## Why no chunking

Environment is a property of the whole recording, not of any moment in it. There
is nothing to align, so cutting into chunks buys nothing and costs a lot. One
signature per video, built from frames spread across its full duration.

## Why keyframes

Measured on a 300 s 1080p file: sampling 64 frames with the normal uniform
sampler takes **28.3 s**, because the fps filter still pushes every frame
through the decoder. Decoding keyframes only (`-skip_frame nokey`) takes
**1.2 s** for 36 frames spanning the same 300 s -- a 24x saving, and for a
scene question the frames are equivalent. Over 766 videos that is the
difference between ~6 hours and ~15 minutes.

Never use keyframes for motion: consecutive ones are seconds apart.

## Why blocking barely matters (and what it IS for)

Comparing only same-domain pairs is the obvious optimisation, but it saves
almost nothing, because the cost is per-VIDEO embedding, not per-PAIR
comparison -- you embed all 766 either way. Measured: all 292,995 pairs cost
~6.5 min at 1.33 ms/pair; blocking by site cuts that to ~43,000 pairs, saving
about six minutes of a multi-hour job.

Blocking is still worth doing, for two better reasons:
  * the output CSV stops containing 250,000 trivially-different pairs nobody
    will read, and
  * the decision threshold then operates where it is hard (same site, different
    station) rather than being flattered by easy cross-industry pairs.

**But the calibration must NOT be blocked.** The whitener and the null need
cross-domain pairs to know what "different" looks like; fit them on a random
sample across the whole corpus, then apply the result within blocks. Blocking
the calibration too would leave no negatives at all.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty.io.decode import sample_keyframes                     # noqa: E402
from novelty.encoders.appearance import static_weight_map          # noqa: E402
from novelty.encoders.base import get_appearance_encoder           # noqa: E402
from novelty.metrics.distance import chamfer, cosine, gaussian_bhattacharyya  # noqa: E402

MAX_FRAMES = 64
#: Frames pulled per video when reading remotely. 20 spread across the span is
#: plenty to characterise a scene, and each costs one HTTP range request.
REMOTE_FRAMES = 20
WIDTH, HEIGHT = 448, 252
READ_PROFILE = os.environ.get("NOVELTY_PROD_PROFILE", "")


def sample_remote(uri: str, n_frames: int = REMOTE_FRAMES,
                  width: int = WIDTH, height: int = HEIGHT,
                  duration: Optional[float] = None) -> np.ndarray:
    """Frames from an S3 video WITHOUT downloading it.

    The episodes are 320-660 MB (median ~430 MB), so fetching all 435 would move
    ~187 GB and take ~3.7 h. ffmpeg can range-request over HTTP, so presigning
    the object and seeking to n timestamps transfers only the bytes around each
    frame: measured **5.7 s for 12 frames** of a 430 MB file, versus ~31 s just
    to download it.

    One ffmpeg call per frame is deliberate. A single call with a sparse `fps`
    or `select` filter still streams the whole file, which is the thing being
    avoided; and `-skip_frame nokey` over HTTP does too, because keyframes are
    spread through the container.
    """
    url = sh(["aws", "s3", "presign", uri, "--expires-in", "3600"],
             profile=READ_PROFILE or None).strip()
    if duration is None:
        try:
            out = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", url])
            duration = float(out.strip())
        except Exception:                                          # noqa: BLE001
            duration = 300.0
    # avoid the very start/end: intros and truncated tails are unrepresentative
    lo, hi = min(2.0, duration * 0.02), max(duration - 2.0, duration * 0.98)
    want = np.linspace(lo, hi, n_frames)
    fb = width * height * 3
    frames = []
    for t in want:
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", url,
             "-frames:v", "1", "-vf",
             f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=180)
        if len(p.stdout) >= fb:
            frames.append(np.frombuffer(p.stdout[:fb], np.uint8).reshape(height, width, 3))
    if not frames:
        raise RuntimeError("no frames read over HTTP")
    return np.ascontiguousarray(np.stack(frames))


def sh(cmd, profile=None):
    if profile:
        cmd = list(cmd) + ["--profile", profile]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}: {p.stderr.strip()[:300]}")
    return p.stdout


# ---------------------------------------------------------------- signatures
def embed_video(path: str, enc, static_mask: bool = True) -> Dict[str, np.ndarray]:
    """One environment signature for a whole video (local path or s3:// uri)."""
    if path.startswith("s3://"):
        frames = sample_remote(path)
        ts = np.linspace(0.0, 1.0, len(frames))
    else:
        frames, ts = sample_keyframes(path, width=WIDTH, height=HEIGHT,
                                      max_frames=MAX_FRAMES, letterbox=True)
    w = None
    if static_mask and getattr(enc, "supports_weights", False) and len(frames) >= 4:
        # down-weights the wearer's own torso/legs, which are a large constant
        # region of every frame they ever shoot and would otherwise make two
        # unrelated jobs by the same person look similar.
        w = static_weight_map(frames)
    E = enc.encode(frames, w)
    return {"frames": E.astype(np.float32),
            "mean": E.mean(0).astype(np.float32),
            "var": E.var(0).astype(np.float32),
            "n_frames": np.int32(len(frames)),
            "span_s": np.float32(float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0)}


def env_scores(a: Dict, b: Dict, wh=None) -> Dict[str, float]:
    """The validated appearance metrics: three, fused, never one."""
    def W(x):
        return wh.apply("app", x) if wh is not None else x
    c = cosine(W(a["mean"]), W(b["mean"]))
    ch = chamfer(W(a["frames"]), W(b["frames"]))
    bh = float(np.exp(-gaussian_bhattacharyya(a["mean"], a["var"],
                                              b["mean"], b["var"])))
    return {"env_cos": c, "env_chamfer": ch, "env_bhat_sim": bh,
            "env_raw": 0.50 * ch + 0.35 * c + 0.15 * bh}


# ------------------------------------------------------------------- phase 1
def cmd_embed(args) -> int:
    os.makedirs(args.sigs, exist_ok=True)
    todo = json.load(open(args.manifest))          # {name: video_uri}
    enc = get_appearance_encoder("dinov2", batch_size=64, dtype="float16")
    done = skipped = failed = 0
    t_start = time.time()
    for i, (name, uri) in enumerate(sorted(todo.items()), 1):
        out = os.path.join(args.sigs, f"{name}.npz")
        if os.path.exists(out):
            skipped += 1
            continue
        work = tempfile.mkdtemp(prefix="env-")
        t0 = time.time()
        try:
            # remote uris are range-read in place; nothing is downloaded
            sig = embed_video(uri, enc, static_mask=not args.no_static_mask)
            np.savez_compressed(out, **sig)
            done += 1
            print(f"  [{i}/{len(todo)}] {name[:56]:<58} "
                  f"{int(sig['n_frames'])} frames  {time.time()-t0:5.1f}s", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"  [{i}/{len(todo)}] {name[:56]:<58} FAILED {str(exc)[:70]}",
                  flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    el = time.time() - t_start
    print(f"\nembedded {done}, cached {skipped}, failed {failed} in {el/60:.1f} min"
          + (f"  ({el/max(done,1):.1f}s per video)" if done else ""))
    return 0


# ------------------------------------------------------------------- phase 2
def load_sigs(sigdir: str) -> Dict[str, Dict]:
    out = {}
    for f in sorted(glob.glob(os.path.join(sigdir, "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = {k: z[k] for k in z.files}
    return out


def strata(name_a, name_b, meta):
    """Which known class a pair belongs to, from metadata alone.

    Only two of these are ground truth. The rest are named so the report can
    show where the UNKNOWN cases fall relative to the known ones.
    """
    a, b = meta.get(name_a) or {}, meta.get(name_b) or {}
    ea, eb = (a.get("environment_l1") or "?").strip(), (b.get("environment_l1") or "?").strip()
    sa, sb = (a.get("site") or "?").strip(), (b.get("site") or "?").strip()
    ja, jb = a.get("_session") or "?", b.get("_session") or "?"
    na, nb = a.get("_nnn"), b.get("_nnn")
    if ea != eb and "?" not in (ea, eb):
        return "diff_industry"                      # GROUND TRUTH: different
    if ja == jb and na is not None and nb is not None and abs(na - nb) <= 1:
        return "same_session_adjacent"              # probably same place
    if ja == jb:
        return "same_session_distant"               # NOT reliable, see docstring
    if sa == sb and "?" not in (sa, sb):
        return "same_site_diff_session"             # THE UNKNOWN we predict
    return "same_industry_diff_site"


def cmd_compare(args) -> int:
    from novelty.calibrate import Whitener

    sigs = load_sigs(args.sigs)
    print(f"{len(sigs)} video signatures")
    if len(sigs) < 8:
        print("need >=8 signatures for a meaningful negative null")
        return 2

    meta = {}
    if args.csv and os.path.exists(args.csv):
        for r in csv.DictReader(open(args.csv)):
            q = (r.get("chunk_s3_path") or "").replace("s3://", "").strip("/").split("/")
            r["_session"] = "/".join(q[1:7]) if len(q) >= 7 else None
            r["_nnn"] = int(q[7]) if len(q) > 7 and q[7].isdigit() else None
            meta[r.get("episode_uuid", "")] = r

    names = sorted(sigs)
    # ---- whitening, fitted on every frame in the corpus --------------------
    X = np.concatenate([sigs[n]["frames"] for n in names], 0).astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    k = len(X) / (len(X) + 40.0)
    sd = np.maximum(k * sd + (1 - k) * sd.mean(), 1e-3)
    wh = Whitener({"app": {"mu": mu.tolist(), "sd": sd.tolist()}})
    print(f"whitener fitted on {len(X)} frame embeddings")

    # ---- ONE-SIDED calibration on certain negatives only -------------------
    # No positive class is asserted. "Same session" is NOT a reliable positive:
    # sessions span up to 47 chunk indices (~hours), and within a single 10 min
    # recording env_raw already decays +0.609 (0-30 s apart) -> +0.271 (2-4 min
    # apart), which is indistinguishable from a different session (+0.285). So
    # the only ground truth available is "different industry => different
    # place", and the threshold is derived from that alone.
    rng = np.random.default_rng(0)
    idx = {n: i for i, n in enumerate(names)}
    negs = [(a, b) for a, b in itertools.combinations(names, 2)
            if strata(a, b, meta) == "diff_industry"]
    if len(negs) < 50:
        print(f"only {len(negs)} certain negatives; need cross-industry pairs")
        return 2
    samp = [negs[i] for i in rng.choice(len(negs), min(len(negs), args.null_pairs),
                                        replace=False)]
    null = np.array([env_scores(sigs[a], sigs[b], wh)["env_raw"] for a, b in samp])
    null.sort()
    print(f"negative null: {len(null)} certain cross-industry pairs "
          f"(of {len(negs)}) mean {null.mean():+.4f} sd {null.std():.4f}")
    for q in (50, 90, 99, 99.9):
        print(f"    p{q:<5} {np.percentile(null, q):+.4f}")

    def pct(v):
        return float(np.searchsorted(null, v) / len(null) * 100.0)

    # ---- split-half reference: the tightest positive obtainable ------------
    # Same episode, first half vs second half. Same room, same minutes. Not
    # ground truth either (the within-recording decay above applies in
    # miniature), but it brackets the scale: this is what same-place looks
    # like at best.
    halves = []
    for n in names:
        F = sigs[n]["frames"]
        if len(F) < 6:
            continue
        h = len(F) // 2
        A = {"frames": F[:h], "mean": F[:h].mean(0), "var": F[:h].var(0)}
        B = {"frames": F[h:], "mean": F[h:].mean(0), "var": F[h:].var(0)}
        halves.append(env_scores(A, B, wh)["env_raw"])
    halves = np.array(halves)
    if len(halves):
        print(f"\nsplit-half reference (same episode, n={len(halves)}): "
              f"median {np.median(halves):+.4f} -> {pct(np.median(halves)):.2f}th pct "
              f"of the negative null")
        print(f"    this is the upper anchor: at best, 'same place' scores here")

    # ---- blocked comparison with an explicit UNRESOLVED band ---------------
    def block_of(n):
        if not args.block_by:
            return "ALL"
        return ((meta.get(n) or {}).get(args.block_by) or "UNKNOWN").strip()

    groups = collections.defaultdict(list)
    for n in names:
        groups[block_of(n)].append(n)
    pairs = [(a, b) for g in groups.values()
             for a, b in itertools.combinations(sorted(g), 2)]
    print(f"\nblocking by {args.block_by or '(none)'}: {len(groups)} groups -> "
          f"{len(pairs):,} pairs")

    rows, byv, bystr = [], collections.Counter(), collections.defaultdict(list)
    for a, b in pairs:
        s = env_scores(sigs[a], sigs[b], wh)
        p = pct(s["env_raw"])
        if p >= args.same_pct:
            v = "SAME_ENV"
        elif p <= args.diff_pct:
            v = "DIFFERENT_ENV"
        else:
            v = "UNRESOLVED"
        byv[v] += 1
        st = strata(a, b, meta)
        bystr[st].append(p)
        rows.append({"video_a": a, "video_b": b, "block": block_of(a),
                     "stratum": st,
                     **{k: round(vv, 4) for k, vv in s.items()},
                     "env_percentile_vs_negatives": round(p, 3),
                     "verdict": v})
    rows.sort(key=lambda r: -r["env_percentile_vs_negatives"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nverdicts: {dict(byv)}")
    print(f"\n{'stratum':<26}{'n':>7}{'median pct':>12}{'p90':>9}  (vs certain negatives)")
    for st in ("same_session_adjacent", "same_session_distant",
               "same_site_diff_session", "same_industry_diff_site", "diff_industry"):
        v = bystr.get(st)
        if v:
            print(f"{st:<26}{len(v):>7}{np.median(v):>12.2f}{np.percentile(v,90):>9.2f}")
    print(f"\nwrote {len(rows)} pairs -> {args.out}")
    print("UNRESOLVED is a real answer: those pairs are neither clearly the same")
    print("place nor clearly different, and no positive ground truth exists to")
    print("split them. Eyeball the top-scoring same_site_diff_session pairs to anchor it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="one environment signature per video")
    e.add_argument("--manifest", required=True, help='JSON {name: video_uri}')
    e.add_argument("--sigs", default="env_sigs")
    e.add_argument("--no-static-mask", action="store_true")
    e.set_defaults(fn=cmd_embed)
    c = sub.add_parser("compare", help="pairwise, blocked, calibrated")
    c.add_argument("--sigs", default="env_sigs")
    c.add_argument("--out", default="env_pairs.csv")
    c.add_argument("--csv", default="rejected_episodes.csv",
                   help="metadata for blocking (needs episode_uuid + block key)")
    c.add_argument("--block-by", default="", help="e.g. site, environment_l1")
    c.add_argument("--null-pairs", type=int, default=4000)
    c.add_argument("--same-pct", type=float, default=99.5,
                   help="percentile of the negative null above which a pair is SAME_ENV")
    c.add_argument("--diff-pct", type=float, default=90.0,
                   help="percentile at or below which a pair is DIFFERENT_ENV")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
