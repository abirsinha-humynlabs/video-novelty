"""v2 pipeline: cycle-aware chunks from hand tracks, then similarity per episode.

v1 cut every video on a strict 30 s grid. v2 cuts where the hand tracks say a
work cycle begins, so each chunk is a whole number of repetitions starting at
the same phase, and chunk-to-chunk comparison is apples-to-apples.

Three phases, deliberately separated because their costs differ by 100x:

  A  segment   NPZ only, no video, ~0.1 s/episode      -> segments.json
  B  embed     needs video: download, cut, DINOv2       -> signatures (slow)
  C  compare   global whitener + null, per-episode CSV  -> seconds

Phase A is cheap enough to re-run freely and tells you which episodes have any
cadence at all -- on sampled data only ~15% do, so phase B only ever runs on
that subset. Phase C must be global: whitening and the null have to be fitted
across episodes, because within one episode every chunk shares the same hands,
bench and lighting, raw cosines saturate at 0.98 with a 0.04 spread, and
mean-centring n chunks mechanically forces the average cosine to -1/(n-1).

Local disk is treated as scratch only: video is deleted as soon as it is cut,
chunks go to S3, and nothing large is kept.

  python scripts/run_v2.py phase-a --npz-dir <dir> --out state/
  python scripts/run_v2.py phase-b --state state/ --workers 2
  python scripts/run_v2.py phase-c --state state/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty import cycles as C                                   # noqa: E402
from scripts.report_csv import S3_DATA_PREFIX, S3_RESULT_PREFIX, check_s3_destination  # noqa: E402

STAGE_HAND = ("s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_mint")
#: The runner's own OUT says model_output, but the files actually land in
#: model_output_fullrate. Verified by listing: model_output holds only a
#: folder marker.
#: BOTH are written to, concurrently, and which one an episode lands in is not
#: predictable: model_output was empty at 17:58 UTC and held 5 files by 18:35
#: while model_output_fullrate held 8. Polling only one silently loses episodes,
#: so every cycle syncs both. They merge into one local tree keyed by episode
#: name, so an episode present in both simply resolves to one entry.
PROD_NPZ_PREFIXES = (
    "s3://prod-egc-stereo-v2-data/work_items/hand_detection/model_output",
    "s3://prod-egc-stereo-v2-data/work_items/hand_detection/model_output_fullrate",
)
N_CYCLES = 6

#: Profile used ONLY for reads from the prod bucket. Writes deliberately keep
#: using the default (instance-role) credentials: an SSO identity is typically
#: far broader than this box needs, and no write path should run through it.
READ_PROFILE = os.environ.get("NOVELTY_PROD_PROFILE", "")


def sh(cmd, profile=None, **kw):
    if profile:
        cmd = list(cmd) + ["--profile", profile]
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:5])}: {p.stderr.strip()[:400]}")
    return p.stdout


def fetch_prod_npz(dest, limit=None):
    """Mirror the dynamically-growing hand-detection output into ``dest``.

    The producing model is still writing into this prefix, so this is written to
    be re-run: `aws s3 sync` skips what is already local, so each call picks up
    whatever has landed since. Never deletes.
    """
    os.makedirs(dest, exist_ok=True)
    per_prefix = {}
    for pref in PROD_NPZ_PREFIXES:
        cmd = ["aws", "s3", "sync", pref, dest,
               # only the fused 21-keypoint file; the 2d/3d siblings triple the
               # transfer and are not used.
               "--exclude", "*", "--include", "*_hand21_keypoints.npz",
               # '..' appears in some keys under these prefixes, which sync
               # would otherwise try to write outside dest
               "--exclude", "*../*",
               "--only-show-errors"]
        try:
            sh(cmd, profile=READ_PROFILE or None)
        except RuntimeError as exc:
            print(f"  sync {pref.rsplit('/', 1)[-1]} failed: {str(exc)[:120]}", flush=True)
        per_prefix[pref.rsplit("/", 1)[-1]] = len(
            glob.glob(os.path.join(dest, "**", "*.npz"), recursive=True))
    got = glob.glob(os.path.join(dest, "**", "*.npz"), recursive=True)
    print(f"  synced (cumulative after each prefix): {per_prefix}", flush=True)
    return got


# ------------------------------------------------------------------ phase A
def phase_a(args):
    os.makedirs(args.out, exist_ok=True)
    if args.sync_prod:
        got = fetch_prod_npz(args.npz_dir)
        print(f"synced {len(got)} npz from prod into {args.npz_dir}", flush=True)
    # accept both the *_hand21_keypoints.npz convention and bare *.npz, since
    # the prod hand-detection output may name files differently
    kps = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*_hand21_keypoints.npz"),
                           recursive=True))
    if not kps:
        kps = [p for p in sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"),
                                           recursive=True))
               if "_head" not in os.path.basename(p)]
    print(f"phase A: {len(kps)} keypoint files", flush=True)
    index = {}
    t0 = time.time()
    for kp in kps:
        name = os.path.basename(kp).replace("_hand21_keypoints.npz", "")
        hd = kp.replace("_hand21_keypoints.npz", "_head.npz")
        rec = {"name": name, "keypoints": kp, "head": hd if os.path.exists(hd) else None}
        try:
            tr = C.load_tracks(kp, rec["head"])
            hand = C.RIGHT if tr.coverage[C.RIGHT] >= tr.coverage[C.LEFT] else C.LEFT
            P, _strength = C.estimate_period(tr, hand=hand)
            an = C.analyse(tr, hand=hand, period_hint=P if P > 0 else None)
            segs = C.segment_by_cycles(an, n_cycles=N_CYCLES)
            rec.update(
                duration_s=round(an.duration_s, 2), fps=tr.fps,
                hand="RIGHT" if hand == C.RIGHT else "LEFT",
                has_head=rec["head"] is not None,
                coverage_l=round(tr.coverage[C.LEFT], 3),
                coverage_r=round(tr.coverage[C.RIGHT], 3),
                period_s=round(an.period_s, 3),
                trackable=round(an.trackable_fraction, 3),
                cadence=round(an.cadence_fraction, 3),
                n_boundaries=len(an.boundaries),
                segments=[[round(s.t0, 3), round(s.t1, 3), s.n_cycles] for s in segs],
            )
        except Exception as exc:                                   # noqa: BLE001
            rec.update(error=repr(exc)[:200], segments=[])
        index[name] = rec
    dt = time.time() - t0
    with open(os.path.join(args.out, "segments.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    withseg = [r for r in index.values() if r.get("segments")]
    nseg = sum(len(r["segments"]) for r in withseg)
    print(f"  {dt:.1f}s total, {dt/max(len(kps),1)*1000:.0f} ms/episode")
    print(f"  episodes with >=1 cycle chunk: {len(withseg)}/{len(index)} "
          f"({len(withseg)/max(len(index),1)*100:.0f}%)")
    print(f"  episodes with >=2 chunks (comparable): "
          f"{sum(1 for r in withseg if len(r['segments']) >= 2)}")
    print(f"  total chunks: {nseg}")
    return 0


# ------------------------------------------------------------------ phase B
def _exists(uri, profile=None):
    bucket, key = uri[len("s3://"):].split("/", 1)
    try:
        sh(["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
           profile=profile)
        return True
    except RuntimeError:
        return False


def find_video(name):
    """Locate the source video for an episode.

    The hand-detection output encodes the episode's own S3 path with '/' turned
    into '__', optionally prefixed 'bitrobot__', so the video path is
    reconstructable rather than needing a lookup table. Prod is tried first
    (48 MB rectified files, read-only) before the much larger stage copies.
    """
    parts = name.split("__")
    # names may or may not carry their collection as the first token; when it is
    # absent the collection is bitrobot (e.g. Indoor_framing-Roses__...), and
    # 'akai' episodes sit under a different collection root entirely.
    COLLECTIONS = ("bitrobot", "akai")
    collection = parts[0] if parts and parts[0] in COLLECTIONS else "bitrobot"
    if parts and parts[0] in COLLECTIONS:
        parts = parts[1:]
    tail = "/".join(parts)
    for root in ("normalized", "chunking"):
        for fn in ("left_rectified.mp4", "left_eye_rectified.mp4"):
            uri = f"s3://prod-egc-stereo-v2-data/{root}/{collection}/{tail}/{fn}"
            if _exists(uri, profile=READ_PROFILE or None):
                return uri, READ_PROFILE or None
    for pref in ("_prod18_input", "_bit28_input", "_prod18eye_input"):
        uri = f"{STAGE_HAND}/{pref}/{name}/left_eye.mp4"
        if _exists(uri):
            return uri, None
    return None, None


def embed_episode(rec, sigdir, upload=True):
    """Download video, cut the cycle chunks, DINOv2 them, push chunks to S3."""
    from novelty.config import Config
    from novelty.signature import build_signatures
    from novelty.encoders.base import get_appearance_encoder

    name, segs = rec["name"], rec["segments"]
    out = os.path.join(sigdir, name)
    if os.path.exists(os.path.join(out, "done")):
        return "cached", 0.0
    vid, vprofile = find_video(name)
    if vid is None:
        return "no-video", 0.0

    t0 = time.time()
    work = tempfile.mkdtemp(prefix="v2-")
    try:
        src = os.path.join(work, "src.mp4")
        sh(["aws", "s3", "cp", vid, src, "--only-show-errors"], profile=vprofile)
        cdir = os.path.join(work, "chunks")
        os.makedirs(cdir)
        for i, (a, b, ncyc) in enumerate(segs, 1):
            nm = f"chunk{i:03d}_{a:.2f}-{b:.2f}_{ncyc}cyc.mp4"
            sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{a:.3f}", "-i", src, "-t", f"{b-a:.3f}",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", os.path.join(cdir, nm)])
        os.remove(src)

        if upload:
            sh(["aws", "s3", "sync", cdir, f"{S3_DATA_PREFIX}/{name}/chunks",
                "--only-show-errors"])

        # appearance only: the task axis is switched off, so V-JEPA2 and the flow
        # tier are 64% of the cost for zero effect on any decision.
        cfg = Config.load("configs/gpu.yaml")
        cfg.motion.enabled = False
        cfg.motion.clip_encoder = None
        enc = get_appearance_encoder(cfg.appearance.encoder, **cfg.appearance.encoder_kwargs)
        os.makedirs(out, exist_ok=True)
        for f in sorted(glob.glob(f"{cdir}/*.mp4")):
            sigs = build_signatures(f, cfg=cfg, appearance_encoder=enc)
            for s in sigs:
                s.save(os.path.join(out, f"{os.path.basename(f)[:-4]}.npz"))
                break
        open(os.path.join(out, "done"), "w").close()
        return "ok", time.time() - t0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def phase_b(args):
    index = json.load(open(os.path.join(args.state, "segments.json")))
    todo = [r for r in index.values() if len(r.get("segments", [])) >= 2]
    todo.sort(key=lambda r: -len(r["segments"]))
    sigdir = os.path.join(args.state, "signatures")
    os.makedirs(sigdir, exist_ok=True)
    print(f"phase B: {len(todo)} episodes with >=2 chunks", flush=True)
    stats = {}
    for i, rec in enumerate(todo, 1):
        try:
            status, dt = embed_episode(rec, sigdir, upload=not args.no_upload)
        except Exception as exc:                                   # noqa: BLE001
            status, dt = f"ERROR {exc!r}"[:90], 0.0
        stats[status.split()[0]] = stats.get(status.split()[0], 0) + 1
        print(f"  [{i}/{len(todo)}] {rec['name'][:52]:<54} "
              f"{len(rec['segments']):>3} chunks  {status:<10} {dt:6.1f}s", flush=True)
    print(f"  {stats}")
    return 0


# ------------------------------------------------------------------ phase C
def phase_c(args):
    import csv
    import itertools
    from novelty.signature import Signature
    from novelty.metrics.distance import cosine, chamfer, gaussian_bhattacharyya

    sigdir = os.path.join(args.state, "signatures")
    eps = sorted(d for d in glob.glob(f"{sigdir}/*") if os.path.isdir(d))
    per_ep = {}
    for d in eps:
        files = sorted(glob.glob(f"{d}/*.npz"))
        if len(files) < 2:
            continue
        per_ep[os.path.basename(d)] = [Signature.load(f) for f in files]
    allsig = [s for v in per_ep.values() for s in v]
    print(f"phase C: {len(per_ep)} episodes, {len(allsig)} chunk signatures")
    if len(allsig) < 4:
        print("  too few signatures to fit a global whitener/null")
        return 2

    # GLOBAL whitener + null -- see module docstring for why this cannot be
    # fitted per episode.
    from novelty.calibrate import fit_whitener, fit_null
    wh = fit_whitener(allsig)
    null = fit_null(allsig, max_pairs=4000, exclude_same_video=True)
    null.whiten = wh.stats

    def env(a, b):
        am, bm = wh.apply("app", a.app_mean), wh.apply("app", b.app_mean)
        af, bf = wh.apply("app", a.app_frames), wh.apply("app", b.app_frames)
        c = cosine(am, bm)
        ch = chamfer(af, bf)
        bh = float(np.exp(-gaussian_bhattacharyya(a.app_mean, a.app_var,
                                                  b.app_mean, b.app_var)))
        raw = 0.50 * ch + 0.35 * c + 0.15 * bh
        return c, ch, bh, raw

    outdir = os.path.join(args.state, "out")
    os.makedirs(outdir, exist_ok=True)
    for name, sigs in per_ep.items():
        rows = []
        for i, j in itertools.combinations(range(len(sigs)), 2):
            c, ch, bh, raw = env(sigs[i], sigs[j])
            rows.append({
                "chunk_a": sigs[i].label, "chunk_b": sigs[j].label,
                "env_cos": round(c, 4), "env_chamfer": round(ch, 4),
                "env_bhat_sim": round(bh, 4), "env_raw": round(raw, 4),
                "env_percentile": round(null.percentile("env_raw", raw), 2),
            })
        p = os.path.join(outdir, f"{name}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    check_s3_destination(S3_RESULT_PREFIX)
    sh(["aws", "s3", "sync", outdir, S3_RESULT_PREFIX, "--only-show-errors"])
    print(f"  wrote {len(per_ep)} CSVs -> {S3_RESULT_PREFIX}")
    return 0


def phase_drain(args):
    """Poll the growing prod prefix and process whatever has arrived.

    The hand-detection model is still writing, so a single batch run would
    either wait idle for hours or miss most episodes. Each cycle re-syncs
    (cheap: sync skips what is local), re-segments (89 ms/episode), embeds only
    newly-eligible episodes (phase B is idempotent via a per-episode `done`
    marker), and rewrites all CSVs. Safe to kill and restart at any point.
    """
    deadline = time.time() + args.max_hours * 3600
    quiet = 0
    seen = 0
    for it in range(1, 10_000):
        if time.time() > deadline:
            print("drain: max-hours reached, stopping", flush=True)
            break
        t0 = time.time()
        try:
            got = fetch_prod_npz(args.npz_dir)
        except RuntimeError as exc:
            print(f"drain[{it}]: sync failed ({str(exc)[:120]}); retrying", flush=True)
            time.sleep(args.interval)
            continue
        new = len(got) - seen
        seen = len(got)
        print(f"\ndrain[{it}] {time.strftime('%H:%M:%S')}  npz={len(got)} (+{new})",
              flush=True)
        a = argparse.Namespace(npz_dir=args.npz_dir, out=args.state, sync_prod=False)
        phase_a(a)
        b = argparse.Namespace(state=args.state, no_upload=args.no_upload)
        phase_b(b)
        try:
            phase_c(argparse.Namespace(state=args.state))
        except SystemExit:
            pass
        except Exception as exc:                                   # noqa: BLE001
            print(f"  phase C deferred: {str(exc)[:150]}", flush=True)
        quiet = quiet + 1 if new == 0 else 0
        if quiet >= args.stop_after_quiet:
            print(f"drain: no new npz for {quiet} cycles, stopping", flush=True)
            break
        sleep = max(args.interval - (time.time() - t0), 5)
        print(f"  cycle took {time.time()-t0:.0f}s; sleeping {sleep:.0f}s", flush=True)
        time.sleep(sleep)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("phase-a"); a.add_argument("--npz-dir", required=True)
    a.add_argument("--sync-prod", action="store_true",
                   help="first mirror new npz from the prod prefix (re-runnable)")
    a.add_argument("--out", default="state"); a.set_defaults(fn=phase_a)
    b = sub.add_parser("phase-b"); b.add_argument("--state", default="state")
    b.add_argument("--no-upload", action="store_true"); b.set_defaults(fn=phase_b)
    c = sub.add_parser("phase-c"); c.add_argument("--state", default="state")
    c.set_defaults(fn=phase_c)
    d = sub.add_parser("drain")
    d.add_argument("--npz-dir", required=True)
    d.add_argument("--state", default="state")
    d.add_argument("--interval", type=float, default=600.0)
    d.add_argument("--max-hours", type=float, default=11.0)
    d.add_argument("--stop-after-quiet", type=int, default=12)
    d.add_argument("--no-upload", action="store_true")
    d.set_defaults(fn=phase_drain)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
