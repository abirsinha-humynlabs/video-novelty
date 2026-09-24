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
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty import cycles as C                                   # noqa: E402
from scripts.report_csv import S3_DATA_PREFIX, S3_TASK2_PREFIX, check_s3_destination  # noqa: E402

STAGE_HAND = ("s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_mint")
#: The runner's own OUT says model_output, but the files actually land in
#: model_output_fullrate. Verified by listing: model_output holds only a
#: folder marker.
#: BOTH are written to, concurrently, and which one an episode lands in is not
#: predictable: model_output was empty at 17:58 UTC and held 5 files by 18:35
#: while model_output_fullrate held 8. Polling only one silently loses episodes,
#: so every cycle syncs both. They merge into one local tree keyed by episode
#: name, so an episode present in both simply resolves to one entry.
#: Every prefix the hand detector writes to. `model_output_30fps` was added
#: 2026-09-22 and is the one actively filling; `model_output` is 766 episodes
#: all at step 10 (3 fps) and is mirrored only so the step gate can reject them
#: with a reason rather than silently seeing nothing. Missing a prefix here is
#: invisible -- the drain simply never notices those episodes exist.
PROD_NPZ_PREFIXES = (
    "s3://prod-egc-stereo-v2-data/work_items/hand_detection/model_output",
    "s3://prod-egc-stereo-v2-data/work_items/hand_detection/model_output_fullrate",
    "s3://prod-egc-stereo-v2-data/work_items/hand_detection/model_output_30fps",
)
N_CYCLES = 6

#: Shortest chunk worth comparing. One second, because the fastest real work
#: cycles are about that: a screw picked from one tray and dropped in the tray
#: beside it, again and again. Jitter at the same rate is rejected by the
#: per-chunk recurrence test in `analyse_adaptive`, not by duration -- a duration
#: floor high enough to exclude jitter would also exclude that work.
MIN_CHUNK_S = 1.0

#: Only full-rate tracks are accepted. Set to 1 deliberately: the hand detector
#: must run at the source frame rate (step=1, 30 fps) for cycle cutting to mean
#: anything.
#:
#: Measured by decimating episodes that work at step=1 and re-running the
#: identical analysis, so everything lost is lost purely to temporal resolution:
#:
#:   cycle    step=1      step=2      step=3      step=5     step=10
#:   1.30 s   2 ch  +-1%  3 ch  +-5%  3 ch  +-8%  3 ch +-12%  1 ch +-25%
#:   0.73 s   9 ch  +-5%  9 ch  +-9%  7 ch +-14%  5 ch +-25%  0 chunks
#:   0.57 s   2 ch  +-6%  1 ch +-14%  0 chunks    0 chunks    0 chunks
#:
#: (+-% = boundary precision as a share of one cycle). Roughly 7 samples per
#: cycle are needed. step=10 produced 0 chunks across 24 real episodes. Phase
#: alignment is the entire purpose of cutting on cycles, so even where coarser
#: strides still yield chunks they give back most of the benefit.
#:
#: Episodes coarser than this are recorded with a reason rather than silently
#: yielding nothing, so "no chunks" is never confused with "no cadence".
MAX_STEP = 1

#: Profile used for reads from the prod bucket.
READ_PROFILE = os.environ.get("NOVELTY_PROD_PROFILE", "")
#: Profile for WRITING to stage. Which credential can write stage differs by
#: machine -- the original GPU box could do it with its instance role, the
#: devbox cannot and needs an SSO profile -- so it is configurable rather than
#: implicit, and an empty value means "use the default credentials".
WRITE_PROFILE = os.environ.get("NOVELTY_STAGE_PROFILE", "")


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
    # Episodes a reviewer has looked at and rejected. Keyed by episode name,
    # valued by the reason. They are still analysed, so the record says what
    # the analysis found, but their chunks are moved aside: phases B and C only
    # act on `segments`, so an excluded episode is never embedded or written,
    # and the status report says why instead of reporting it as no-cadence.
    exclude = {}
    if getattr(args, "exclude", None):
        exclude = json.load(open(args.exclude))
        print(f"  excluding {len(exclude)} reviewed episode(s)", flush=True)
    index = {}
    t0 = time.time()
    for kp in kps:
        name = os.path.basename(kp).replace("_hand21_keypoints.npz", "")
        hd = kp.replace("_hand21_keypoints.npz", "_head.npz")
        rec = {"name": name, "keypoints": kp, "head": hd if os.path.exists(hd) else None}
        try:
            step = int(np.load(kp, allow_pickle=True)["step"])
        except Exception:                                          # noqa: BLE001
            step = 1
        rec["step"] = step
        if step > MAX_STEP:
            rec.update(segments=[], skipped=f"step={step} ({30/step:.0f} fps) "
                       f"coarser than MAX_STEP={MAX_STEP}; re-run at step<=3")
            index[name] = rec
            continue
        try:
            tr = C.load_tracks(kp, rec["head"])
            # Let the analysis pick the hand AND the channel: it tries both
            # hands across the wrist, whole-hand articulation, hand rotation
            # and each of the five fingers, at the strictest gate first, and
            # only loosens the gate when nothing is found above it. Choosing
            # here on coverage alone was close but not the same thing, and
            # passing a period_hint computed for the pre-chosen hand would have
            # pinned the answer to it.
            an = C.analyse_adaptive(tr, n_cycles=N_CYCLES, min_chunks=2,
                                    min_chunk_s=MIN_CHUNK_S)
            # Ship only an ACCEPTED result's chunks. When nothing passes the
            # recurrence test the analysis still comes back -- the best
            # near-miss, so the status report can say why -- and its chunks
            # must not be mistaken for accepted ones. And never a fresh
            # segment_by_cycles, which would put the rejected chunks back.
            accepted = an.recurrence_p <= C.RECURRENCE_ALPHA
            segs = an.segments if accepted else []
            rec.update(
                duration_s=round(an.duration_s, 2), fps=tr.fps,
                hand="RIGHT" if an.hand == C.RIGHT else "LEFT",
                has_head=rec["head"] is not None,
                coverage_l=round(tr.coverage[C.LEFT], 3),
                coverage_r=round(tr.coverage[C.RIGHT], 3),
                period_s=round(an.period_s, 3),
                trackable=round(an.trackable_fraction, 3),
                cadence=round(an.cadence_fraction, 3),
                n_boundaries=len(an.boundaries),
                # HOW the cadence was found, not just that it was. A clip found
                # on the wrist at 0.45 and one found on the ring finger at 0.25
                # both produce chunks and are not equally good evidence; without
                # these two the CSVs cannot tell them apart.
                signal=an.signal,
                min_strength=round(an.min_strength, 3),
                # the physical check on the cuts: pooled pose recurrence at the
                # cut points and its permutation p-value against random cuts
                recurrence=(round(float(an.recurrence), 3)
                            if np.isfinite(an.recurrence) else None),
                recurrence_p=round(float(an.recurrence_p), 3),
                # each segment carries its OWN period, not the clip median: the
                # operator speeds up, slows down and adds sub-steps, so a
                # constant figure here is what made a 6-cycle chunk read as a
                # 6-second cycle in the output.
                segments=[[round(s.t0, 3), round(s.t1, 3), s.n_cycles,
                           round(s.period_s, 3), round(s.recurrence, 3)]
                          for s in segs],
            )
        except Exception as exc:                                   # noqa: BLE001
            rec.update(error=repr(exc)[:200], segments=[])
        if name in exclude:
            rec.update(excluded=exclude[name],
                       excluded_segments=rec.get("segments", []), segments=[])
        index[name] = rec
    dt = time.time() - t0
    with open(os.path.join(args.out, "segments.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    skipped = [r for r in index.values() if r.get("skipped")]
    if skipped:
        print(f"  SKIPPED {len(skipped)} episodes for coarse frame stride "
              f"(step>{MAX_STEP}); they cannot produce cycle chunks")
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


#: One encoder for the whole process, and one lock around the GPU section.
#: Workers overlap the parts that are I/O and CPU -- the video download and the
#: ffmpeg cut, which together are the bulk of an episode -- while the DINOv2
#: pass stays serialised. Sharing a torch module across threads without the lock
#: is the kind of thing that works until it silently does not, and the GPU is
#: not the bottleneck anyway (2.9 s/chunk against 5.7 s of ffmpeg).
_ENC = None
_ENC_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def _encoder(cfg):
    global _ENC
    with _ENC_LOCK:
        if _ENC is None:
            from novelty.encoders.base import get_appearance_encoder
            _ENC = get_appearance_encoder(cfg.appearance.encoder,
                                          **cfg.appearance.encoder_kwargs)
    return _ENC


def embed_episode(rec, sigdir, upload=True, data_prefix=None):
    """Download video, cut the cycle chunks, DINOv2 them, push chunks to S3."""
    from novelty.config import Config
    from novelty.signature import build_signatures

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
        for i, seg in enumerate(segs, 1):
            a, b, ncyc = seg[0], seg[1], seg[2]
            # The period goes in the NAME. A chunk's span is n_cycles x the
            # period, so 6 cycles of a 1.3 s cycle is ~8 s -- and with only the
            # span in the filename that reads as an 8 s cycle, which is exactly
            # how the shipped CSVs were misread. Carrying both is self-explaining.
            per = seg[3] if len(seg) > 3 else 0.0
            nm = (f"chunk{i:03d}_{a:.2f}-{b:.2f}_{ncyc}cyc"
                  f"{f'_{per:.2f}s' if per else ''}.mp4")
            sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{a:.3f}", "-i", src, "-t", f"{b-a:.3f}",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", os.path.join(cdir, nm)])
        os.remove(src)

        if upload:
            dest = (data_prefix or S3_DATA_PREFIX).rstrip("/")
            sh(["aws", "s3", "sync", cdir, f"{dest}/{name}/chunks",
                "--only-show-errors"], profile=WRITE_PROFILE or None)

        # appearance only: the task axis is switched off, so V-JEPA2 and the flow
        # tier are 64% of the cost for zero effect on any decision.
        cfg = Config.load("configs/gpu.yaml")
        cfg.motion.enabled = False
        cfg.motion.clip_encoder = None
        enc = _encoder(cfg)
        os.makedirs(out, exist_ok=True)
        for f in sorted(glob.glob(f"{cdir}/*.mp4")):
            with _GPU_LOCK:
                sigs = build_signatures(f, cfg=cfg, appearance_encoder=enc)
            for s in sigs:
                s.save(os.path.join(out, f"{os.path.basename(f)[:-4]}.npz"))
                break
        open(os.path.join(out, "done"), "w").close()
        return "ok", time.time() - t0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def phase_b(args):
    import concurrent.futures as cf

    index = json.load(open(os.path.join(args.state, "segments.json")))
    todo = [r for r in index.values() if len(r.get("segments", [])) >= 2]
    todo.sort(key=lambda r: -len(r["segments"]))
    sigdir = os.path.join(args.state, "signatures")
    os.makedirs(sigdir, exist_ok=True)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    print(f"phase B: {len(todo)} episodes with >=2 chunks, {workers} worker(s)",
          flush=True)
    stats = {}
    done = 0

    def run(rec):
        try:
            return rec, *embed_episode(rec, sigdir, upload=not args.no_upload,
                                       data_prefix=getattr(args, "data_prefix", None))
        except Exception as exc:                                   # noqa: BLE001
            return rec, f"ERROR {exc!r}"[:90], 0.0

    # Each episode is independent and writes only under its own name, so the
    # only shared state is the GPU (locked inside embed_episode) and the disk.
    # Workers past ~2-3 do not help on a 4 vCPU box: one ffmpeg already
    # multithreads across every core, measured at 19.8 s for 9 chunks on 4
    # parallel against 20.1 s serial. The win here is overlapping the S3
    # download of one episode with the cut of another.
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for rec, status, dt in ex.map(run, todo):
            done += 1
            stats[status.split()[0]] = stats.get(status.split()[0], 0) + 1
            print(f"  [{done}/{len(todo)}] {rec['name'][:52]:<54} "
                  f"{len(rec['segments']):>3} chunks  {status:<10} {dt:6.1f}s",
                  flush=True)
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
    # task2 subprefix, not the novelty_result_v2 root: these are per-episode
    # within-video CSVs and must not land beside task1's whole-video pair files.
    dest = (getattr(args, "s3_prefix", None) or S3_TASK2_PREFIX).rstrip("/")
    check_s3_destination(dest)
    sh(["aws", "s3", "sync", outdir, dest, "--only-show-errors"],
       profile=WRITE_PROFILE or None)
    print(f"  wrote {len(per_ep)} CSVs -> {dest}")
    return 0


def _comparable_set(state):
    """The episodes that currently qualify for phase B, as a comparable key.

    Includes the chunk count, so an episode that gains chunks from new
    keypoints counts as a change even though it was already eligible.
    """
    try:
        seg = json.load(open(os.path.join(state, "segments.json")))
    except Exception:                                              # noqa: BLE001
        return frozenset()
    return frozenset((n, len(r.get("segments") or []))
                     for n, r in seg.items() if len(r.get("segments") or []) >= 2)


def phase_drain(args):
    """Poll the growing prod prefix and process only what actually changed.

    The hand-detection model writes over hours, so a single batch run would
    either idle or miss most episodes. Each cycle re-syncs (cheap: sync skips
    what is local) and re-segments (~20 ms/episode, so always worth doing).

    Phases B and C then run ONLY if the set of comparable episodes changed.
    Phase B was already idempotent via its per-episode `done` marker, but phase
    C is not: it refits the whitener and null and rewrites every CSV, so a
    fully-caught-up drain was re-uploading identical files every cycle -- 6 CSVs
    every 15 minutes, indefinitely, which burns S3 requests and makes the
    modified time on every object a lie about when its content was produced.
    """
    deadline = time.time() + args.max_hours * 3600
    quiet = 0
    seen = 0
    last = _comparable_set(args.state)
    if last:
        print(f"drain: resuming with {len(last)} comparable episodes already done",
              flush=True)
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
        a = argparse.Namespace(npz_dir=args.npz_dir, out=args.state, sync_prod=False,
                               exclude=getattr(args, "exclude", None))
        phase_a(a)

        now = _comparable_set(args.state)
        if now == last:
            print(f"  no change: {len(now)} comparable episodes, nothing to "
                  f"embed or upload", flush=True)
        else:
            added = {n for n, _ in now} - {n for n, _ in last}
            print(f"  changed: {len(now)} comparable ({len(added)} new episode(s))",
                  flush=True)
            b = argparse.Namespace(state=args.state, no_upload=args.no_upload,
                                   workers=getattr(args, "workers", 1),
                                   data_prefix=getattr(args, "data_prefix", None))
            phase_b(b)
            try:
                phase_c(argparse.Namespace(state=args.state,
                                           s3_prefix=getattr(args, "s3_prefix", None)))
            except SystemExit:
                pass
            except Exception as exc:                               # noqa: BLE001
                print(f"  phase C deferred: {str(exc)[:150]}", flush=True)
            last = now

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
    a.add_argument("--out", default="state")
    a.add_argument("--exclude", default=None,
                   help="JSON {episode name: reason} of reviewed episodes whose "
                        "chunks must not ship")
    a.set_defaults(fn=phase_a)
    b = sub.add_parser("phase-b"); b.add_argument("--state", default="state")
    b.add_argument("--no-upload", action="store_true")
    b.add_argument("--workers", type=int, default=1,
                   help="episodes in flight at once; 2-3 is the useful range on "
                        "a 4 vCPU box, see phase_b")
    b.add_argument("--data-prefix", default=None,
                   help="S3 prefix for the cut chunk media; default S3_DATA_PREFIX")
    b.set_defaults(fn=phase_b)
    c = sub.add_parser("phase-c"); c.add_argument("--state", default="state")
    c.add_argument("--s3-prefix", default=None,
                   help="S3 prefix for the per-episode CSVs; default "
                        "S3_TASK2_PREFIX. Must be inside the write allowlist.")
    c.set_defaults(fn=phase_c)
    d = sub.add_parser("drain")
    d.add_argument("--npz-dir", required=True)
    d.add_argument("--state", default="state")
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--data-prefix", default=None)
    d.add_argument("--s3-prefix", default=None)
    d.add_argument("--interval", type=float, default=600.0)
    d.add_argument("--max-hours", type=float, default=11.0)
    d.add_argument("--stop-after-quiet", type=int, default=12)
    d.add_argument("--no-upload", action="store_true")
    d.add_argument("--exclude", default=None,
                   help="as for phase-a; carried into every cycle")
    d.set_defaults(fn=phase_drain)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
