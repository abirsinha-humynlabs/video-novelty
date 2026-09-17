"""Command line entry point.

    novelty index   VIDEO...        build signatures into an index
    novelty calibrate               fit the corpus null model (DO THIS SECOND)
    novelty compare A B             explain one pair
    novelty search  VIDEO           nearest neighbours in the index
    novelty gate    VIDEO           accept/reject a candidate for ingest
    novelty select                  pick a maximally-diverse subset
    novelty report                  write a self-contained HTML report
    novelty encoders                list available appearance backbones

Run `novelty <cmd> --help` for flags.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from .calibrate import ENV_COMPONENTS, TASK_COMPONENTS, evaluate_axis, fit_null, fit_weights
from .config import Config
from .encoders.base import get_appearance_encoder, list_appearance_encoders
from .metrics.fuse import Label, compare as compare_sigs, raw_scores
from .select import facility_location_greedy, gate as gate_fn
from .signature import Signature, build_signatures
from .store.numpy_store import Index

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}


# --------------------------------------------------------------------------- utils
def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files)
                        if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        else:
            out.append(p)
    return out


def _cfg(args) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "encoder", None):
        cfg.appearance.encoder = args.encoder
    if getattr(args, "window", None):
        cfg.segment.seconds = float(args.window)
    if getattr(args, "hop", None):
        cfg.segment.hop = float(args.hop)
    if getattr(args, "no_motion", False):
        cfg.motion.enabled = False
    return cfg


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- index
def cmd_index(args) -> int:
    cfg = _cfg(args)
    idx = Index(args.index).create(cfg)
    paths = _expand(args.videos)
    if not paths:
        _eprint("no videos found")
        return 2
    enc = get_appearance_encoder(cfg.appearance.encoder, **cfg.appearance.encoder_kwargs)
    total = 0
    for n, p in enumerate(paths, 1):
        t = time.time()
        try:
            sigs = build_signatures(p, cfg=cfg, appearance_encoder=enc)
        except Exception as exc:                                  # noqa: BLE001
            _eprint(f"[{n}/{len(paths)}] SKIP {os.path.basename(p)}: {exc}")
            continue
        idx.add(sigs)
        total += len(sigs)
        _eprint(f"[{n}/{len(paths)}] {os.path.basename(p)}: {len(sigs)} signature(s) "
                f"in {time.time() - t:.1f}s")
    print(f"indexed {total} signature(s) from {len(paths)} file(s) -> {idx.root}")
    print("NEXT: run `novelty calibrate --index {}` before trusting any score.".format(idx.root))
    return 0


# ----------------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> int:
    idx = Index(args.index)
    sigs = idx.signatures()
    if len(sigs) < 2:
        _eprint("need >= 2 signatures in the index")
        return 2

    def prog(k, n):
        if k % max(n // 20, 1) == 0:
            _eprint(f"  null pairs {k}/{n}")

    null = fit_null(sigs, max_pairs=args.max_pairs, progress=prog)
    idx.save_null(null)
    if len(sigs) < 50:
        _eprint(f"WARNING: only {len(sigs)} signatures. The null distribution and the "
                f"whitener are both fitted on your corpus, so with a corpus this small "
                f"they mostly describe noise. Percentiles below are indicative, not "
                f"trustworthy. Aim for >= a few hundred signatures (use --window to "
                f"split long files).")
    print(f"fitted null model over {null.n_pairs} random pairs -> {idx.null_path}")
    for m in ("env_raw", "task_raw"):
        q = null.quantiles[m]
        print(f"  {m:9s} null: mean={null.mean[m]:.4f} sd={null.std[m]:.4f} "
              f"p50={q[100]:.4f} p95={q[190]:.4f} p99={q[198]:.4f}")

    if args.labels:
        import yaml
        with open(args.labels) as fh:
            spec = yaml.safe_load(fh)
        # A label is attached to a FILE, but the index holds one signature per
        # WINDOW. Expanding each labelled file pair into every cross-window pair
        # is both more data and more honest: it measures the property the label
        # actually asserts ("these two recordings show the same place/work"),
        # instead of whichever single window happened to be looked up first.
        by_file = collections.defaultdict(list)
        for s_ in sigs:
            by_file[os.path.basename(s_.path)].append(s_)
        rows = []
        for pair in spec.get("pairs", []):
            A, B = by_file.get(pair["a"], []), by_file.get(pair["b"], [])
            if not A or not B:
                _eprint(f"  label pair skipped (not indexed): {pair['a']} / {pair['b']}")
                continue
            for x in A:
                for y in B:
                    rows.append((pair, raw_scores(x, y, whitener=null.whitener)))
        if rows:
            print(f"\nlabelled pairs: {len(rows)}")
            fitted = {}
            for axis, key, comps, field in (
                ("environment", "env_raw", ENV_COMPONENTS, "same_env"),
                ("task", "task_raw", TASK_COMPONENTS, "same_task"),
            ):
                lab = np.array([1 if p.get(field) else 0 for p, _ in rows])
                if len(np.unique(lab)) < 2:
                    print(f"  {axis:11s}: need both positive and negative pairs to evaluate")
                    continue
                w = fit_weights([(r, int(l)) for (_, r), l in zip(rows, lab)], axis, comps)
                if w and not args.no_fit_weights:
                    fitted["env" if axis == "environment" else "task"] = w
                    print(f"  {axis:11s}: fitted weights " +
                          " ".join(f"{k}={v:.2f}" for k, v in w.items()))
                sc = np.array([null.percentile(key, r[key]) for _, r in rows])
                rep = evaluate_axis(sc, lab, axis, key)
                print(f"  {axis:11s}: AUC={rep.auc:.3f}  best-F1={rep.best_f1:.3f} "
                      f"@ percentile>={rep.best_threshold:.1f}  "
                      f"separation={rep.separation_sigma:.2f} sigma  "
                      f"(n+={rep.n_pos}, n-={rep.n_neg})")
            if fitted:
                null.weights = fitted
                # the null distribution must be re-fitted under the new fusion
                # weights, otherwise percentiles refer to a scale that no
                # longer exists.
                _eprint("refitting null under the fitted weights ...")
                from .calibrate import fit_null as _fn
                null2 = _fn(sigs, max_pairs=args.max_pairs)
                null2.weights = fitted
                idx.save_null(null2)
                print("saved fitted weights + refitted null")
    return 0


# ------------------------------------------------------------------------ compare
def _sig_for(path_or_id: str, idx: Optional[Index], cfg: Config, enc=None) -> Signature:
    if idx is not None:
        for s in idx.signatures():
            if s.segment_id == path_or_id or os.path.abspath(s.path) == os.path.abspath(path_or_id):
                return s
    sigs = build_signatures(path_or_id, cfg=cfg, appearance_encoder=enc)
    return sigs[0]


def cmd_compare(args) -> int:
    idx = Index(args.index) if args.index and os.path.exists(args.index) else None
    cfg = idx.config() if idx else _cfg(args)
    if getattr(args, "config", None):
        cfg = _cfg(args)
    enc = get_appearance_encoder(cfg.appearance.encoder, **cfg.appearance.encoder_kwargs)
    a = _sig_for(args.a, idx, cfg, enc)
    b = _sig_for(args.b, idx, cfg, enc)
    null = idx.null() if idx else None
    v = compare_sigs(a, b, null=null, decision=cfg.decision, hashing=cfg.hashing)
    if args.json:
        print(json.dumps(v.as_dict(), indent=2, default=float))
    else:
        print(v.explain())
    return 0


# ------------------------------------------------------------------------- search
def cmd_search(args) -> int:
    idx = Index(args.index)
    sigs = idx.signatures()
    if not sigs:
        _eprint("empty index")
        return 2
    cfg = idx.config()
    enc = get_appearance_encoder(cfg.appearance.encoder, **cfg.appearance.encoder_kwargs)
    q = _sig_for(args.video, idx, cfg, enc)
    null = idx.null()
    rows = []
    for s in sigs:
        if s.segment_id == q.segment_id:
            continue
        v = compare_sigs(q, s, null=null, decision=cfg.decision, hashing=cfg.hashing)
        rows.append(v)
    rows.sort(key=lambda v: -(0.5 * (v.env_score + v.task_score)))
    print(f"query: {q.label}   (null model: {'yes' if null else 'NO - scores are raw'})")
    print(f"{'env':>7} {'task':>7}  {'verdict':<20} clip")
    for v in rows[: args.k]:
        print(f"{v.env_score:7.2f} {v.task_score:7.2f}  {v.label:<20} {v.b}")
    return 0


# --------------------------------------------------------------------------- gate
def cmd_gate(args) -> int:
    idx = Index(args.index)
    sigs = idx.signatures()
    cfg = idx.config()
    enc = get_appearance_encoder(cfg.appearance.encoder, **cfg.appearance.encoder_kwargs)
    null = idx.null()
    cands = build_signatures(args.video, cfg=cfg, appearance_encoder=enc)
    rc = 0
    for q in cands:
        sims, labels = [], []
        wh = null.whitener if null is not None else None
        for s in sigs:
            r = raw_scores(q, s, whitener=wh, weights=(null.weights if null else None))
            if null is not None:
                env = null.percentile("env_raw", r["env_raw"]) / 100.0
                task = null.percentile("task_raw", r["task_raw"]) / 100.0
            else:
                env, task = (r["env_raw"] + 1) / 2, (r["task_raw"] + 1) / 2
            sims.append(0.5 * (env + task))
            labels.append(s.label)
        d = gate_fn(np.asarray(sims), labels, redundancy_threshold=args.threshold)
        print(f"{q.label}: {'ACCEPT' if d.accept else 'REJECT'}  novelty={d.novelty:.3f}  "
              f"nearest={d.nearest}  ({d.reason})")
        if not d.accept:
            rc = 1
    return rc


# ------------------------------------------------------------------------- select
def cmd_select(args) -> int:
    idx = Index(args.index)
    sigs = idx.signatures()
    if not sigs:
        _eprint("empty index")
        return 2
    S = idx.similarity_matrix(kind=args.kind)
    res = facility_location_greedy(S, budget=args.budget)
    frac = res.fraction_covered()
    print(f"facility-location selection over {len(sigs)} signature(s), kind={args.kind}")
    print(f"{'#':>3} {'gain':>8} {'cum.cov':>8}  clip")
    for k, i in enumerate(res.order, 1):
        print(f"{k:3d} {res.gains[k-1]:8.3f} {frac[k-1]*100:7.2f}%  {sigs[i].label}")
    knee = res.knee(tol=args.knee_tol)
    print(f"\nknee at {knee} clip(s): after this, each extra clip adds "
          f"< {args.knee_tol*100:.1f}% of total coverage.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"order": [sigs[i].label for i in res.order],
                       "gains": res.gains, "coverage_fraction": frac,
                       "knee": knee}, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------------- report
def cmd_report(args) -> int:
    from .report import write_report
    idx = Index(args.index)
    out = write_report(idx, args.out, kind=args.kind)
    print(f"wrote {out}")
    return 0


def cmd_encoders(_args) -> int:
    for name in list_appearance_encoders():
        print(name)
    return 0


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="novelty", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, index_default=".novelty"):
        sp.add_argument("--index", default=index_default, help="index directory")
        sp.add_argument("--config", help="YAML config (see configs/)")

    sp = sub.add_parser("index", help="build signatures")
    common(sp)
    sp.add_argument("videos", nargs="+", help="video files or directories")
    sp.add_argument("--encoder", help="override appearance encoder")
    sp.add_argument("--window", type=float, help="segment length in seconds (0 = whole file)")
    sp.add_argument("--hop", type=float, help="segment hop in seconds")
    sp.add_argument("--no-motion", action="store_true", help="skip tier 2 (much faster, much dumber)")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("calibrate", help="fit the corpus null model")
    common(sp)
    sp.add_argument("--max-pairs", type=int, default=2000)
    sp.add_argument("--labels", help="YAML of labelled pairs, e.g. eval/pairs.yaml")
    sp.add_argument("--no-fit-weights", action="store_true",
                    help="evaluate against labels but keep the default fusion weights")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("compare", help="compare two clips")
    common(sp, index_default=".novelty")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--encoder")
    sp.add_argument("--no-motion", action="store_true")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("search", help="nearest neighbours")
    common(sp)
    sp.add_argument("video")
    sp.add_argument("-k", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("gate", help="accept/reject a candidate clip")
    common(sp)
    sp.add_argument("video")
    sp.add_argument("--threshold", type=float, default=0.97,
                    help="reject when fused similarity to the nearest corpus clip exceeds this")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("select", help="diverse subset via facility location")
    common(sp)
    sp.add_argument("--budget", type=int, default=None)
    sp.add_argument("--kind", choices=("env", "task", "fused"), default="fused")
    sp.add_argument("--knee-tol", type=float, default=0.01)
    sp.add_argument("--out", help="write selection order as JSON")
    sp.set_defaults(func=cmd_select)

    sp = sub.add_parser("report", help="self-contained HTML report")
    common(sp)
    sp.add_argument("--out", default="novelty-report.html")
    sp.add_argument("--kind", choices=("env", "task", "fused"), default="fused")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("encoders", help="list appearance backbones")
    sp.set_defaults(func=cmd_encoders)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
