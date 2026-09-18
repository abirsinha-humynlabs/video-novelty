"""Ingest one recording end to end without leaving video on local disk.

    download -> verify -> chunk -> upload chunks to S3 -> index -> delete

The box runs near full and a single 10-minute 1080p recording is ~430 MB, with
another ~390 MB once chunked. Keeping either around is what fills the disk, and
neither needs to persist: the chunks live in S3, and everything downstream
(`compare`, `select`, `report_csv`) reads the *signatures* in the index, not the
video. Signatures are ~210 KB each, so a 36-chunk corpus costs ~8 MB locally.

    python scripts/ingest_session.py --url "<presigned url>" --index .novelty-2sess

The directory layout is derived from the URL's own key, so the S3 mirror matches
the source tree without anyone typing a path.

Add --keep-local only when you are actively debugging decode behaviour and need
the files; it defeats the entire point of this script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.report_csv import S3_DATA_PREFIX, check_s3_destination  # noqa: E402


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd[:4])}...\n{p.stderr.strip()[:800]}")
    return p.stdout


def key_from_url(url: str) -> str:
    """Path part of the S3 key, e.g. normalized/bitrobot/.../000/left_rectified.mp4."""
    path = urllib.parse.urlparse(url).path.lstrip("/")
    return urllib.parse.unquote(path)


def duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="presigned S3 URL of left_rectified.mp4")
    ap.add_argument("--index", default=".novelty-2sess")
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--end", type=float, default=None, help="default: last whole chunk")
    ap.add_argument("--chunk", type=float, default=30.0)
    ap.add_argument("--s3", default=S3_DATA_PREFIX)
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    if args.s3:
        check_s3_destination(args.s3)

    key = key_from_url(args.url)
    rel = os.path.dirname(key)                     # normalized/.../000
    work = tempfile.mkdtemp(prefix="novelty-ingest-")
    src = os.path.join(work, os.path.basename(key))
    # Chunks are staged under data/<rel>/chunks even though they are deleted
    # afterwards, because `index` records the file path INTO the signature and
    # that path is the only provenance a signature carries. Staging in the temp
    # dir stamps every signature with /tmp/novelty-ingest-XXXX/..., which is
    # both meaningless and gone by the time anything reads it -- session
    # grouping in report_csv then silently lumps unrelated recordings together.
    chunks = os.path.abspath(os.path.join("data", rel, "chunks"))
    os.makedirs(chunks, exist_ok=True)

    try:
        print(f"[1/5] downloading -> {src}")
        run(["curl", "-sS", "-f", "-o", src, args.url])
        size = os.path.getsize(src)

        print(f"[2/5] verifying decode ({size/1e6:.0f} MB)")
        # A truncated upload still reports a full duration from metadata -- only
        # a real decode pass catches it. This has bitten this project once.
        run(["ffmpeg", "-v", "error", "-i", src, "-map", "0:v", "-f", "null", "-"])
        dur = duration(src)
        end = args.end if args.end is not None else args.start + \
            int((dur - args.start) // args.chunk) * args.chunk
        n = int((end - args.start) // args.chunk)
        print(f"      duration {dur:.1f}s, cutting {n} x {args.chunk:g}s "
              f"from {args.start:g}s to {end:g}s")

        print(f"[3/5] chunking (stream copy)")
        for i in range(n):
            s = args.start + i * args.chunk
            name = f"chunk{i+1:02d}_{int(s):03d}-{int(s+args.chunk):03d}.mp4"
            # -c copy cuts on keyframes; these sources key every 1.0s so a
            # whole-second boundary is frame-exact and costs no re-encode.
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(s), "-i", src, "-t", str(args.chunk), "-an",
                 "-c", "copy", os.path.join(chunks, name)])
        os.remove(src)                              # source is not needed again

        if args.s3:
            dest = f"{args.s3.rstrip('/')}/{rel}/chunks"
            print(f"[4/5] uploading {n} chunks -> {dest}")
            run(["aws", "s3", "sync", chunks, dest, "--only-show-errors"])
        else:
            print("[4/5] skipping upload (--s3 '')")

        if not args.skip_index:
            print(f"[5/5] indexing into {args.index}")
            run([sys.executable, "-m", "novelty.cli", "index", chunks,
                 "--index", args.index, "--config", args.config,
                 "--window", str(args.chunk), "--hop", str(args.chunk)])
        else:
            print("[5/5] skipping index (--skip-index)")

        if args.keep_local:
            print(f"kept local copy at {chunks}")
        else:
            shutil.rmtree(chunks, ignore_errors=True)
            # prune now-empty parents, but never the data/ root itself
            d = os.path.dirname(chunks)
            while os.path.abspath(d) != os.path.abspath("data") and not os.listdir(d):
                os.rmdir(d)
                d = os.path.dirname(d)
        where = f"chunks are in S3 under {rel}/chunks" if args.s3 else "chunks were NOT uploaded"
        print(f"\ndone. video removed from local disk; {where}")
        print("NEXT: novelty calibrate --index "
              f"{args.index} && python scripts/report_csv.py --index {args.index}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
