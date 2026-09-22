"""Reconcile Claude Code session directories after the repo path changes.

Claude Code stores sessions under `~/.claude/projects/<escaped-cwd>/`, where the
escaped name is the absolute working directory with both `/` and `_` replaced by
`-`. Clone the repo somewhere else and the same session resolves to a different
directory, so `claude --resume` from the new path sees nothing.

Renaming the directory to match the new path is the right fix, but it cannot be
done while a session is live: the running process is bound to the old name, so
it recreates that directory and starts a FRESH transcript under the same
filename. The history then sits in one directory and the tail in the other --
which is what this script repairs.

It is idempotent and append-only:

  * the canonical directory is the one matching the current working directory
  * for every other directory belonging to this repo, each `*.jsonl` is merged
    into its namesake under the canonical directory
  * lines are deduplicated on the `uuid` field (a resumed session replays a few
    lines, so straight concatenation would double them) and written in
    timestamp order
  * `memory/`, `MEMORY.md` and per-session subdirectories are copied across only
    where the canonical side does not already have them

Nothing is deleted by default. `--prune` removes a stray directory, and only
when nothing in it has been touched for `--quiet-minutes` (default 10) -- a
live session writes to an open file handle, and unlinking that file sends its
remaining output nowhere.

Usage:
    python scripts/merge_session_dirs.py            # report only
    python scripts/merge_session_dirs.py --apply    # merge
    python scripts/merge_session_dirs.py --apply --prune
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

PROJECTS = os.path.expanduser("~/.claude/projects")


def escaped(path: str) -> str:
    """The directory name Claude Code derives from a working directory.

    Both separators collapse to a dash. Getting only the slashes -- the obvious
    reading -- produces a name that looks right and matches nothing.
    """
    return path.replace("/", "-").replace("_", "-")


def repo_identity(cwd: str) -> str:
    """Last path component, used to recognise this repo's other spellings."""
    return os.path.basename(cwd.rstrip("/"))


def read_lines(path: str) -> List[str]:
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(line)
    return out


def key_of(line: str) -> Tuple[str, str]:
    """(uuid, timestamp) for dedup and ordering; falls back to the raw line."""
    try:
        d = json.loads(line)
    except Exception:                                              # noqa: BLE001
        return (line[:200], "")
    return (str(d.get("uuid") or line[:200]), str(d.get("timestamp") or ""))


def newest_mtime(root: str) -> float:
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
            except OSError:
                pass
    return newest


def merge_jsonl(src: str, dst: str, apply: bool) -> Tuple[int, int]:
    """Append lines of src missing from dst. Returns (added, skipped)."""
    src_lines = read_lines(src)
    dst_lines = read_lines(dst) if os.path.exists(dst) else []
    have = {key_of(l)[0] for l in dst_lines}
    new = [l for l in src_lines if key_of(l)[0] not in have]
    if new and apply:
        merged = dst_lines + new
        merged.sort(key=lambda l: key_of(l)[1])
        tmp = dst + ".merging"
        with open(tmp, "w") as fh:
            fh.write("\n".join(merged) + "\n")
        os.replace(tmp, dst)
    return len(new), len(src_lines) - len(new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cwd", default=os.getcwd(),
                    help="working directory whose session dir is canonical")
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--prune", action="store_true",
                    help="remove a stray dir once merged AND quiet")
    ap.add_argument("--quiet-minutes", type=float, default=10.0)
    args = ap.parse_args()

    if not os.path.isdir(PROJECTS):
        print(f"no {PROJECTS} — nothing to do")
        return 0

    canonical_name = escaped(os.path.abspath(args.cwd))
    canonical = os.path.join(PROJECTS, canonical_name)
    ident = repo_identity(os.path.abspath(args.cwd))

    strays = []
    for name in sorted(os.listdir(PROJECTS)):
        full = os.path.join(PROJECTS, name)
        if name == canonical_name or not os.path.isdir(full) or os.path.islink(full):
            continue
        if name.endswith("-" + ident):            # same repo, different path
            strays.append(name)

    print(f"canonical : {canonical_name}"
          f"{'' if os.path.isdir(canonical) else '   (does not exist yet)'}")
    if not strays:
        print("strays    : none — nothing to reconcile")
        return 0
    print(f"strays    : {', '.join(strays)}")
    if not args.apply:
        print("\n(dry run — pass --apply to merge)")

    os.makedirs(canonical, exist_ok=True)
    now = time.time()
    for name in strays:
        src_dir = os.path.join(PROJECTS, name)
        quiet_for = (now - newest_mtime(src_dir)) / 60.0
        print(f"\n{name}   last written {quiet_for:.1f} min ago")
        moved_any = False
        for entry in sorted(os.listdir(src_dir)):
            s = os.path.join(src_dir, entry)
            d = os.path.join(canonical, entry)
            if entry.endswith(".jsonl"):
                added, dup = merge_jsonl(s, d, args.apply)
                print(f"  {entry}: +{added} new, {dup} already present")
                moved_any = moved_any or added > 0
            elif os.path.isdir(s):
                if os.path.exists(d):
                    print(f"  {entry}/: canonical already has it, left alone")
                else:
                    print(f"  {entry}/: copying")
                    if args.apply:
                        shutil.copytree(s, d)
            else:
                if os.path.exists(d):
                    print(f"  {entry}: canonical already has it, left alone")
                else:
                    print(f"  {entry}: copying")
                    if args.apply:
                        shutil.copy2(s, d)
        if args.prune:
            if quiet_for < args.quiet_minutes:
                print(f"  NOT pruning: written {quiet_for:.1f} min ago, under "
                      f"--quiet-minutes {args.quiet_minutes}. A live session holds "
                      f"an open handle here; deleting now discards its remaining "
                      f"output. Re-run once that session has exited.")
            elif args.apply:
                shutil.rmtree(src_dir)
                print("  pruned")
            else:
                print("  would prune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
