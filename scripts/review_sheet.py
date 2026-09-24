"""Join the customer QA workbook to the task 2 run, one row per episode.

The status report (`task2_status.py`) says *what* happened to each episode in
our terms -- untrackable, no_cadence, too_few_chunks, ok. It deliberately
carries almost no context, because it is keyed on our own paths. To decide
whether `untrackable` means "the hands really are out of frame" or "upstream
keypoints are bad", someone has to open the video and look, and for that they
need the customer's own columns next to our verdict: the QC flags, the reviewer
notes, the task description, the site.

So this writes our outcome columns first, then every column of the workbook's
'Rejected Episodes' sheet, plus `left_stream_s3` from the master list -- the
exact clip the pipeline read, which the sheet's `chunk_s3_path` does not give
(it stops at the take, without the `seg_NNN`).

The default cohort is the 143 episodes that actually ran: the 433 customer-
marked-repetitive ones, minus the 290 whose keypoints only exist at step 10
(3 fps), which never reached the detector. `--status` overrides it.

PRESIGNED LINKS. The sheet's '▶ Watch' and 'headpose_*' cells are hyperlinks
whose targets are presigned prod S3 URLs -- working credentials for anyone
holding them, IAM notwithstanding. The cell *text* is just the label, so the
default output carries no credential and is safe to move around. `--with-links`
resolves the real targets into extra columns; the result is a credential file,
must not be committed or shared, and is why the output name is gitignored.

Usage:
    python scripts/review_sheet.py --state ~/task2run/run143 --out review.csv
    python scripts/review_sheet.py ... --with-links     # adds watch URLs
    python scripts/review_sheet.py ... --status ok,too_few_chunks
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.rejected_ground_truth import (  # noqa: E402
    REPETITIVE, SHEET, WORKBOOK, M, R, Workbook,
)
from scripts.task2_status import classify, episode_key  # noqa: E402

#: The episodes that reached the detector. `coarse_rate` and `no_keypoints`
#: are excluded because there is nothing to look at: the run never saw them.
RAN = "untrackable,no_cadence,too_few_chunks,ok"

#: Headers the sheet leaves blank. They are free-text reviewer columns and
#: carry the only human judgement in the file, so they are named rather than
#: dropped -- 4 of the 966 rows have something in them and they are the rows
#: most worth reading.
BLANK_HEADERS = {5: "reviewer_note", 6: "reviewer_comment"}


def hyperlinks(wb: Workbook, sheet_name: str) -> dict:
    """{cell ref -> target URL} for one sheet.

    The target lives in the sheet's own rels part, not in the cell, which is
    why the cell value reads '▶ Watch' and not a URL.
    """
    path = wb.sheets[sheet_name]
    rels_path = os.path.join(os.path.dirname(path), "_rels",
                             os.path.basename(path) + ".rels").replace(os.sep, "/")
    if rels_path not in wb.z.namelist():
        return {}
    rels = {r.get("Id"): r.get("Target")
            for r in ET.fromstring(wb.z.read(rels_path))}
    out = {}
    for h in ET.fromstring(wb.z.read(path)).iter(M + "hyperlink"):
        target = rels.get(h.get(R + "id")) or h.get("location") or ""
        if target:
            out[h.get("ref")] = target
    return out


def col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True, help="dir holding segments.json")
    ap.add_argument("--workbook", default=WORKBOOK)
    ap.add_argument("--sheet", default=SHEET)
    ap.add_argument("--reason", default=REPETITIVE)
    ap.add_argument("--master", default=os.path.join(ROOT, "delivered_1000h_master_list.csv"))
    ap.add_argument("--status", default=RAN, help=f"comma-separated; default {RAN}")
    ap.add_argument("--with-links", action="store_true",
                    help="resolve presigned hyperlink targets (credentials)")
    ap.add_argument("--out", default="task2_review.csv")
    args = ap.parse_args()

    wb = Workbook(args.workbook)
    rows = wb.rows(args.sheet)
    header = [h or BLANK_HEADERS.get(i, f"col_{col_letter(i)}")
              for i, h in enumerate(rows[0])]
    uu = header.index("episode_uuid")
    links = hyperlinks(wb, args.sheet) if args.with_links else {}
    link_cols = [i for i, h in enumerate(header)
                 if any(f"{col_letter(i)}{n}" in links for n in range(2, len(rows) + 1))]

    # master list: episode_uuid -> the clip path, and through it our key
    by_uuid = {}
    for r in csv.DictReader(open(args.master)):
        by_uuid[r["episode_uuid"]] = r

    seg = json.load(open(os.path.join(args.state, "segments.json")))
    by_key = {episode_key(name): rec for name, rec in seg.items()}

    want = {s.strip() for s in args.status.split(",") if s.strip()}
    out_rows, unmatched = [], 0
    for n, row in enumerate(rows[1:], start=2):
        if (row[uu] if uu < len(row) else "").strip() != "" and \
                (row[header.index("reject_reason")] if header.index("reject_reason") < len(row) else "") != args.reason:
            continue
        uuid = row[uu].strip()
        if not uuid:
            continue
        m = by_uuid.get(uuid)
        if not m:
            unmatched += 1
            continue
        key = episode_key(m["left_stream_s3"])
        rec = by_key.get(key)
        status = classify(rec)
        if status not in want:
            continue
        rec = rec or {}
        out = {
            "status": status,
            "episode_key": key,
            "left_stream_s3": m["left_stream_s3"],
            "fps": rec.get("fps", ""),
            "step": rec.get("step", ""),
            "run_duration_s": round(float(rec.get("duration_s") or 0), 1),
            "hand": rec.get("hand", ""),
            "coverage_l": round(float(rec.get("coverage_l") or 0), 3),
            "coverage_r": round(float(rec.get("coverage_r") or 0), 3),
            "trackable": round(float(rec.get("trackable") or 0), 3),
            "cadence": round(float(rec.get("cadence") or 0), 4),
            "period_s": round(float(rec.get("period_s") or 0), 3),
            "n_chunks": len(rec.get("segments") or []),
        }
        for i, h in enumerate(header):
            out[h] = row[i] if i < len(row) else ""
        for i in link_cols:
            out[header[i] + "_url"] = links.get(f"{col_letter(i)}{n}", "")
        out_rows.append(out)

    if not out_rows:
        raise SystemExit("no rows matched")

    # Ordered for the person doing the looking, not for the machine: the
    # episodes that produced something first, then the ones that nearly did,
    # and inside each group the best-tracked first -- so an `untrackable` row
    # near the top with trackable 0.49 is the one that argues the floor is
    # wrong, and it is not buried 90 rows down.
    rank = {"ok": 0, "too_few_chunks": 1, "no_cadence": 2, "untrackable": 3,
            "coarse_rate": 4, "no_keypoints": 5}
    out_rows.sort(key=lambda r: (rank.get(r["status"], 9), -r["trackable"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    tally: dict = {}
    for r in out_rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"{len(out_rows)} rows, {len(out_rows[0])} columns -> {args.out}")
    for k in ("ok", "too_few_chunks", "no_cadence", "untrackable",
              "coarse_rate", "no_keypoints"):
        if k in tally:
            print(f"  {k:<16} {tally[k]:>5}")
    if unmatched:
        print(f"  {unmatched} sheet rows not in the master list")
    if args.with_links:
        print("  CONTAINS PRESIGNED PROD URLS -- do not commit or share")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
