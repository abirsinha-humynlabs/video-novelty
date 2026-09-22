"""Extract the customer's rejection labels from the Opeth QA workbook.

This is the ground truth for task 2: the customer rejected 766 delivered
episodes, and the reason string says which ones they considered repetitive.

    433  Repetitive motion or repeated simple work; Needs a shorter usable segment
    306  Idle or stalled task progress; Needs a shorter usable segment
     19  Working hands are not visible enough
      6  Idle ...; Working hands ...; Needs a shorter usable segment
      1  Repetitive motion ...; Idle or stalled task progress; Needs a shorter ...
    200  (blank)

The 433 exact matches are the positive class. Every one of them is also in
`delivered_1000h_master_list.csv`, so the delivered list is NOT a clean
negative set -- the rejected episodes are inside it and must be subtracted
before the remainder is used as negatives.

WHY A SCRIPT AND NOT A ONE-OFF: the workbook is the only copy of these labels.
It is not in S3 -- `novelty_data_v2` holds chunk media, not ground truth, and a
scan of all 188,711 keys under stage `labelling_results/` found nothing --
and it is gitignored because it carries per-episode customer data. So the
labels have to be re-derived from the sheet on every machine, and the parsing
has two traps worth encoding:

  * The sheet has five tabs; the labels are on 'Rejected Episodes' (966 rows),
    not on 'Sheet1' or 'Summary', which look plausible and are not it.
  * `reject_reason` is a semicolon-joined list, and the join order varies, so
    the exact string misses two rows that also say repetitive:

        --reason "..Needs a shorter usable segment"   exact      433
        --reason "Repetitive motion" --contains                  435

    The 435 is what `manifest_435.json` in the old handoff holds -- verified
    identical -- so that file was built with the loose reading. Which to use is
    a judgement: 433 is one clean reason, 435 includes two episodes the
    customer flagged for repetitiveness *and* something else. Exact is the
    default because a mixed reason is weaker evidence that repetition is what
    made it unusable.

Reading it without openpyxl is deliberate: an xlsx is a zip of XML, the shared
string table plus one sheet is all that is needed, and the dependency is not
worth it for one file.

Usage:
    python scripts/rejected_ground_truth.py --out repetitive_433.json
    python scripts/rejected_ground_truth.py --reasons        # just the tally
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKBOOK = os.path.join(ROOT, "Opeth Rejected Episodes.xlsx")
SHEET = "Rejected Episodes"
REPETITIVE = ("Repetitive motion or repeated simple work; "
              "Needs a shorter usable segment")

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class Workbook:
    """Minimal read-only xlsx reader: shared strings plus one sheet."""

    def __init__(self, path: str):
        self.z = zipfile.ZipFile(path)
        self.shared: List[str] = []
        if "xl/sharedStrings.xml" in self.z.namelist():
            root = ET.fromstring(self.z.read("xl/sharedStrings.xml"))
            self.shared = ["".join(t.text or "" for t in si.iter(M + "t"))
                           for si in root.iter(M + "si")]
        wb = ET.fromstring(self.z.read("xl/workbook.xml"))
        rels = {r.get("Id"): r.get("Target")
                for r in ET.fromstring(self.z.read("xl/_rels/workbook.xml.rels"))}
        self.sheets: Dict[str, str] = {}
        for s in wb.iter(M + "sheet"):
            target = (rels.get(s.get(R + "id")) or "").lstrip("/")
            target = target if target.startswith("xl/") else "xl/" + target
            self.sheets[s.get("name")] = target

    def _value(self, cell) -> str:
        v = cell.find(M + "v")
        if v is None or v.text is None:
            # an inline string carries its text under <is>, not <v>
            is_ = cell.find(M + "is")
            return "".join(t.text or "" for t in is_.iter(M + "t")) if is_ is not None else ""
        if cell.get("t") == "s" and v.text.isdigit():
            i = int(v.text)
            return self.shared[i] if i < len(self.shared) else ""
        return v.text

    def rows(self, sheet_name: str) -> List[List[str]]:
        """Rows as lists of strings, positioned by column letter.

        Empty cells are omitted from the XML entirely, so reading cells in
        document order silently shifts every value left of a gap into the wrong
        column. The cell reference (``A3``, ``AB7``) is the only reliable
        position, so it is decoded rather than trusted to ordering.
        """
        path = self.sheets.get(sheet_name)
        if not path or path not in self.z.namelist():
            raise SystemExit(f"sheet {sheet_name!r} not in workbook; have "
                             f"{sorted(self.sheets)}")
        out: List[List[str]] = []
        for row in ET.fromstring(self.z.read(path)).iter(M + "row"):
            cells: Dict[int, str] = {}
            for c in row:
                ref = c.get("r") or ""
                letters = re.match(r"([A-Z]+)", ref)
                if not letters:
                    continue
                idx = 0
                for ch in letters.group(1):
                    idx = idx * 26 + (ord(ch) - 64)
                cells[idx - 1] = self._value(c)
            if cells:
                out.append([cells.get(i, "") for i in range(max(cells) + 1)])
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workbook", default=WORKBOOK)
    ap.add_argument("--sheet", default=SHEET)
    ap.add_argument("--reason", default=REPETITIVE)
    ap.add_argument("--contains", action="store_true",
                    help="substring match instead of exact; also catches the "
                         "combined Repetitive+Idle rows")
    ap.add_argument("--reasons", action="store_true",
                    help="print the reason tally and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.workbook):
        raise SystemExit(f"{args.workbook} not found. It is gitignored (customer "
                         f"data) and is the only copy of these labels -- get it "
                         f"from the QA spreadsheet.")

    rows = Workbook(args.workbook).rows(args.sheet)
    header = rows[0]
    idx = {h: i for i, h in enumerate(header) if h}
    for need in ("episode_uuid", "reject_reason"):
        if need not in idx:
            raise SystemExit(f"column {need!r} not on sheet {args.sheet!r}; "
                             f"have {[h for h in header if h][:12]}")

    def cell(row: List[str], name: str) -> str:
        i = idx[name]
        return row[i].strip() if i < len(row) else ""

    body = rows[1:]
    if args.reasons:
        tally = collections.Counter(cell(r, "reject_reason") for r in body)
        print(f"{len(body)} rows on {args.sheet!r}")
        for reason, n in tally.most_common():
            print(f"  {n:>5}  {reason or '(blank)'}")
        return 0

    want = args.reason
    picked = []
    for r in body:
        reason = cell(r, "reject_reason")
        hit = (want in reason) if args.contains else (reason == want)
        if hit and cell(r, "episode_uuid"):
            picked.append({
                "episode_uuid": cell(r, "episode_uuid"),
                "reject_reason": reason,
                "chunk_s3_path": cell(r, "chunk_s3_path") if "chunk_s3_path" in idx else "",
            })

    print(f"{len(picked)} episodes match "
          f"({'contains' if args.contains else 'exact'}): {want!r}")
    missing = sum(1 for p in picked if not p["chunk_s3_path"])
    if missing:
        print(f"  {missing} of them have no chunk_s3_path")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(picked, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
