"""VLM caption reading and TASK similarity between whole videos.

**Separate from the chunk-diversity pipeline on purpose.** `cycles.py` and
`scripts/run_v2.py` answer "how much variety is inside ONE video", cutting it
into chunks. This module and `scripts/match_videos.py` answer "are these TWO
videos the same work in the same place". Different question, different inputs,
deliberately no shared state -- mixing them was causing confusion.

Input is the captioning pipeline's per-segment parquet:

```
captioning/<collection>/<Industry>/<date>/<SITE>/<session>/<NNN>/segments/<seg>/
    captions.parquet.zst     one row per VLM segment (see FIELDS below)
    objects.parquet.zst      per-segment object inventory
    captions.summary.json    total_segments, unique_verbs, unique_stations, ...
```

Two things about those files that will waste your time otherwise:

* **`.zst` is a lie.** They are plain parquet (magic bytes `PAR1`) using
  parquet-internal compression. `pandas.read_parquet` opens them directly;
  running them through a zstd decompressor fails with "Unknown frame
  descriptor".
* **The VLM already segments by activity**, into variable-length spans (21 s,
  23 s observed) rather than a fixed grid. Those boundaries are *semantic* --
  they move when the work changes -- so they are better chunk boundaries than
  either a 30 s grid or hand-cadence cutting, and they come for free.

## Why the task axis comes from text and not from pixels

Every pixel-based attempt at task similarity in this project failed and the
measurements are in WORK.md: optical-flow rhythm separated tasks by 0.11 sigma,
and V-JEPA 2's embedding correlates **+0.55** with the appearance signal even
within a single session -- it was largely re-describing the room. It also
ordered whole industries wrongly, rating an automobile plant as more
task-similar to pipe-factory A than pipe-factory B was.

A VLM `verb` is a direct semantic read on the action, and it cannot leak
appearance the way a video encoder does.

## Match on structured fields, NOT on wording

The tempting approach -- "the VLM should produce the same wording for the same
job" -- does not hold. VLMs paraphrase across runs, so lexical or
sentence-embedding matching produces false negatives on identical work. The
structured fields (`verb`, `hand_*_action`, `object_name`) are a small, nearly
closed vocabulary and are far more stable, so they are the primary signal and
`caption` free text is only ever a tie-breaker.

## Duration weighting

Segments differ in length, so every distribution here is weighted by segment
duration: a verb occupying 60 s of a video counts more than one occupying 3 s.
Comparing raw counts would let a burst of short segments outvote the activity
that actually filled the recording.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Columns the captioning pipeline emits (stereo-pipeline-v2-captioning-3.0).
FIELDS = (
    "segment_id", "start_frame", "end_frame", "start_time", "end_time",
    "verb", "object_name", "object_color", "object_shape", "object_context",
    "caption", "hand_left_action", "hand_left_object",
    "hand_right_action", "hand_right_object", "station", "confidence",
    "model", "error",
)

#: Values that mean "nothing here" and must not be treated as a real token.
NULL_TOKENS = {"", "none", "n/a", "na", "null", "unknown", "nan", "-"}


def _norm(v) -> str:
    """Normalise a VLM string field to a comparable token."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in NULL_TOKENS:
        return ""
    # collapse whitespace and strip trailing punctuation the model sometimes adds
    return re.sub(r"\s+", " ", s).strip(" .,;:")


def _parse_time(v) -> Optional[float]:
    """'00:21.00' or '1:02:03.5' or a bare number of seconds -> seconds."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:                       # mm:ss.s, hh:mm:ss.s
        total = total * 60.0 + n
    return total


@dataclass
class CaptionSegment:
    """One VLM segment: an activity span, not a fixed-length chunk."""
    segment_id: int
    t0: float
    t1: float
    station: str = ""
    verb: str = ""
    object_name: str = ""
    caption: str = ""
    hand_left_action: str = ""
    hand_left_object: str = ""
    hand_right_action: str = ""
    hand_right_object: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        # Normalise here, not only in load_captions: a segment built directly
        # would otherwise keep raw values, and "none" (the VLM saying a hand is
        # empty) would be counted as an action called "none".
        for f in ("station", "verb", "object_name", "hand_left_action",
                  "hand_left_object", "hand_right_action", "hand_right_object"):
            setattr(self, f, _norm(getattr(self, f)))

    @property
    def seconds(self) -> float:
        return max(self.t1 - self.t0, 0.0)

    @property
    def actions(self) -> Tuple[str, ...]:
        """Every action token this segment asserts, deduplicated.

        `verb` is the segment-level action; the per-hand actions often add
        detail the top-level verb omits (a segment verbed 'walking' can still
        carry hand actions), so all three inform the task axis.
        """
        return tuple(dict.fromkeys(
            t for t in (self.verb, self.hand_left_action, self.hand_right_action) if t))

    @property
    def objects(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(
            t for t in (self.object_name, self.hand_left_object,
                        self.hand_right_object) if t))


@dataclass
class VideoCaptions:
    """All VLM segments for one video, plus duration-weighted summaries."""
    video_id: str
    segments: List[CaptionSegment] = field(default_factory=list)
    source: str = ""

    @property
    def duration(self) -> float:
        return sum(s.seconds for s in self.segments)

    def _weighted(self, key) -> Dict[str, float]:
        """token -> seconds spent on it, normalised to sum 1."""
        acc: Dict[str, float] = {}
        for s in self.segments:
            toks = key(s)
            if not toks:
                continue
            # split a segment's time evenly across the tokens it asserts, so a
            # segment naming two objects does not count double
            share = s.seconds / len(toks)
            for t in toks:
                acc[t] = acc.get(t, 0.0) + share
        total = sum(acc.values())
        return {k: v / total for k, v in acc.items()} if total > 0 else {}

    def action_profile(self) -> Dict[str, float]:
        return self._weighted(lambda s: s.actions)

    def object_profile(self) -> Dict[str, float]:
        return self._weighted(lambda s: s.objects)

    def station_profile(self) -> Dict[str, float]:
        return self._weighted(lambda s: (s.station,) if s.station else ())

    def dominant_station(self) -> str:
        p = self.station_profile()
        return max(p, key=p.get) if p else ""

    def mean_confidence(self) -> float:
        cs = [s.confidence for s in self.segments if s.confidence]
        return float(np.mean(cs)) if cs else 0.0


# ------------------------------------------------------------------ loading
def load_captions(captions_parquet: str, video_id: str = "") -> VideoCaptions:
    """Read one `captions.parquet(.zst)`.

    Despite the `.zst` suffix these are plain parquet; pandas handles the
    internal compression.
    """
    import pandas as pd

    df = pd.read_parquet(captions_parquet)
    segs: List[CaptionSegment] = []
    for i, row in df.iterrows():
        def g(c, default=""):
            return row[c] if c in df.columns else default

        if _norm(g("error")):
            continue                     # failed segment: no usable content
        t0 = _parse_time(g("start_time"))
        t1 = _parse_time(g("end_time"))
        if t0 is None or t1 is None or t1 <= t0:
            continue
        try:
            conf = float(g("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        segs.append(CaptionSegment(
            segment_id=int(g("segment_id", i) or i), t0=t0, t1=t1,
            station=_norm(g("station")), verb=_norm(g("verb")),
            object_name=_norm(g("object_name")),
            caption=str(g("caption") or "").strip(),
            hand_left_action=_norm(g("hand_left_action")),
            hand_left_object=_norm(g("hand_left_object")),
            hand_right_action=_norm(g("hand_right_action")),
            hand_right_object=_norm(g("hand_right_object")),
            confidence=conf,
        ))
    segs.sort(key=lambda s: s.t0)
    vid = video_id or _video_id_from_path(captions_parquet)
    return VideoCaptions(video_id=vid, segments=segs, source=captions_parquet)


def _video_id_from_path(path: str) -> str:
    """.../<Industry>/<date>/<SITE>/<session>/<NNN>/segments/<seg>/captions.parquet
    -> Industry__date__SITE__session__NNN__seg, matching the hand-detection and
    chunk_s3_path naming so records can be joined across pipelines.
    """
    p = os.path.abspath(path).replace(os.sep, "/").split("/")
    if "captioning" in p:
        parts = p[p.index("captioning") + 1:-1]
    else:
        parts = p[-8:-1]
    parts = [x for x in parts if x not in ("segments",)]
    return "__".join(parts)


def load_summary(summary_json: str) -> Dict:
    with open(summary_json) as fh:
        return json.load(fh)


# -------------------------------------------------------------- similarity
def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    u = np.array([a.get(k, 0.0) for k in keys])
    v = np.array([b.get(k, 0.0) for k in keys])
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / n) if n > 0 else 0.0


def _jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fit_idf(videos: Sequence[VideoCaptions], floor: float = 0.05
            ) -> Dict[str, float]:
    """Inverse document frequency over action and object tokens.

    This is the text-side equivalent of the corpus whitening in calibrate.py,
    and it exists for the same measured reason. Generic tokens -- "holding",
    "handling", "moving" -- appear in nearly every video, contribute nothing to
    telling jobs apart, and dominate a raw overlap score. Without IDF, a
    same-place-different-job pair scored *lower* (0.292) than an unrelated pair
    (0.315) purely because the unrelated video happened to have more distinct
    actions, diluting the shared "holding".

    Weight is ``log(N / df)``, floored so a token present in every video still
    counts for a little rather than vanishing.
    """
    n = max(len(videos), 1)
    df: Dict[str, int] = {}
    for v in videos:
        seen = set()
        for s in v.segments:
            seen.update(s.actions)
            seen.update(s.objects)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    return {t: max(float(np.log(n / c)), floor) for t, c in df.items()}


def _apply_idf(profile: Dict[str, float], idf: Optional[Dict[str, float]]
               ) -> Dict[str, float]:
    if not idf:
        return profile
    out = {k: v * idf.get(k, 1.0) for k, v in profile.items()}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else out


def task_similarity(a: VideoCaptions, b: VideoCaptions,
                    idf: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Task-axis similarity between two videos, from structured VLM fields.

    Four complementary measurements rather than one, because they fail
    differently:

    ``action_cos``      duration-weighted overlap of what was DONE. Sensitive to
                        how much time each action took.
    ``action_jaccard``  did the same actions occur at all, ignoring duration.
                        Catches "same job, different pace".
    ``object_cos``      duration-weighted overlap of what was HANDLED.
    ``object_jaccard``  set overlap of handled objects.

    ``task_raw`` weights actions above objects: the same objects can be used for
    different jobs (a box gets packed, stacked, or opened), whereas the action
    is the job. Objects break the tie and guard against a verb vocabulary that
    is too coarse ("handling" covers everything).

    Pass ``idf`` from :func:`fit_idf` whenever you have a corpus. Without it,
    ubiquitous tokens like "holding" dominate and the ordering can invert --
    see fit_idf for the measured case.

    Deliberately NOT included: caption free-text similarity. VLMs paraphrase the
    same work across runs, so text matching produces false negatives on
    identical jobs. Use ``caption_similarity`` separately and only as a
    tie-breaker.
    """
    aa = _apply_idf(a.action_profile(), idf)
    ba = _apply_idf(b.action_profile(), idf)
    ao = _apply_idf(a.object_profile(), idf)
    bo = _apply_idf(b.object_profile(), idf)
    out = {
        "action_cos": _cosine(aa, ba),
        "action_jaccard": _jaccard(aa, ba),
        "object_cos": _cosine(ao, bo),
        "object_jaccard": _jaccard(ao, bo),
    }
    out["task_raw"] = (0.40 * out["action_cos"] + 0.25 * out["action_jaccard"]
                       + 0.20 * out["object_cos"] + 0.15 * out["object_jaccard"])
    return out


def station_agreement(a: VideoCaptions, b: VideoCaptions) -> Dict[str, float]:
    """Text-level environment cross-check, independent of any image embedding.

    `station` is the VLM's own name for the workspace ("construction yard").
    It is NOT a replacement for the appearance embedding -- it is a second,
    independent opinion. When the embedding says "same place" and the stations
    agree, two unrelated channels concur. When they disagree, that is worth
    surfacing rather than averaging away, because one of them is wrong.
    """
    sa, sb = a.station_profile(), b.station_profile()
    return {
        "station_cos": _cosine(sa, sb),
        "station_jaccard": _jaccard(sa, sb),
        "same_dominant_station": float(
            bool(a.dominant_station()) and a.dominant_station() == b.dominant_station()),
    }


def caption_similarity(a: VideoCaptions, b: VideoCaptions,
                       embed=None) -> float:
    """Tie-breaker only. See the warning in ``task_similarity``.

    ``embed`` maps a list of strings to an (n, d) array; if omitted, falls back
    to token-overlap over the captions, which needs no model and no network --
    the same dependency-free-fallback principle as the ``gist`` encoder.
    """
    ta = [s.caption for s in a.segments if s.caption]
    tb = [s.caption for s in b.segments if s.caption]
    if not ta or not tb:
        return 0.0
    if embed is None:
        def bag(texts):
            acc: Dict[str, float] = {}
            for t in texts:
                for w in re.findall(r"[a-z]+", t.lower()):
                    if len(w) > 3:
                        acc[w] = acc.get(w, 0.0) + 1.0
            return acc
        return _cosine(bag(ta), bag(tb))
    ea, eb = np.asarray(embed(ta)), np.asarray(embed(tb))
    ea /= np.linalg.norm(ea, axis=1, keepdims=True) + 1e-8
    eb /= np.linalg.norm(eb, axis=1, keepdims=True) + 1e-8
    # best-match both ways, so a long video is not penalised for containing more
    S = ea @ eb.T
    return float(0.5 * (S.max(1).mean() + S.max(0).mean()))
