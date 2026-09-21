"""Turning raw distances into a decision you can defend.

The output of a comparison is NOT a number. It is a 2-D point -- (environment
similarity, task similarity) -- plus a separate, non-semantic duplicate flag.
Those three facts support four genuinely different curation actions:

                       task similar        task different
    env similar    |  REDUNDANT          |  SAME_PLACE_NEW_TASK   <- keep!
    env different  |  SAME_TASK_NEW_PLACE|  NOVEL                 <- keep!
                       ^ keep!

Collapsing this to one scalar destroys the two "keep" quadrants, which are
precisely the samples a physical-AI dataset is short of. That is the concrete
cost of the "one embedding, one cosine, one threshold" design.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

from .distance import (
    chamfer, cosine, dtw_cost, gaussian_bhattacharyya, period_agreement, spectrum_cosine,
)

#: every raw measurement we record for a pair. Calibration builds a null
#: distribution for each of these, so adding one here makes it automatically
#: available as a calibrated percentile.
RAW_METRICS = (
    "env_cos", "env_chamfer", "env_bhat_sim",
    "task_cos", "task_dtw_sim", "rhythm_cos", "period_agree",
    "env_raw", "task_raw",
)

#: weights for the two fused scores. These are the ONLY hand-set numbers in the
#: pipeline; `novelty calibrate --labels ...` replaces them with a logistic fit
#: whenever you give it labelled pairs.
ENV_WEIGHTS = {"env_chamfer": 0.50, "env_cos": 0.35, "env_bhat_sim": 0.15}
TASK_WEIGHTS = {"task_cos": 0.40, "task_dtw_sim": 0.30, "rhythm_cos": 0.20, "period_agree": 0.10}


class Label:
    DUPLICATE_SOURCE = "DUPLICATE_SOURCE"
    REDUNDANT = "REDUNDANT"
    SAME_PLACE_NEW_TASK = "SAME_PLACE_NEW_TASK"
    SAME_TASK_NEW_PLACE = "SAME_TASK_NEW_PLACE"
    NOVEL = "NOVEL"

    KEEP = {SAME_PLACE_NEW_TASK, SAME_TASK_NEW_PLACE, NOVEL}
    DROP = {DUPLICATE_SOURCE, REDUNDANT}


@dataclass
class Verdict:
    a: str
    b: str
    label: str
    env_score: float          # calibrated percentile 0-100 if a null model was given,
    task_score: float         # else the raw fused score scaled to 0-100
    calibrated: bool
    raw: Dict[str, float] = field(default_factory=dict)
    duplicate: Dict[str, float] = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def keep(self) -> bool:
        return self.label in Label.KEEP

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def explain(self) -> str:
        lines = [
            f"{self.a}  vs  {self.b}",
            f"  verdict      : {self.label}   ({'keep' if self.keep else 'drop'})",
            f"  environment  : {self.env_score:6.2f}  {'(percentile vs corpus null)' if self.calibrated else '(RAW - uncalibrated)'}",
            f"  task         : {self.task_score:6.2f}  {'(percentile vs corpus null)' if self.calibrated else '(RAW - uncalibrated)'}",
        ]
        if self.duplicate.get("overlap_seconds", 0) > 0:
            lines.append(
                f"  shared source: {self.duplicate['overlap_seconds']:.1f}s "
                f"at offset {self.duplicate['offset_seconds']:+.1f}s"
            )
        lines.append("  raw          : " + "  ".join(f"{k}={v:.4f}" for k, v in self.raw.items()))
        for n in self.notes:
            lines.append(f"  note         : {n}")
        return "\n".join(lines)


def _weighted(raw: Dict[str, float], weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    return float(sum(raw.get(k, 0.0) * w for k, w in weights.items()) / total)


def fuse_axes(raw: Dict[str, float], weights=None) -> Dict[str, float]:
    """(Re)compute env_raw / task_raw from components with the given weights."""
    weights = weights or {}
    env_w = weights.get("env") or ENV_WEIGHTS
    task_w = weights.get("task") or TASK_WEIGHTS
    return {"env_raw": _weighted(raw, env_w), "task_raw": _weighted(raw, task_w)}


def raw_scores(a, b, *, dtw_band: float = 0.25, whitener=None, weights=None) -> Dict[str, float]:
    """All pairwise measurements between two Signatures, before score calibration.

    ``whitener`` (a :class:`novelty.calibrate.Whitener`) standardises features
    against the corpus before any dot product. Pass it whenever you have one --
    without it every score in a homogeneous corpus sits at 0.95+ and carries
    almost no information. ``NullModel.whitener`` gives you the fitted one.
    """
    out: Dict[str, float] = {}

    def W(name, X):
        return whitener.apply(name, X) if (whitener is not None and X is not None) else X

    a_mean, b_mean = W("app", a.app_mean), W("app", b.app_mean)
    a_fr, b_fr = W("app", a.app_frames), W("app", b.app_frames)

    # ---- tier 1: environment ------------------------------------------
    out["env_cos"] = cosine(a_mean, b_mean)
    out["env_chamfer"] = chamfer(a_fr, b_fr)
    bhat = gaussian_bhattacharyya(a.app_mean, a.app_var, b.app_mean, b.app_var)
    out["env_bhat_sim"] = float(np.exp(-bhat))          # distance -> similarity in (0, 1]

    # ---- tier 2: task -------------------------------------------------
    if a.motion_pooled is not None and b.motion_pooled is not None:
        out["task_cos"] = cosine(W("motion_pooled", a.motion_pooled),
                                 W("motion_pooled", b.motion_pooled))
        out["task_dtw_sim"] = 1.0 - dtw_cost(W("motion_seq", a.motion_seq),
                                             W("motion_seq", b.motion_seq),
                                             band_frac=dtw_band)
        ra, rb = np.sqrt(np.maximum(a.rhythm, 0.0)), np.sqrt(np.maximum(b.rhythm, 0.0))
        out["rhythm_cos"] = cosine(W("rhythm", ra), W("rhythm", rb))
        out["period_agree"] = period_agreement(
            a.period_s, a.period_strength, b.period_s, b.period_strength
        )
    else:
        out.update(task_cos=0.0, task_dtw_sim=0.0, rhythm_cos=0.0, period_agree=0.0)

    out.update(fuse_axes(out, weights))
    return out


def duplicate_scores(a, b, *, max_hamming: int = 8) -> Dict[str, float]:
    from ..encoders.hashing import source_overlap
    if a.phash is None or b.phash is None:
        return {"overlap_seconds": 0.0, "offset_seconds": 0.0, "matched_frames": 0.0,
                "fraction_of_a": 0.0, "fraction_of_b": 0.0}
    return source_overlap(a.phash, a.phash_ts, b.phash, b.phash_ts,
                          max_hamming=max_hamming).as_dict()


def decide(env: float, task: float, decision) -> str:
    """Percentiles -> label, the one place the decision rule lives.

    With ``use_task_axis`` off this collapses to environment alone, because a
    task axis that orders whole industries wrongly cannot be allowed to name a
    quadrant. See config.DecisionConfig for the measurements behind that.
    """
    same_env = env >= decision.env_percentile
    if not getattr(decision, "use_task_axis", False):
        return Label.REDUNDANT if same_env else Label.NOVEL
    same_task = task >= decision.task_percentile
    return (
        Label.REDUNDANT if (same_env and same_task) else
        Label.SAME_PLACE_NEW_TASK if (same_env and not same_task) else
        Label.SAME_TASK_NEW_PLACE if (same_task and not same_env) else
        Label.NOVEL
    )


def compare(a, b, *, null=None, decision=None, hashing=None) -> Verdict:
    """Full comparison of two Signatures.

    ``null`` is a :class:`novelty.calibrate.NullModel`. Without it the scores
    are raw and the verdict is a guess -- the function says so loudly via
    ``Verdict.calibrated`` and a note. Do not ship uncalibrated verdicts.
    """
    from ..config import DecisionConfig, HashConfig
    decision = decision or DecisionConfig()
    hashing = hashing or HashConfig()

    raw = raw_scores(a, b,
                     whitener=(null.whitener if null is not None else None),
                     weights=(null.weights if null is not None else None))
    dup = duplicate_scores(a, b, max_hamming=hashing.max_hamming) if hashing.enabled else {}
    notes = []

    if null is not None:
        env = null.percentile("env_raw", raw["env_raw"])
        task = null.percentile("task_raw", raw["task_raw"])
        calibrated = True
    else:
        env = 100.0 * (raw["env_raw"] + 1.0) / 2.0
        task = 100.0 * (raw["task_raw"] + 1.0) / 2.0
        calibrated = False
        notes.append("no null model: scores are RAW, thresholds are meaningless. "
                     "Run `novelty calibrate` first.")

    frac = max(dup.get("fraction_of_a", 0.0), dup.get("fraction_of_b", 0.0))
    if (dup.get("overlap_seconds", 0.0) >= hashing.min_overlap_seconds
            and frac >= getattr(hashing, "min_overlap_fraction", 0.0)):
        label = Label.DUPLICATE_SOURCE
    else:
        label = decide(env, task, decision)
        if dup.get("overlap_seconds", 0.0) >= hashing.min_overlap_seconds:
            notes.append(
                f"tier 0 found {dup['overlap_seconds']:.1f}s of matching frames "
                f"({frac:.0%} of a clip) but that is below min_overlap_fraction "
                f"-- most likely repeated action, not shared footage")

    if a.motion_pooled is None or b.motion_pooled is None:
        notes.append("motion tier disabled: task score is not meaningful.")
    if not getattr(decision, "use_task_axis", False):
        notes.append("task axis reported but NOT gating the verdict "
                     "(decision.use_task_axis=false) -- see config.DecisionConfig")

    return Verdict(a=a.label, b=b.label, label=label, env_score=float(env),
                   task_score=float(task), calibrated=calibrated,
                   raw=raw, duplicate=dup, notes=notes)


def compare_windows(A, B, *, null=None, decision=None, hashing=None) -> Verdict:
    """Compare two *files* by every cross-window pair, not one window each.

    A 60 s file windowed at 30 s/15 s is four signatures. Scoring one window
    against one window throws away fifteen sixteenths of the evidence and swings
    the percentile by tens of points -- on this repo's own footage, a chunk pair
    that sits at the 71st percentile aggregated reads 49th on the single pair
    that happened to be first. That is the same failure that made file-level
    labels read AUC 0.222 before `calibrate --labels` was taught to expand them.

    Percentiles are averaged and the label re-derived from the average, so the
    verdict describes the files. ``DUPLICATE_SOURCE`` is the exception: shared
    footage anywhere in either file makes the pair a duplicate, so it wins
    whenever any window pair reports it.
    """
    from ..config import DecisionConfig
    decision = decision or DecisionConfig()
    pairs = [compare(a, b, null=null, decision=decision, hashing=hashing)
             for a in A for b in B]
    if len(pairs) == 1:
        return pairs[0]

    env = float(np.mean([v.env_score for v in pairs]))
    task = float(np.mean([v.task_score for v in pairs]))
    raw = {k: float(np.mean([v.raw[k] for v in pairs])) for k in pairs[0].raw}
    dup = max((v for v in pairs), key=lambda v: v.duplicate.get("overlap_seconds", 0.0)).duplicate

    if any(v.label == Label.DUPLICATE_SOURCE for v in pairs):
        label = Label.DUPLICATE_SOURCE
    else:
        label = decide(env, task, decision)

    notes = list(dict.fromkeys(n for v in pairs for n in v.notes))
    env_sd = float(np.std([v.env_score for v in pairs]))
    notes.append(f"aggregated over {len(pairs)} window pairs "
                 f"({len(A)}x{len(B)}); env percentile sd={env_sd:.1f}")
    return Verdict(a=A[0].label, b=B[0].label, label=label, env_score=env,
                   task_score=task, calibrated=pairs[0].calibrated,
                   raw=raw, duplicate=dup, notes=notes)
