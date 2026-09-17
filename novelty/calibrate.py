"""Calibration -- the step that turns a number into a measurement.

The problem this solves, stated plainly:

    Every clip shot in the same factory will score >0.9 cosine against every
    other clip from that factory. The metric saturates. A threshold of 0.95
    then encodes nothing about your data -- only about how the backbone
    happens to compress grey plastic and human hands. Move to a different
    site and the "right" threshold moves with it, silently.

The fix is to stop reading raw scores and start reading them *relative to a
null distribution built from your own corpus*: sample random pairs, record
every metric, and thereafter report where a new pair falls in that
distribution. "99.4th percentile" means "closer than 99.4% of random pairs in
this dataset" and it transfers across sites, backbones and resolutions.

If you also hand it labelled pairs (``eval/pairs.yaml``), it fits a logistic
model per axis and reports the achievable separation, so you find out whether
your features can do the job *before* you trust them on a million clips.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics.fuse import RAW_METRICS, raw_scores

QUANTILE_GRID = np.linspace(0.0, 100.0, 201)

WHITEN_FIELDS = ("app", "motion_pooled", "motion_seq", "rhythm")


@dataclass
class Whitener:
    """Per-dimension standardisation fitted on YOUR corpus.

    The second half of the anti-saturation story. Percentile calibration fixes
    the *scores*; whitening fixes the *features* they are computed from.

    Any embedding of a homogeneous corpus has a large shared component -- the
    "this is a factory, shot on a head camera, in daylight" direction that
    every single clip has in common. That component contributes nothing to
    telling clips apart but dominates the dot product, which is why raw
    cosines pile up at 0.95+. Subtracting the corpus mean and dividing by the
    per-dimension standard deviation removes exactly that shared component and
    rescales each dimension to contribute in proportion to how much it actually
    varies in your data.

    This is the same trick that makes centred CLIP retrieval work, and it costs
    one pass over the signatures.
    """
    stats: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.stats

    def apply(self, name: str, X: np.ndarray, renorm: bool = True) -> np.ndarray:
        st = self.stats.get(name)
        if st is None or X is None:
            return X
        mu = np.asarray(st["mu"], np.float32)
        sd = np.asarray(st["sd"], np.float32)
        if X.shape[-1] != mu.shape[0]:
            return X
        Z = (X - mu) / sd
        if renorm:
            Z = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-8)
        return Z.astype(np.float32)

    def to_dict(self):
        return {"stats": self.stats}


def _shrunk_sd(X: np.ndarray, shrink_k: float = 40.0, eps_floor: float = 1e-3) -> np.ndarray:
    """Per-dimension std with shrinkage toward the average variance.

    With few samples relative to dimensions, some dimensions have a near-zero
    sample std purely by luck; dividing by it explodes that dimension and the
    whitened cosine becomes noise. Shrinking toward the mean variance with
    weight n/(n+k) is the cheap diagonal analogue of Ledoit-Wolf and makes the
    whitener safe to fit on a small index while converging to the plain
    estimate as the corpus grows.
    """
    n = len(X)
    var = X.var(0)
    alpha = n / (n + shrink_k)
    var = alpha * var + (1.0 - alpha) * float(np.mean(var))
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.maximum(sd, max(eps_floor, float(np.median(sd)) * 1e-2))


def fit_whitener(signatures: Sequence, *, eps_floor: float = 1e-3,
                 shrink_k: float = 40.0) -> Whitener:
    """One pass over the corpus -> mean/std for every whitenable feature block."""
    stats: Dict[str, Dict[str, List[float]]] = {}

    frames = [s.app_frames for s in signatures if s.app_frames is not None and len(s.app_frames)]
    if frames:
        X = np.concatenate(frames, 0).astype(np.float64)
        stats["app"] = {"mu": X.mean(0).tolist(),
                        "sd": _shrunk_sd(X, shrink_k, eps_floor).tolist()}

    # per-frame motion descriptors live in their own space (131-d), separate
    # from the pooled clip descriptor -- DTW compares sequences, so it needs
    # its own whitener or the alignment cost stays pinned near a constant.
    seqs = [s.motion_seq for s in signatures if getattr(s, "motion_seq", None) is not None]
    if seqs:
        X = np.concatenate(seqs, 0).astype(np.float64)
        stats["motion_seq"] = {"mu": X.mean(0).tolist(),
                               "sd": _shrunk_sd(X, shrink_k, eps_floor).tolist()}

    vals = [s.motion_pooled for s in signatures if getattr(s, "motion_pooled", None) is not None]
    if len(vals) >= 2:
        X = np.stack(vals).astype(np.float64)
        stats["motion_pooled"] = {"mu": X.mean(0).tolist(),
                                  "sd": _shrunk_sd(X, shrink_k, eps_floor).tolist()}

    # Rhythm spectra are probability distributions, so we work in Hellinger
    # space (sqrt) and only CENTRE them -- rescaling per band would destroy the
    # distributional meaning. Centring asks the right question: "how does this
    # clip's rhythm differ from the average rhythm in this corpus", which is
    # what discriminates tasks inside one factory.
    rvals = [np.sqrt(np.maximum(s.rhythm, 0.0)) for s in signatures
             if getattr(s, "rhythm", None) is not None]
    if len(rvals) >= 2:
        X = np.stack(rvals).astype(np.float64)
        stats["rhythm"] = {"mu": X.mean(0).tolist(), "sd": np.ones(X.shape[1]).tolist()}
    return Whitener(stats=stats)


@dataclass
class NullModel:
    """Empirical distribution of every raw metric over random corpus pairs."""
    n_pairs: int
    quantiles: Dict[str, List[float]] = field(default_factory=dict)   # metric -> 201 values
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    encoder: str = "?"
    whiten: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def whitener(self) -> Whitener:
        return Whitener(stats=self.whiten)

    # -- queries ---------------------------------------------------------
    def percentile(self, metric: str, value: float) -> float:
        """Where ``value`` falls in the null, 0-100. Monotone, bounded, comparable."""
        q = self.quantiles.get(metric)
        if not q:
            return float("nan")
        return float(np.interp(value, np.asarray(q), QUANTILE_GRID,
                               left=0.0, right=100.0))

    def z(self, metric: str, value: float) -> float:
        """How many corpus-null standard deviations above chance this pair is."""
        s = self.std.get(metric, 0.0)
        if s <= 0:
            return float("nan")
        return float((value - self.mean[metric]) / s)

    # -- io ---------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"n_pairs": self.n_pairs, "quantiles": self.quantiles,
                       "mean": self.mean, "std": self.std, "encoder": self.encoder,
                       "whiten": self.whiten, "weights": self.weights}, fh)

    @classmethod
    def load(cls, path: str) -> "NullModel":
        with open(path) as fh:
            return cls(**json.load(fh))


def fit_null(signatures: Sequence, *, max_pairs: int = 2000, seed: int = 0,
             exclude_same_video: bool = True, progress=None) -> NullModel:
    """Build the null distribution from random pairs of corpus signatures.

    ``exclude_same_video=True`` is important and easy to get wrong: two windows
    of the same recording are not a "random pair", and letting them into the
    null inflates it, which then makes genuinely similar pairs look ordinary.
    """
    n = len(signatures)
    if n < 2:
        raise ValueError("need at least 2 signatures to fit a null model")
    rng = random.Random(seed)
    all_pairs = list(itertools.combinations(range(n), 2))
    if exclude_same_video:
        filtered = [(i, j) for i, j in all_pairs
                    if signatures[i].video_id != signatures[j].video_id]
        if len(filtered) >= 8:
            all_pairs = filtered
    rng.shuffle(all_pairs)
    pairs = all_pairs[:max_pairs]

    # Pass 1: feature-space whitening, fitted on the signatures themselves.
    # Must happen before the pair loop, because every raw score is computed in
    # the whitened space.
    wh = fit_whitener(signatures)

    acc: Dict[str, List[float]] = {m: [] for m in RAW_METRICS}
    for k, (i, j) in enumerate(pairs):
        r = raw_scores(signatures[i], signatures[j], whitener=wh)
        for m in RAW_METRICS:
            acc[m].append(r.get(m, 0.0))
        if progress:
            progress(k + 1, len(pairs))

    quantiles, mean, std = {}, {}, {}
    for m, vals in acc.items():
        v = np.asarray(vals, np.float64)
        quantiles[m] = np.percentile(v, QUANTILE_GRID).tolist()
        mean[m] = float(v.mean())
        std[m] = float(v.std())
    enc = getattr(signatures[0], "app_encoder", "?")
    return NullModel(n_pairs=len(pairs), quantiles=quantiles, mean=mean, std=std,
                     encoder=enc, whiten=wh.stats)


# ---------------------------------------------------------------------------
# supervised threshold fitting
# ---------------------------------------------------------------------------

@dataclass
class AxisReport:
    axis: str
    metric: str
    auc: float
    best_threshold: float
    best_f1: float
    n_pos: int
    n_neg: int
    separation_sigma: float


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (== probability a random positive outranks a random negative)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_axis(scores: np.ndarray, labels: np.ndarray, axis: str, metric: str) -> AxisReport:
    """Sweep every candidate threshold and report the best F1 plus the AUC."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    best_f1, best_t = -1.0, float("nan")
    for t in np.unique(scores):
        pred = scores >= t
        tp = float(np.sum(pred & (labels == 1)))
        fp = float(np.sum(pred & (labels == 0)))
        fn = float(np.sum(~pred & (labels == 1)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    pooled = np.sqrt(0.5 * (pos.var() + neg.var())) if len(pos) and len(neg) else 0.0
    sep = float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else float("nan")
    return AxisReport(axis=axis, metric=metric, auc=_auc(pos, neg), best_threshold=best_t,
                      best_f1=best_f1, n_pos=int((labels == 1).sum()),
                      n_neg=int((labels == 0).sum()), separation_sigma=sep)


# ---------------------------------------------------------------------------
# supervised fusion weights
# ---------------------------------------------------------------------------

ENV_COMPONENTS = ("env_cos", "env_chamfer", "env_bhat_sim")
TASK_COMPONENTS = ("task_cos", "task_dtw_sim", "rhythm_cos", "period_agree")


def fit_weights(rows, axis: str, components) -> Optional[Dict[str, float]]:
    """Fit non-negative fusion weights for one axis from labelled pairs.

    ``rows`` is a list of ``(raw_scores_dict, label01)``. We fit a logistic
    regression and keep the positive part of the coefficients, renormalised to
    sum to 1. Negative coefficients are clipped to zero rather than kept: a
    component that *anti*-correlates with the label on a handful of pairs is
    almost always noise, and letting it subtract makes the fused score
    unstable on data you have not labelled.

    Returns ``None`` when there is not enough signal to fit -- in which case
    the hand-set defaults in ``metrics.fuse`` stand, and you should go label
    more pairs rather than trusting a fit from six examples.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:                                            # pragma: no cover
        return None
    X = np.array([[r.get(c, 0.0) for c in components] for r, _ in rows], float)
    y = np.array([lab for _, lab in rows], int)
    if len(np.unique(y)) < 2 or len(y) < 2 * len(components):
        return None
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
    w = np.clip(clf.coef_[0], 0, None)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    return {c: float(v) for c, v in zip(components, w)}
