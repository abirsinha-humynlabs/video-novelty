"""Coverage selection -- what you actually wanted instead of dedup.

The stated goal is "monotonous task data stops helping the model, so give me
variety". Threshold dedup is a bad proxy for that, in a specific way:

* It is **order dependent**. Whichever near-duplicate you happened to ingest
  first survives; the better-lit, longer, better-framed twin gets deleted.
* It is **locally greedy about the wrong thing**. Deleting everything within
  epsilon of a kept clip removes the rare clip that merely sits near a common
  one, while ten mediocre clips scattered across empty space all survive.
* It cannot answer the question you will actually be asked, which is not "is
  this a duplicate" but "which 200 of my 2000 hours should I label first".

The right objective is **facility location**, a monotone submodular function:

    f(S) = sum over all clips v of  max over selected s in S of  sim(v, s)

"how well is every clip in my corpus represented by something I selected".
Greedy maximisation is within (1 - 1/e) ~= 63% of the optimum -- a guarantee
threshold dedup does not have -- and the marginal gain of each pick falls
monotonically, which gives you the diminishing-returns curve that answers
"where should I stop collecting this task?" directly from data.

Dedup falls out for free: a clip whose marginal gain is ~0 is redundant.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SelectionResult:
    order: List[int]          # corpus indices, in the order greedy picked them
    gains: List[float]        # marginal gain contributed by each pick
    coverage: List[float]     # cumulative f(S) after each pick
    total: float              # f(V), the ceiling (everything selected)

    def fraction_covered(self) -> List[float]:
        return [c / self.total if self.total > 0 else 0.0 for c in self.coverage]

    def knee(self, tol: float = 0.01) -> int:
        """First pick after which each extra clip adds < ``tol`` of total coverage."""
        for k, g in enumerate(self.gains):
            if self.total > 0 and g / self.total < tol:
                return k
        return len(self.gains)


def facility_location_greedy(
    S: np.ndarray,
    budget: Optional[int] = None,
    *,
    preselected: Optional[Sequence[int]] = None,
) -> SelectionResult:
    """Lazy greedy maximisation of facility location over similarity matrix ``S``.

    ``S[i, j]`` = similarity of clip i to clip j, ideally in [0, 1] and
    symmetric, but neither is required. ``preselected`` lets you extend an
    existing dataset rather than re-choosing it.

    Lazy (accelerated) greedy: because f is submodular, a candidate's gain can
    only shrink as S grows, so a stale gain that still tops the heap after
    re-evaluation is provably the true best. In practice this turns O(n^2) per
    pick into something closer to O(n log n).
    """
    n = S.shape[0]
    if n == 0:
        return SelectionResult([], [], [], 0.0)
    budget = n if budget is None else min(budget, n)

    best = np.full(n, -np.inf)
    chosen: List[int] = []
    if preselected:
        for p in preselected:
            best = np.maximum(best, S[:, p])
            chosen.append(int(p))
    base = float(best.sum()) if np.isfinite(best).all() else 0.0
    if not np.isfinite(best).all():
        best = np.full(n, -np.inf)
        base = 0.0

    def gain_of(j: int) -> float:
        return float(np.maximum(best, S[:, j]).sum() - (best.sum() if np.isfinite(best).all() else 0.0))

    # initial gains (best is -inf everywhere when nothing is preselected)
    if np.isfinite(best).all():
        cur = float(best.sum())
        heap = [(-(float(np.maximum(best, S[:, j]).sum()) - cur), j) for j in range(n) if j not in chosen]
    else:
        heap = [(-float(S[:, j].sum()), j) for j in range(n)]
        cur = 0.0
        best = np.zeros(n)
    heapq.heapify(heap)

    order, gains, coverage = [], [], []
    selected = set(chosen)
    while heap and len(order) < budget:
        neg_g, j = heapq.heappop(heap)
        if j in selected:
            continue
        true_gain = float(np.maximum(best, S[:, j]).sum() - cur)
        if heap and true_gain < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-true_gain, j))     # stale: re-queue with the fresh value
            continue
        best = np.maximum(best, S[:, j])
        cur = float(best.sum())
        selected.add(j)
        order.append(j)
        gains.append(true_gain)
        coverage.append(cur)

    return SelectionResult(order=order, gains=gains, coverage=coverage,
                           total=float(S.max(axis=1).sum()))


def marginal_gain(S_corpus_to_new: np.ndarray, best_so_far: np.ndarray) -> float:
    """Gain from adding one new clip, given the corpus's current coverage vector.

    ``S_corpus_to_new[i]`` = similarity of corpus clip i to the new clip.
    ``best_so_far[i]``     = how well corpus clip i is already represented.
    """
    return float(np.maximum(best_so_far, S_corpus_to_new).sum() - best_so_far.sum())


@dataclass
class GateDecision:
    accept: bool
    reason: str
    max_similarity: float
    nearest: Optional[str]
    novelty: float            # 1 - max_similarity, in [0, 1] for similarity in [0, 1]


def gate(
    sims: np.ndarray,
    labels: Sequence[str],
    *,
    redundancy_threshold: float,
) -> GateDecision:
    """Accept/reject a candidate clip against an existing corpus.

    Deliberately simple and nearest-neighbour based, because at ingest time you
    usually do not have the corpus in RAM. For batch curation prefer
    :func:`facility_location_greedy`, which is strictly better informed.
    """
    if len(sims) == 0:
        return GateDecision(True, "corpus is empty", 0.0, None, 1.0)
    k = int(np.argmax(sims))
    mx = float(sims[k])
    accept = mx < redundancy_threshold
    reason = (
        f"nearest corpus clip is {mx:.3f} similar (threshold {redundancy_threshold:.3f})"
        f" -> {'novel enough' if accept else 'redundant'}"
    )
    return GateDecision(accept, reason, mx, labels[k], 1.0 - mx)
