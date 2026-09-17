import numpy as np
import pytest

from novelty.select import facility_location_greedy, gate, marginal_gain


def _blocks(sizes, within=0.95, between=0.10):
    n = sum(sizes)
    S = np.full((n, n), between, np.float32)
    o = 0
    for s in sizes:
        S[o:o + s, o:o + s] = within
        o += s
    np.fill_diagonal(S, 1.0)
    return S


def test_greedy_covers_every_cluster_first():
    """The property that makes this better than threshold dedup.

    With three tight clusters, the first three picks must come from three
    DIFFERENT clusters. Threshold dedup gives no such guarantee -- it depends
    on ingest order.
    """
    sizes = [6, 6, 6]
    S = _blocks(sizes)
    res = facility_location_greedy(S, budget=3)
    cluster = lambda i: 0 if i < 6 else (1 if i < 12 else 2)   # noqa: E731
    assert len({cluster(i) for i in res.order}) == 3


def test_gains_are_non_increasing():
    """Submodularity: marginal value can never rise as the selection grows."""
    S = _blocks([4, 4, 4, 4])
    res = facility_location_greedy(S)
    g = np.array(res.gains)
    assert np.all(np.diff(g) <= 1e-6)


def test_coverage_is_monotone_and_reaches_one():
    S = _blocks([3, 3])
    res = facility_location_greedy(S)
    frac = res.fraction_covered()
    assert np.all(np.diff(frac) >= -1e-9)
    assert frac[-1] == pytest.approx(1.0, abs=1e-6)


def test_knee_flags_saturation():
    S = _blocks([8, 8])
    res = facility_location_greedy(S)
    assert res.knee(tol=0.01) <= 4       # two clusters -> saturates almost immediately


def test_preselected_extends_an_existing_dataset():
    S = _blocks([5, 5, 5])
    res = facility_location_greedy(S, budget=2, preselected=[0])
    assert 0 not in res.order
    assert all(i >= 5 for i in res.order)   # never re-picks the covered cluster


def test_marginal_gain_is_tiny_for_a_covered_clip():
    """A clip inside an already-represented cluster adds almost nothing.

    Not exactly zero: the matrix has a 1.0 diagonal above the 0.95
    within-cluster similarity, so a clip always covers itself slightly better
    than its neighbour does. That residual is the honest answer, and the gap
    against an uncovered clip is what the gate keys on.
    """
    S = _blocks([4, 4])
    best = np.maximum(S[:, 0], S[:, 4])
    covered = marginal_gain(S[:, 1], best)
    fresh = marginal_gain(np.ones(8), best)
    assert covered < 0.1
    assert fresh > 5 * covered


def test_gate_accepts_novel_rejects_redundant():
    labels = [f"c{i}" for i in range(4)]
    assert gate(np.array([0.2, 0.3, 0.1, 0.25]), labels, redundancy_threshold=0.9).accept
    d = gate(np.array([0.2, 0.99, 0.1, 0.25]), labels, redundancy_threshold=0.9)
    assert not d.accept and d.nearest == "c1"


def test_gate_on_empty_corpus():
    assert gate(np.zeros(0), [], redundancy_threshold=0.9).accept


def test_empty_matrix():
    res = facility_location_greedy(np.zeros((0, 0)))
    assert res.order == [] and res.total == 0.0
