import numpy as np
import pytest

from novelty.metrics.distance import (
    chamfer, cosine, dtw_cost, gaussian_bhattacharyya, period_agreement, spectrum_cosine,
)


def _unit(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_cosine_bounds():
    a = np.array([1.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, -a) == pytest.approx(-1.0)
    assert cosine(a, np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_chamfer_is_symmetric_and_ordered():
    A = _unit(20, 8, 0)
    B = np.concatenate([A[10:], _unit(5, 8, 1)])
    C = _unit(15, 8, 2)
    assert chamfer(A, B) == pytest.approx(chamfer(B, A))
    assert chamfer(A, A) == pytest.approx(1.0, abs=1e-5)
    assert chamfer(A, B) > chamfer(A, C)


def test_chamfer_beats_mean_cosine_on_partial_overlap():
    """The 'cut a long video in half' case, in miniature.

    Two halves of one trajectory have DIFFERENT mean vectors (the trajectory
    moved) but every frame of each still has a close counterpart in the other.
    Mean-cosine understates the relationship; chamfer does not.
    """
    t = np.linspace(0, 1, 40)[:, None]
    traj = np.concatenate([np.cos(t * 3), np.sin(t * 3), t], axis=1)
    traj = traj / np.linalg.norm(traj, axis=1, keepdims=True)
    A, B = traj[:22], traj[18:]
    assert chamfer(A, B) > cosine(A.mean(0), B.mean(0))


def test_chamfer_handles_empty():
    assert chamfer(np.zeros((0, 4)), _unit(3, 4)) == 0.0


def test_bhattacharyya_zero_for_identical():
    mu, var = np.zeros(6), np.ones(6)
    assert gaussian_bhattacharyya(mu, var, mu, var) == pytest.approx(0.0, abs=1e-9)


def test_bhattacharyya_detects_spread_difference():
    """Same centre, different spread -- invisible to cosine, visible here."""
    mu = np.zeros(6)
    assert gaussian_bhattacharyya(mu, np.ones(6), mu, np.ones(6) * 9) > 0.1


def test_dtw_tolerates_time_warp_but_not_reordering():
    A = _unit(30, 6, 3)
    slow = np.repeat(A, 2, axis=0)[:40]        # same sequence, ~2x slower
    shuffled = A[np.random.default_rng(0).permutation(len(A))]
    assert dtw_cost(A, slow) < dtw_cost(A, shuffled)


def test_dtw_is_length_normalised():
    A = _unit(20, 5, 4)
    B = _unit(60, 5, 5)
    c = dtw_cost(A, B)
    assert 0.0 <= c <= 1.5


def test_spectrum_cosine_on_distributions():
    p = np.zeros(24); p[5] = 1.0
    q = np.zeros(24); q[5] = 1.0
    r = np.zeros(24); r[18] = 1.0
    assert spectrum_cosine(p, q) == pytest.approx(1.0, abs=1e-6)
    assert spectrum_cosine(p, r) == pytest.approx(0.0, abs=1e-6)


def test_period_agreement_rewards_harmonics_not_noise():
    assert period_agreement(4.0, 0.8, 4.05, 0.8) > 0.7
    assert period_agreement(4.0, 0.8, 8.0, 0.8) > 0.5      # 2x harmonic still same task
    assert period_agreement(4.0, 0.8, 6.3, 0.8) < 0.2      # unrelated cadence
    assert period_agreement(0.0, 0.0, 4.0, 0.8) == 0.0     # no cycle -> no evidence
