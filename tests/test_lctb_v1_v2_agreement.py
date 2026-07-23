"""Regression guard: LCTB and LCTB_v2 must agree.

LCTB_v2 is a memory/speed optimization of LCTB, not a different
procedure. The existing test_lctb_v2.py is a smoke test only -- it
checks shapes and grid length but never compares against v1, so a
divergence could ship silently. This test closes that gap.
"""
import numpy as np
import pytest

from src.LCTB import lct_threshold_bootstrap as v1
from src.LCTB_v2 import lct_threshold_bootstrap as v2
from src.Simulate import make_block_cov, sample_gaussian


def _fixture(p=100, n=120, rho=0.6, block=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = sample_gaussian(n, make_block_cov(p, rho, block), seed=seed + 1)
    return X, Y


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_v1_v2_identical_strong_signal(alpha):
    """Same t_hat and rejection set on a strong-signal scenario."""
    X, Y = _fixture()
    t1, m1, _ = v1(X, Y, alpha=alpha, B=50, var_method="cai_liu", rng=0)
    t2, m2, _ = v2(X, Y, alpha=alpha, B=50, var_method="cai_liu", rng=0,
                   coarse_grid=None)
    assert abs(t1 - t2) < 1e-9, f"t_hat drift at alpha={alpha}: {t1} vs {t2}"
    assert np.array_equal(m1, m2), f"rejection sets differ at alpha={alpha}"


def test_v1_v2_identical_under_null():
    """Both should agree when neither finds anything (t_hat = inf)."""
    rng = np.random.default_rng(7)
    p, n = 60, 80
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, p))
    t1, m1, _ = v1(X, Y, alpha=0.05, B=50, var_method="cai_liu", rng=0)
    t2, m2, _ = v2(X, Y, alpha=0.05, B=50, var_method="cai_liu", rng=0,
                   coarse_grid=None)
    assert (np.isinf(t1) and np.isinf(t2)) or abs(t1 - t2) < 1e-9
    assert np.array_equal(m1, m2)


def test_coarse_grid_conservative_and_close():
    """coarse_grid restricts the threshold to ~K quantiles of |T|. Since
    selection takes the infimum of the qualifying set, a coarser grid can
    only land at or above the exact threshold -- never below. That makes
    it conservative (fewer rejections, FDR still controlled). This test
    pins the direction, which is the correctness property, plus a loose
    magnitude bound to catch gross divergence."""
    X, Y = _fixture()
    t_exact, m_exact, _ = v2(X, Y, alpha=0.05, B=50, rng=0, coarse_grid=None)
    t_coarse, m_coarse, _ = v2(X, Y, alpha=0.05, B=50, rng=0, coarse_grid=200)

    # Direction: coarse grid never undershoots the exact threshold.
    assert t_coarse >= t_exact - 1e-9, f"coarse below exact: {t_coarse} < {t_exact}"

    # Consequence: coarse is never more liberal.
    n_exact, n_coarse = int(m_exact.sum()), int(m_coarse.sum())
    assert n_coarse <= n_exact, f"coarse rejected more: {n_coarse} > {n_exact}"

    # Magnitude: grid spacing in the tail, not a free parameter. Loose.
    assert abs(t_coarse - t_exact) < 0.5, f"{t_exact} vs {t_coarse}"
    assert (n_exact - n_coarse) <= max(10, 0.10 * n_exact), \
        f"power loss too large: {n_exact} -> {n_coarse}"