import numpy as np
from src.LCT import lct_edge_stat
from src.Simulate import make_block_cov, sample_gaussian

def test_lct_null_calibration_cai_liu_small_p():
    rng = np.random.default_rng(0)
    p, n1, n2 = 60, 120, 120
    X = rng.normal(size=(n1, p))
    Y = rng.normal(size=(n2, p))
    T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu")
    iu, ju = np.triu_indices(p, 1)
    t = T[iu, ju]
    # mean near 0, sd near 1 under H0 (tolerances generous for CI over finite m)
    assert abs(t.mean()) < 0.08
    assert 0.85 < t.std(ddof=1) < 1.15

def test_lct_null_at_nonzero_common_rho():
    """H0 with BOTH groups sharing a nonzero correlation structure.

    T must still be ~N(0,1) here. The rho=0 test above cannot detect
    variance-formula errors, since most candidate formulas agree at
    rho=0 and only diverge as |rho| grows -- which is where the
    alternative lives.

    Averaged over replicates: edges within one realisation are strongly
    dependent (they share the same 20 variables and the same two draws),
    so a single realisation's edge-mean has effective sample size ~1 and
    can sit well away from zero.
    """
    p, n, block, reps = 40, 200, 10, 30
    Sigma = make_block_cov(p, rho=0.6, block_size=block)
    iu, ju = np.triu_indices(p, 1)
    blk = (iu < block) & (ju < block)

    means, sds = [], []
    for r in range(reps):
        X = sample_gaussian(n, Sigma, seed=2 * r)
        Y = sample_gaussian(n, Sigma, seed=2 * r + 1)
        T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu")
        t = T[iu, ju][blk]
        means.append(t.mean())
        sds.append(t.std(ddof=1))

    assert abs(np.mean(means)) < 0.20, f"mean {np.mean(means):.3f}"
    assert 0.80 < np.mean(sds) < 1.30, f"sd {np.mean(sds):.3f}"