# src/LCTB.py
import numpy as np
from joblib import Parallel, delayed
from src.LCT import lct_edge_stat
"""
LCT-B (bootstrap threshold) — original implementation.

Implements Cai & Liu (2016), Sec. 2, Eq. (10)–(12): pooled-sample bootstrap
under H0 to estimate the tail probability q*(t) = P*(|T*| >= t), then choose
the threshold as the smallest t in the grid such that the estimated FDR
M * q*(t) / R(t) is at most alpha.

For a faster streaming implementation, see src.LCTB_v2.
"""
from scipy.stats import norm  # only for a fallback if needed

def _compute_T_from_indices(pooled, idx1, idx2, var_method):
    Xb = pooled[idx1]
    Yb = pooled[idx2]
    Tb, _, _ = lct_edge_stat(Xb, Yb, var_method=var_method)
    p = Tb.shape[0]
    iu, ju = np.triu_indices(p, 1)
    return np.abs(Tb[iu, ju])

def lct_threshold_bootstrap(
    X, Y,
    alpha=0.05,
    B=200,
    var_method="cai_liu",   # same variance choices: "cai_liu" | "gaussian" | "jackknife"
    replace=False,          # False = permutation-style split without replacement
    n_jobs=-1,
    rng=None,
):
    """
    Bootstrap LCT threshold (LCT-B) by resampling under H0 (pooled rows).
    Returns:
      t_hat (float),
      reject_mask (1D bool over upper-tri),
      info dict (t_grid, fdr_hat(t) curve, q_hat(t) curve, boot_tail_counts, etc.)
    """
    rng = np.random.default_rng(rng)
    n1, p = X.shape
    n2 = Y.shape[0]
    pooled = np.vstack([X, Y])
    N = pooled.shape[0]

    # 1) Original T and vectorized upper-tri
    T, _, _ = lct_edge_stat(X, Y, var_method=var_method)
    iu, ju = np.triu_indices(p, 1)
    absT = np.abs(T[iu, ju])
    M = absT.size

    # 2) Build bootstrap index pairs under H0
    idxs = []
    for _ in range(B):
        if replace:
            idx1 = rng.integers(0, N, size=n1)
            idx_rest = rng.integers(0, N, size=n2)
        else:
            perm = rng.permutation(N)
            idx1 = perm[:n1]
            idx_rest = perm[n1:n1+n2]
        idxs.append((idx1, idx_rest))

    # 3) Compute |T| for each bootstrap (parallel)
    boot_absTs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_T_from_indices)(pooled, idx1, idx2, var_method)
        for (idx1, idx2) in idxs
    )
    boot_absTs = np.asarray(boot_absTs)   # shape (B, M)

    # 4) Empirical tail function \hat q(t)
    t_grid = np.unique(np.sort(absT))           # scan only at observed |T|
    # For each t, count how many boot |T| exceed t
    # Vectorized: for each bootstrap, sort and use searchsorted would be fastest,
    # but a direct thresholding is fine for moderate M/B.
    q_hat = []
    for t in t_grid:
        exceed = (boot_absTs >= t).sum()
        q_hat.append(exceed / (B * M))
    q_hat = np.asarray(q_hat)

    # 5) FDP estimate and threshold selection
    # Vectorized rejection counts: sort once, then searchsorted over the whole
    # grid. O(M log M) instead of the O(M^2) Python loop this replaces.
    _absT_sorted = np.sort(absT)
    R_t = (M - np.searchsorted(_absT_sorted, t_grid, side="left")).astype(float)
    fdr_hat = (M * q_hat) / np.maximum(R_t, 1.0)

    # Scan ascending (Cai & Liu, 2016, Eq. 12 / Sec. 2 discussion):
    # pick the smallest t with FDR_hat(t) <= alpha. This is the infimum,
    # giving the largest rejection set consistent with the FDR bound.
    t_hat, reject_mask = None, None
    for t, fdr_val in zip(t_grid, fdr_hat):
        R = int((absT >= t).sum())
        if R == 0:
            continue
        if fdr_val <= alpha:
            t_hat = float(t)
            reject_mask = (absT >= t)
            break

    if t_hat is None:
        # No threshold controls FDR at level alpha; reject nothing.
        t_hat = float("inf")
        reject_mask = np.zeros_like(absT, dtype=bool)

    info = {
        "t_grid": t_grid,
        "absT": absT,
        "q_hat": q_hat,
        "fdr_hat": fdr_hat,
        "R_t": R_t,
        "M": M,
        "B": B,
    }
    return float(t_hat), reject_mask, info
