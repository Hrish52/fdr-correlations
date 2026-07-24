import numpy as np
from scipy.stats import norm

# ---------- helpers ----------

def _zscore_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)   # avoid divide-by-zero
    return (X - mu) / sd

def _corr_from_z(Xz: np.ndarray) -> np.ndarray:
    n = Xz.shape[0]
    R = (Xz.T @ Xz) / (n - 1)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -0.999999, 0.999999)

def _kappa_hat(Xz: np.ndarray) -> float:
    """
    Cai-Liu kurtosis parameter (Sec. 2):

        kappa = (1/3) E(X_i - mu_i)^4 / [E(X_i - mu_i)^2]^2

    estimated by averaging the standardised fourth moment over the p
    columns. Equals 1 for Gaussian data, larger for heavy tails.
    Scale-invariant, so computing it on z-scored columns is equivalent
    to the paper's raw-scale formula.
    """
    n = Xz.shape[0]
    s2 = (Xz ** 2).sum(axis=0)
    m4 = (Xz ** 4).sum(axis=0)
    ratio = n * m4 / np.maximum(s2 ** 2, 1e-300)
    return float(np.mean(ratio) / 3.0)

def _var_r_jackknife(Xz: np.ndarray, R: np.ndarray) -> np.ndarray:
    # Jackknife variance for r_ij (more robust, slower: O(n p^2)). Use for small p.
    n, p = Xz.shape
    XY = Xz.T @ Xz
    jk_vals = np.empty((n, p, p), float)
    for k in range(n):
        outer = np.outer(Xz[k], Xz[k])      # x_k y_k
        num = XY - outer                    # sum_{t≠k} x_t y_t
        r_k = num / (n - 2)                 # columns are z-scored (sd≈1)
        np.fill_diagonal(r_k, 1.0)
        jk_vals[k] = np.clip(r_k, -0.999999, 0.999999)
    r_bar = jk_vals.mean(axis=0)
    diff = jk_vals - r_bar
    V = (n - 1) * diff.var(axis=0, ddof=0)
    np.fill_diagonal(V, 0.0)
    return V

def _rho_tilde_sq(R1, R2, kappa1, kappa2, n1, n2, p):
    """
    Thresholded sample correlations (Cai-Liu Sec. 2, paragraph after Eq. 5):

        rho_tilde_ijl = rho_hat_ijl * I{ |rho_hat_ijl| / sqrt(kappa_l/n_l
                        * (1-rho_hat_ijl^2)^2) >= 2 sqrt(log p)}

    then rho_tilde^2_ij = max(rho_tilde^2_ij1, rho_tilde^2_ij2), which is
    substituted for BOTH rho^2_ij1 and rho^2_ij2 in Eq. (4). Using the max
    shrinks the denominator under the alternative, which is what makes T
    more powerful than the naive per-group plug-in.
    """
    def _thr(R, kappa, n):
        se = np.sqrt(kappa / n) * (1.0 - R ** 2)
        stat = np.abs(R) / np.maximum(se, 1e-300)
        keep = stat >= 2.0 * np.sqrt(np.log(max(p, 2)))
        return np.where(keep, R, 0.0)

    return np.maximum(_thr(R1, kappa1, n1) ** 2, _thr(R2, kappa2, n2) ** 2)

# ---------- main API ----------

def lct_edge_stat(X: np.ndarray, Y: np.ndarray, var_method: str = "cai_liu", winsorize=None):
    """
    Cai-Liu (2016) Eq. (4)-(5) edge statistic for H_0,ij: rho_ij1 = rho_ij2.

        T_ij = (r1 - r2) / sqrt( k1/n1 * (1-rt^2)^2 + k2/n2 * (1-rt^2)^2 )

    where k_l is the kurtosis parameter (1 for Gaussian, larger for heavy
    tails) estimated per group, and rt^2 = max of the two thresholded
    sample correlations. Both matter: k_l is what Fisher-z implicitly
    assumes is 1, and the max in rt^2 shrinks the denominator under the
    alternative, which is what gives T its power advantage.

    Under H_0 and condition (C2), T_ij is asymptotically N(0,1).

    Parameters
    ----------
    X, Y : (n1, p), (n2, p) ndarray
    var_method : {"cai_liu", "gaussian", "jackknife"}
        "cai_liu"   - Eq. (4) with kappa estimated from the data.
        "gaussian"  - same, kappa forced to 1 (ablation; anticonservative
                      under heavy tails).
        "jackknife" - assumption-free but O(n p^2); small p only.
    winsorize : float or None
        Clip standardised entries to [-c, c] before correlating. Not part
        of Cai-Liu; an empirical robustness knob (try ~5 for heavy tails).

    Returns
    -------
    T : (p, p) statistics, zero diagonal
    R1, R2 : (p, p) sample correlation matrices

    Notes
    -----
    A variance formula correct only at rho=0 passes null calibration while
    destroying power, since nulls sit near rho=0 and alternatives do not.
    See docs/patches/patch10 and the nonzero-common-rho test.
    """
    # z-score columns
    Xz = _zscore_columns(X)
    Yz = _zscore_columns(Y)

    # optional robustness under heavy tails
    if winsorize is not None:
        c = float(winsorize)
        Xz = np.clip(Xz, -c, c)
        Yz = np.clip(Yz, -c, c)

    # correlations
    R1 = _corr_from_z(Xz)
    R2 = _corr_from_z(Yz)

    # per-edge variances
    if var_method in ("cai_liu", "gaussian"):
        n1, n2 = Xz.shape[0], Yz.shape[0]
        p = Xz.shape[1]
        if var_method == "cai_liu":
            k1, k2 = _kappa_hat(Xz), _kappa_hat(Yz)
        else:                       # 'gaussian': assume kappa = 1
            k1 = k2 = 1.0
        rt2 = _rho_tilde_sq(R1, R2, k1, k2, n1, n2, p)
        shared = (1.0 - rt2) ** 2
        V1 = (k1 / n1) * shared
        V2 = (k2 / n2) * shared
        np.fill_diagonal(V1, 0.0)
        np.fill_diagonal(V2, 0.0)
    elif var_method == "jackknife":
        V1 = _var_r_jackknife(Xz, R1)
        V2 = _var_r_jackknife(Yz, R2)
    else:
        raise ValueError("var_method must be 'cai_liu', 'gaussian', or 'jackknife'.")

    # studentized difference
    denom = np.sqrt(np.maximum(V1 + V2, 1e-12))
    T = (R1 - R2) / denom
    np.fill_diagonal(T, 0.0)
    return T, R1, R2

def lct_threshold_normal(T: np.ndarray, alpha: float = 0.05):
    """
    LCT-N threshold via the normal-tail FDR estimator (Cai & Liu, 2016, Eq. 9):

        t_hat = inf { t in [0, b_p] : est_FDR(t) <= alpha }

    where est_FDR(t) = M * q(t) / max(R(t), 1),
          q(t)      = 2 * (1 - Phi(t)),
          R(t)      = #{ |T_ij| >= t } on the upper-tri,
          M         = p * (p - 1) / 2.

    We scan the unique values of |T| in ascending order and return the FIRST
    t at which est_FDR(t) <= alpha; that is the smallest qualifying threshold
    and therefore yields the largest rejection set consistent with the FDR
    bound. If no such t exists we return t_hat = np.inf and an empty mask.

    Parameters
    ----------
    T : (p, p) ndarray
        Symmetric edge statistic with zero diagonal.
    alpha : float
        Nominal FDR level in (0, 1).

    Returns
    -------
    t_hat : float
        Chosen threshold; np.inf if no threshold controls FDR at level alpha.
    reject_mask : 1-D bool ndarray of length M = p*(p-1)/2
        True at upper-tri edges whose |T_ij| >= t_hat.
    """
    p = T.shape[0]
    iu, ju = np.triu_indices(p, 1)
    absT = np.abs(T)[iu, ju]
    M = absT.size
    if M == 0:
        return np.inf, np.zeros(0, dtype=bool)

    t_grid = np.unique(absT)                      # ascending, deduplicated
    absT_sorted = np.sort(absT)

    # Vectorized: rejection counts and normal-tail FDR over the entire grid
    # at once. O(M log M) instead of an O(M^2) Python loop.
    R_all = M - np.searchsorted(absT_sorted, t_grid, side="left")
    q_all = 2.0 * (1.0 - norm.cdf(t_grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_all = (M * q_all) / np.maximum(R_all, 1)

    # Infimum: first grid point with R > 0 and est_FDR <= alpha.
    ok = (R_all > 0) & (fdr_all <= alpha)
    if ok.any():
        t = float(t_grid[int(np.argmax(ok))])
        return t, (absT >= t)

    return np.inf, np.zeros_like(absT, dtype=bool)