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

def _var_r_gaussian_approx(R: np.ndarray, n: int) -> np.ndarray:
    # Var(r_ij) ≈ (1 - r_ij^2)^2 / (n - 1)   (fast; good under elliptical/Gaussian)
    V = (1.0 - R**2)**2 / max(n - 1, 1)
    np.fill_diagonal(V, 0.0)
    return V

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

def _var_r_cai_liu(Xz: np.ndarray) -> np.ndarray:
    """
    Cai–Liu style plug-in variance for Pearson r using 2nd/4th moments:
      Let u_k = z_{ki} z_{kj}.
      A = E[u] ≈ (Xz^T Xz)/n
      B = E[u^2] ≈ ((Xz^2)^T (Xz^2))/n
      Var(u) ≈ B - A^2
      Var(r_ij) ≈ Var(u)/n = (B - A^2)/n
    """
    n = Xz.shape[0]
    A = (Xz.T @ Xz) / n
    X2 = Xz**2
    B = (X2.T @ X2) / n
    V = (B - A**2) / n
    V = np.maximum(V, 0.0)   # numeric safety
    np.fill_diagonal(V, 0.0)
    return V

# ---------- main API ----------

def lct_edge_stat(X: np.ndarray, Y: np.ndarray, var_method: str = "cai_liu", winsorize=None):
    """
    LCT-style statistic (LCT-N/LCT-B backbone):
        T_ij = (r1_ij - r2_ij) / sqrt( Var(r1_ij) + Var(r2_ij) )

    var_method: "cai_liu" (moment plug-in), "gaussian" (approx), "jackknife" (robust, slower)
    winsorize: if not None, clip standardized entries in X and Y to [-winsorize, winsorize]
               (use values like 5 or 6 under heavy tails; default None leaves data unchanged)

    Returns:
        T  : (p,p) symmetric matrix, 0 diagonal
        R1 : (p,p) correlations for X
        R2 : (p,p) correlations for Y
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
    if var_method == "cai_liu":
        V1 = _var_r_cai_liu(Xz)
        V2 = _var_r_cai_liu(Yz)
    elif var_method == "gaussian":
        V1 = _var_r_gaussian_approx(R1, X.shape[0])
        V2 = _var_r_gaussian_approx(R2, Y.shape[0])
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