# Patch 7 — Vectorized Threshold Scan

**Files:** `src/LCT.py`, `src/LCTB.py`, `src/LCTB_v2.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. 14/14 tests pass. Threshold values verified identical to Patch 1 reference.

---

## 1. Summary

The threshold-selection routines computed rejection counts with a Python loop over the threshold grid, where each iteration performed a full O(M) comparison against all M edge statistics. Since the grid has length ~M, the scan was O(M²). At p = 500 this dominated runtime and made the paper-scale simulation grid effectively unrunnable on modest hardware.

The fix sorts |T| once and uses `np.searchsorted` to obtain all rejection counts in a single vectorized call — O(M log M). This is the same trick already used elsewhere in `LCTB.py` for the pooled bootstrap tail. The patch is behavior-preserving: threshold values and rejection sets are bit-identical to the pre-patch code.

---

## 2. The Bottleneck

### 2.1 Where the time went

All three selectors contained a variant of:

```python
R_t = np.array([(absT >= t).sum() for t in t_grid], dtype=float)
```

`t_grid` is `np.unique(absT)`. For continuous-valued statistics essentially every value is distinct, so `len(t_grid) ≈ M`. Each loop iteration compares all M entries. Total cost: O(M²).

| p | M = p(p−1)/2 | M² | Approx. time per scan |
|---|---|---|---|
| 250 | 31,125 | 9.7 × 10⁸ | ~2 s |
| 500 | 124,750 | 1.6 × 10¹⁰ | ~30 s |
| 1000 | 499,500 | 2.5 × 10¹¹ | ~8 min |

The scan runs once per (α, B) pair per replicate. A null-calibration call with 2 α-values and 3 B-values performs 6 scans per replicate. At p = 500, reps = 30, four marginal families, that is roughly six hours — matching the observed "taking forever" behaviour.

### 2.2 What was *not* the bottleneck

The matrix products inside `lct_edge_stat` (the `Xz.T @ Xz` and `X2.T @ X2` calls) are handled by BLAS and take on the order of 10 ms at p = 500, n = 80. Bootstrap resampling and the pooled-tail counting were already vectorised. Profiling attention had previously gone to those paths; the scan was the actual hot spot.

### 2.3 Why it went unnoticed

At p = 250 — the size used for all development and smoke testing — the scan takes about two seconds, which reads as ordinary overhead. The quadratic term only becomes visible at larger p, and the first serious p = 500 run was attempted during Week 2 result regeneration.

---

## 3. The Fix

### 3.1 Core idea

For a sorted array `s` and threshold `t`, `np.searchsorted(s, t, side="left")` gives the number of entries strictly less than `t`. Therefore `M − searchsorted(s, t)` is the number of entries `≥ t`, which is exactly `R(t)`. Because `searchsorted` accepts an array of query points, the entire grid is evaluated in one call.

```python
_absT_sorted = np.sort(absT)
R_t = (M - np.searchsorted(_absT_sorted, t_grid, side="left")).astype(float)
```

Cost: one O(M log M) sort plus one O(M log M) batched binary search.

### 3.2 Additional change in `LCT.py`

`LCT.py` also called `scipy.stats.norm.cdf` once per grid point inside the loop. SciPy's `norm.cdf` is vectorised, so the whole grid is now evaluated in a single call:

```python
q_all = 2.0 * (1.0 - norm.cdf(t_grid))
```

The infimum is then located with a boolean mask and `argmax`:

```python
ok = (R_all > 0) & (fdr_all <= alpha)
if ok.any():
    t = float(t_grid[int(np.argmax(ok))])
    return t, (absT >= t)
return np.inf, np.zeros_like(absT, dtype=bool)
```

`np.argmax` on a boolean array returns the index of the first `True`, which is precisely the ascending-scan-with-early-return semantics established in Patch 1.

### 3.3 Why this is behavior-preserving

The loop and the vectorised form compute the same quantity for every grid point. Patch 1 defined the selection rule as "the smallest `t` in the grid with `R(t) > 0` and `est_FDR(t) ≤ α`". The vectorised form evaluates the predicate over the whole grid and takes the first index satisfying it. Identical rule, identical result.

---

## 4. Verification

### 4.1 Numerical equivalence — A/B test in a fixed environment

Because library versions affect the generated data (see below), equivalence was verified by running the identical script against pre- and post-patch code *within the same environment*, rather than against previously recorded values.

Reference configuration: p = 100, n = 120, ρ = 0.6, block = 20, `var_method="cai_liu"`, B = 50, seed 0.

| | LCT-N `t̂` / rej | LCT-B v1 | LCT-B v2 |
|---|---|---|---|
| Pre-patch, α=0.05 | 3.108033 / 190 | 3.149995 / 188 | 3.149995 / 188 |
| Post-patch, α=0.05 | 3.108033 / 190 | 3.149995 / 188 | 3.149995 / 188 |
| Pre-patch, α=0.10 | 2.855381 / 213 | 2.895726 / 208 | 2.895726 / 208 |
| Post-patch, α=0.10 | 2.855381 / 213 | 2.895726 / 208 | 2.895726 / 208 |

Identical to six decimal places. The optimisation changes no output.

**Reproducibility note.** These values differ from those recorded in `patch01_threshold_selection.md` §3.3 (3.1212 / 186 at α = 0.05). That drift is not caused by this patch — it was introduced when `requirements.txt` landed in Patch 5 and upgraded numpy/scipy. `sample_gaussian` calls `rng.multivariate_normal`, whose internal SVD is BLAS/LAPACK-dependent; a library bump perturbs the generated matrix in the final digits, which propagates through `absT` to shift `t̂` by roughly 0.4% and the rejection count by a few edges.

The implication for the paper is that a fixed seed alone does not guarantee bit-reproducibility across environments. Final paper runs should be accompanied by a `requirements-lock.txt` recording exact versions, and the paper's reproducibility statement should name the numpy/scipy versions used. Aggregate quantities (empirical FDR, power) averaged over hundreds of replications are insensitive to this drift; only individual `t̂` values are affected.

### 4.2 Unit tests

All 14 tests in `tests/` pass.

### 4.3 Runtime

A single `lct_threshold_bootstrap` call at p = 500, n = 80, B = 100 now completes in seconds rather than minutes. The p = 500 and p = 1000 grids move from impractical to routine.

---

## 5. Implications

### 5.1 What is unaffected

All threshold values, rejection sets, FDR figures, power figures, and `defaults.json` entries. Any result produced by the pre-patch code remains valid — it was correct, merely slow.

### 5.2 What this enables

- p = 500 and p = 1000 grids become feasible on modest hardware.
- Replication counts can rise from the 20–30 range (Monte Carlo SE ≈ 0.04, too coarse to resolve FDR differences near α = 0.05) into the 300–500 range (SE ≈ 0.010), which is what a calibration-focused paper requires.
- The `coarse_grid` option in `LCTB_v2` becomes a refinement rather than a necessity.

### 5.3 Recommended grid design

Given the speedup, a defensible design is p ∈ {250, 500} at 300–500 replications as the main result, plus a single p = 1000 configuration at ~50 replications as an explicitly-labelled scalability demonstration. This keeps the headline calibration numbers statistically tight while still covering the dimension range used in Cai & Liu's own simulations.

---

## 6. Follow-up (deferred)

- A runtime regression test asserting that a p = 500, B = 100 call completes under a fixed wall-clock budget, to prevent silent reintroduction of a quadratic path.
- Profiling the remaining hot spots at p = 1000, where the per-bootstrap `lct_edge_stat` calls become the dominant cost and may benefit from further vectorisation or parallelism.

---

## 7. References

- **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *Journal of the American Statistical Association*, 111(513), 229–240. Eq. (9) and Eq. (10)–(12) define the threshold rules this patch optimises.
- `docs/patches/patch01_threshold_selection.md` — establishes the infimum semantics and the reference values used for equivalence checking here.

---

*End of Patch 7 documentation.*