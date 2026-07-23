# Patch 9 — Bootstrap Caching Across α Levels

**Files:** `src/LCTB.py`, `src/LCTB_v2.py`, `scripts/run_null_calibration.py`, `scripts/run_power_curves.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Tests pass. ~2× speedup on LCT-B in the two patched drivers.

---

## 1. Summary

The bootstrap tail estimate in LCT-B does not depend on the nominal FDR level α — only the threshold *selection* does. The driver scripts were nonetheless calling `lct_threshold_bootstrap` once per (B, α) pair, recomputing B full bootstrap resamples for each α to arrive at an identical tail. With two α levels this doubled the cost of the dominant operation.

This patch adds `select_threshold_from_info()`, which applies the Cai–Liu Eq. (9) infimum rule to a cached `info` dict. Drivers now bootstrap once per B and derive thresholds for every α from that single result.

A separate bug was found and fixed in the same area: `run_null_calibration.py` accepted `--winsorize` but never forwarded it to `lct_threshold_bootstrap`, so the flag silently affected LCT-N only.

---

## 2. The Redundancy

### 2.1 What depends on α and what does not

The LCT-B procedure computes, over a grid of candidate thresholds:

q̂(t) = P*(|T*| ≥ t) bootstrap tail — independent of α
R(t) = #{ |T_ij| ≥ t } observed rejections — independent of α
FDP̂(t) = M · q̂(t) / max(R(t), 1) estimated FDP — independent of α
t̂ = inf{ t : FDP̂(t) ≤ α } selection — depends on α

Only the final step consults α. Everything upstream — including the B bootstrap resamples, which dominate runtime — is shared.

### 2.2 The pattern in the drivers

`run_null_calibration.py` before this patch:

```python
for B in B_list:
    for alpha_s in alpha_list:
        t_b, mask_b, _ = lct_threshold_bootstrap(X, Y, alpha=a, B=B, ...)
```

With `alpha_list = ["0.05", "0.10"]`, the inner loop performs B resamples, discards the `info` dict, and performs B more resamples for the second α — producing the same `q̂` both times. The same structure appeared in `run_power_curves.py`.

### 2.3 Cost

At p = 500, B = 100, the bootstrap accounts for the large majority of a `lct_threshold_bootstrap` call (roughly 10s of the ~12s measured after Patch 7). Doubling it for a second α is a direct 2× penalty on the driver's dominant cost, for no information gain.

---

## 3. The Fix

### 3.1 `select_threshold_from_info()`

Added to `src/LCTB_v2.py`:

```python
def select_threshold_from_info(info: dict, alpha: float):
    t_grid  = info["t_grid"]
    fdr_hat = info["fdr_hat"]
    R_t     = info["R_t"]
    absT    = info["absT"]

    ok = (R_t > 0) & (fdr_hat <= float(alpha))
    if ok.any():
        t = float(t_grid[int(np.argmax(ok))])
        return t, (absT >= t)
    return float("inf"), np.zeros_like(absT, dtype=bool)
```

This is the same selection rule used inside the main routines: ascending scan, first grid point with `R > 0` and `FDP̂ ≤ α`, `np.inf` fallback. The `np.argmax` on a boolean array returns the first `True`, matching the early-return semantics established in Patch 1 and the vectorised form introduced in Patch 7.

### 3.2 `absT` added to `info`

Both `LCTB.py` and `LCTB_v2.py` now include `absT` in their returned `info` dict. Without it the helper would need to recompute the edge statistic, which would defeat the purpose. `t_grid`, `fdr_hat`, and `R_t` were already present.

### 3.3 Driver changes

Both patched drivers now follow:

```python
for B in B_list:
    _, _, info = lct_threshold_bootstrap(X, Y, alpha=<any>, B=B, ...)
    for alpha in alpha_list:
        t_b, mask_b = select_threshold_from_info(info, alpha)
        ...
```

The `alpha` passed to the initial call is irrelevant to the cached arrays; it only determines the `t_hat` in the discarded return value.

### 3.4 Drivers deliberately not patched

`run_sim_gaussian.py` and `run_robustness.py` were left unchanged. Both resolve `--use-defaults` inside the α loop, meaning `B`, `coarse_grid`, `winsorize`, and `var_method` can all differ per α. Caching there requires either resolving defaults before the loop or keying the cache by the resolved parameter tuple. The gain is the same 2×, but the change is not a one-liner and carries more risk of silently mixing configurations. Deferred.

---

## 4. Bug Fix: `--winsorize` Ignored by LCT-B

While restructuring the loop, the LCT-B call in `run_null_calibration.py` was found to omit `winsorize`:

```python
t_b, mask_b, _ = lct_threshold_bootstrap(
    X, Y, alpha=a, B=B, var_method=var_method, n_jobs=n_jobs, rng=seed
)   # winsorize never passed
```

The LCT-N call in the same function did pass it. So `--winsorize 5` produced winsorised LCT-N columns alongside unwinsorised LCT-B columns in the same CSV row — silently, with no error.

This matters because winsorisation is one of the tuning knobs the calibration study is meant to evaluate, and `make_defaults.py` records a `winsorize` value into `defaults.json`. Any conclusion drawn about winsorisation's effect on LCT-B from a run with `--winsorize` set would have been unfounded.

**Any prior `nullcal_*.csv` generated with `--winsorize` set should be regenerated.** Runs at the default (`winsorize=None`) are unaffected, since the omitted argument defaulted to `None` anyway.

---

## 5. Verification

### 5.1 Unit tests

Full suite passes. The new `tests/test_lctb_v1_v2_agreement.py` (added alongside this patch) provides additional coverage by comparing the two implementations directly.

### 5.2 Column schema unchanged

A smoke run at p = 100, reps = 3 produces the same `t_lctb_*`, `R_lctb_*`, `fdp_lctb_*`, `any_reject_lctb_*`, and `R_over_M_lctb_*` columns as before. Downstream notebooks and `make_defaults.py` require no changes.

### 5.3 Runtime

Per-replicate wall time in the patched drivers drops by roughly half at two α levels, consistent with eliminating one of two identical bootstrap passes.

### 5.4 Equivalence of the selection rule

`select_threshold_from_info` uses the same predicate and the same `argmax`-based first-true lookup as the in-routine selection introduced in Patch 7. Given identical `t_grid`, `fdr_hat`, `R_t`, and `absT`, it returns identical output by construction.

---

## 6. Implications

### 6.1 What is unaffected

Threshold values, rejection sets, FDR and power figures, CSV schema, `defaults.json` format. This is a pure compute-time optimisation for runs that did not use `--winsorize`.

### 6.2 What needs regeneration

Any `nullcal_*.csv` produced with `--winsorize` set, for the reason in §4.

### 6.3 Combined effect with Patch 7

Patch 7 removed the O(M²) threshold scan; Patch 9 removes the duplicated bootstrap. Together they bring p = 500 from impractical to routine and make the higher replication counts (300–500) required for meaningful Monte Carlo precision achievable.

---

## 7. Follow-up (deferred)

- Apply the same caching to `run_sim_gaussian.py` and `run_robustness.py`, handling per-α defaults resolution (§3.4).
- Consider caching across `var_method` in `run_robustness.py`, which currently loops three variance estimators — though unlike α, that loop genuinely changes the edge statistic and cannot share a bootstrap.

---

## 8. References

- **Cai, T. T. & Liu, W. (2016).** Eq. (10)–(12) define the bootstrap tail and threshold rule.
- `docs/patches/patch01_threshold_selection.md` — infimum semantics.
- `docs/patches/patch07_vectorized_scan.md` — the prior speedup; together with this patch determines current runtime.

---

*End of Patch 9 documentation.*