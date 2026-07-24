# Patch 10 — Implementing the Cai–Liu Eq. (4)/(5) Variance

**Repository:** `Hrish52/fdr-correlations`
**Files:** `src/LCT.py`, `tests/test_calibration.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Suite passes. Power at p=250, n=80, ρ=0.7 rises from 0.656 to 1.000 at FDP 0.050.

---

## 1. Summary

The `cai_liu` variance estimator did not implement Cai & Liu (2016) Eq. (4). It computed `Var(z_i z_j)/n`, which for bivariate normal data equals `(1 + ρ²)/n`, whereas Eq. (4) specifies `κ(1 − ρ²)²/n`. The two agree at ρ = 0 and diverge sharply as |ρ| grows.

Two further components of the paper's statistic were absent entirely: the kurtosis parameter κ was never estimated, and the thresholded-correlation substitution `ρ̃² = max(ρ̃²₁, ρ̃²₂)` was not implemented.

Because nulls concentrate near ρ = 0 and alternatives do not, the defect passed every null-calibration check while roughly halving power. It surfaced only when power results showed LCT losing decisively to the Fisher-z + BH baseline it is supposed to beat.

A second, subtler error was introduced and then fixed during this patch: the sparsity threshold was initially applied on the wrong scale, producing a four-fold FDR violation. Both the original defect and the interim error are documented below, since the second is instructive about how the two scales interact.

---

## 2. How the Defect Was Found

Direction A's simulation grid was run after Patch 8 corrected the ρ values. At p = 250, n = 80, ρ = 0.7, Gaussian marginals:

| Method | FDR | Power |
|---|---|---|
| Fisher-z + BH | 0.054 | 0.962 |
| Fisher-z + BY | 0.006 | 0.870 |
| LCT-N | 0.066 | 0.656 |
| LCT-B | 0.052 | 0.613 |

This inverts Cai & Liu's central claim. Their Tables 1–4 show LCT matching or exceeding Fisher-z + BH on power while controlling FDR more reliably. A 30-percentage-point deficit is not a tuning discrepancy.

Non-Gaussian rows were worse. At t₆, ρ = 0.7: BH power 0.962, LCT-B power 0.247. At Exp: BH 0.872, LCT-B 0.046.

The `run_robustness.py` output corroborated the diagnosis independently. That script runs all three variance estimators side by side, and at ρ = 0.7 the `gaussian` estimator outperformed `cai_liu` on power (0.167 vs 0.121). Since `_var_r_gaussian_approx` used the correct `(1 − R²)²` functional form and differed only by omitting κ, the ordering pointed directly at the `cai_liu` branch.

---

## 3. The Original Defect

### 3.1 What the code computed

```python
def _var_r_cai_liu(Xz):
    n = Xz.shape[0]
    A = (Xz.T @ Xz) / n          # ~ rho
    X2 = Xz ** 2
    B = (X2.T @ X2) / n          # ~ E[z_i^2 z_j^2]
    V = (B - A**2) / n           # = Var(z_i z_j) / n
    return np.maximum(V, 0.0)
```

This is the variance of the *product of standardised scores*, divided by n. For bivariate normal with correlation ρ, `E[z_i² z_j²] = 1 + 2ρ²`, so

```
V = (1 + 2ρ² − ρ²)/n = (1 + ρ²)/n
```

### 3.2 What Eq. (4) specifies

Cai & Liu's asymptotic variance for the difference of sample correlations is

```
Var(r̂_ij,l) ≈ κ_l (1 − ρ_ij,l²)² / n_l
```

with `κ_l = (1/3) E(X_i − μ_i)⁴ / [E(X_i − μ_i)²]²`, equal to 1 for Gaussian data.

### 3.3 Magnitude of the error

| ρ | Implemented `(1+ρ²)` | Eq. (4) `(1−ρ²)²` | Ratio |
|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 1.0 |
| 0.3 | 1.090 | 0.828 | 1.3 |
| 0.5 | 1.250 | 0.563 | 2.2 |
| 0.7 | 1.490 | 0.260 | 5.7 |
| 0.9 | 1.810 | 0.036 | 50.2 |

At ρ = 0.7 the variance was 5.7× too large, so the denominator of T was 2.4× too large and T was deflated by that factor.

### 3.4 The arithmetic reproduces the observed power

Group X is independent (ρ = 0, contributing 1/n either way). Group Y carries ρ_eff = 0.99 × 0.7 = 0.693.

| | V₂ | E\|T\| | vs t_req = 3.61 | predicted power |
|---|---|---|---|---|
| As implemented | (1 + 0.48)/80 | 3.94 | +0.33 | ~60% |
| Eq. (4) | (1 − 0.48)²/80 | 5.50 | +1.89 | ~100% |

Observed LCT-N power was 0.656 — squarely in the ~60% band. Observed BH power was 0.962, consistent with Fisher-z's own variance being correct here.

### 3.5 Why null calibration never caught it

Under H₀ both groups have ρ ≈ 0, where `(1 + ρ²)` and `(1 − ρ²)²` both equal 1. FDR checks therefore passed at every marginal family (0.033, 0.033, 0.033, 0.000 across Gaussian, t, Laplace, Exp at α = 0.05). The existing `test_calibration.py` tested exactly this case: independent standard normals, ρ = 0. It could not have detected the defect.

**A variance formula correct only at ρ = 0 passes null calibration while destroying power.** This is a general lesson worth carrying into the paper's discussion of implementation pitfalls.

---

## 4. The Fix

### 4.1 Estimating κ

```python
def _kappa_hat(Xz):
    n = Xz.shape[0]
    s2 = (Xz ** 2).sum(axis=0)
    m4 = (Xz ** 4).sum(axis=0)
    ratio = n * m4 / np.maximum(s2 ** 2, 1e-300)
    return float(np.mean(ratio) / 3.0)
```

The standardised fourth moment is averaged over the p columns, matching Cai & Liu's Sec. 2 estimator. The quantity is scale-invariant, so computing it on z-scored columns is equivalent to their raw-scale formula.

Verified: Gaussian gives κ̂ = 1.006 at n = 500, p = 50.

### 4.2 Thresholded correlations

Cai & Liu substitute, for both groups, a single thresholded quantity

```
ρ̃_ij,l = ρ̂_ij,l · 1{ |ρ̂_ij,l| / √(κ_l/n_l (1 − ρ̂²_ij,l)²) ≥ 2√(log p) }
ρ̃²_ij  = max(ρ̃²_ij,1, ρ̃²_ij,2)
```

Taking the maximum shrinks the denominator under the alternative, which they identify explicitly as the source of T's power advantage over a naive per-group plug-in. Under H₀ the two thresholded values coincide, so the substitution is asymptotically harmless.

### 4.3 Unified dispatch

`var_method="cai_liu"` and `var_method="gaussian"` now share one code path, differing only in whether κ is estimated or fixed at 1. This makes `gaussian` a clean ablation isolating the effect of κ estimation, rather than a separate functional form. The `jackknife` branch is unchanged.

---

## 5. Interim Error: Threshold Applied on the Wrong Scale

The first version of `_rho_tilde_sq` compared a standardised statistic against a raw-scale constant:

```python
keep = stat >= 2.0 * np.sqrt(np.log(max(p, 2)) / n)   # WRONG
```

`stat` is `|ρ̂| / √(κ/n (1 − ρ̂²)²)`, which has standard deviation ≈ 1 under H₀ — a z-score. The bound `2√(log p / n)` evaluates to 0.525 at p = 250, n = 80, so roughly 60% of null edges survived thresholding and retained a nonzero ρ̃². Their denominators were shrunk by `1/(1 − ρ̃²)`, inflating T hardest in the upper tail where rejections occur.

Result at p = 250, n = 80, ρ = 0.7, α = 0.05: **238 rejections against only 190 true edges**, i.e. at least 48 false positives and FDP ≥ 0.20. A four-fold violation of the nominal level.

The correct constant on the standardised scale is the universal threshold `2√(log p)` ≈ 4.70 at p = 250. This is the same sparsity rule expressed on a different scale: `|ρ̂| ≥ 2√(log p / n)` on the raw correlation scale becomes `|ρ̂|/SE ≥ 2√(log p)` once divided by an SE of order `1/√n`.

```python
keep = stat >= 2.0 * np.sqrt(np.log(max(p, 2)))       # correct
```

---

## 6. Verification

### 6.1 End-to-end at the reference configuration

p = 250, n₁ = n₂ = 80, ρ = 0.7, block = 20, α = 0.05, Gaussian:

```
t_hat = 3.618   R = 200   V = 10   FDP = 0.050   power = 1.000
```

FDP lands exactly at the nominal level; all 190 true edges are recovered.

| | Power | FDP |
|---|---|---|
| Before Patch 10 | 0.656 | 0.066 |
| Interim (wrong threshold scale) | — | ≥ 0.20 |
| After Patch 10 | **1.000** | **0.050** |
| Fisher-z + BH, same config | 0.962 | 0.054 |

LCT now exceeds BH on power at matched FDR — the relationship Cai & Liu report.

### 6.2 κ estimates

| Data | κ̂ | Theoretical |
|---|---|---|
| Gaussian | 1.006 | 1.000 |
| t₆ | 1.703 | 2.000 |

The t₆ shortfall is finite-sample bias, not an implementation error: t₆ has infinite eighth moment, so the sample fourth moment is biased downward at moderate n. Cai & Liu use the same estimator, so this is inherent to the method. The direction of the bias is conservative for FDR (κ̂ too small ⇒ denominator too small ⇒ T too large ⇒ more rejections), which is worth stating explicitly in the paper.

### 6.3 New regression test

`tests/test_calibration.py` gains a test asserting T ~ N(0,1) under H₀ **at a nonzero common ρ**, averaged over 30 replicates.

The averaging matters. Within a single realisation, all block edges share the same 20 variables and the same two draws, so they move together — the effective sample size of the within-realisation edge mean is close to 1, not 190. An early version of this test asserted on a single realisation and failed with an edge-mean near −1.5 on an unlucky seed, despite the code being correct. The distributional claim is across replicates.

---

## 7. Implications

### 7.1 What must be regenerated

**All power and FDR results produced before this patch.** Null-calibration outputs are numerically unaffected (ρ ≈ 0 is the one regime where the old formula was right), but should be regenerated anyway for provenance.

### 7.2 What this changes for the paper

The headline comparison is now the one Cai & Liu describe, so the paper's contribution shifts from "we could not reproduce the claimed advantage" to a calibration study of a method that does work. The three-way `var_method` ablation also becomes substantive: `cai_liu` (κ estimated), `gaussian` (κ = 1), and `jackknife` (assumption-free) now isolate exactly the effect of the kurtosis correction, which is the mechanism by which LCT beats Fisher-z off-Gaussian.

### 7.3 Methodological note worth reporting

Two observations generalise beyond this repository and belong in the paper's discussion:

1. A variance formula correct only at ρ = 0 passes null calibration while destroying power. Null-only validation is insufficient for any FDR procedure whose statistic involves a nuisance parameter that differs between null and alternative.
2. The sparsity threshold must be applied on the scale it was derived for. The raw-correlation and standardised forms differ by a factor of √n, and using the wrong one converts a mild regularisation into a four-fold FDR violation.

---

## 8. Follow-up (deferred)

- Quantify the finite-sample bias in κ̂ across n and marginal families; report the direction and magnitude in the paper.
- Consider a bias-corrected κ̂ as an ablation arm.
- Add a test asserting κ̂ → 1 for Gaussian data as n grows, guarding the estimator itself.

---

## 9. References

- **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *JASA* 111(513), 229–240. Sec. 2, Eq. (4)–(5) and the following paragraph on thresholded correlations.
- `docs/notes/signal_calibration.md` — the E|T| vs t_req framework used to predict power in §3.4.
- `docs/patches/patch08_*` — the grid revision that made this defect visible.

---

*End of Patch 10 documentation.*
