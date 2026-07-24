# Patch 12 — Cai–Liu Sec. 5.1 Generators

**Repository:** `Hrish52/fdr-correlations`
**Files:** `src/Simulate.py`, all four driver scripts, `tests/test_generators.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Adds models `t_cl`, `exp_cl`, `nmix`. Existing samplers retained.

---

## 1. Summary

The repository's non-Gaussian samplers construct heavy-tailed and skewed data in ways that happen to *satisfy* Cai & Liu's moment condition (C2), rather than violate it. Cai & Liu's own Sec. 5.1 constructions do violate it. Because (C2) is precisely the assumption that separates their method from Fisher-z + BH, our simulations were not exercising the contrast the paper is about.

This patch adds three generators matching their design — two non-elliptical, one elliptical with κ ≠ 1 — while retaining the existing samplers, since the elliptical-versus-not contrast is itself worth reporting.

---

## 2. How This Was Found

The Week 2 summary showed Fisher-z + BH achieving FDR = 0.000 on t₆, Laplace, and Exp at p = 250 and p = 500. Cai & Liu report FDR between 0.32 and 0.99 for the same baseline under their non-Gaussian settings (their Tables 1 and 3).

A baseline that is supposed to break, and doesn't, means the setting isn't stressing it. Inspection of `src/Simulate.py` confirmed why.

---

## 3. The Discrepancy

### 3.1 What condition (C2) requires

Cai & Liu's Condition (C2) states that fourth moments factorise as

```
E[(X_i−μ_i)(X_j−μ_j)(X_k−μ_k)(X_l−μ_l)] = κ(σ_ij σ_kl + σ_ik σ_jl + σ_il σ_jk)
```

for a single constant κ. This holds for all elliptically contoured distributions, of which the multivariate normal (κ = 1) and multivariate t are members. It is the condition under which the Eq. (4) variance and the asymptotic normality of T_ij hold.

### 3.2 What our samplers produce

| Sampler | Construction | Elliptical? | (C2)? |
|---|---|---|---|
| `sample_t` | Z ~ N(0,Σ), s ~ χ²_df/df, T = Z/√s | Yes | Satisfied |
| `sample_laplace` | Gaussian copula + Laplace inverse-CDF | No | Approximately satisfied |
| `sample_exp` | Gaussian copula + Exp inverse-CDF | No | Approximately satisfied |

`sample_t` is the standard multivariate t — a scale mixture of normals, hence elliptical by construction. It has heavy tails but κ is well-defined and Fisher-z's failure mode is only mildly triggered.

The copula-based samplers give correct marginals but impose Gaussian dependence. The resulting fourth-moment structure stays close enough to the elliptical form that (C2) is not meaningfully violated.

### 3.3 What Cai & Liu use

Their Sec. 5.1 constructions are:

- **t distribution:** `X = Σ^{1/2} Z` with Z having **iid t₆ components**. The linear transform of independent heavy-tailed variables is *not* elliptical.
- **Exponential:** `X = Σ^{1/2} Z` with Z having **iid Exp(1) components**. Also not elliptical, and skewed.
- **Normal mixture:** `X = U · Z` with U ~ Uniform(0,1) scalar per observation and Z ~ N(0, Σ). This *is* elliptical, so (C2) holds — but with κ = 9/5 rather than 1.

They state explicitly that the t and exponential constructions do not satisfy (C2), and that the t case additionally violates their exponential-tail condition (C3). The point of including them is to test robustness beyond the assumptions.

### 3.4 Why the normal mixture is the sharpest case

The normal mixture is the cleanest separator, and the one where Cai & Liu report the largest gap (Fisher-z + BH FDR of 0.95–0.99 against a nominal 0.2).

It satisfies (C2), so Cai–Liu's theory applies in full — but κ = 9/5 ≠ 1. Fisher's z-transformation has variance `1/(n−3)` derived under normality, which implicitly assumes κ = 1. When κ = 9/5 the true variance is 1.8× larger, so Fisher-z p-values are anticonservative by a large factor and BH inherits the error.

Cai–Liu's estimated κ̂ absorbs this exactly. **This is the single case that isolates the mechanism by which LCT beats Fisher-z**, with no confounding from tail weight or skew.

---

## 4. The Fix

Three generators added to `src/Simulate.py`:

```python
sample_t_cl(n, df, Sigma, rng)        # X = Σ^{1/2} Z, Z iid t_df      — non-elliptical
sample_exp_cl(n, rate, Sigma, rng)    # X = Σ^{1/2} Z, Z iid Exp(rate) — non-elliptical
sample_normal_mixture(n, Sigma, rng)  # X = U·Z, U ~ Unif(0,1)         — elliptical, κ = 9/5
```

Each scales its components to unit variance before the linear transform, so the resulting correlation matrix matches Σ.

Registered as models `t_cl`, `exp_cl`, `nmix` in all four drivers.

The existing `sample_t`, `sample_laplace`, and `sample_exp` are retained. Running both families gives a designed contrast — same nominal marginal, (C2) satisfied versus violated — which is more informative than replacing one with the other.

---

## 5. Verification

`tests/test_generators.py` gains:

- A shape and finiteness check for all three new samplers.
- A test asserting κ̂ for `sample_normal_mixture` lies in (1.4, 2.4) at n = 4000, against a theoretical 1.8. This directly verifies the property that makes the mixture a separating case, and would catch a construction error that produced merely-Gaussian data.

Smoke runs of `run_null_calibration.py --model nmix` and `run_power_curves.py --models nmix` complete and write CSVs.

---

## 6. Implications

### 6.1 Expected results

On `nmix`, Fisher-z + BH should exceed the nominal FDR substantially while LCT-B holds near α. That contrast is the paper's headline. On `t_cl` and `exp_cl`, both methods face genuine (C2) violation, and the question becomes how gracefully each degrades — a question our original samplers could not pose.

### 6.2 What this adds to the paper

The simulation section can now report a two-by-two structure: elliptical versus non-elliptical, and κ = 1 versus κ ≠ 1. That decomposition attributes any FDR failure to a specific assumption violation, rather than reporting an undifferentiated "non-Gaussian" result.

### 6.3 Compute

Six non-Gaussian marginal families instead of three roughly doubles the non-Gaussian arm of the grid. Given the Patch 7 and Patch 9 speedups this is affordable, but the paper need not report all six at full replication — the three Cai–Liu families as the primary result, with the copula-based ones as a supplementary contrast, is a reasonable allocation.

---

## 7. Follow-up (deferred)

- Empirically confirm (C2) violation for `t_cl` and `exp_cl` by estimating the fourth-moment factorisation error directly, rather than relying on the construction argument.
- Add Cai & Liu's Model 2 covariance (block sizes m₁ = 80, m₂ = 40 with ρ = 0.6) as a strong-correlation setting.

---

## 8. References

- **Cai, T. T. & Liu, W. (2016).** *JASA* 111(513), 229–240. Sec. 5.1 for the four distributions; Sec. 3, Condition (C2) for the moment assumption; Tables 1–5 for the FDR figures our baseline should be reproducing.
- `docs/patches/patch10_cai_liu_variance.md` — the κ estimator that the normal-mixture case is designed to exercise.

---

*End of Patch 12 documentation.*
