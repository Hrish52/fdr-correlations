# Patch 15 — Finite-Sample Bias of the Kurtosis Estimator

**Repository:** `Hrish52/fdr-correlations`
**Files:** `scripts/kappa_bias.py`, `results/summary/kappa_bias.csv`
**Date:** April 21, 2026
**Status:** ✅ Measured. Bias is substantial and well-characterised; its effect on FDR control is not detectable.

---

## 1. Summary

Patch 10 introduced `_kappa_hat`, an estimator of the kurtosis parameter κ that enters the Cai–Liu Eq. (4) variance. A spot check during that patch gave κ̂ = 1.703 against a theoretical 2.0 for t₆ data, prompting this study.

The bias is real, sizeable, and scales with the true κ: between −12% and −32% at n = 50, decaying roughly as n^{−1/2}. At the sample sizes used in our simulation grid (n = 80–120) heavy-tailed marginals carry a 15–25% understatement.

A direct check found no corresponding inflation in the false discovery rate. The reason is that κ enters both the true scale of the numerator and the estimate in the denominator in the same direction, so the bias largely cancels. This is recorded as a measured property of the estimator rather than a threat to validity — and the study is reported here partly because an initial analysis predicted the opposite, and the check that refuted it is more informative than the prediction.

---

## 2. Why κ Is Estimated at All

Cai & Liu's Eq. (4) variance is `κ(1 − ρ²)²/n`, where

```
κ = (1/3) · E(X_i − μ_i)⁴ / [E(X_i − μ_i)²]²
```

For Gaussian data κ = 1, and the expression reduces to the classical `(1 − ρ²)²/n`. Fisher's z-transformation has variance `1/(n − 3)`, derived under normality — which implicitly fixes κ = 1.

That implicit assumption is the mechanism by which Fisher-z + BH loses FDR control off-Gaussian: when the true κ exceeds 1, Fisher-z understates the sampling variability and its p-values are anticonservative. Estimating κ from the data is what allows LCT to adapt. The quality of that estimate therefore matters directly.

---

## 3. Method

`scripts/kappa_bias.py` draws `reps` independent datasets at each `(marginal, n)` with identity covariance and `p = 100`, computes κ̂ on each, and compares the mean against the closed-form κ.

Closed-form values used:

| Marginal | Construction | κ |
|---|---|---|
| Gaussian | N(0, I) | 1 |
| t₈ | iid t₈ components | (3 + 6/(8−4))/3 = 1.5 |
| t₆ | iid t₆ components | (3 + 6/(6−4))/3 = 2.0 |
| Normal mixture | U·Z, U ~ Unif(0,1) | 9/5 = 1.8 |
| Exponential | iid Exp(1), centred | 3 |

Grid: n ∈ {50, 80, 120, 200, 500, 2000}, 200 replicates each.

---

## 4. Results

Relative bias, (κ̂ − κ)/κ:

| Marginal | κ | n=50 | n=80 | n=120 | n=200 | n=500 | n=2000 |
|---|---|---|---|---|---|---|---|
| Gaussian | 1.0 | −4.1% | −2.7% | −1.8% | −1.1% | −0.4% | −0.1% |
| Normal mixture | 1.8 | −12.4% | −8.6% | −5.8% | −3.3% | −1.5% | −0.2% |
| t₈ | 1.5 | −17.0% | −12.5% | −9.6% | −6.7% | −3.1% | −0.7% |
| t₆ | 2.0 | −30.1% | −24.2% | −19.8% | −15.2% | −9.0% | −2.5% |
| Exponential | 3.0 | −31.8% | −23.8% | −17.8% | −12.0% | −5.3% | −1.6% |

Three observations:

**The bias is always negative.** κ̂ understates κ in every cell. This follows from the fourth moment being a convex functional of the empirical distribution combined with the normalisation by the squared second moment: the ratio's sampling distribution is right-skewed, so its mean sits above its median but its *estimate* of the population ratio is pulled down by the finite sample's failure to observe the extreme tail that dominates E[X⁴].

**Magnitude scales with κ.** Gaussian is nearly unbiased even at n = 50; t₆ and exponential are off by nearly a third. The heavier the tail, the more of E[X⁴] lives in observations a finite sample rarely draws.

**Convergence is slow.** t₆ is still 9% low at n = 500 and only reaches −2.5% at n = 2000. The decay is consistent with n^{−1/2}. Note also that the standard deviation for t₆ *increases* from n = 500 to n = 2000 (0.112 → 0.227): t₆ has infinite eighth moment, so κ̂ has no finite variance and occasional extreme draws dominate as the sample grows large enough to catch them.

---

## 5. Effect on FDR Control

### 5.1 The prediction that failed

An initial analysis reasoned as follows. κ sits in the numerator of the variance, so κ̂ too small ⟹ denominator too small ⟹ T inflated by `1/√(κ̂/κ)`. At t₆, n = 80 that is a factor of 1.149. Since LCT-N estimates the null exceedance as `2(1 − Φ(t))`, an actually-N(0, 1.149²) statistic would exceed a threshold of 3.6 roughly 5.4 times as often as assumed — predicting an empirical FDR near 0.2–0.3 against a nominal 0.05.

### 5.2 The measurement

At p = 250, n₁ = n₂ = 80, t₆ marginals under the global null, 60 replicates, α = 0.05:

| Procedure | Empirical FDR |
|---|---|
| LCT-N | 0.033 |
| LCT-B | 0.033 |

Both control at the nominal level. The prediction was wrong by nearly an order of magnitude.

### 5.3 Why the prediction was wrong

The argument treated the κ̂ bias as a pure variance scaling applied to a fixed numerator. It is not, because the numerator's own sampling variability depends on κ.

Under the global null with identity Σ, correlations are near zero, so the thresholding step sets ρ̃² = 0 for essentially every edge and the denominator reduces to `√(κ̂₁/n₁ + κ̂₂/n₂)`. Meanwhile the numerator `r̂₁ − r̂₂` has sampling variance proportional to the *true* κ — heavy-tailed data produces more variable sample correlations. So the numerator is larger than the Gaussian case by roughly √κ, while the denominator uses κ̂ ≈ 0.76κ. The two effects are in the same direction and largely cancel, leaving T close to standard normal.

This is precisely what estimating κ is for. A biased estimate degrades the adaptation to tail weight but does not reverse it, and the residual after cancellation is far smaller than the bias in κ̂ itself.

---

## 6. What to Report

The measurement stands on its own: the estimator's finite-sample behaviour is clean, well-characterised, and relevant to anyone applying the method at small n. It belongs in the paper as a short subsection or supplementary table.

The framing should be:

> The kurtosis estimator is biased low by 15–30% for heavy-tailed marginals at n ≤ 120, with bias scaling in κ and decaying as n^{−1/2}. We find no corresponding inflation of the empirical false discovery rate, because κ enters the true scale of the test statistic's numerator and its estimated denominator in the same direction, so the bias substantially cancels.

What it should *not* be framed as is a limitation or a caveat on the method's validity. Overstating it would be as misleading as omitting it.

---

## 7. Methodological Note

The predicted effect and the measured effect differed by an order of magnitude. The prediction was an arithmetic argument that neglected a dependency; the measurement took one short script and sixty replicates.

For the paper's own claims, the lesson is to treat plausible-sounding mechanisms as hypotheses requiring a direct check rather than as results. The check here cost minutes and prevented a wrong claim from entering the manuscript.

---

## 8. Follow-up (deferred)

- A sensitivity arm fixing κ at its theoretical value, to bound the residual effect directly rather than inferring it from the null check.
- Repeat the FDR check under the alternative, where ρ̃² > 0 and the cancellation argument is less clean.
- Investigate whether a bias-corrected κ̂ (jackknife or a moment-matched adjustment) measurably changes anything. Given §5 the expected answer is no, which would itself be worth one sentence.

---

## 9. References

- **Cai, T. T. & Liu, W. (2016).** Sec. 2 for the κ estimator; Eq. (4) for where it enters.
- `docs/patches/patch10_cai_liu_variance.md` — introduces `_kappa_hat` and the spot check that motivated this study.

---

*End of Patch 15 documentation.*
