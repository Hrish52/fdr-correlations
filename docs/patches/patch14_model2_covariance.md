# Patch 14 — Cai–Liu Model 2 Covariance

**Repository:** `Hrish52/fdr-correlations`
**Files:** `src/Simulate.py`, `tests/test_generators.py`
**Date:** April 21, 2026
**Status:** ✅ Generator and truth mask added and tested. Not yet wired into the drivers — see §5.

---

## 1. Summary

Cai & Liu (2016) Sec. 5.1 defines two covariance models. The repository implemented only Model 1 — a single correlated block against an identity matrix. This patch adds Model 2, in which both groups are block-diagonal but with *different block sizes*, producing a much denser and stronger differential correlation matrix.

The generator and the corresponding truth mask are added and verified against each other. Wiring Model 2 into the drivers is deferred, because it changes an assumption the current dataset interface is built on.

---

## 2. The Two Models

**Model 1 (implemented).** Group 1 is `I_p`. Group 2 has a single `k × k` block with off-diagonal ρ in the top-left corner. The differential correlation matrix is sparse: `k(k−1)/2` non-zero entries out of `p(p−1)/2`. At the repository's usual `k = 20, p = 250` that is 190 true edges out of 31,125, a signal density of 0.6%.

**Model 2 (this patch).** Both groups are block-diagonal with off-diagonal ρ, but group 1 uses blocks of size `m₁` and group 2 blocks of size `m₂`. Cai & Liu use `(m₁, m₂) = (80, 40)` with ρ = 0.6.

An edge is a true discovery exactly when it falls inside a block for one grouping but not the other. With `m₂` dividing `m₁`, every group-2 block is nested inside a group-1 block, so the true edges are those within an `m₁`-block but spanning two different `m₂`-blocks. At `p = 240, m₁ = 80, m₂ = 40` that is 3 × 40 × 40 = 4,800 true edges out of 28,680 — a signal density of 17%, nearly thirty times denser than Model 1.

---

## 3. Why Model 2 Matters

Three reasons it is worth having:

**Signal density.** Cai & Liu's Theorem 1 assumes `q₁ ≤ cq` for some `c < 1`, i.e. the true alternatives are not overwhelming. Model 1 sits far inside that condition; Model 2 approaches it. The FDR estimator uses `M` in place of the unknown number of true nulls `q₀`, which is conservative when `q₀/M ≈ 1` and progressively less so as density rises. Model 2 is where that approximation is actually tested.

**Strong dependence.** Both groups carry substantial within-block correlation, so the edge statistics are far more strongly dependent than under Model 1, where one group is independent by construction. Cai & Liu's Sec. 6.2 argues that stronger correlation among the test statistics *improves* FDP control — a claim their Model 2 is designed to exhibit and which we currently have no setting to check.

**Sparse versus dense alternatives.** Both the Vesely and the Sánchez-Gómez/Zhang/Liu papers distinguish methods by their behaviour under sparse versus dense alternatives. Having both models available gives that axis in our own results, which matters for the eventual comparison against those two frameworks.

---

## 4. Implementation

```python
make_model2_covs(p, rho=0.6, m1=80, m2=40) -> (Sigma1, Sigma2)
truth_mask_model2(p, m1=80, m2=40)         -> 1-D bool over upper-tri edges
```

The mask is derived by the same block-membership logic as the covariances (`iu // m == ju // m` for each block size, then exclusive-or), rather than by comparing the two matrices numerically. The accompanying test closes that loop by asserting the two agree exactly:

```python
differs = ~np.isclose(S1[iu, ju], S2[iu, ju])
assert np.array_equal(differs, truth_mask_model2(p, m1, m2))
```

This guards against the two drifting apart if either is edited later — a mask that silently disagrees with the data-generating covariance would corrupt every FDR and power figure computed from it, without raising an error.

---

## 5. Why the Drivers Are Not Yet Wired

Every driver currently builds a replicate as:

```python
X = _dataset(x_model, n1, p, np.eye(p), seed,     extra)   # identity
Y = _dataset(model,   n2, p, Sigma,     seed+off, extra)   # structured
```

The identity covariance for group X is baked into the call site, and the truth mask is always `truth_mask_block`. Model 2 requires both groups to carry structure and a different mask, so supporting it means either a second code path in four drivers or a refactor of the dataset interface to take `(Sigma1, Sigma2, truth_mask)` as a unit.

The refactor is the right answer, but it touches every driver at a point where the immediate priority is regenerating results with the corrections from Patches 10–12. Adding the generator now — tested, with a verified mask — means the work is available the moment Model 2 is needed, without holding up the reruns.

---

## 6. Follow-up

- Introduce a `Scenario` object bundling `(Sigma1, Sigma2, truth_mask, label)` and refactor the four drivers to consume it. Model 1 becomes `(I, block, truth_mask_block)`; Model 2 becomes `(m1-blocks, m2-blocks, truth_mask_model2)`.
- Note that `p` should be a multiple of `m₁` for Model 2, otherwise the trailing variables fall outside every block. Cai & Liu use `p ∈ {250, 500, 1000}` with `m₁ = 80`, none of which divide evenly; their handling of the remainder is worth checking against the paper before running.
- Decide whether Model 2 appears in the paper. If the contribution stays focused on marginal-distribution robustness, Model 1 alone may suffice and Model 2 becomes supplementary.

---

## 7. References

- **Cai, T. T. & Liu, W. (2016).** Sec. 5.1 for both covariance models; Sec. 3 Theorem 1 for the `q₁ ≤ cq` condition; Sec. 6.2 for the argument that stronger correlation aids FDP control.

---

*End of Patch 14 documentation.*
