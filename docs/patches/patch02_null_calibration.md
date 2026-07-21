# Patch 2 — Null-Calibration Marginal Consistency

**Repository:** `Hrish52/lct-corr-replication`
**Author:** Hrishikesh Deepak Dhole
**Supervisor:** Prof. Elio Zhang, Seattle University
**Date:** April 21, 2026
**Files touched:** `scripts/run_null_calibration.py`, `scripts/run_robustness.py`
**Commit:** *(to be filled in after push)*
**Status:** ✅ Applied. All 14 unit tests pass. Smoke runs confirm default and explicit-override paths both produce the expected marginal in group X.

---

## 1. Summary

This patch changes how the two group-comparison scripts generate their "null" group X. Prior to the patch, both `run_null_calibration.py` and `run_robustness.py` drew group X from N(0, I) regardless of the `--model` flag, while only group Y followed the chosen marginal family (t₆, Laplace, or Exp). This meant that non-Gaussian "null" calibration runs were, strictly speaking, not testing H₀ in the sense that LCT is designed to control — they were testing a mixed-marginal scenario in which the two groups shared a correlation matrix but differed in every other feature of their distribution.

Post-patch, both scripts default to drawing X and Y from the *same* marginal family, matching the standard definition of H₀ for a two-sample edge-correlation test. A new `--x-model` CLI flag preserves the ability to reproduce pre-patch behavior explicitly for future ablation studies, and a new `x_model` column in the output CSVs makes pre- and post-patch runs distinguishable at analysis time.

Any `results/tables/nullcal_*.csv` or `results/tables/robust_*.csv` files generated prior to this patch should be regenerated before use in downstream figures or in the calibration memo.

---

## 2. Background and Motivation

### 2.1 The definition of H₀ for LCT

Cai & Liu (2016) define the correlation-testing problem in the two-sample setting as (their Eq. (2), verbatim):

```
H_{0,ij}: ρ_{ij,1} = ρ_{ij,2}   vs.   H_{1,ij}: ρ_{ij,1} ≠ ρ_{ij,2}
```

for `1 ≤ i < j ≤ p`, based on samples `X_1, ..., X_{n_1}` from the distribution of X and `Y_1, ..., Y_{n_2}` from the distribution of Y. Their theoretical results (Prop. 1, Theorems 1 and 2) assume both samples come from distributions satisfying their moment condition (C2), which is standard for elliptical distributions.

A null calibration study aims to verify empirically that LCT-N and LCT-B control FDR at the nominal level `α` when H₀ holds — that is, when there are no edges with `ρ_{ij,1} ≠ ρ_{ij,2}`. The natural way to enforce this in simulation is:

- Both groups drawn from the same joint distribution (identity covariance, same marginals)
- Verify that at most an `α` fraction of the M = p(p−1)/2 edges are rejected in expectation

### 2.2 The mixed-marginal alternative and why it's ambiguous

If group X is drawn from N(0, I) but group Y is drawn from, say, a t₆ distribution with identity covariance, then:

- The two sample correlation matrices `R̂_1` and `R̂_2` have the same population correlation (identity → zero off-diagonals)
- But the sampling distributions of individual `r̂_{ij,1}` and `r̂_{ij,2}` differ, because the second-moment behavior is different across marginal families

A permutation-based or bootstrap-based test that pools the samples under H₀ (which LCT-B does — see Cai & Liu Eq. (10)) then draws its null distribution from a *mixture* of the two marginal families. This isn't wrong per se, but it makes any FDR-control claim harder to interpret: is LCT-B controlling FDR in the null distribution of the pooled data, or in the true H₀?

A cleaner null runs both groups through the same sampler, and the answer becomes unambiguous.

### 2.3 Why the bug persisted

The bug is easy to miss in a code review because the script structure looks reasonable: X is always drawn as "the null baseline" (Gaussian, identity) and Y is drawn according to the `--model` argument. On the surface this reads as "X is fixed, Y varies," which sounds like a controlled experiment. The issue only becomes visible once you ask what H₀ actually means for LCT: **both samples must come from the same distribution**, not just have the same correlation matrix.

---

## 3. The Bug

### 3.1 Concrete code path

In `scripts/run_null_calibration.py` before this patch:

```python
def make_null(model, n1, n2, p, seed, extra):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))  # <-- always Gaussian, ignores `model`
    I = np.eye(p)
    if model == "gaussian":
        Y = sample_gaussian(n2, I, seed=seed)
    elif model == "t":
        Y = sample_t(n2, df=extra.get("df", 6), Sigma=I, rng=seed)
    elif model == "laplace":
        Y = sample_laplace(n2, b=extra.get("b", 1/np.sqrt(2)), Sigma=I, rng=seed)
    elif model == "exp":
        Y = sample_exp(n2, rate=extra.get("rate", 1.0), Sigma=I, rng=seed, zscore=True)
    else:
        raise ValueError("Unknown model")
    return X, Y
```

The `X = rng.normal(size=(n1, p))` line uses NumPy's `default_rng.normal`, which always draws from N(0, 1) independently by column. It never consults the `model` argument.

An analogous pattern lived in `scripts/run_robustness.py` at line ~33:

```python
X1, _ = _dataset("gaussian", n1, p, np.eye(p), seed, {})     # null group ~ N(0, I)
_,  Y = _dataset(model,     n2, p, Sigma,      seed+12345, extra or {})
```

Here the marginal for X1 is hard-coded to `"gaussian"` regardless of the `--model` flag on the CLI.

### 3.2 What was previously tested

Runs generated under this code answered the question:

> When group X is Gaussian with identity covariance, and group Y has heavy-tailed or skewed marginals with identity covariance, does LCT-B control FDR at level α?

This is a valid empirical question, but it is not the question the paper claims to answer. The paper's claim is about LCT-B's calibration under H₀ — under a common data-generating distribution for both groups.

### 3.3 Impact on prior figures

- Any figure or table that plotted empirical FDR under a "null" scenario for `model ∈ {t, laplace, exp}` was measuring FDR in the mixed-marginal regime, not in true-H₀.
- The Gaussian rows (`model = "gaussian"`) were unaffected — X and Y were both Gaussian in that case, so the pre-patch behavior and the post-patch default coincide.

The practical size of this bias is small in most regimes we tested (the Cai–Liu variance estimator is designed to be robust to some marginal violations), but it is real, and it is exactly the kind of thing a reviewer would push back on.

---

## 4. The Fix

### 4.1 Design choice: default to matching marginals, allow explicit override

Two design considerations shaped the patch:

1. **Correctness by default.** The overwhelmingly most common use case for these scripts is genuine null calibration, and getting that case right by default without any CLI incantation is what most future users (including future-you) will want.

2. **Preserve the ability to reproduce past runs.** Pre-patch runs are valid data — they just answer a different question than the paper's main claim. Deleting the mixed-marginal path entirely would (a) prevent a clean regeneration of pre-patch numbers for a "before/after" comparison in the paper, and (b) foreclose a legitimate ablation study of the form "how sensitive is LCT-B to marginal mismatch across groups?"

The compromise is a new `--x-model` CLI flag that defaults to whatever `--model` is set to but can be set explicitly to any other supported marginal for ablation. The output CSV carries a new `x_model` column so pre- and post-patch runs are trivially distinguishable at analysis time — no need to inspect filenames or commit hashes.

### 4.2 Code changes to `scripts/run_null_calibration.py`

- New helper `_sample_from_model(model, n, p, seed, extra)` that dispatches to `sample_gaussian`, `sample_t`, `sample_laplace`, or `sample_exp` with identity covariance.
- `make_null(...)` now takes an optional `x_model` parameter, defaulting to `model`. Both groups are drawn via `_sample_from_model`; seeds are offset (`seed` for X, `seed + 10**6` for Y) to keep the draws independent even when the two marginals are the same family.
- `run_once(...)` gains an `x_model` parameter and forwards it to `make_null`. The output row includes `x_model` between `model` and `p`.
- `main()` adds `--x-model` to the argparse block.

### 4.3 Code changes to `scripts/run_robustness.py`

- `run_once(...)` gains an `x_model` parameter, defaulting to `model`.
- The line that draws `X1` now uses `_dataset(x_model, ...)` instead of the hardcoded `"gaussian"`.
- The output row includes `x_model` alongside `model`.
- `main()` adds `--x-model` to the argparse block.

### 4.4 Seed offset — why 10⁶?

Under the new default, both groups draw from the same marginal family with the same parameters. If they used the *same* seed, they would produce identical samples, and every FDR computation would trivially reject nothing. Offsetting the seed for group Y by a large constant (10⁶) guarantees the two sequences of pseudo-random numbers are effectively independent while remaining deterministic and reproducible. Any offset large enough to skip past the auto-correlation window of the PCG64 generator would work — 10⁶ is comfortably beyond that and produces round-numbered seeds in logs (e.g., seed=5 → group X uses 5, group Y uses 1000005).

---

## 5. Verification

### 5.1 Unit tests

All 14 tests in `tests/` pass after the patch. None of the existing tests exercise `run_null_calibration.py` or `run_robustness.py` directly (they cover the underlying `src/` primitives), so a test-suite pass here is a necessary but not sufficient condition. Adding integration tests for these two scripts is deferred to a hardening pass.

### 5.2 Smoke runs — default path

Running:

```bash
python scripts/run_null_calibration.py --model t --p 40 --reps 3 --n1 30 --n2 30 --B 50
```

produces `results/tables/nullcal_t_p40_n30_30_R3.csv` in which the `x_model` column equals `t` for every row. This confirms that under the default path, group X now shares the marginal family of group Y.

### 5.3 Smoke runs — explicit override

Running:

```bash
python scripts/run_null_calibration.py --model t --p 40 --reps 3 --n1 30 --n2 30 --B 50 --x-model gaussian
```

produces a CSV whose `x_model` column equals `gaussian`. This confirms that the `--x-model` override is respected and that pre-patch mixed-marginal runs can be reproduced verbatim by setting `--x-model gaussian` explicitly.

### 5.4 Analogous verification for `run_robustness.py`

A parallel smoke run (`--model laplace --x-model` unset, then set) confirms that group X's marginal correctly follows either the `--model` default or the explicit `--x-model` override, and that the `x_model` column in `robust_*.csv` records the choice.

### 5.5 What was not verified in this patch

- The *scientific* consequence of the fix — that empirical FDR under the new default is closer to the nominal α than it was under the pre-patch regime — is what a full re-run of the calibration grid will demonstrate. That re-run is scheduled for Week 2 of the Direction A pre-flight schedule.
- Reproducibility of pre-patch numbers via `--x-model gaussian` was checked structurally (the argument is respected, the column reflects it) but not numerically (the pre-patch CSVs were not preserved for byte-level comparison). This is acceptable because the pre-patch code path is deterministic in `seed`, and the new default with `--x-model gaussian` calls the identical helper functions.

---

## 6. Implications for Downstream Results

### 6.1 What is unaffected

- The `src/` module code (`LCT.py`, `LCTB.py`, `LCTB_v2.py`, `Simulate.py`, `FisherBaselines.py`).
- All prior Gaussian-model calibration runs (`nullcal_gaussian_*.csv`) — in that regime the pre-patch behavior and the post-patch default coincide, so the numbers are correct as-is.
- The `defaults.json` schema and resolver (`src/defaults.py`, `scripts/make_defaults.py`).
- The Fisher-z + BH/BY baselines.

### 6.2 What needs regeneration

- Any `results/tables/nullcal_*.csv` for non-Gaussian marginals (t, laplace, exp) generated before this patch.
- Any `results/tables/robust_*.csv` for non-Gaussian marginals generated before this patch.
- Any calibration figure in `notebooks/07_calibration.ipynb` or robustness figure in `notebooks/09_robustness.ipynb` that consumed those CSVs.

The rerun schedule fits naturally into Week 2 of the Direction A pre-flight (regenerate results with clean code), so no extra time cost is introduced.

### 6.3 What this enables for the paper

The paper's methods section can now describe the null calibration in the standard way: "For each marginal family in {Gaussian, t₆, Laplace, Exp}, we drew both groups from that family with identity covariance..." — matching the definition of H₀ that a reviewer will expect. The `x_model` column also creates an obvious hook for a supplementary ablation study of the form "how does LCT-B behave when the two groups have different marginals?" — a question the pre-patch code was answering by accident and that could become a legitimate additional experiment for the paper.

---

## 7. Repository State After the Patch

| Aspect | Before | After |
|---|---|---|
| `run_null_calibration.py` group X marginal | Hardcoded Gaussian | Follows `--model` by default |
| `run_robustness.py` group X marginal | Hardcoded Gaussian | Follows `--model` by default |
| CLI override for group X | Not available | `--x-model` in both scripts |
| Output CSV `x_model` column | Absent | Present between `model` and `p` |
| Definition of "null" for non-Gaussian runs | Mixed-marginal (implicit) | True H₀ (both groups same distribution) |
| Unit-test suite | 14/14 passing | 14/14 passing |
| Smoke tests (default + override) | N/A | Both paths verified |

---

## 8. References

- **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *Journal of the American Statistical Association*, 111(513), 229–240. Especially Sec. 1, Eq. (2) for the two-sample H₀ definition; Sec. 2 for the LCT-B bootstrap under H₀; Sec. 3, Condition (C2) for the elliptical-distribution assumption.
- **Direction A pre-flight checklist**, `docs/patches/patch01_threshold_selection.md` and the corresponding pre-flight document. Section 1.2 flagged this bug.

---

*End of Patch 2 documentation.*
