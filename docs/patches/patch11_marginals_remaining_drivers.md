# Patch 11 — Completing the Matching-Marginal Fix

**Repository:** `Hrish52/fdr-correlations`
**Files:** `scripts/run_sim_gaussian.py`, `scripts/run_power_curves.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Suite passes. `x_model` column now present in all four drivers' output.

---

## 1. Summary

Patch 2 corrected the null-group marginal in `run_null_calibration.py` and `run_robustness.py` but did not touch `run_sim_gaussian.py` or `run_power_curves.py`. Both continued to draw group X from `rng.normal(...)` regardless of the `--model` flag, so every non-Gaussian row those two scripts produced was a mixed-marginal comparison rather than the hypothesis Cai & Liu Eq. (2) describes.

This patch applies the same treatment to the two remaining drivers: group X now follows `x_model` (defaulting to `model`), a `--x-model` flag preserves the previous behaviour for ablation, and an `x_model` column records the choice in every output CSV.

---

## 2. Why Patch 2 Was Incomplete

Patch 2 was scoped from the pre-flight checklist, which flagged the hardcoded Gaussian draw in `run_null_calibration.py` (§1.2) and noted the analogous line in `run_robustness.py`. The checklist did not examine `run_sim_gaussian.py` or `run_power_curves.py` for the same pattern, because those scripts were reviewed for their grid definitions rather than their data generation.

The omission surfaced when reviewing the Week 2 summary output. The `x_model` column introduced by Patch 2 was present in `nullcal_*.csv` and `robust_*.csv` but absent from the power and simulation CSVs — which is itself a useful property of that column: it makes the scope of a fix visible in the data.

---

## 3. The Defect

Both scripts contained the same shape:

```python
def _dataset(model, n1, n2, p, rho, block, seed, extra):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))              # always Gaussian
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    if model == "gaussian":
        Y = sample_gaussian(n2, Sigma, seed=seed)
    elif model == "t":
        Y = sample_t(n2, df=extra.get("df", 6), Sigma=Sigma, rng=seed)
    ...
```

Group X is drawn from N(0, I) and never consults `model`. For `--model t`, the comparison is therefore Gaussian-versus-t rather than t-versus-t. The correlation structures differ as intended (identity versus block), but so do the marginals, which is not part of the intended design.

### 3.1 Consequence

For the alternative, this is less severe than in the null case Patch 2 addressed — the block correlation is a genuine signal either way, and power figures remain interpretable as "power against a block alternative." But the FDR figures are contaminated: null edges in a mixed-marginal comparison are not draws from H₀, so a rejection there is not unambiguously a false positive in the sense the paper defines.

The Gaussian rows are unaffected, since X and Y are both Gaussian in that case.

---

## 4. The Fix

Both scripts now factor the sampler dispatch into a `_sample(model, n, p, Sigma, seed, extra)` helper and build the dataset as:

```python
def _dataset(model, n1, n2, p, rho, block, seed, extra, x_model=None):
    x_model = x_model or model
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    X = _sample(x_model, n1, p, np.eye(p), seed, extra)
    Y = _sample(model,   n2, p, Sigma,     seed + 10**6, extra)
    return X, Y
```

Mirroring Patch 2: seeds are offset by 10⁶ so the two draws stay independent when the marginals coincide, `x_model` defaults to `model`, and the value is written to the output row.

`run_sim_gaussian.py` carries `x_model` through a module-level `_X_MODEL` global, consistent with how it already handles `_SKIP_LCTB`, `_B_LIST`, and `_N_JOBS`. `run_power_curves.py` passes it as an ordinary keyword argument through `run_once`.

Both gain `--x-model`, defaulting to `None`.

---

## 5. Verification

- Full test suite passes.
- Smoke runs of both scripts produce CSVs whose `x_model` column equals `model` under the default and equals the override when `--x-model` is supplied.
- Column ordering places `x_model` immediately after `model`, matching the convention Patch 2 established, so downstream aggregation code that selects by name is unaffected.

---

## 6. Implications

### 6.1 What needs regenerating

Any `power_*.csv` or model-grid CSV for a non-Gaussian marginal produced before this patch. In practice this overlaps entirely with the regeneration already required by Patch 10, so no additional compute is introduced.

### 6.2 Process note

The `x_model` column functioned as an audit trail: its presence or absence across CSV families made the incompleteness of Patch 2 visible without reading any code. Recording the configuration in the output — rather than only in the script — is what made that possible, and is worth doing for other design choices that could silently vary between runs.

---

## 7. References

- `docs/patches/patch02_null_calibration.md` — the original fix and its rationale.
- **Cai, T. T. & Liu, W. (2016).** Sec. 1, Eq. (2) for the two-sample hypothesis.

---

*End of Patch 11 documentation.*
