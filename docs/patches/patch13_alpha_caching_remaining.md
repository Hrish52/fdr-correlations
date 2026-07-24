# Patch 13 — Bootstrap Caching in the Remaining Drivers

**Repository:** `Hrish52/fdr-correlations`
**Files:** `scripts/run_sim_gaussian.py`, `scripts/run_robustness.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Suite passes.

---

## 1. Summary

Patch 9 cached the LCT-B bootstrap across significance levels in `run_null_calibration.py` and `run_power_curves.py`, but deliberately skipped these two drivers. Both support `--use-defaults`, which resolves `B`, `coarse_grid`, `winsorize` and `var_method` from a JSON file *inside* the α loop, so those parameters may legitimately differ between levels. A naive cache would silently reuse a bootstrap computed under one configuration to select a threshold under another.

This patch keys the cache by the fully resolved parameter tuple, so it reuses the bootstrap when the configuration is shared across α and correctly recomputes when it is not.

---

## 2. Why These Two Were Deferred

The pattern Patch 9 replaced was straightforward in the two simpler drivers:

```python
for B in B_list:
    for alpha in alpha_list:
        t, mask, _ = lct_threshold_bootstrap(X, Y, alpha=alpha, B=B, ...)
```

Every argument except `alpha` is loop-invariant, so hoisting the call out of the inner loop is safe.

In `run_sim_gaussian.py` and `run_robustness.py` the body is:

```python
for B in B_list:
    for alpha in (0.05, 0.10):
        if use_defaults:
            d = get_defaults_for(p, alpha, ...)
            B_eff = d["B"]; coarse = d["coarse_grid"]
            wins = d["winsorize"]; vm = d["var_method"]
        t, mask, _ = lct_threshold_bootstrap(X, Y, alpha=alpha, B=B_eff,
                                             var_method=vm, winsorize=wins, ...)
```

`get_defaults_for(p, alpha)` is indexed by α. Nothing prevents `defaults.json` from specifying `B = 100` at α = 0.05 and `B = 200` at α = 0.10 — indeed `make_defaults.py` selects B per (p, α) independently, so divergence is the expected case rather than a pathological one.

Hoisting the call would therefore have produced thresholds selected from a bootstrap run under the wrong configuration, with no error raised. That is a worse failure mode than the redundant computation it replaces.

---

## 3. The Fix

The cache is keyed by the tuple that determines the bootstrap's output:

```python
_cache = {}
for B in B_list:
    for alpha in (0.05, 0.10):
        # ... resolve B_eff, vm, wins, kwargs_extra from defaults ...
        key = (B_eff, vm, wins, kwargs_extra.get("coarse_grid"))
        if key not in _cache:
            _, _, _cache[key] = lctb(X, Y, alpha=alpha, B=B_eff,
                                     var_method=vm, winsorize=wins,
                                     n_jobs=n_jobs, rng=seed, **kwargs_extra)
        t_b, mask_b = select_threshold_from_info(_cache[key], alpha)
```

`alpha` is deliberately absent from the key: the bootstrap tail `q̂` and the derived `fdr_hat` do not depend on it, which is the premise of the whole optimisation. The `alpha` passed to the initial call only affects the discarded `t_hat` in its return value.

Three properties worth noting:

- **Declared inside `run_once`.** The cache is per-replicate, so it cannot leak across seeds. A module-level cache would have been a correctness bug, since `X` and `Y` change every replicate.
- **Correct in both directions.** With uniform defaults across α the cache hits and the driver saves one full bootstrap per B. With divergent defaults it misses, and the behaviour is identical to pre-patch.
- **`rng=seed` is fixed per replicate**, so a cache hit and a recomputation would produce the same result anyway; the key is about configuration, not randomness.

---

## 4. Verification

- Full test suite passes.
- Smoke runs of both drivers with and without `--use-defaults` produce the expected CSV columns.
- Output schema is unchanged, so downstream notebooks require no edits.

Expected speedup is approximately 2× on the LCT-B portion when defaults are uniform across the two α levels, and nil otherwise. Since LCT-B dominates runtime in both drivers, that is close to a 2× on total wall time in the common case.

---

## 5. Implications

Purely a compute-time optimisation. Threshold values, rejection sets, and all reported metrics are unchanged. No results require regeneration on account of this patch.

---

## 6. Follow-up (deferred)

`run_robustness.py` also loops over three variance estimators. Unlike α, that loop genuinely changes the edge statistic, so no bootstrap can be shared between arms — the `var_method` component of the cache key already reflects this correctly.

---

## 7. References

- `docs/patches/patch09_alpha_caching.md` — the original optimisation and the argument that `q̂` is α-independent.
- **Cai, T. T. & Liu, W. (2016).** Eq. (10)–(12).

---

*End of Patch 13 documentation.*
