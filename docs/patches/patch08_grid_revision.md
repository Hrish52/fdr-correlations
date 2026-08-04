# Patch 8 — Simulation Grid Revised for Detectable Power

**Repository:** `Hrish52/fdr-correlations`
**Files:** `scripts/run_sim_gaussian.py`, `scripts/run_power_curves.py`
**Date:** April 21, 2026
**Status:** ✅ Applied. Test suite passes. Grids now produce non-zero power.

## What

Raised the correlation strength and, at p = 500, the sample size in the hardcoded simulation grids so that configurations land in the informative power regime:

| | Before | After |
|---|---|---|
| p = 250 | rho = 0.30, n = 80 | rho = 0.70, n = 80 |
| p = 500 | rho = 0.25, n = 80 | rho = 0.65, n = 120 |
| power-curve rho sweep | 0.20–0.35 | 0.50–0.80 |
| power-curve n sweep | 60, 80, 120 | 80, 120, 200 |

## Why

The prior correlation strengths placed the expected edge statistic for a true discovery far below the smallest threshold capable of controlling FDR. The governing quantities are the achievable signal,

    E|T| ≈ rho_eff * sqrt(n) / sqrt(2 + rho_eff^2),   rho_eff = 0.99 * rho,

and the required threshold,

    t_req = Phi^{-1}(1 − alpha * m1 / (2M)),

where M = p(p−1)/2 is the number of edges and m1 the number of true edges. At p = 500, n = 80, rho = 0.25 this gives E|T| ≈ 1.5 against t_req ≈ 3.96 — the signal is more than two standard normal units short of the threshold, so every method (LCT-N, LCT-B, and both Fisher-z baselines) returns zero rejections. The resulting power curves would have been flat at zero for all methods: an uninformative comparison.

The 1% ridge applied by `make_block_cov` for positive-definiteness means the realised correlation is 0.99 * rho; this is now documented and reflected in `rho_eff`.

## The fix

Correlation strengths and sample sizes were chosen so that E|T| sits roughly 0.3–1.2 units above t_req — high enough for real power, low enough to avoid saturating at 1 for every method. A helper, `scripts/calibrate_grid.py`, computes E|T| against t_req for any configuration and is intended to be run before committing compute to a new grid.

## Impact

- Power curves now traverse from near zero to near one across the rho sweep, which is the shape required to compare methods.
- Any power or FDR result generated before this patch at rho ≤ 0.30 is uninformative (all methods at zero) and should not appear in figures.
- Null-calibration results are unaffected (they use identity covariance, so rho does not enter).

## Follow-up

Full derivation, margin tables across (rho, n, p), and the t_req-versus-p relationship are documented in `docs/notes/signal_calibration.md`. The relationship between required threshold and dimension is itself a candidate secondary finding for the paper.

---

*End of Patch 8 documentation.*
