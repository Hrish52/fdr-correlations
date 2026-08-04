# Patch 4 — Notebook Renumbering

**Repository:** `Hrish52/fdr-correlations`
**Files:** all `notebooks/*.ipynb`
**Date:** April 20, 2026
**Status:** ✅ Applied. All tests pass.

## What

Compressed the notebook numbering to remove gaps in the 05–12 range:

| Before | After |
|---|---|
| 06_non_gaussian | 05_non_gaussian |
| 07_calibration | 06_calibration |
| 08_power_curves | 07_power_curves |
| 09_robustness | 08_robustness |
| 10_scaling | 09_scaling |
| 11_defaults | 10_defaults |

Notebooks 01, 03, 04 retain their numbers (early-project milestones). The `13_realdata_experiment` notebook was drafted locally but never committed to the repo; slot 11 is reserved for it and will be filled when the real-data experiment is built out.

## Why

Non-contiguous notebook numbering (missing 05, 12) signals an incomplete or abandoned pipeline. Contiguous numbering communicates a deliberate sequence to a reviewer or replicator.

## Impact

No code, results, or test changes. No notebook references another by filename, so nothing internal breaks.

## Follow-up

If notebooks 01/03/04 are ever revisited, consider renumbering them to 01/02/03 for full contiguity. Deferred because the git-mv cost is not worth the small aesthetic gain, and slot 11 is reserved for the real-data experiment.

---

*End of Patch 4 documentation.*
