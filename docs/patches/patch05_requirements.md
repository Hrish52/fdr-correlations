# Patch 5 — Pinned Runtime Dependencies

**Repository:** `Hrish52/fdr-correlations`
**Files:** `requirements.txt` (new), `tests/test_imports.py` (edited)
**Date:** April 20, 2026
**Status:** ✅ Applied. Test suite passes on a fresh `pip install -r requirements.txt`.

## What

- New `requirements.txt` listing the minimal runtime dependencies with loose lower bounds (Python 3.10+): numpy, scipy, pandas, matplotlib, joblib, jupyter, ipykernel, pytest.
- `tests/test_imports.py` no longer imports numba, dask, or networkx — none of which are used anywhere in the codebase.

## Why

Prior to this patch there was no pinned dependency file at all; a reviewer had to reverse-engineer the imports. `test_imports.py` further claimed the code depended on numba/dask/networkx, which is misleading — grepping `src/`, `scripts/`, and `notebooks/` shows they are never used. The mismatch would have failed pytest on a fresh install unless the reviewer over-installed.

## Design choices

- **Loose lower bounds, not exact pins.** Pin the API level (numpy 1.24+ for `default_rng`, pandas 2.0+ for arrow-backed dtypes) but let the resolver pick patch versions. Exact pins are for deployment, not research reproducibility; the exact environment for the paper's numbers is recorded separately in `requirements-lock.txt`.
- **No `pyproject.toml` yet.** The repo is not distributed as an installable package; `requirements.txt` plus a real README suffices.

## Impact

A fresh clone now runs `python -m venv .venv && pip install -r requirements.txt && pytest` as a one-line reproducibility path. All tests pass on a clean install.

## Follow-up (deferred)

- `pyproject.toml` with entry points and version metadata, if the code is ever released as a package.
- `requirements-lock.txt` (added later) pins exact versions for the final paper-scale runs.

---

*End of Patch 5 documentation.*
