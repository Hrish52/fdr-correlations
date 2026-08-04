# Patch 3 — Repo Hygiene

**Repository:** `Hrish52/fdr-correlations`
**Files:** `.gitignore` (edit), `~WRL2773.tmp` (delete), `notebooks/Test_file.ipynb` (delete)
**Date:** April 20, 2026
**Status:** ✅ Applied. All tests pass.

## What

- `.gitignore` extended to ignore Word autosave files (`~$*`, `~WRL*.tmp`, `*.tmp`) and OS junk (`.DS_Store`, `Thumbs.db`).
- Removed `~WRL2773.tmp` — a Word autosave scratch file for the scope document that was accidentally committed.
- Removed `notebooks/Test_file.ipynb` — a one-off CSV-schema sanity check with a hardcoded Windows absolute path (`C:\Users\hirshikesh\...`) that no other user could run.

## Why

Committed scratch files and OS-specific paths are the first thing a reviewer or replicator notices, and they signal that the repository is not ready. Removing them costs nothing — the schema check in `Test_file.ipynb` duplicates coverage already in the `tests/` suite — and makes the repo look intentional.

## Impact

None on results, code, or tests. Purely cosmetic.

## Follow-up (deferred)

If a CSV-schema regression test is genuinely useful, its proper home is `tests/test_csv_schema.py` — a pytest that loads a fixture CSV and asserts on its columns — rather than a notebook with an absolute path. Not written in this patch because the existing pipeline tests already exercise the CSV-producing code paths.

---

*End of Patch 3 documentation.*
