# Patch 6 — Real README.md

**Repository:** `Hrish52/fdr-correlations`
**Files:** `README.md`
**Date:** April 20, 2026
**Status:** ✅ Applied. All tests pass.

## What

Replaced the placeholder README with a full document covering project purpose, quick-start install, repository layout, figure-reproduction workflow, key methodological decisions (each linked to the patch note that made it), and references to the three foundational papers.

## Why

The repository landing page is the single highest-visibility artifact in the project. A one-sentence stub signals "work in progress, not ready to review." A well-organised README signals a serious project with a plan. Same repository, same code, opposite reader reaction.

## Impact

- The landing page is now a coherent project introduction.
- Quick-start instructions are testable: `git clone && venv && pip install && pytest` produces a green suite.
- Every methodological decision links back to the patch note that documents it, so a reader can drill down without reading source.

## Follow-up

- Add badges (build status, Python version, license) once GitHub Actions CI is set up.
- Add a citation `.bib` entry once the paper is on arXiv.
- Add a representative figure once the paper's headline result is finalised.

---

*End of Patch 6 documentation.*
