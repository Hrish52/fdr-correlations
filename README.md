# fdr-correlations

Replication and empirical stress-test of the LCT-N and LCT-B procedures for large-scale multiple testing of correlations (Cai & Liu, 2016), with comparison against Fisher-z + BH/BY baselines under Gaussian and non-Gaussian marginals.

**Author:** Hrishikesh Dhole (Data Analyst & Data Scientist)  
**Supervisor:** Prof. Elio Zhang  
**Status:** Active development. Paper draft in progress — see `docs/patches/` for the change log.

---

## What this repository does

Given two independent samples $X \in \mathbb{R}^{n_1 \times p}$ and $Y \in \mathbb{R}^{n_2 \times p}$, we test edge-level equality of correlations:

$$H_{0,ij}: \rho_{ij,1} = \rho_{ij,2} \text{ for all } i < j$$

while controlling the false discovery rate at level $\alpha$. Four procedures are implemented and compared:

*   **LCT-N:** Normal-tail threshold (Cai & Liu 2016, Eq. 9)
*   **LCT-B:** Bootstrap threshold (Cai & Liu 2016, Eq. 10–12)
*   **Fisher-z + BH:** Benjamini–Hochberg on Fisher-transformed correlations
*   **Fisher-z + BY:** Benjamini–Yekutieli on Fisher-transformed correlations

Simulations cover Gaussian, $t_6$, Laplace, and Exp(1) marginals across sample sizes $n_1 = n_2 \in \{60, 80, 120\}$, dimensions $p \in \{250, 500, 1000\}$, and multiple block-correlation structures (simple block, block-AR(1), block-decay).

---

## Quick start

```bash
# Clone and set up
git clone https://github.com/Hrish52/fdr-correlations.git
cd fdr-correlations
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Verify install
pytest tests/ -v

# Run a small null-calibration
python scripts/run_null_calibration.py --model t --p 250 --reps 10 --B 100

# Run a small power-curves grid
python scripts/run_power_curves.py --models gaussian --p 250 --reps 20
```

> **Note:** Requires Python 3.10 or later.

---

## Repository layout

```text
src/                          # Core algorithms
├── LCT.py                    # LCT-N: edge statistic + normal-tail threshold
├── LCTB.py                   # LCT-B (original): pooled bootstrap threshold
├── LCTB_v2.py                # LCT-B (optimized): streaming, coarse-grid, float32
├── FisherBaselines.py        # Fisher-z + BH/BY baselines
├── Simulate.py               # Gaussian and non-Gaussian samplers, block covariances
├── Evaluate.py               # FDR/power utilities
├── defaults.py               # Loader/resolver for results/defaults.json
└── Plots.py                  # Heatmap helpers

scripts/                      # Driver scripts (CLI, results to results/tables/)
├── run_sim_gaussian.py       # Full Gaussian + non-Gaussian grid
├── run_null_calibration.py   # Pure-null runs to calibrate B and defaults
├── run_power_curves.py       # Power/FDR vs (n, ρ) grids
├── run_robustness.py         # Ablations over covariance kind and variance estimator
├── make_defaults.py          # Build results/defaults.json from calibration runs
└── summarize_csvs.py         # Quick-look summaries over results/tables/*.csv

notebooks/                    # Analysis and figures
├── 01_fisher_baselines
├── 03_simulations
├── 04_lct_sandbox
├── 05_non_gaussian
├── 06_calibration
├── 07_power_curves
├── 08_robustness
├── 09_scaling
├── 10_defaults
└── 11_realdata_experiment    # Week 3 — in progress

tests/                        # Pytest suite (14 tests, ~10s to run)

results/                      # Outputs
├── defaults.json             # Locked per-(p, α) defaults for B, coarse_grid, winsorize
├── tables/                   # Raw CSVs from driver scripts (git-ignored)
└── summary/                  # Aggregated tables checked in (git-tracked)

docs/patches/                 # Change-log research notes, one per patch
```

---

## Reproducing published figures

1. **Run the full simulation grid:**
   ```bash
   python scripts/run_sim_gaussian.py
   python scripts/run_power_curves.py --reps 50
   python scripts/run_robustness.py --model gaussian --cov-kind block_ar1
   ```
2. **Build the calibration defaults:**
   ```bash
   python scripts/run_null_calibration.py --p 250 --reps 50
   python scripts/run_null_calibration.py --p 500 --reps 30
   python scripts/make_defaults.py --winsorize 5
   ```
3. **Analyze:** Open the analysis notebooks in Jupyter and run them top-to-bottom. Each notebook's first cell locates `results/tables/` automatically.

Every driver logs its grid and seeds; re-running with the same seed reproduces the exact CSV.

---

## Key methodological decisions

*   **Null definition (Patch 2):** Non-Gaussian null runs draw both groups from the same marginal family with identity covariance, matching Cai & Liu's $H_0$ definition. Prior mixed-marginal behavior ($X$ Gaussian regardless of $Y$) is preserved as an ablation via `--x-model gaussian`.
*   **Threshold selection (Patch 1):** All three implementations (LCT-N, LCT-B, LCT-B v2) return the infimum of $\{t : \text{est}_{\text{FDR}}(t) \leq \alpha\}$ per Cai & Liu Eq. (9), by ascending scan with early return.
*   **Bootstrap defaults:** `B=100` for $p \approx 250$; `B=200` for $p \geq 500$ or non-Gaussian marginals with noisy null control. Cai–Liu variance estimator by default; light winsorize (5) recommended for heavy-tailed data. Locked in `results/defaults.json` via `scripts/make_defaults.py`.
*   **Windows-friendly parallelism:** All scripts default to `n_jobs=1` on Windows to avoid `joblib` spawn overhead. Linux/macOS get full parallelism automatically.

---

## Change log

Every non-trivial change lands with a research note in `docs/patches/`. See:

*   `patch01_threshold_selection.md` — consistent Eq. (9) infimum semantics across LCT-N/B/v2
*   `patch02_null_calibration.md` — matching-marginal $H_0$ by default
*   `patch03_repo_hygiene.md` — remove committed scratch files
*   `patch04_notebook_renumbering.md` — contiguous notebook sequence
*   `patch05_requirements.md` — pin runtime dependencies
*   `patch06_readme.md` — this README

---

## References

*   **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *Journal of the American Statistical Association*, 111(513), 229–240.
*   **Sánchez-Gómez, J. A., Zhang, E., & Liu, Y. (2025).** Effective Permutation Tests for Differences Across Multiple High-Dimensional Correlation Matrices.
*   **Vesely, A., Finos, L., & Goeman, J. J. (2023).** Permutation-based true discovery guarantee by sum tests. *Journal of the Royal Statistical Society Series B*, 85(3), 664–683.

---

## License

See `LICENSE` in the repository root.

## Contact

Hrishikesh Dhole — [hrishikeshdhole0@gmail.com](mailto:hrishikeshdhole0@gmail.com)
