"""Finite-sample bias of the Cai-Liu kurtosis estimator.

kappa_hat averages the standardised fourth moment across columns. For
heavy-tailed marginals the sample fourth moment is biased downward at
moderate n, so kappa_hat understates kappa. Since kappa sits in the
numerator of the Eq. (4) variance, understating it shrinks the denominator
of T and makes the procedure ANTI-conservative. This script measures the
size of that effect.

Usage:
    python scripts/kappa_bias.py --reps 200
"""
import sys, pathlib, argparse
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.LCT import _kappa_hat, _zscore_columns
from src.Simulate import (
    sample_gaussian, sample_t, sample_t_cl, sample_exp_cl,
    sample_normal_mixture,
)

# Theoretical kappa = (1/3) E(X-mu)^4 / [E(X-mu)^2]^2
#   Gaussian:       E z^4 = 3            -> kappa = 1
#   t_df:           excess kurtosis 6/(df-4) -> kappa = (3 + 6/(df-4))/3
#   Exp(1):         E z^4 = 9            -> kappa = 3
#   U*Z mixture:    kappa = 9/5
THEORY = {
    "gaussian": 1.0,
    "t6":       (3 + 6 / (6 - 4)) / 3,   # = 2.0
    "t8":       (3 + 6 / (8 - 4)) / 3,   # = 1.5
    "exp_cl":   3.0,
    "nmix":     9 / 5,
}


def _draw(name, n, p, seed):
    I = np.eye(p)
    if name == "gaussian": return sample_gaussian(n, I, seed=seed)
    if name == "t6":       return sample_t_cl(n, 6, I, rng=seed)
    if name == "t8":       return sample_t_cl(n, 8, I, rng=seed)
    if name == "exp_cl":   return sample_exp_cl(n, 1.0, I, rng=seed)
    if name == "nmix":     return sample_normal_mixture(n, I, rng=seed)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default="50,80,120,200,500,2000")
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/summary/kappa_bias.csv")
    args = ap.parse_args()

    ns = [int(x) for x in args.n_list.split(",")]
    rows = []
    for name, k_true in THEORY.items():
        for n in ns:
            est = [_kappa_hat(_zscore_columns(_draw(name, n, args.p, r)))
                   for r in range(args.reps)]
            est = np.asarray(est)
            rows.append({
                "marginal": name, "n": n, "kappa_true": k_true,
                "kappa_mean": est.mean(), "kappa_sd": est.std(ddof=1),
                "bias": est.mean() - k_true,
                "rel_bias": (est.mean() - k_true) / k_true,
            })
            print(f"{name:9s} n={n:5d}  kappa_hat={est.mean():.3f} "
                  f"(sd {est.std(ddof=1):.3f})  true={k_true:.3f}  "
                  f"rel_bias={(est.mean()-k_true)/k_true:+.1%}")

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()