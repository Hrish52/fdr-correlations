"""Compute achievable |T| vs required threshold for a simulation config.

Usage:
    python scripts/calibrate_grid.py --p 500 --n 120 --block 20
"""
import argparse
import numpy as np
from scipy.stats import norm


def signal_T(rho, n, shrink=0.99):
    """Expected |T| for a true edge: X ~ N(0,I) vs Y with correlation rho."""
    rho_eff = shrink * rho
    return rho_eff * np.sqrt(n) / np.sqrt(2.0 + rho_eff ** 2)


def required_t(p, block, alpha):
    """Smallest threshold that can control FDR, assuming every rejection is true."""
    M = p * (p - 1) // 2
    m1 = block * (block - 1) // 2
    q_max = alpha * m1 / M
    return float(norm.isf(q_max / 2.0)), M, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rho-list", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    t_req, M, m1 = required_t(args.p, args.block, args.alpha)
    print(f"p={args.p}  n={args.n}  block={args.block}  alpha={args.alpha}")
    print(f"M = {M:,} edges, m1 = {m1} true edges")
    print(f"Required threshold: t >= {t_req:.3f}\n")
    print(f"{'rho':>6} {'|T|':>8} {'margin':>8}  regime")
    print("-" * 42)
    for rho in [float(x) for x in args.rho_list.split(",")]:
        T = signal_T(rho, args.n)
        margin = T - t_req
        if margin < -0.5:
            regime = "no power"
        elif margin < 0.3:
            regime = "threshold (~50%)"
        elif margin < 1.2:
            regime = "good power"
        else:
            regime = "saturated (~100%)"
        print(f"{rho:>6.2f} {T:>8.2f} {margin:>+8.2f}  {regime}")

    ceiling = signal_T(0.999, args.n)
    print(f"\nCeiling as rho -> 1: |T| = {ceiling:.2f}")
    if ceiling < t_req + 0.3:
        print(f"WARNING: unreachable at n={args.n}. Raise n or block size.")


if __name__ == "__main__":
    main()