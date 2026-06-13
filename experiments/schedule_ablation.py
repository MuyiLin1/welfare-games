"""Experiment: schedule ablation -- the p = 1/2 phase transition (Lemma 8).

Sweeps the shaping-decay exponent ``eta_t = c * t^{-p}`` over
``p in {0.25, 0.5, 0.75, 1.0}`` and reports both the final equilibrium (CCE) gap
and the final welfare gap. Lemma 8 (`lem:sched`) predicts a phase transition at
``p = 1/2``:

  * ``p < 1/2``: the alpha-weighted shaping bias dominates the V-learning
    ``1/sqrt(T)`` term, degrading the equilibrium gap to ``O(T^{-p})``.
  * ``p > 1/2``: the gap rate is preserved but the integrated steering
    ``sum_k L_k eta_k`` stops diverging, so welfare selection dies (welfare gap
    stays large).

Only ``p = 1/2`` balances both. Run:
    python -m experiments.schedule_ablation --game coordination --seeds 12
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsvl import games, oracle  # noqa: E402
from wsvl.vlearning import RunConfig, run_many  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="coordination", choices=sorted(games.REGISTRY))
    ap.add_argument("--ps", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--eta0", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    g = games.get(args.game)
    opt = oracle.opt_cce(g)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for p in args.ps:
        cfg = RunConfig(eta0=args.eta0, n_epochs=args.epochs, p_exponent=p)
        res = run_many(g, opt.welfare, cfg, args.seeds)
        T = res[0].T
        gap = np.vstack([r.cce_gap for r in res]).mean(axis=0)
        wg = np.vstack([r.welfare_gap for r in res]).mean(axis=0)
        ls = "-" if abs(p - 0.5) < 1e-9 else "--"
        lw = 2.4 if abs(p - 0.5) < 1e-9 else 1.4
        axes[0].plot(T, gap, ls, lw=lw, label=fr"$p={p}$")
        axes[1].plot(T, wg, ls, lw=lw, label=fr"$p={p}$")
        print(f"p={p}: final CCE gap={gap[-1]:.4f}  final welfare gap={wg[-1]:.4f}")

    axes[0].set(xscale="log", yscale="log", xlabel="episodes $T$",
                ylabel="CCE gap", title="(a) equilibrium gap vs schedule")
    axes[1].set(xscale="log", xlabel="episodes $T$",
                ylabel=r"welfare gap", title="(b) welfare selection vs schedule")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(fr"Schedule ablation on {args.game}: $p=1/2$ is the knife edge",
                 y=1.02)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"schedule_ablation_{args.game}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
