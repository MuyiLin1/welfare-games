"""Experiment: drift margin vs shaping strength and welfare headroom.

Probes how the welfare gap closes as a function of the initial shaping strength
``eta0``, across games with different unshaped welfare headroom ``delta``. This is
the proof-enabling characterization: it traces, empirically, how much steering
budget is needed to discharge Assumption 10, and how that scales with the
primitive ``delta = OPT_CCE - SW(unshaped CCE)`` -- the quantity the paper's
"What remains" section conjectures controls the drift margin ``Delta_w``.

Run:
    python -m experiments.drift_vs_eta0 --seeds 12
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
    ap.add_argument("--games", nargs="+",
                    default=["coordination", "stag_hunt", "trap"])
    ap.add_argument("--etas", nargs="+", type=float,
                    default=[0, 1, 2, 5, 10, 20, 40, 80])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))
    for gi, gname in enumerate(args.games):
        g = games.get(gname)
        opt = oracle.opt_cce(g)
        unshaped = run_many(g, opt.welfare, RunConfig(eta0=0.0, n_epochs=args.epochs),
                            args.seeds)
        delta = float(np.mean([r.welfare_gap[-1] for r in unshaped]))

        means, ses = [], []
        for eta0 in args.etas:
            res = run_many(g, opt.welfare,
                           RunConfig(eta0=eta0, n_epochs=args.epochs), args.seeds)
            wg = np.array([r.welfare_gap[-1] for r in res])
            means.append(wg.mean())
            ses.append(wg.std() / np.sqrt(len(wg)))
        means, ses = np.array(means), np.array(ses)

        ax.errorbar(args.etas, means, yerr=ses, marker="o", capsize=3,
                    color=f"C{gi}",
                    label=fr"{gname} ($\delta={delta:.2f}$)")
        print(f"[{gname}] delta={delta:.3f}  "
              + "  ".join(f"eta0={e:g}:{m:.3f}"
                          for e, m in zip(args.etas, means)))

    ax.set(xlabel=r"initial shaping strength $\eta_0$",
           ylabel=r"final welfare gap $\mathrm{OPT}_{CCE}-\mathrm{SW}$",
           title="Steering budget needed to discharge Assumption 10")
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.legend()
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "drift_vs_eta0.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
