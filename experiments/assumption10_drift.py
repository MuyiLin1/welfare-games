"""Experiment: empirical evidence for Assumption 10 (welfare drift with margin).

Validates the three separable sub-claims of ``ass:drift`` in paper.tex on the
Borowski normal-form game, where the welfare-optimal CCE is a ground-truth LP:

  (A) Landing   -- the welfare gap WG(T) = OPT_CCE - SW(certified) drops toward 0
                   under WSVL-C1, while vanilla V-learning stalls at a
                   lower-welfare CCE.
  (B) Margin    -- the per-epoch contraction Delta_w_hat_k stays bounded below by
                   a positive constant while WG > eps; this constant IS the
                   empirical drift margin Delta_w.
  (C) Free      -- the equilibrium gap stays on the V-learning O(1/sqrt(T)) curve
                   throughout, so shaping does not break the CCE guarantee
                   (Lemmas 7-8).

Run:
    python -m experiments.assumption10_drift --game borowski --seeds 20
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

from wsvl import games, oracle, plots  # noqa: E402
from wsvl.vlearning import RunConfig, run_many  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="borowski", choices=sorted(games.REGISTRY))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--eta0", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05, help="target accuracy band")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    game = games.get(args.game)
    opt = oracle.opt_cce(game)
    print(f"[{game.name}] OPT_CCE welfare = {opt.welfare:.4f}")
    print(f"[{game.name}] OPT_CCE mu =\n{np.round(opt.mu, 3)}")

    vanilla_cfg = RunConfig(eta0=0.0, n_epochs=args.epochs)
    wsvl_cfg = RunConfig(eta0=args.eta0, n_epochs=args.epochs)

    vanilla = run_many(game, opt.welfare, vanilla_cfg, args.seeds)
    wsvl = run_many(game, opt.welfare, wsvl_cfg, args.seeds)

    # Report the unshaped welfare headroom delta = OPT - SW(unshaped CCE): the
    # primitive Assumption 10 conjectures Delta_w should scale with.
    v_final_wg = np.mean([r.welfare_gap[-1] for r in vanilla])
    w_final_wg = np.mean([r.welfare_gap[-1] for r in wsvl])
    delta = v_final_wg  # OPT - SW(unshaped) at the final iterate
    print(f"[{game.name}] welfare gap (vanilla, final) = {v_final_wg:.4f}")
    print(f"[{game.name}] welfare gap (WSVL-C1, final) = {w_final_wg:.4f}")
    print(f"[{game.name}] unshaped welfare headroom delta = {delta:.4f}")

    # Report the empirical drift margin over the *contracting* regime only
    # (epochs whose starting welfare gap still exceeds eps): this is the regime
    # Assumption 10 makes a claim about. Converged epochs (gap < eps) are excluded.
    contracting = []
    for r in wsvl:
        start_wg = r.epoch_wg[:-1]
        mask = (start_wg > args.eps) & ~np.isnan(r.delta_w_hat)
        contracting.append(r.delta_w_hat[mask])
    contracting = np.concatenate(contracting)
    print(f"[{game.name}] measured drift margin Delta_w_hat over contracting "
          f"regime: median={np.median(contracting):.4f}  "
          f"min={np.min(contracting):.4f} (>0 validates Assumption 10)")
    _plot(game.name, opt.welfare, vanilla, wsvl, args.eps, args.outdir)


def _plot(name, opt_welfare, vanilla, wsvl, eps, outdir) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel A: welfare gap vs T.
    ax = axes[0]
    for res, label, color in [(vanilla, "vanilla V-learning", "C1"),
                              (wsvl, "WSVL-C1", "C0")]:
        T, mean, se = plots.stack_curves(res, "welfare_gap")
        ax.plot(T, mean, color=color, label=label)
        ax.fill_between(T, mean - se, mean + se, color=color, alpha=0.2)
    ax.axhline(eps, ls="--", color="gray", lw=1, label=fr"$\epsilon={eps}$")
    ax.set(xscale="log", xlabel="episodes $T$",
           ylabel=r"welfare gap $\mathrm{OPT}_{CCE}-\mathrm{SW}$",
           title="(A) Landing: drift closes the welfare gap")
    ax.legend(fontsize=8)

    # Panel B: measured per-epoch drift margin.
    ax = axes[1]
    x, mean, se = plots.stack_epoch(wsvl, "delta_w_hat")
    x = wsvl[0].epoch_T[1:]
    ax.errorbar(x, mean[: len(x)] if len(mean) > len(x) else mean,
                yerr=se[: len(x)] if len(se) > len(x) else se,
                marker="o", color="C0", capsize=3)
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.set(xscale="log",
           xlabel="episodes $T$ (epoch boundary)",
           ylabel=r"measured margin $\widehat{\Delta}_w$",
           title="(B) Margin: per-epoch contraction stays positive")

    # Panel C: equilibrium gap vs T with O(1/sqrt(T)) reference.
    ax = axes[2]
    for res, label, color in [(vanilla, "vanilla V-learning", "C1"),
                              (wsvl, "WSVL-C1", "C0")]:
        T, mean, se = plots.stack_curves(res, "cce_gap")
        ax.plot(T, mean, color=color, label=label)
        ax.fill_between(T, mean - se, mean + se, color=color, alpha=0.2)
    T = wsvl[0].T
    ref = mean[0] * np.sqrt(T[0]) / np.sqrt(T)
    ax.plot(T, ref, ls=":", color="black", lw=1.2, label=r"$\propto 1/\sqrt{T}$")
    ax.set(xscale="log", yscale="log", xlabel="episodes $T$",
           ylabel="CCE gap of certified policy",
           title="(C) Free: gap stays at the V-learning rate")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Assumption 10 (welfare drift) on the {name} game", y=1.02, fontsize=12
    )
    fig.tight_layout()
    out = os.path.join(outdir, f"assumption10_{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
