# WSVL empirical testbed

A small, self-contained testbed for the empirical claims in
[`paper.tex`](paper.tex) (*Self-Play Learning of Welfare-Maximizing Equilibria*).
Its purpose is to supply evidence for **Assumption 10** (`ass:drift`, "welfare
drift with margin") — the single non-rigorous link the main upper bound
(Theorem 9) rests on — and to validate the equilibrium-side machinery
(Lemmas 7–8).

Everything runs in the degenerate normal-form slice `H = S = 1`, where the
welfare-optimal CCE is a polynomial-size linear program (Proposition 6(a)). That
gives a **ground-truth `OPT_CCE`** to measure welfare drift against, so every
claim is exactly checkable rather than self-certified.

## Layout

```
wsvl/
  games.py       normal-form game registry (Borowski, coordination, stag hunt, trap, ...)
  oracle.py      LP for welfare-optimal CCE + CCE-gap evaluator (ground truth)
  vlearning.py   V-learning core (EXP3 bandit + alpha-weighted certified policy)
                 + WSVL-C1 scalar welfare bonus + doubling-epoch schedule
  plots.py       seed-averaging helpers
experiments/
  assumption10_drift.py   the headline 3-panel drift figure
  drift_vs_eta0.py        steering budget vs welfare headroom across games
  schedule_ablation.py    the Lemma 8 p = 1/2 phase transition
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the figures

```bash
# (1) Headline: welfare drift discharges Assumption 10 on a recoverable game.
python -m experiments.assumption10_drift --game coordination --seeds 20 --epochs 14 --eta0 20

# (1b) Falsification: a genuine coordination trap where drift STALLS.
python -m experiments.assumption10_drift --game stag_hunt --seeds 20 --epochs 14 --eta0 20

# (2) Steering budget vs welfare headroom across three regimes.
python -m experiments.drift_vs_eta0 --games coordination stag_hunt trap --seeds 8 --epochs 13

# (3) Schedule ablation: the p = 1/2 knife edge (Lemma 8).
python -m experiments.schedule_ablation --game coordination --seeds 10 --epochs 14 --eta0 20
```

Figures are written to `figures/`.

## What the experiments show

**Assumption 10 has three separable sub-claims; the headline figure lands all
three** on the `coordination` game (`OPT_CCE` welfare 2.0; unshaped V-learning
stalls at the welfare-1.2 trap):

- **(A) Landing** — WSVL-C1 drives the welfare gap `OPT_CCE − SW` from ≈0.66
  (vanilla) to <0.01, i.e. it reaches the welfare-optimal CCE.
- **(B) Margin** — the measured per-epoch contraction
  `Δ̂_w = (WG_start − WG_end) / (η_k L_k)` stays strictly positive throughout the
  sub-optimal regime, then decays to 0 at convergence — exactly the "contracts
  until `SW ≥ OPT − ε`" wording of the assumption.
- **(C) Free** — the CCE gap stays on the vanilla V-learning `O(1/√T)` curve, so
  welfare shaping does not break the equilibrium guarantee (Lemmas 7–8).

**Where Assumption 10 fails.** On `stag_hunt`, the social action has no positive
welfare externality (switching to Stag against a Hare opponent *lowers* realized
welfare), so the scalar bonus gets no drift signal: the welfare gap plateaus and
some epochs show a *negative* margin. This is the coordination trap the paper
flags, and characterizing it is itself a contribution — the drift margin tracks
**externality alignment**, not the size of the welfare headroom `δ`.

**Steering budget.** `drift_vs_eta0` shows the final welfare gap decreasing
monotonically in `η_0` on recoverable games (the `poly(1/Δ_w)` factor of
Theorem 9 made visible) while `stag_hunt` floors out — recoverable vs
non-recoverable games separate cleanly.

**Schedule.** `schedule_ablation` confirms the Lemma 8 prediction that welfare
selection dies for `η_t ∼ t^{-p}` with `p > 1/2` (the integrated steering
`Σ_k L_k η_k` stops diverging) while `p ≤ 1/2` still selects the optimum.

## Caveats / honest notes

- These are normal-form (`H = S = 1`) games — the slice where the theory is
  already unconditional. Lifting to multi-state, multi-step Markov games
  (`S = 2–3`, `H = 3`) is the natural next step.
- The bandit is EXP3 (an adversarial no-regret learner consistent with
  V-learning's analysis), not V-learning's exact FTRL instantiation; the
  certified-policy `α`-weighting matches the paper.
- The `p < 1/2` half of the Lemma 8 transition (gap degradation) is mild at these
  game sizes because the shaping-bias term `B_w·η` is small; it should sharpen on
  larger action sets.
