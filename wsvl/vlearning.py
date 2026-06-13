"""V-learning and Welfare-Shaped V-Learning (WSVL) for normal-form games.

At ``H = S = 1`` a tabular Markov game collapses to a single adversarial
contextual bandit per player, and V-learning reduces to each player running a
no-regret bandit while the *certified policy* is the alpha-weighted mixture of
the per-round play distributions (Section ``sec:vlearning``). Because each round
draws actions from a product distribution but the certified mixture shares the
round index across players, the certified joint policy is a correlated mixture of
product distributions -- exactly a T-sparse CCE.

WSVL adds the scalar-bonus channel C1: at a realized joint action the broadcast
welfare-to-go ``w = (1/m) sum_j r_j`` shapes each player's reward to
``g~_i = g^_i + eta_k * w`` (Section ``sec:wsvl``), with ``eta_k`` decaying on a
doubling-epoch schedule ``L_k = 2^k``, ``eta_k = eta_0 * 2^{-k/2}``
(Equation ``eq:sched``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .games import Game
from .oracle import cce_gap, social_welfare


class Exp3Bandit:
    """Anytime EXP3 adversarial bandit over ``A`` actions (gain formulation).

    Rewards (gains) are in ``[0, 1]``. The exploration rate follows the standard
    anytime schedule ``gamma_t = min(1, sqrt(A ln A / ((e-1) t)))`` so the
    per-round play distribution is no-regret without knowing the horizon.
    """

    def __init__(self, A: int, rng: np.random.Generator):
        self.A = A
        self.rng = rng
        self.log_w = np.zeros(A)  # log multiplicative weights
        self.t = 0

    def _gamma(self) -> float:
        t = max(self.t, 1)
        return min(1.0, math.sqrt(self.A * math.log(self.A) / ((math.e - 1) * t)))

    def play_dist(self) -> np.ndarray:
        """Return the play distribution used this round (mixture with uniform)."""
        gamma = self._gamma()
        w = np.exp(self.log_w - self.log_w.max())
        theta = w / w.sum()
        return (1.0 - gamma) * theta + gamma / self.A

    def update(self, action: int, gain: float, p: np.ndarray) -> None:
        """Importance-weighted EXP3 update for the played ``action``."""
        gamma = self._gamma()
        ghat = gain / p[action]
        self.log_w[action] += gamma * ghat / self.A
        self.t += 1


@dataclass
class RunConfig:
    """Configuration for a single WSVL run."""

    eta0: float = 1.0          # initial shaping strength (eta0 = 0 -> vanilla)
    n_epochs: int = 14         # K; total episodes T = sum_k 2^k = 2^{K+1} - 2
    base_len: int = 1          # epoch length multiplier: L_k = base_len * 2^k
    p_exponent: float = 0.5    # eta_t ~ t^{-p}; p = 1/2 is the Lemma 8 threshold
    seed: int = 0


@dataclass
class RunResult:
    """Per-checkpoint and per-epoch telemetry from a run."""

    T: np.ndarray                 # episode counts at each checkpoint
    welfare_gap: np.ndarray       # OPT_CCE - SW(certified policy)
    cce_gap: np.ndarray           # equilibrium gap of certified policy
    social_welfare: np.ndarray    # SW(certified policy)
    epoch_T: np.ndarray           # episode count at each epoch boundary
    epoch_eta: np.ndarray         # eta_k used in each epoch
    epoch_len: np.ndarray         # L_k for each epoch
    epoch_wg: np.ndarray          # welfare gap at each epoch boundary
    delta_w_hat: np.ndarray = field(default=None)  # measured per-epoch margin

    def __post_init__(self) -> None:
        # Per-epoch empirical drift margin:
        #   Delta_w_hat_k = (WG_start - WG_end) / (eta_k * L_k),
        # the realized contraction normalised by the shaping budget spent.
        wg = self.epoch_wg
        start, end = wg[:-1], wg[1:]
        budget = self.epoch_eta[1:] * self.epoch_len[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.delta_w_hat = np.where(budget > 0, (start - end) / budget, np.nan)


def _eta_schedule(cfg: RunConfig, epoch: int) -> float:
    """Shaping strength for ``epoch`` (1-indexed): eta_k = eta0 * L_k^{-p}.

    With ``p = 1/2`` and ``L_k = 2^k`` this is the paper's
    ``eta_k = eta0 * 2^{-k/2}`` (Equation ``eq:sched``); other ``p`` values are
    used by the schedule ablation probing the Lemma 8 phase transition.
    """
    Lk = cfg.base_len * (2 ** epoch)
    return cfg.eta0 * (Lk ** (-cfg.p_exponent))


def run_wsvl(
    game: Game,
    opt_welfare: float,
    cfg: RunConfig,
    n_checkpoints: int = 60,
) -> RunResult:
    """Run (Welfare-Shaped) V-learning on a normal-form ``game``.

    Set ``cfg.eta0 = 0`` to recover vanilla V-learning. Returns telemetry on the
    *certified* correlated policy ``mu_hat`` (the alpha-weighted product mixture),
    which is the object the theory certifies as an (approximate) CCE.
    """
    rng = np.random.default_rng(cfg.seed)
    A1, A2 = game.A1, game.A2

    b1 = Exp3Bandit(A1, rng)
    b2 = Exp3Bandit(A2, rng)

    # Certified joint policy, maintained via the V-learning alpha recursion
    #   mu_hat <- (1 - alpha_t) mu_hat + alpha_t (p1 (x) p2),  alpha_t = 2/(1+t)
    # (H = 1 so alpha_t = (H+1)/(H+t) = 2/(1+t)). Initialised to uniform (x) uniform.
    mu_hat = np.outer(np.full(A1, 1.0 / A1), np.full(A2, 1.0 / A2))

    total_T = sum(cfg.base_len * (2 ** k) for k in range(1, cfg.n_epochs + 1))
    checkpoints = sorted(
        set(np.unique(np.geomspace(1, total_T, n_checkpoints).astype(int)))
    )
    cp_set = set(checkpoints)

    T_log, wg_log, gap_log, sw_log = [], [], [], []
    epoch_T, epoch_eta, epoch_len, epoch_wg = [], [], [], []

    # Record the initial (pre-learning) epoch boundary.
    epoch_T.append(0)
    epoch_eta.append(0.0)
    epoch_len.append(0)
    epoch_wg.append(opt_welfare - social_welfare(mu_hat, game))

    t = 0
    for k in range(1, cfg.n_epochs + 1):
        eta_k = _eta_schedule(cfg, k)
        Lk = cfg.base_len * (2 ** k)
        for _ in range(Lk):
            t += 1
            p1 = b1.play_dist()
            p2 = b2.play_dist()
            a1 = rng.choice(A1, p=p1)
            a2 = rng.choice(A2, p=p2)

            r1 = game.u1[a1, a2]
            r2 = game.u2[a1, a2]
            w = 0.5 * (r1 + r2)  # broadcast scalar welfare-to-go (m = 2, H = 1)

            # C1 shaped reward, renormalised to [0, 1].
            g1 = (r1 + eta_k * w) / (1.0 + eta_k)
            g2 = (r2 + eta_k * w) / (1.0 + eta_k)

            b1.update(a1, g1, p1)
            b2.update(a2, g2, p2)

            alpha_t = 2.0 / (1.0 + t)
            mu_hat = (1.0 - alpha_t) * mu_hat + alpha_t * np.outer(p1, p2)

            if t in cp_set:
                sw = social_welfare(mu_hat, game)
                T_log.append(t)
                sw_log.append(sw)
                wg_log.append(opt_welfare - sw)
                gap_log.append(cce_gap(mu_hat, game))

        epoch_T.append(t)
        epoch_eta.append(eta_k)
        epoch_len.append(Lk)
        epoch_wg.append(opt_welfare - social_welfare(mu_hat, game))

    return RunResult(
        T=np.asarray(T_log),
        welfare_gap=np.asarray(wg_log),
        cce_gap=np.asarray(gap_log),
        social_welfare=np.asarray(sw_log),
        epoch_T=np.asarray(epoch_T),
        epoch_eta=np.asarray(epoch_eta),
        epoch_len=np.asarray(epoch_len),
        epoch_wg=np.asarray(epoch_wg),
    )


def run_many(
    game: Game,
    opt_welfare: float,
    cfg: RunConfig,
    n_seeds: int,
    n_checkpoints: int = 60,
) -> list[RunResult]:
    """Run ``n_seeds`` independent replications differing only by RNG seed."""
    results = []
    for s in range(n_seeds):
        c = RunConfig(**{**cfg.__dict__, "seed": cfg.seed + s})
        results.append(run_wsvl(game, opt_welfare, c, n_checkpoints))
    return results
