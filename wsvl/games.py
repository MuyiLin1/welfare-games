"""Normal-form game registry (H = S = 1).

Each game is a pair of payoff tensors ``(u1, u2)`` of shape ``(A1, A2)`` with
entries in ``[0, 1]`` (the reward range V-learning assumes). Row index = player 1
action, column index = player 2 action.

The registry deliberately covers the four regimes the drift story needs:

* ``borowski``  -- correlation strictly beats every product/Nash distribution;
                   the welfare-optimal CCE is the 50/50 anti-diagonal mix with
                   per-player value 0.8 (the paper's reference example).
* ``stag_hunt`` -- a Pareto-dominated risk-dominant trap; vanilla self-play
                   tends to the low-welfare equilibrium, shaping should escape it.
* ``chicken``   -- canonical anti-coordination; correlation helps.
* ``trap``      -- a deep low-welfare basin used as a *falsification* target,
                   where C1 shaping is expected to stall (Assumption 10 fails).
* ``prisoners`` -- sanity check: a dominant-strategy game with a unique CCE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Game:
    """A two-player normal-form game with payoffs in [0, 1]."""

    name: str
    u1: np.ndarray  # shape (A1, A2)
    u2: np.ndarray  # shape (A1, A2)

    @property
    def A1(self) -> int:
        return self.u1.shape[0]

    @property
    def A2(self) -> int:
        return self.u1.shape[1]

    def welfare_tensor(self) -> np.ndarray:
        """Per-cell social welfare ``u1 + u2``."""
        return self.u1 + self.u2

    def __post_init__(self) -> None:
        assert self.u1.shape == self.u2.shape, "payoff tensors must match"
        for u in (self.u1, self.u2):
            if u.min() < -1e-9 or u.max() > 1 + 1e-9:
                raise ValueError(f"{self.name}: payoffs must lie in [0, 1]")


def _g(name: str, u1, u2) -> Game:
    return Game(name=name, u1=np.asarray(u1, float), u2=np.asarray(u2, float))


# --- Borowski reference example -------------------------------------------------
# Anti-coordination: the diagonal is worthless, so the welfare-optimal CCE must
# correlate the two off-diagonal cells. The 50/50 mix 0.5*(T,R) + 0.5*(B,L) gives
# each player (0.9 + 0.7) / 2 = 0.8 -- the paper's quoted value -- and strictly
# beats any product distribution, which cannot avoid the zero diagonal.
BOROWSKI = _g(
    "borowski",
    u1=[[0.0, 0.9],
        [0.7, 0.0]],
    u2=[[0.0, 0.7],
        [0.9, 0.0]],
)

# --- Coordination game with a positive welfare externality (DRIFT FIRES) -------
# Symmetric stag-hunt-like game (row/col 0 = Stag, 1 = Hare) engineered so that
# Assumption 10 *holds*: both (Stag,Stag) and (Hare,Hare) are CCE, vanilla
# self-play is drawn to the low-welfare (Hare,Hare) trap, but switching to Stag
# carries a positive welfare externality -- w(Stag,Hare) = 0.725 > w(Hare,Hare)
# = 0.6 -- so the C1 welfare bonus creates a strictly positive drift margin that
# pulls the certified policy out of the trap toward the (Stag,Stag) optimum.
#   OPT_CCE = (Stag,Stag) welfare 2.0; trap (Hare,Hare) welfare 1.2; delta = 0.8.
COORDINATION = _g(
    "coordination",
    u1=[[1.00, 0.50],
        [0.95, 0.60]],
    u2=[[1.00, 0.95],
        [0.50, 0.60]],
)

# --- Stag hunt: a risk-dominant low-welfare trap (DRIFT STALLS) ----------------
# Classic stag hunt with *no* welfare externality for the social action:
# unilaterally switching to Stag against a Hare opponent *lowers* realized
# welfare (w(Stag,Hare) = 0.3 < w(Hare,Hare) = 0.6), so the C1 bonus offers no
# drift and the certified policy stalls at (Hare,Hare). This is the genuine
# coordination trap where Assumption 10 fails -- used as a falsification target.
#   OPT_CCE = 1.6, trap = 1.2, delta = 0.4.
STAG_HUNT = _g(
    "stag_hunt",
    u1=[[0.8, 0.0],
        [0.6, 0.6]],
    u2=[[0.8, 0.6],
        [0.0, 0.6]],
)

# --- Chicken -------------------------------------------------------------------
CHICKEN = _g(
    "chicken",
    u1=[[0.6, 0.2],
        [0.7, 0.0]],
    u2=[[0.6, 0.7],
        [0.2, 0.0]],
)

# --- Deep coordination trap (DRIFT FAILS HARD; falsification target) ----------
# A strongly risk-dominant low-welfare equilibrium (0,0) with a negative welfare
# externality: unilaterally switching to the social action (1) against a greedy
# (0) opponent *lowers* welfare -- w(social,greedy) = 0.9 < w(greedy,greedy) =
# 1.2 -- and costs the deviator heavily (0.0 vs 0.6), so the trap basin is deep.
# The C1 bonus gets no positive drift signal and stalls regardless of eta0.
# Crucially the welfare headroom here (delta = 0.8) EXCEEDS coordination's, yet
# recovery is worse -- showing the drift margin tracks externality alignment, not
# headroom size. OPT_CCE = (1,1) welfare 2.0; trap (0,0) welfare 1.2.
TRAP = _g(
    "trap",
    u1=[[0.6, 0.9],
        [0.0, 1.0]],
    u2=[[0.6, 0.0],
        [0.9, 1.0]],
)

# --- Prisoner's dilemma (sanity) -----------------------------------------------
PRISONERS = _g(
    "prisoners",
    u1=[[0.6, 0.0],
        [1.0, 0.2]],
    u2=[[0.6, 1.0],
        [0.0, 0.2]],
)


REGISTRY: dict[str, Game] = {
    g.name: g for g in (BOROWSKI, COORDINATION, STAG_HUNT, CHICKEN, TRAP, PRISONERS)
}


def get(name: str) -> Game:
    """Look up a game by name."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown game {name!r}; available: {sorted(REGISTRY)}"
        ) from exc
