"""Ground-truth equilibrium oracle for normal-form games (H = S = 1).

At ``H = S = 1`` the welfare-optimal CCE is a polynomial-size linear program
(Proposition 6(a) / ``prop:gate`` in paper.tex): coarse deviations fix an action,
so each no-deviation constraint is linear in the joint distribution ``mu``. This
module computes that optimum and the CCE gap, giving the experiment a checkable
target to measure welfare drift against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .games import Game


def social_welfare(mu: np.ndarray, game: Game) -> float:
    """Utilitarian social welfare ``sum_a mu(a) (u1 + u2)``."""
    return float(np.sum(mu * game.welfare_tensor()))


def cce_gap(mu: np.ndarray, game: Game) -> float:
    """CCE gap: the largest gain any single player gets from a fixed-action
    coarse deviation, ``max_i max_{a_i'} E_mu[u_i(a_i', a_{-i}) - u_i(a)]``.
    """
    marg1 = mu.sum(axis=1)  # player 1's action marginal, shape (A1,)
    marg2 = mu.sum(axis=0)  # player 2's action marginal, shape (A2,)

    val1 = float(np.sum(mu * game.u1))
    val2 = float(np.sum(mu * game.u2))

    # Best fixed-action deviation value for each player against the opponent
    # marginal induced by mu.
    dev1 = float(np.max(game.u1 @ marg2))  # max_{a1'} sum_{a2} marg2[a2] u1[a1',a2]
    dev2 = float(np.max(marg1 @ game.u2))  # max_{a2'} sum_{a1} marg1[a1] u2[a1,a2']

    return max(dev1 - val1, dev2 - val2)


def _cce_constraints(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Build the CCE no-deviation inequality system ``A_ub @ vec(mu) <= 0``.

    For player 1 deviating to action ``a1'``:
        sum_{x,y} mu[x,y] (u1[a1',y] - u1[x,y]) <= 0.
    For player 2 deviating to action ``b'``:
        sum_{x,y} mu[x,y] (u2[x,b'] - u2[x,y]) <= 0.
    """
    A1, A2 = game.A1, game.A2
    rows = []

    for a1p in range(A1):
        # coefficient on mu[x, y] is u1[a1', y] - u1[x, y]
        coef = game.u1[a1p, :][None, :] - game.u1  # shape (A1, A2)
        rows.append(coef.ravel())

    for b1p in range(A2):
        coef = game.u2[:, b1p][:, None] - game.u2  # shape (A1, A2)
        rows.append(coef.ravel())

    A_ub = np.asarray(rows, float)
    b_ub = np.zeros(A_ub.shape[0])
    return A_ub, b_ub


@dataclass(frozen=True)
class OptCCE:
    """Result of the welfare-optimal-CCE linear program."""

    welfare: float
    mu: np.ndarray  # the maximizing joint distribution, shape (A1, A2)


def opt_cce(game: Game) -> OptCCE:
    """Solve for the welfare-maximizing CCE via linear programming.

    Maximizes ``sum_a mu(a)(u1 + u2)`` subject to the CCE no-deviation
    constraints, ``mu >= 0``, and ``sum_a mu(a) = 1``.
    """
    A1, A2 = game.A1, game.A2
    n = A1 * A2

    c = -game.welfare_tensor().ravel()  # linprog minimizes; we maximize welfare
    A_ub, b_ub = _cce_constraints(game)
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"opt_cce LP failed for {game.name}: {res.message}")

    mu = res.x.reshape(A1, A2)
    mu = np.clip(mu, 0.0, None)
    mu /= mu.sum()
    return OptCCE(welfare=social_welfare(mu, game), mu=mu)
