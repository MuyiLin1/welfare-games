"""Welfare-Shaped V-Learning (WSVL) empirical testbed.

A minimal, self-contained package for validating the welfare-drift mechanism
(Assumption 10 / `ass:drift` in paper.tex) on normal-form games (H=S=1), where
the welfare-optimal CCE is a polynomial-size linear program and therefore a
checkable ground truth.
"""

from . import games, oracle, vlearning

__all__ = ["games", "oracle", "vlearning"]
