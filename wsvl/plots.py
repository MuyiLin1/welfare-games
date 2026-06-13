"""Plotting helpers for WSVL experiments."""

from __future__ import annotations

import numpy as np


def stack_curves(results, attr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error across seeds for a per-checkpoint attribute.

    Returns ``(T, mean, stderr)`` using the first run's checkpoint grid (all runs
    share it because checkpoints are deterministic in T).
    """
    T = getattr(results[0], "T")
    vals = np.vstack([getattr(r, attr) for r in results])
    mean = vals.mean(axis=0)
    stderr = vals.std(axis=0) / np.sqrt(vals.shape[0])
    return T, mean, stderr


def stack_epoch(results, attr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error across seeds for a per-epoch attribute."""
    x = getattr(results[0], "epoch_T")
    vals = np.vstack([getattr(r, attr) for r in results])
    mean = np.nanmean(vals, axis=0)
    stderr = np.nanstd(vals, axis=0) / np.sqrt(vals.shape[0])
    return x, mean, stderr
