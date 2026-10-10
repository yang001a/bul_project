"""Internal validation helpers for Phase 3 privacy mechanisms."""

from __future__ import annotations

import math
from numbers import Real

from bul_project.config import EXPERIMENT_SEEDS


def validate_l_max(l_max: int) -> int:
    """Return a validated positive integer trace-contribution bound."""

    if isinstance(l_max, bool) or not isinstance(l_max, int) or l_max < 1:
        raise ValueError("l_max must be a positive integer.")
    return l_max


def validate_t_max(t_max: float) -> float:
    """Return a validated finite, non-negative time-contribution bound."""

    if isinstance(t_max, bool) or not isinstance(t_max, Real):
        raise ValueError("t_max must be a finite non-negative number.")
    value = float(t_max)
    if not math.isfinite(value) or value < 0:
        raise ValueError("t_max must be a finite non-negative number.")
    return value


def validate_epsilon(epsilon: float, name: str = "epsilon") -> float:
    """Return a validated finite positive privacy budget."""

    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    value = float(epsilon)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def validate_sensitivity(sensitivity: float) -> float:
    """Return a validated finite non-negative global sensitivity."""

    if isinstance(sensitivity, bool) or not isinstance(sensitivity, Real):
        raise ValueError("sensitivity must be a finite non-negative number.")
    value = float(sensitivity)
    if not math.isfinite(value) or value < 0:
        raise ValueError("sensitivity must be a finite non-negative number.")
    return value


def validate_experiment_seed(seed: int) -> int:
    """Validate the paper's reproducibility seed range, 0 through 29."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in EXPERIMENT_SEEDS:
        raise ValueError(
            f"seed must be one of the configured experiment seeds "
            f"{EXPERIMENT_SEEDS[0]}..{EXPERIMENT_SEEDS[-1]}."
        )
    return seed
