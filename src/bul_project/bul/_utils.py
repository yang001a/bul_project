"""Shared validation for Phase 5 loss aggregation."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from numbers import Real
from typing import TypeVar

import numpy as np

from bul_project.config import WEIGHT_SUM_TOLERANCE


KeyT = TypeVar("KeyT", bound=Hashable)


def validate_loss(value: float, name: str) -> float:
    """Validate one already normalized higher-is-worse loss."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite loss in [0,1].")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 1:
        raise ValueError(f"{name} must be a finite loss in [0,1].")
    return numeric


def validate_weights(
    weights: Mapping[KeyT, float], expected_keys: Sequence[KeyT], name: str
) -> dict[KeyT, float]:
    """Validate a complete non-negative weight vector that sums to one."""

    expected = tuple(expected_keys)
    if set(weights) != set(expected):
        raise ValueError(f"{name} keys must be exactly {expected!r}.")
    checked: dict[KeyT, float] = {}
    for key in expected:
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} values must be finite and non-negative.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} values must be finite and non-negative.")
        checked[key] = numeric
    if not math.isclose(
        sum(checked.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_SUM_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to 1.")
    return checked


def weighted_loss(
    components: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Combine validated loss components without changing their formulas."""

    checked_weights = validate_weights(weights, tuple(components), "component weights")
    total = sum(
        validate_loss(value, key) * checked_weights[key]
        for key, value in components.items()
    )
    return float(np.clip(total, 0.0, 1.0))
