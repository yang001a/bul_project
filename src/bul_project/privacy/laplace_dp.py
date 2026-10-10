"""Scalar Laplace mechanism with the paper-required post-processing."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal, Protocol

import numpy as np

from ._validation import (
    validate_epsilon,
    validate_experiment_seed,
    validate_sensitivity,
)


class _LaplaceRNG(Protocol):
    def laplace(self, loc: float, scale: float) -> float:
        """Draw one Laplace variate."""


def laplace_release(
    value: float,
    sensitivity: float,
    epsilon: float,
    *,
    statistic: Literal["count", "time"] = "count",
    seed: int | None = None,
    rng: _LaplaceRNG | None = None,
) -> int | float:
    """Release one count or time sum with calibrated Laplace noise.

    Input
    -----
    value:
        Non-negative clipped count or transition-time sum.
    sensitivity:
        Paper-defined global sensitivity for the statistic family.
    epsilon:
        Positive component privacy budget, never ``epsilon_total`` unless the
        selected mechanism explicitly assigns the full budget.
    statistic:
        ``count`` applies non-negative truncation and integer rounding;
        ``time`` applies non-negative truncation without rounding.
    seed, rng:
        Supply exactly one reproducible experiment seed or a shared random
        generator. A shared generator lets vector coordinates receive
        independent draws while the whole release remains reproducible.

    Output
    ------
    int | float
        ``value + Lap(sensitivity/epsilon)`` after the required deterministic
        post-processing.

    Paper experiment step
    ---------------------
    Overall Step 5/7: apply the Laplace mechanism, then round count statistics
    and truncate negative count or time-sum outputs to zero. Post-processing
    consumes no additional privacy budget.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be a finite non-negative number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError("value must be a finite non-negative number.")
    delta = validate_sensitivity(sensitivity)
    component_epsilon = validate_epsilon(epsilon)
    if statistic not in ("count", "time"):
        raise ValueError("statistic must be either 'count' or 'time'.")
    if seed is not None and rng is not None:
        raise ValueError("Provide seed or rng, not both.")
    if seed is None and rng is None:
        raise ValueError("A seed or shared rng is required for reproducibility.")
    if rng is None:
        rng = np.random.default_rng(validate_experiment_seed(seed))  # type: ignore[arg-type]

    noise = 0.0 if delta == 0 else float(rng.laplace(0.0, delta / component_epsilon))
    released = max(numeric_value + noise, 0.0)
    if statistic == "count":
        return int(math.floor(released + 0.5))
    return float(released)
