"""One-dimensional Earth Mover's/Wasserstein distance for transition times."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import wasserstein_distance

from bul_project.config import NEMD_ZERO_SCALE

from ._validation import SampleInput, validate_samples


def calculate_emd(reference: SampleInput, comparison: SampleInput) -> float:
    """Calculate raw one-dimensional Earth Mover's Distance.

    Input
    -----
    reference, comparison:
        Non-empty, finite, non-negative transition-time samples for the same
        frozen DFG edge.

    Output
    ------
    float
        The established 1D EMD/Wasserstein-1 distance in the samples' time
        unit. It is already loss-directed but is not yet scale-normalized.

    Paper experiment step
    ---------------------
    Overall Step 9.2: compare transition-level waiting-time distributions per
    edge without mixing samples from different edge meanings.
    """

    left = validate_samples(reference, "reference")
    right = validate_samples(comparison, "comparison")
    return float(wasserstein_distance(left, right))


def calculate_normalized_emd(
    reference: SampleInput,
    comparison: SampleInput,
    normalization_scale: float,
) -> float:
    """Normalize EMD with one pre-fixed edge scale and return a bounded loss.

    Input
    -----
    reference, comparison:
        Transition-time samples accepted by ``calculate_emd``.
    normalization_scale:
        Precomputed, epsilon-independent ``Z_e`` for this edge. The caller must
        freeze it before comparing methods/seeds. A zero scale is replaced by
        the paper-required value 1.

    Output
    ------
    float
        ``nEMD = EMD/Z_e`` clipped to ``[0,1]``.

    Paper experiment step
    ---------------------
    Overall Step 9.2/10: transform raw EMD into a comparable bounded loss using
    a fixed scale; never recompute ``Z_e`` separately for each epsilon.
    """

    if isinstance(normalization_scale, bool):
        raise ValueError("normalization_scale must be finite and non-negative.")
    try:
        scale = float(normalization_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalization_scale must be finite and non-negative."
        ) from exc
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("normalization_scale must be finite and non-negative.")
    effective_scale = NEMD_ZERO_SCALE if scale == 0 else scale
    normalized = calculate_emd(reference, comparison) / effective_scale
    return float(np.clip(normalized, 0.0, 1.0))
