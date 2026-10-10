"""Jensen-Shannon divergence from Lin (1991)."""

from __future__ import annotations

import numpy as np

from bul_project.config import DELTA_PROB

from ._validation import VectorInput, align_vectors, normalize_distribution


def calculate_js(reference: VectorInput, comparison: VectorInput) -> float:
    """Calculate base-2 Jensen-Shannon divergence on one aligned space.

    Input
    -----
    reference, comparison:
        Non-negative probability or count mappings/vectors. Mapping inputs are
        aligned on their key union; vector inputs must have equal length.

    Output
    ------
    float
        ``0.5*KL(P||M) + 0.5*KL(Q||M)`` in ``[0,1]`` using base-2 logs. This
        is JS divergence, not the square-root Jensen-Shannon distance.

    Paper experiment step
    ---------------------
    Overall Step 9/10: measure probability-distribution loss in the established
    JS loss direction. ``DELTA_PROB=1e-10`` prevents ``log(0)`` and the result
    is clipped only for numerical roundoff.
    """

    left, right = align_vectors(reference, comparison)
    p = normalize_distribution(left, "reference")
    q = normalize_distribution(right, "comparison")

    # Apply the fixed probability smoothing symmetrically, then renormalize.
    p = normalize_distribution(p + DELTA_PROB, "smoothed reference")
    q = normalize_distribution(q + DELTA_PROB, "smoothed comparison")
    midpoint = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / midpoint)))
    kl_qm = float(np.sum(q * np.log2(q / midpoint)))
    divergence = 0.5 * kl_pm + 0.5 * kl_qm
    return float(np.clip(divergence, 0.0, 1.0))
