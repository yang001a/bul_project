"""Shannon entropy and bounded entropy-deviation loss."""

from __future__ import annotations

import numpy as np

from bul_project.config import DELTA_DENOM

from ._validation import VectorInput, align_vectors, normalize_distribution


def _entropy(probabilities: np.ndarray) -> float:
    """Return entropy.

    Inputs:
        probabilities: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Behavioral utility loss component evaluation.
    """
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def calculate_entropy_loss(
    reference: VectorInput, comparison: VectorInput
) -> float:
    """Calculate bounded Shannon entropy deviation from a reference.

    Input
    -----
    reference, comparison:
        Non-negative probability or count mappings/vectors on one aligned
        activity or edge space.

    Output
    ------
    float
        ``|H(reference)-H(comparison)| / max(H(reference), DELTA_DENOM)``
        clipped to ``[0,1]``.

    Paper experiment step
    ---------------------
    Overall Step 9.3/10: measure information-distribution loss with Shannon
    entropy deviation in the established higher-is-worse direction.
    """

    left, right = align_vectors(reference, comparison)
    p = normalize_distribution(left, "reference")
    q = normalize_distribution(right, "comparison")
    reference_entropy = _entropy(p)
    comparison_entropy = _entropy(q)
    loss = abs(reference_entropy - comparison_entropy) / max(
        reference_entropy, DELTA_DENOM
    )
    return float(np.clip(loss, 0.0, 1.0))
