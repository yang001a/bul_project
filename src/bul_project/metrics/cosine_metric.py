"""Cosine similarity and its explicit BUL loss-direction conversion."""

from __future__ import annotations

import numpy as np

from bul_project.config import DELTA_DENOM

from ._validation import VectorInput, align_vectors


def calculate_cosine_similarity(
    reference: VectorInput, comparison: VectorInput
) -> float:
    """Calculate cosine similarity for aligned non-negative behavior vectors.

    Input
    -----
    reference, comparison:
        Non-negative count/probability mappings or equal-length vectors.

    Output
    ------
    float
        Cosine similarity in ``[0,1]``. Two all-zero vectors are treated as
        identical (similarity 1); exactly one all-zero vector has similarity 0.

    Paper experiment step
    ---------------------
    Overall Step 9.1: compute the established cosine similarity before its
    explicit conversion to the BUL loss direction.
    """

    left, right = align_vectors(reference, comparison)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    denominator = max(left_norm * right_norm, DELTA_DENOM)
    similarity = float(np.dot(left, right) / denominator)
    return float(np.clip(similarity, 0.0, 1.0))


def calculate_cosine_loss(
    reference: VectorInput, comparison: VectorInput
) -> float:
    """Convert cosine similarity into the required loss direction.

    Input
    -----
    reference, comparison:
        Inputs accepted by ``calculate_cosine_similarity``.

    Output
    ------
    float
        ``L_cos = 1 - Cos(reference, comparison)`` in ``[0,1]``. Identical
        vectors therefore produce loss 0, never similarity 1 as a loss value.

    Paper experiment step
    ---------------------
    Overall Step 10: convert a higher-is-better similarity into a
    higher-is-worse loss before any later BUL aggregation.
    """

    similarity = calculate_cosine_similarity(reference, comparison)
    return float(np.clip(1.0 - similarity, 0.0, 1.0))
