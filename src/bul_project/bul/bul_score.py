"""Equal and fixed-weight aggregation of the three reported BUL dimensions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from bul_project.config import BUL_DIMENSION_WEIGHTS

from ._utils import validate_loss, validate_weights


_DIMENSIONS = ("structure", "temporal", "information")


def calculate_BUL(
    l_structure: float,
    l_temporal: float,
    l_information: float,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Aggregate three loss dimensions into one BUL summary.

    Input
    -----
    l_structure, l_temporal, l_information:
        Independently reported, normalized losses in ``[0,1]``.
    weights:
        Optional fixed weights with keys ``structure``, ``temporal`` and
        ``information`` summing to one. ``None`` uses equal thirds.

    Output
    ------
    float
        ``BUL_equal=(L_structure+L_temporal+L_information)/3`` by default, or
        the corresponding pre-fixed weighted robustness score.

    Paper experiment step
    ---------------------
    Overall Step 11: use equal dimension weights for the main result and allow
    only weights fixed independently of the evaluated epsilon/noise draw for
    robustness analysis. Component losses remain separately reportable.
    """

    losses = {
        "structure": validate_loss(l_structure, "l_structure"),
        "temporal": validate_loss(l_temporal, "l_temporal"),
        "information": validate_loss(l_information, "l_information"),
    }
    selected = BUL_DIMENSION_WEIGHTS if weights is None else weights
    checked = validate_weights(selected, _DIMENSIONS, "BUL weights")
    return float(
        np.clip(
            sum(losses[dimension] * checked[dimension] for dimension in _DIMENSIONS),
            0.0,
            1.0,
        )
    )


def calculate_bul(
    l_structure: float,
    l_temporal: float,
    l_information: float,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Lowercase alias of ``calculate_BUL``.

    Input
    -----
    l_structure, l_temporal, l_information, weights:
        The same dimension losses and optional fixed weights accepted by
        ``calculate_BUL``.

    Output
    ------
    float
        The same bounded aggregate returned by ``calculate_BUL``.

    Paper experiment step
    ---------------------
    Overall Step 11: BUL dimension aggregation.
    """

    return calculate_BUL(
        l_structure,
        l_temporal,
        l_information,
        weights=weights,
    )


def calculate_fixed_entropy_weights(
    reference_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, float]:
    """Fit entropy-method dimension weights once on a reference matrix.

    Input
    -----
    reference_matrix:
        A fixed ``n x 3`` matrix ordered as structure, temporal, information.
        It must contain at least two finite rows of losses in ``[0,1]`` and
        must be independent of the results later evaluated with these weights.

    Output
    ------
    dict[str, float]
        Fixed non-negative dimension weights summing to one. Constant or
        otherwise uninformative columns receive zero divergence; if every
        column is uninformative, equal weights are returned.

    Paper experiment step
    ---------------------
    Overall Step 11 and entropy-weight robustness analysis: calculate weights
    once from a predetermined reference matrix, never again per epsilon,
    method, or random seed.
    """

    try:
        matrix = np.asarray(reference_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("reference_matrix must be a numeric n x 3 matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != 3:
        raise ValueError("reference_matrix must have at least two rows and exactly three columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("reference_matrix values must be finite losses in [0,1].")

    minima = np.min(matrix, axis=0)
    ranges = np.max(matrix, axis=0) - minima
    normalized = np.divide(
        matrix - minima,
        ranges,
        out=np.zeros_like(matrix),
        where=ranges > 0,
    )
    column_sums = np.sum(normalized, axis=0)
    proportions = np.divide(
        normalized,
        column_sums,
        out=np.zeros_like(normalized),
        where=column_sums > 0,
    )
    entropy = np.ones(3, dtype=float)
    coefficient = 1.0 / np.log(matrix.shape[0])
    for column in range(3):
        positive = proportions[:, column][proportions[:, column] > 0]
        if positive.size:
            entropy[column] = -coefficient * float(
                np.sum(positive * np.log(positive))
            )
    divergence = np.clip(1.0 - entropy, 0.0, None)
    if float(np.sum(divergence)) == 0.0:
        weights = np.full(3, 1.0 / 3.0)
    else:
        weights = divergence / np.sum(divergence)
    result = dict(zip(_DIMENSIONS, (float(value) for value in weights)))
    validate_weights(result, _DIMENSIONS, "fixed entropy weights")
    return result
