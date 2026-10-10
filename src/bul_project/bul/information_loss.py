"""Information-distribution aggregation without duplicate Variant weighting."""

from __future__ import annotations

from collections.abc import Mapping

from bul_project.config import INFORMATION_COMPONENT_WEIGHTS
from bul_project.metrics import calculate_entropy_loss, calculate_js
from bul_project.metrics._validation import VectorInput

from ._utils import weighted_loss


def calculate_information_loss(
    reference_activities: VectorInput,
    comparison_activities: VectorInput,
    reference_edges: VectorInput,
    comparison_edges: VectorInput,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Calculate the paper's information-distribution loss dimension.

    Input
    -----
    reference_activities, comparison_activities:
        Activity count/probability vectors on one fixed activity space.
    reference_edges, comparison_edges:
        Edge count/probability vectors on one fixed DFG edge space.
    weights:
        Optional complete weights for ``activity_entropy_loss``,
        ``edge_entropy_loss`` and ``activity_js``. They must sum to one; the
        default is transparent equal weighting.

    Output
    ------
    float
        ``L_information`` in ``[0,1]`` with higher values indicating loss.

    Paper experiment step
    ---------------------
    Overall Step 9.3/10: combine Activity entropy deviation, Edge entropy
    deviation and Activity JS. Variant JS is deliberately excluded here
    because it is already counted in the structure/path dimension.
    """

    components = {
        "activity_entropy_loss": calculate_entropy_loss(
            reference_activities, comparison_activities
        ),
        "edge_entropy_loss": calculate_entropy_loss(
            reference_edges, comparison_edges
        ),
        "activity_js": calculate_js(
            reference_activities, comparison_activities
        ),
    }
    selected_weights = INFORMATION_COMPONENT_WEIGHTS if weights is None else weights
    return weighted_loss(components, selected_weights)
