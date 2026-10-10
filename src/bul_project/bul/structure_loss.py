"""Structure-dimension aggregation from established Phase 4 metrics."""

from __future__ import annotations

from collections.abc import Mapping

from bul_project.config import STRUCTURE_COMPONENT_WEIGHTS
from bul_project.metrics import calculate_cosine_loss, calculate_js
from bul_project.metrics._validation import VectorInput

from ._utils import weighted_loss


def calculate_structure_loss(
    reference_edges: VectorInput,
    comparison_edges: VectorInput,
    reference_variants: VectorInput,
    comparison_variants: VectorInput,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Calculate structural behavior loss on fixed edge and variant spaces.

    Input
    -----
    reference_edges, comparison_edges:
        DFG edge count/probability vectors projected onto one frozen edge set.
    reference_variants, comparison_variants:
        Variant count/probability vectors projected onto the same fixed
        Top-k+OTHER candidate space.
    weights:
        Optional complete weights for ``edge_js``, ``edge_cosine_loss`` and
        ``variant_js``. They must be non-negative and sum to one; the default
        gives the three stated main components equal weight.

    Output
    ------
    float
        ``L_structure`` in ``[0,1]``, where larger means greater loss.

    Paper experiment step
    ---------------------
    Overall Step 9.1/10: aggregate Edge JS divergence, ``1-DFG cosine`` and
    Variant JS divergence after all three are aligned to the loss direction.
    No EdgeRecall or newly defined distance is introduced.
    """

    components = {
        "edge_js": calculate_js(reference_edges, comparison_edges),
        "edge_cosine_loss": calculate_cosine_loss(
            reference_edges, comparison_edges
        ),
        "variant_js": calculate_js(reference_variants, comparison_variants),
    }
    selected_weights = STRUCTURE_COMPONENT_WEIGHTS if weights is None else weights
    return weighted_loss(components, selected_weights)
