"""Phase 3 public API: case-level clipping and differential privacy."""

from .budget import (
    allocate_privacy_budget,
    edge_only_epsilon,
    split_temporal_budget,
)
from .clipping import clip_trace, clip_traces
from .laplace_dp import laplace_release
from .models import DPReleaseStatistics, PrivacyBudget, SensitivityBounds
from .release import dp_edge_only_release, dp_full_release, release_g_dp
from .sensitivity import calculate_sensitivity

__all__ = [
    "DPReleaseStatistics",
    "PrivacyBudget",
    "SensitivityBounds",
    "allocate_privacy_budget",
    "calculate_sensitivity",
    "clip_trace",
    "clip_traces",
    "dp_edge_only_release",
    "dp_full_release",
    "edge_only_epsilon",
    "laplace_release",
    "release_g_dp",
    "split_temporal_budget",
]
