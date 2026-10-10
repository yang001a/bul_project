"""Typed ``G_raw`` component structures for Phase 2 behavior statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Edge = tuple[str, str]
Variant = tuple[str, ...]


def _edge_rows(
    counts: dict[Edge, int], probabilities: dict[Edge, float]
) -> list[dict[str, Any]]:
    """Convert an edge mapping to deterministic JSON-compatible records.

    Input: edge counts and probabilities. Output: sorted record list.
    Paper step: Overall Step 3, DFG edge representation in ``G_raw``.
    """

    return [
        {
            "source": source,
            "target": target,
            "count": count,
            "probability": probabilities.get((source, target), 0.0),
        }
        for (source, target), count in sorted(counts.items())
    ]


def _variant_rows(
    counts: dict[Variant, int], probabilities: dict[Variant, float]
) -> list[dict[str, Any]]:
    """Convert variant mappings to deterministic JSON-compatible records.

    Input: variant counts/probabilities. Output: activity-list records.
    Paper step: Overall Step 3, trace variant representation in ``G_raw``.
    """

    return [
        {
            "activities": list(variant),
            "count": count,
            "probability": probabilities.get(variant, 0.0),
        }
        for variant, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


@dataclass(frozen=True)
class ActivityStatistics:
    """Activity counts and their normalized empirical distribution."""

    counts: dict[str, int]
    probabilities: dict[str, float]
    total_events: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible activity statistics for ``G_raw`` output."""

        return {
            "total_events": self.total_events,
            "activities": [
                {
                    "activity": activity,
                    "count": count,
                    "probability": self.probabilities[activity],
                }
                for activity, count in self.counts.items()
            ],
        }


@dataclass(frozen=True)
class DFGStatistics:
    """Full DFG counts plus a separate low-frequency-filtered view."""

    counts: dict[Edge, int]
    probabilities: dict[Edge, float]
    total_edge_occurrences: int
    min_support: int | None
    filtered_counts: dict[Edge, int]
    filtered_probabilities: dict[Edge, float]
    dropped_edge_types: int
    dropped_edge_occurrences: int

    def to_dict(self) -> dict[str, Any]:
        """Return full and filtered DFG views without discarding raw counts."""

        return {
            "total_edge_occurrences": self.total_edge_occurrences,
            "min_support": self.min_support,
            "edges": _edge_rows(self.counts, self.probabilities),
            "filtered_edges": _edge_rows(
                self.filtered_counts, self.filtered_probabilities
            ),
            "dropped_edge_types": self.dropped_edge_types,
            "dropped_edge_occurrences": self.dropped_edge_occurrences,
        }


@dataclass(frozen=True)
class VariantStatistics:
    """Full variant counts and a Top-k candidate space with ``OTHER``."""

    full_counts: dict[Variant, int]
    full_probabilities: dict[Variant, float]
    total_cases: int
    unique_variants: int
    top_k: int | None
    top_counts: dict[Variant, int]
    top_probabilities: dict[Variant, float]
    other_count: int
    other_probability: float
    other_label: str = "OTHER"

    def to_dict(self) -> dict[str, Any]:
        """Return full and Top-k+OTHER variant distributions."""

        return {
            "total_cases": self.total_cases,
            "unique_variants": self.unique_variants,
            "top_k": self.top_k,
            "candidate_scope": "G_raw preview; final release domain must be frozen after clipping",
            "full_variants": _variant_rows(
                self.full_counts, self.full_probabilities
            ),
            "top_variants": _variant_rows(
                self.top_counts, self.top_probabilities
            ),
            "other": {
                "label": self.other_label,
                "count": self.other_count,
                "probability": self.other_probability,
            },
        }


@dataclass(frozen=True)
class TemporalStatistics:
    """Transition-level empirical time distributions and filtered edge view."""

    samples: dict[Edge, tuple[float, ...]]
    edge_support_counts: dict[Edge, int]
    sample_counts: dict[Edge, int]
    sums: dict[Edge, float]
    observed_means: dict[Edge, float | None]
    edge_normalized_means: dict[Edge, float]
    medians: dict[Edge, float | None]
    total_transition_pairs: int
    valid_interval_pairs: int
    missing_interval_pairs: int
    min_edge_support: int | None
    effective_samples: dict[Edge, tuple[float, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return edge-level time samples and their non-destructive filtered view."""

        rows = []
        for edge, values in self.samples.items():
            rows.append(
                {
                    "source": edge[0],
                    "target": edge[1],
                    "edge_support": self.edge_support_counts[edge],
                    "sample_count": self.sample_counts[edge],
                    "sum_seconds": self.sums[edge],
                    "observed_mean_seconds": self.observed_means[edge],
                    "edge_normalized_mean_seconds": self.edge_normalized_means[edge],
                    "median_seconds": self.medians[edge],
                    "samples_seconds": list(values),
                }
            )
        effective_edges = [
            {"source": edge[0], "target": edge[1], "samples_seconds": list(values)}
            for edge, values in self.effective_samples.items()
        ]
        return {
            "total_transition_pairs": self.total_transition_pairs,
            "valid_interval_pairs": self.valid_interval_pairs,
            "missing_interval_pairs": self.missing_interval_pairs,
            "min_edge_support": self.min_edge_support,
            "edges": rows,
            "effective_edges": effective_edges,
        }


@dataclass(frozen=True)
class GRawStatistics:
    """Complete Phase 2 output assembled from four behavior-statistics families."""

    activity: ActivityStatistics
    dfg: DFGStatistics
    variants: VariantStatistics
    temporal: TemporalStatistics
    trace_count: int
    top_k_variants: int | None
    min_edge_support: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible ``G_raw`` structure for later phases."""

        return {
            "name": "G_raw",
            "trace_count": self.trace_count,
            "statistical_space": {
                "top_k_variants": self.top_k_variants,
                "min_edge_support": self.min_edge_support,
            },
            "activity": self.activity.to_dict(),
            "dfg": self.dfg.to_dict(),
            "variants": self.variants.to_dict(),
            "temporal": self.temporal.to_dict(),
        }
