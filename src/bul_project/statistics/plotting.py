"""Deterministic, paper-oriented figures for original Phase 7."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .statistical_tests import DATASET_ORDER, EPSILON_ORDER, STRATEGY_ORDER


STRATEGY_LABELS = {
    "uniform": "Uniform",
    "activity_focus": "Activity focus",
    "edge_focus": "Edge focus",
    "variant_focus": "Variant focus",
    "time_focus": "Time focus",
}
METRIC_LABELS = {
    "L_structure": "Structure loss",
    "L_temporal": "Temporal loss",
    "L_information": "Information loss",
    "BUL_equal": "BUL (equal weights)",
    "BUL_wo_structure": "Without structure",
    "BUL_wo_temporal": "Without temporal",
    "BUL_wo_information": "Without information",
}
STAGE_LABELS = {
    "clipping": "Clipping",
    "dp_noise": "DP transformation",
    "total": "Total",
}

# Okabe-Ito inspired palette. Every multi-series figure also uses distinct
# markers and line styles, so interpretation never depends on color alone.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "grey": "#6B6B6B",
}
STRATEGY_STYLES = {
    "uniform": {"color": COLORS["blue"], "marker": "o", "linestyle": "-"},
    "activity_focus": {"color": COLORS["orange"], "marker": "s", "linestyle": "--"},
    "edge_focus": {"color": COLORS["green"], "marker": "^", "linestyle": "-."},
    "variant_focus": {"color": COLORS["vermillion"], "marker": "D", "linestyle": ":"},
    "time_focus": {"color": COLORS["purple"], "marker": "P", "linestyle": (0, (5, 1, 1, 1))},
}
METRIC_STYLES = {
    "L_structure": {"color": COLORS["blue"], "marker": "o", "linestyle": "-"},
    "L_temporal": {"color": COLORS["orange"], "marker": "s", "linestyle": "--"},
    "L_information": {"color": COLORS["green"], "marker": "^", "linestyle": "-."},
}
STAGE_STYLES = {
    "clipping": {"color": COLORS["blue"], "marker": "s", "linestyle": "-"},
    "dp_noise": {"color": COLORS["orange"], "marker": "^", "linestyle": "--"},
    "total": {"color": COLORS["green"], "marker": "o", "linestyle": "-."},
}
ABLATION_STYLES = {
    "BUL_equal": {"color": COLORS["black"], "marker": "o", "linestyle": "-"},
    "BUL_wo_structure": {"color": COLORS["orange"], "marker": "s", "linestyle": "--"},
    "BUL_wo_temporal": {"color": COLORS["green"], "marker": "^", "linestyle": "-."},
    "BUL_wo_information": {"color": COLORS["vermillion"], "marker": "D", "linestyle": ":"},
}
DATASET_STYLES = {
    "BPIC2012": {"color": COLORS["blue"], "marker": "o", "linestyle": "-"},
    "BPIC2017": {"color": COLORS["orange"], "marker": "s", "linestyle": "--"},
    "SEPSIS": {"color": COLORS["green"], "marker": "^", "linestyle": "-."},
}


class Phase7PlotError(ValueError):
    """Raised when a plot input does not contain the required Phase 7 rows."""


def _style() -> None:
    """Apply.

    Inputs:
        None.
    Outputs:
        None.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "bul-project-sci",
        }
    )


def _require_columns(frame: pd.DataFrame, required: set[str], role: str) -> None:
    """Validate that all required table columns are present.

    Inputs:
        frame: Tabular data processed by the operation.
        required: Input value used by this operation.
        role: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    missing = required - set(frame.columns)
    if missing:
        raise Phase7PlotError(f"{role} is missing columns: {sorted(missing)!r}")


def _epsilon_axis(axis: Any) -> None:
    """Return epsilon axis.

    Inputs:
        axis: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    axis.set_xscale("log", base=2)
    axis.set_xticks(EPSILON_ORDER)
    axis.set_xticklabels([str(value) for value in EPSILON_ORDER])
    axis.set_xlabel(r"Privacy budget $\epsilon$ (log$_2$ scale)")


def _series_with_ci(
    axis: Any,
    frame: pd.DataFrame,
    *,
    label: str,
    marker: str = "o",
    color: str | None = None,
    linestyle: Any = "-",
) -> None:
    """Return series with ci.

    Inputs:
        axis: Input value used by this operation.
        frame: Tabular data processed by the operation.
        label: Input value used by this operation.
        marker: Input value used by this operation.
        color: Input value used by this operation.
        linestyle: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    ordered = frame.sort_values("epsilon", kind="mergesort")
    if tuple(ordered["epsilon"].astype(float)) != EPSILON_ORDER:
        raise Phase7PlotError(f"Incomplete epsilon series for {label!r}.")
    x = ordered["epsilon"].to_numpy(dtype=float)
    y = ordered["mean"].to_numpy(dtype=float)
    lower = ordered["ci_lower"].to_numpy(dtype=float)
    upper = ordered["ci_upper"].to_numpy(dtype=float)
    line = axis.plot(
        x,
        y,
        marker=marker,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=0.55,
        label=label,
        zorder=3,
    )[0]
    axis.fill_between(
        x,
        lower,
        upper,
        color=line.get_color(),
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )


def _padded_limits(
    frame: pd.DataFrame,
    *,
    include_zero: bool,
    padding_fraction: float = 0.08,
) -> tuple[float, float]:
    """Return shared journal-figure limits containing every confidence interval."""

    lower = float(frame["ci_lower"].min())
    upper = float(frame["ci_upper"].max())
    span = max(upper - lower, 1e-9)
    bottom = lower - padding_fraction * span
    top = upper + padding_fraction * span
    if include_zero:
        bottom = min(bottom, 0.0)
    return max(0.0, bottom), top


def plot_epsilon_bul(summary: pd.DataFrame) -> Figure:
    """Plot DP-full equal-weight BUL versus epsilon for all strategies.

    Input: Phase 7 descriptive summary. Output: three-panel Figure with 95% CIs.
    Paper step: Phase 7 RQ1 privacy-budget/BUL main figure.
    """

    _require_columns(
        summary,
        {"analysis", "dataset", "metric", "budget_strategy", "epsilon", "mean", "ci_lower", "ci_upper"},
        "descriptive summary",
    )
    _style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.2),
        sharey=True,
        constrained_layout=True,
    )
    source = summary[
        summary["analysis"].eq("dp_full") & summary["metric"].eq("BUL_equal")
    ]
    for axis, dataset in zip(axes, DATASET_ORDER):
        dataset_rows = source[source["dataset"].eq(dataset)]
        for strategy in STRATEGY_ORDER:
            style = STRATEGY_STYLES[strategy]
            _series_with_ci(
                axis,
                dataset_rows[dataset_rows["budget_strategy"].eq(strategy)],
                label=STRATEGY_LABELS[strategy],
                **style,
            )
        axis.set_title(dataset)
        _epsilon_axis(axis)
        axis.set_ylabel("Behavioral utility loss")
    axes[0].set_ylim(*_padded_limits(source, include_zero=False))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        columnspacing=1.4,
        handlelength=2.8,
    )
    figure.suptitle("Equal-weight behavioral utility loss across privacy budgets")
    return figure


def plot_epsilon_dimensions(summary: pd.DataFrame) -> Figure:
    """Plot the three BUL dimensions for the uniform budget strategy.

    Input: Phase 7 descriptive summary. Output: a 3-by-3 Figure with separate
    row scales and 95% CIs, preventing temporal loss from visually suppressing
    the structure and information dimensions. Paper step: Phase 7 RQ1/RQ3
    dimension decomposition.
    """

    _require_columns(
        summary,
        {"analysis", "dataset", "metric", "budget_strategy", "epsilon", "mean", "ci_lower", "ci_upper"},
        "descriptive summary",
    )
    _style()
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.4, 8.0),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    metrics = ("L_structure", "L_temporal", "L_information")
    source = summary[
        summary["analysis"].eq("dp_full") & summary["budget_strategy"].eq("uniform")
    ]
    for row, metric in enumerate(metrics):
        metric_rows = source[source["metric"].eq(metric)]
        limits = _padded_limits(metric_rows, include_zero=metric != "L_temporal")
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row, column]
            dataset_rows = metric_rows[metric_rows["dataset"].eq(dataset)]
            style = METRIC_STYLES[metric]
            _series_with_ci(
                axis,
                dataset_rows,
                label=METRIC_LABELS[metric],
                **style,
            )
            axis.set_ylim(*limits)
            if row == 0:
                axis.set_title(dataset)
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            if row == len(metrics) - 1:
                _epsilon_axis(axis)
            else:
                axis.set_xlabel("")
                axis.tick_params(labelbottom=False)
    figure.suptitle(
        "Dimension-specific losses under uniform budget allocation\n"
        "(row-specific y-axis scales)"
    )
    return figure


def plot_stage_losses(summary: pd.DataFrame) -> Figure:
    """Plot clipping, DP-noise, and total BUL as independent comparisons.

    Input: Phase 7 stage summary. Output: three-panel Figure; it intentionally
    does not stack the stages because total loss is not their arithmetic sum.
    Paper step: Phase 7 RQ2 distortion-stage comparison.
    """

    _require_columns(
        summary,
        {"dataset", "epsilon", "stage", "metric", "mean", "ci_lower", "ci_upper", "additive_decomposition_claim"},
        "stage summary",
    )
    if summary["additive_decomposition_claim"].astype(bool).any():
        raise Phase7PlotError("Stage summary must not claim additive decomposition.")
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True, constrained_layout=True)
    source = summary[summary["metric"].eq("BUL_equal")]
    for axis, dataset in zip(axes, DATASET_ORDER):
        dataset_rows = source[source["dataset"].eq(dataset)]
        for stage in ("clipping", "dp_noise", "total"):
            style = STAGE_STYLES[stage]
            _series_with_ci(
                axis,
                dataset_rows[dataset_rows["stage"].eq(stage)],
                label=STAGE_LABELS[stage],
                **style,
            )
        axis.set_title(dataset)
        _epsilon_axis(axis)
        axis.set_ylabel("Behavioral utility loss")
    axes[0].set_ylim(bottom=0.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle(
        "Clipping, DP-transformation, and total loss\n"
        "(independent comparisons; not an additive decomposition)"
    )
    return figure


def plot_ablation(summary: pd.DataFrame) -> Figure:
    """Plot dimension-level ablation for the uniform strategy.

    Input: Phase 7 ablation summary. Output: three-panel Figure with full and
    one-dimension-removed BUL. Paper step: Phase 7 RQ3 ablation analysis.
    """

    _require_columns(
        summary,
        {"dataset", "epsilon", "budget_strategy", "metric", "mean", "ci_lower", "ci_upper"},
        "ablation summary",
    )
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True, constrained_layout=True)
    source = summary[summary["budget_strategy"].eq("uniform")]
    metrics = (
        "BUL_equal",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
    )
    for axis, dataset in zip(axes, DATASET_ORDER):
        dataset_rows = source[source["dataset"].eq(dataset)]
        for metric in metrics:
            style = ABLATION_STYLES[metric]
            _series_with_ci(
                axis,
                dataset_rows[dataset_rows["metric"].eq(metric)],
                label=METRIC_LABELS[metric],
                **style,
            )
        axis.set_title(dataset)
        _epsilon_axis(axis)
        axis.set_ylabel("Composite loss")
    axes[0].set_ylim(bottom=0.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.suptitle(
        "Dimension-level BUL ablation under uniform allocation\n"
        "(weights of retained dimensions are renormalized)"
    )
    return figure


def plot_sensitivity_heatmap(summary: pd.DataFrame) -> Figure:
    """Plot the BPIC2017 Top-k/tau sensitivity surface for BUL_equal.

    Input: Phase 7 sensitivity summary. Output: three heatmaps for epsilon
    0.5, 2, and 8. Paper step: Phase 7 RQ5 parameter sensitivity.
    """

    _require_columns(
        summary,
        {"epsilon", "top_k_variants", "min_edge_support", "metric", "mean"},
        "sensitivity summary",
    )
    _style()
    epsilons = (0.5, 2.0, 8.0)
    top_k = (50, 100, 200)
    tau = (5, 10, 20)
    source = summary[summary["metric"].eq("BUL_equal")]
    arrays: list[np.ndarray] = []
    for epsilon in epsilons:
        pivot = source[source["epsilon"].eq(epsilon)].pivot(
            index="top_k_variants", columns="min_edge_support", values="mean"
        ).reindex(index=top_k, columns=tau)
        if pivot.isna().any().any():
            raise Phase7PlotError(f"Incomplete sensitivity grid at epsilon={epsilon}.")
        arrays.append(pivot.to_numpy(dtype=float))
    minimum = min(float(values.min()) for values in arrays)
    maximum = max(float(values.max()) for values in arrays)
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    image = None
    for axis, epsilon, values in zip(axes, epsilons, arrays):
        image = axis.imshow(values, cmap="viridis", vmin=minimum, vmax=maximum, aspect="auto")
        normalization = plt.Normalize(vmin=minimum, vmax=maximum)
        color_map = plt.get_cmap("viridis")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                red, green, blue, _ = color_map(normalization(values[row, column]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "black" if luminance > 0.56 else "white"
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        axis.add_patch(
            Rectangle(
                (0.5, 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="white",
                linewidth=2.2,
                linestyle="--",
            )
        )
        axis.set_title(rf"$\epsilon={epsilon:g}$")
        axis.set_xticks(range(len(tau)), [str(value) for value in tau])
        axis.set_yticks(range(len(top_k)), [str(value) for value in top_k])
        axis.set_xlabel(r"Edge threshold $\tau$")
        axis.set_ylabel("Top-k variants")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Mean BUL (equal weights)", shrink=0.82)
    figure.suptitle(
        "BPIC2017 Top-k and edge-threshold sensitivity\n"
        r"Dashed cell: frozen reference ($k=100$, $\tau=10$); "
        "cell values are 30-seed means"
    )
    return figure


def plot_cross_dataset_trends(summary: pd.DataFrame) -> Figure:
    """Plot within-dataset relative BUL changes for the uniform strategy.

    Input: Phase 7 descriptive summary. Output: one descriptive Figure indexed
    to epsilon=0.5 within each dataset. This supports response-shape comparison
    without implying that dataset-specific absolute BUL scales are exchangeable.
    Paper step: Phase 7 RQ5 cross-dataset robustness comparison.
    """

    _require_columns(
        summary,
        {"analysis", "dataset", "metric", "budget_strategy", "epsilon", "mean"},
        "descriptive summary",
    )
    _style()
    figure, axis = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    source = summary[
        summary["analysis"].eq("dp_full")
        & summary["metric"].eq("BUL_equal")
        & summary["budget_strategy"].eq("uniform")
    ]
    for dataset in DATASET_ORDER:
        ordered = source[source["dataset"].eq(dataset)].sort_values(
            "epsilon", kind="mergesort"
        )
        if tuple(ordered["epsilon"].astype(float)) != EPSILON_ORDER:
            raise Phase7PlotError(f"Incomplete epsilon series for {dataset!r}.")
        x = ordered["epsilon"].to_numpy(dtype=float)
        mean = ordered["mean"].to_numpy(dtype=float)
        baseline = float(mean[0])
        relative = 100.0 * (mean / baseline - 1.0)
        axis.plot(
            x,
            relative,
            label=dataset,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
            **DATASET_STYLES[dataset],
        )
    _epsilon_axis(axis)
    axis.axhline(0.0, color=COLORS["grey"], linewidth=0.9, linestyle=":", zorder=0)
    axis.set_ylabel(r"Change in BUL from $\epsilon=0.5$ (%)")
    axis.legend(ncol=3, loc="lower left")
    axis.set_title("Within-dataset relative BUL trends under uniform allocation")
    return figure


def _figure_bytes(figure: Figure, *, output_format: str) -> bytes:
    """Render one figure without closing it, using deterministic metadata."""

    if output_format not in {"png", "pdf"}:
        raise Phase7PlotError(f"Unsupported paper-figure format: {output_format!r}.")
    stream = io.BytesIO()
    options: dict[str, Any] = {
        "format": output_format,
        "bbox_inches": "tight",
    }
    if output_format == "png":
        options.update(
            {
                "dpi": 300,
                "metadata": {"Software": "BUL SCI statistical analysis"},
            }
        )
    else:
        options["metadata"] = {
            "Creator": "BUL Project",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        }
    figure.savefig(stream, **options)
    return stream.getvalue()


def figure_png_bytes(figure: Figure) -> bytes:
    """Render one Figure as deterministic high-resolution PNG bytes.

    Input: completed matplotlib Figure. Output: PNG bytes with fixed metadata.
    Paper step: Phase 7 reproducible figure serialization.
    """

    payload = _figure_bytes(figure, output_format="png")
    plt.close(figure)
    return payload


def figure_pdf_bytes(figure: Figure) -> bytes:
    """Render one Figure as a vector PDF with embedded TrueType fonts."""

    payload = _figure_bytes(figure, output_format="pdf")
    plt.close(figure)
    return payload


def build_figure_payloads(
    *,
    descriptive_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> dict[str, bytes]:
    """Build every frozen Phase 7 paper figure without writing files.

    Input: four Phase 7 summary tables. Output: filename-to-byte mapping for
    300-dpi PNG previews and vector PDF submission figures. Paper step:
    Phase 7 figure generation and artifact verification.
    """

    figures = {
        "figure_epsilon_bul.png": plot_epsilon_bul(descriptive_summary),
        "figure_epsilon_dimensions.png": plot_epsilon_dimensions(descriptive_summary),
        "figure_stage_losses.png": plot_stage_losses(stage_summary),
        "figure_dimension_ablation.png": plot_ablation(ablation_summary),
        "figure_parameter_sensitivity.png": plot_sensitivity_heatmap(sensitivity_summary),
        "figure_cross_dataset_trends.png": plot_cross_dataset_trends(descriptive_summary),
    }
    payloads: dict[str, bytes] = {}
    for png_name, figure in figures.items():
        payloads[png_name] = _figure_bytes(figure, output_format="png")
        payloads[Path(png_name).with_suffix(".pdf").name] = _figure_bytes(
            figure,
            output_format="pdf",
        )
        plt.close(figure)
    return payloads


def figure_contract() -> dict[str, dict[str, Any]]:
    """Return expected titles and axis labels for automated visual checks.

    Input: none. Output: machine-readable plot contract. Paper step: Phase 7
    figure-file and axis-label validation.
    """

    return {
        "figure_epsilon_bul.png": {
            "x_label": r"Privacy budget $\epsilon$ (log$_2$ scale)",
            "y_label": "Behavioral utility loss",
            "panels": 3,
            "formats": ["png", "pdf"],
        },
        "figure_epsilon_dimensions.png": {
            "x_label": r"Privacy budget $\epsilon$ (log$_2$ scale)",
            "y_labels": [
                "Structure loss",
                "Temporal loss",
                "Information loss",
            ],
            "panels": 9,
            "row_specific_y_scales": True,
            "formats": ["png", "pdf"],
        },
        "figure_stage_losses.png": {
            "x_label": r"Privacy budget $\epsilon$ (log$_2$ scale)",
            "y_label": "Behavioral utility loss",
            "panels": 3,
            "stacked": False,
            "formats": ["png", "pdf"],
        },
        "figure_dimension_ablation.png": {
            "x_label": r"Privacy budget $\epsilon$ (log$_2$ scale)",
            "y_label": "Composite loss",
            "panels": 3,
            "retained_dimension_weights_renormalized": True,
            "formats": ["png", "pdf"],
        },
        "figure_parameter_sensitivity.png": {
            "x_label": r"Edge threshold $\tau$",
            "y_label": "Top-k variants",
            "panels": 3,
            "reference_cell": {"top_k": 100, "tau": 10},
            "formats": ["png", "pdf"],
        },
        "figure_cross_dataset_trends.png": {
            "x_label": r"Privacy budget $\epsilon$ (log$_2$ scale)",
            "y_label": r"Change in BUL from $\epsilon=0.5$ (%)",
            "panels": 1,
            "absolute_dataset_ranking_claim": False,
            "within_dataset_reference_epsilon": 0.5,
            "formats": ["png", "pdf"],
        },
    }


# Natural public error name; the historical class remains for compatibility.
PlottingError = Phase7PlotError
