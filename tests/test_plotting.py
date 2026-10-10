from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from bul_project.config import (
    DERIVED_ANALYSIS_RESULT_DIR,
    MAIN_EXPERIMENT_RESULT_DIR,
    SENSITIVITY_ANALYSIS_RESULT_DIR,
)
from bul_project.statistics.plotting import (
    figure_contract,
    figure_pdf_bytes,
    figure_png_bytes,
    plot_ablation,
    plot_cross_dataset_trends,
    plot_epsilon_bul,
    plot_epsilon_dimensions,
    plot_sensitivity_heatmap,
    plot_stage_losses,
)
from bul_project.statistics.statistical_tests import (
    build_ablation_summary,
    build_descriptive_summary,
    build_sensitivity_summary,
    build_stage_summary,
)


@pytest.fixture(scope="module")
def summaries() -> dict[str, pd.DataFrame]:
    derived = DERIVED_ANALYSIS_RESULT_DIR
    reference = MAIN_EXPERIMENT_RESULT_DIR
    sensitivity = SENSITIVITY_ANALYSIS_RESULT_DIR
    required = (
        derived / "bul_enriched_result.csv",
        reference / "experiment_stage_result.csv",
        derived / "dimension_ablation.csv",
        sensitivity / "sensitivity_result.csv",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("requires completed rerun outputs for plotting")
    return {
        "descriptive": build_descriptive_summary(
            pd.read_csv(derived / "bul_enriched_result.csv"),
            n_resamples=100,
        ),
        "stage": build_stage_summary(
            pd.read_csv(reference / "experiment_stage_result.csv"),
            n_resamples=100,
        ),
        "ablation": build_ablation_summary(
            pd.read_csv(derived / "dimension_ablation.csv"),
            n_resamples=100,
        ),
        "sensitivity": build_sensitivity_summary(
            pd.read_csv(sensitivity / "sensitivity_result.csv"),
            n_resamples=100,
        ),
    }


def test_paper_figures_have_expected_panel_axis_labels(
    summaries: dict[str, pd.DataFrame],
) -> None:
    cases = [
        (plot_epsilon_bul(summaries["descriptive"]), "figure_epsilon_bul.png"),
        (
            plot_epsilon_dimensions(summaries["descriptive"]),
            "figure_epsilon_dimensions.png",
        ),
        (plot_stage_losses(summaries["stage"]), "figure_stage_losses.png"),
        (plot_ablation(summaries["ablation"]), "figure_dimension_ablation.png"),
        (
            plot_sensitivity_heatmap(summaries["sensitivity"]),
            "figure_parameter_sensitivity.png",
        ),
        (
            plot_cross_dataset_trends(summaries["descriptive"]),
            "figure_cross_dataset_trends.png",
        ),
    ]
    contract = figure_contract()
    for figure, filename in cases:
        expected = contract[filename]
        data_axes = figure.axes[: expected["panels"]]
        assert len(data_axes) == expected["panels"]
        if filename == "figure_epsilon_dimensions.png":
            assert all(
                axis.get_xlabel() == expected["x_label"] for axis in data_axes[-3:]
            )
            assert [
                data_axes[index].get_ylabel() for index in (0, 3, 6)
            ] == expected["y_labels"]
        else:
            assert all(axis.get_xlabel() == expected["x_label"] for axis in data_axes)
            assert all(axis.get_ylabel() == expected["y_label"] for axis in data_axes)
        png = figure_png_bytes(figure)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 10_000
    plt.close("all")


def test_vector_pdf_is_generated_for_submission(
    summaries: dict[str, pd.DataFrame],
) -> None:
    figure = plot_cross_dataset_trends(summaries["descriptive"])
    pdf = figure_pdf_bytes(figure)

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5_000


def test_shared_y_axis_contains_every_confidence_interval(
    summaries: dict[str, pd.DataFrame],
) -> None:
    figure = plot_epsilon_bul(summaries["descriptive"])
    source = summaries["descriptive"]
    source = source[
        source["analysis"].eq("dp_full") & source["metric"].eq("BUL_equal")
    ]
    lower, upper = figure.axes[0].get_ylim()

    assert lower <= source["ci_lower"].min()
    assert upper >= source["ci_upper"].max()
    plt.close(figure)


def test_stage_figure_is_not_stacked(summaries: dict[str, pd.DataFrame]) -> None:
    figure = plot_stage_losses(summaries["stage"])
    assert figure_contract()["figure_stage_losses.png"]["stacked"] is False
    assert all(len(axis.containers) == 0 for axis in figure.axes[:3])
    plt.close(figure)


def test_main_figure_does_not_rely_on_color_alone(
    summaries: dict[str, pd.DataFrame],
) -> None:
    figure = plot_epsilon_bul(summaries["descriptive"])
    lines = figure.axes[0].lines

    assert len({line.get_marker() for line in lines}) == 5
    assert len({line.get_linestyle() for line in lines}) >= 4
    plt.close(figure)


def test_cross_dataset_figure_is_within_dataset_relative_change(
    summaries: dict[str, pd.DataFrame],
) -> None:
    figure = plot_cross_dataset_trends(summaries["descriptive"])
    data_lines = [line for line in figure.axes[0].lines if len(line.get_xdata()) == 5]

    assert len(data_lines) == 3
    assert all(line.get_ydata()[0] == pytest.approx(0.0) for line in data_lines)
    assert figure_contract()["figure_cross_dataset_trends.png"][
        "absolute_dataset_ranking_claim"
    ] is False
    plt.close(figure)
