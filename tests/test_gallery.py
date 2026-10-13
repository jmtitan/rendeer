"""The gallery must exercise every advertised grammar and retain synthetic provenance."""

import gallery
import numpy as np


def test_gallery_has_a_working_example_for_every_public_plot():
    expected = {
        "line",
        "scatter",
        "bar",
        "interval",
        "area",
        "hist",
        "kde",
        "box",
        "violin",
        "ecdf",
        "heatmap",
        "correlation",
        "confusion",
        "pareto",
        "roc",
        "precision_recall",
        "calibration",
        "facet",
        "pairplot",
        "composite",
    }
    figures = gallery.make_gallery()
    assert expected <= set(figures)
    for name, fig in figures.items():
        report = fig.validate()
        assert report.passed, (name, report.to_dict())


def test_synthetic_fixture_is_reproducible_without_mutating_global_rng():
    state = np.random.get_state()
    first, second = gallery.synthetic_data(), gallery.synthetic_data()
    assert first["curves"].equals(second["curves"])
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
