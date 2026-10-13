"""Each advertised family has a rendering case and a semantic boundary case."""

import numpy as np
import pandas as pd
import pytest

import rendeer as rd


@pytest.mark.parametrize("kind", ["hist", "kde", "box", "violin", "ecdf"])
def test_distribution_families_render(kind):
    f = getattr(rd, kind)(x=[1.0, 2.0, 2.5, 3.0, 4.0], xlabel="Latency (ms)")
    assert f.validate().passed
    assert f.ax.get_title() == ""
    assert f.ax.get_ylabel()


def test_histogram_and_ecdf_count_observations_without_inventing_samples():
    f = rd.hist(x=[0, 0, 1, 2], bins=[0, 1, 2, 3], xlabel="Value")
    np.testing.assert_allclose([p.get_height() for p in f.ax.patches], [2, 1, 1])
    g = rd.ecdf(x=[3, 1, 1, 2], xlabel="Value")
    np.testing.assert_allclose(g.ax.lines[0].get_ydata(), [0, 0.5, 0.75, 1.0])


def test_distribution_missing_policy_is_explicit():
    with pytest.raises(ValueError, match="missing"):
        rd.hist(x=[1, np.nan, 2])
    f = rd.hist(x=[1, np.nan, 2], missing="drop", xlabel="Value")
    assert f.provenance()["plots"][0]["missing_dropped"] == 1
    with pytest.raises(ValueError, match="distinct"):
        rd.kde(x=[1, 1, 1])


def test_comparison_histograms_share_bin_edges():
    data = pd.DataFrame({"value": [0, 1, 10, 11], "group": ["A", "A", "B", "B"]})
    f = rd.hist(data, x="value", hue="group", bins=2)
    # Step histograms use identical boundaries across groups.
    assert f.provenance()["plots"][0]["bin_edges"] == [0, 5.5, 11]


def test_heatmap_masks_missing_values_and_shared_colorbar_rejects_different_scales():
    f = rd.heatmap([[1, np.nan], [2, 3]], xlabel="Predicted", ylabel="Actual", cbar_label="Count")
    assert np.ma.getmaskarray(f.ax.images[0].get_array())[0, 1]
    assert f.validate().passed
    g = rd.figure(layout=[["a", "b"]], width="double")
    g["a"].heatmap([[1, 2]], colorbar=False)
    g["b"].heatmap([[3, 4]], colorbar=False)
    with pytest.raises(ValueError, match="normalization"):
        g.colorbar(panels=["a", "b"], label="Value")


def test_confusion_rows_with_no_observations_remain_missing():
    f = rd.confusion([[2, 2], [0, 0]], labels=["A", "B"], normalize="true")
    image = f.ax.images[0].get_array()
    np.testing.assert_allclose(image[0], [0.5, 0.5])
    assert np.ma.getmaskarray(image)[1].all()
    assert f.ax.get_xlabel() == "Predicted class"
    with pytest.raises(ValueError):
        rd.confusion([[1, -1], [2, 3]])


def test_correlation_requires_missing_policy_and_uses_signed_scale():
    f = rd.correlation(pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}))
    np.testing.assert_allclose(f.ax.images[0].get_array(), [[1, -1], [-1, 1]])
    assert f.ax.images[0].get_clim() == (-1, 1)
    with pytest.raises(ValueError, match="missing"):
        rd.correlation(pd.DataFrame({"A": [1, np.nan, 3], "B": [1, 2, 3]}))


def test_pareto_frontier_does_not_connect_dominated_points():
    f = rd.pareto(x=[1, 2, 3, 2], y=[2, 1, 4, 3], xlabel="Compute", ylabel="Score")
    front = f.ax.lines[0]
    np.testing.assert_allclose(front.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(front.get_ydata(), [2, 3, 4])
    assert len(f.ax.collections[0].get_offsets()) == 4


def test_roc_and_pr_group_tied_scores():
    f = rd.roc(y_true=[0, 1, 0, 1], y_score=[0.1, 0.8, 0.8, 0.9])
    np.testing.assert_allclose(f.ax.lines[0].get_xdata(), [0, 0, 0.5, 1])
    np.testing.assert_allclose(f.ax.lines[0].get_ydata(), [0, 0.5, 1, 1])
    assert f.provenance()["plots"][0]["auc"] == pytest.approx(0.875)
    g = rd.precision_recall(y_true=[0, 1, 0, 1], y_score=[0.1, 0.8, 0.8, 0.9])
    np.testing.assert_allclose(g.ax.lines[0].get_xdata(), [0, 0.5, 1, 1])
    np.testing.assert_allclose(g.ax.lines[0].get_ydata(), [1, 1, 2 / 3, 0.5])
    for function in [rd.roc, rd.precision_recall]:
        with pytest.raises(ValueError, match="(?i)both"):
            function(y_true=[1, 1], y_score=[0.5, 0.6])


def test_calibration_uses_mean_probability_not_bin_midpoint_and_keeps_empty_bins():
    f = rd.calibration(y_true=[0, 1, 1], y_prob=[0.01, 0.09, 0.95], bins=3)
    record = f.provenance()["plots"][-1]
    assert record["counts"] == [2, 0, 1]
    np.testing.assert_allclose(f.ax.lines[0].get_xdata()[[0, 2]], [0.05, 0.95])
    assert np.isnan(f.ax.lines[0].get_xdata()[1])
    with pytest.raises(ValueError, match="probabil"):
        rd.calibration(y_true=[0, 1], y_prob=[-1, 2])


def test_facets_share_semantics_and_do_not_add_titles():
    data = pd.DataFrame({"step": [0, 1, 0, 1], "score": [1, 2, 3, 4], "task": ["A", "A", "B", "B"]})
    f = rd.facet(data, col="task", kind="line", x="step", y="score")
    assert list(f.panels) == ["A", "B"]
    assert all(p.ax.get_title() == "" for p in f.panels.values())
    assert f["A"].ax.get_ylim() == f["B"].ax.get_ylim()
    assert f["A"].ax.lines[0].get_color() == rd.color_for("A")
    assert f["B"].ax.lines[0].get_color() == rd.color_for("B")
    assert f.validate().passed


def test_pairplot_and_justified_dual_axis():
    f = rd.pairplot(pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 4]}))
    assert len(f.panels) == 4
    assert not f.mpl.legends
    assert f.validate().passed
    g = rd.line(x=[0, 1], y=[1, 2], xlabel="Time (s)", ylabel="Power (W)")
    with pytest.raises(ValueError, match="justification"):
        g["a"].dual_axis(justification="", ylabel="Temperature (K)")
    right = g["a"].dual_axis(
        justification="Two sensors observed at identical times", ylabel="Temperature (K)"
    )
    right.line(x=[0, 1], y=[300, 310], ylabel="Temperature (K)")
    assert "dual_axis" in {w.code for w in g.validate().warnings}
