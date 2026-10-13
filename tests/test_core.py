"""Contracts: style isolation, semantic identity, composition and data honesty."""

import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

import rendeer as rd


def test_configuration_does_not_mutate_matplotlib():
    before = dict(mpl.rcParams)
    importlib.reload(rd)
    rd.configure(theme="nature", font="DejaVu Sans", strict=True)
    f = rd.line(x=[0, 1], y=[1, 2], xlabel="Step", ylabel="Score")
    f.render()
    assert dict(mpl.rcParams) == before
    with pytest.raises(FrozenInstanceError):
        rd.get_config().strict = False


def test_context_restores_even_after_exception():
    original = rd.get_config()
    with pytest.raises(RuntimeError), rd.context(theme="slides"):
        assert rd.get_config().theme == "slides"
        raise RuntimeError("example")
    assert rd.get_config() == original


def test_config_rejects_invalid_values():
    for kw in [{"theme": "unknown"}, {"dpi": 0}, {"palette": "rainbow"}]:
        with pytest.raises(ValueError):
            rd.configure(**kw)


def test_semantic_color_and_marker_survive_reordering():
    rd.set_colors({"Ours": "primary", "Baseline": "neutral"})
    frame = pd.DataFrame(
        {"x": [0, 1, 0, 1], "y": [1, 2, 2, 3], "method": ["Ours", "Ours", "Baseline", "Baseline"]}
    )
    f = rd.line(frame, x="x", y="y", hue="method")
    g = rd.line(frame.iloc[::-1], x="x", y="y", hue="method")

    def style(fig):
        return {line.get_label(): (line.get_color(), line.get_marker()) for line in fig.ax.lines}

    assert style(f) == style(g)
    assert style(f)["Ours"][0] == "#4678ae"
    assert f.ax.get_title() == ""
    assert all(f.ax.get_xlabel() for f in [f, g])


def test_color_is_stable_in_another_process():
    result = subprocess.check_output(
        [
            sys.executable,
            "-B",
            "-c",
            "import rendeer as r; print(r.color_for('unregistered-model'))",
        ],
        text=True,
    )
    assert result.strip() == rd.color_for("unregistered-model")


def test_duplicate_colors_are_reported():
    rd.set_colors({"A": "primary", "B": "primary"})
    f = rd.line(pd.DataFrame({"x": [1, 1], "y": [2, 3], "g": ["A", "B"]}), x="x", y="y", hue="g")
    assert "ambiguous_colors" in {i.code for i in f.validate().warnings}


def test_missing_value_stays_a_gap_and_data_is_not_modified():
    frame = pd.DataFrame({"step": [0, 1, 2], "score": [1, np.nan, 3]})
    before = frame.copy(deep=True)
    f = rd.line(frame, x="step", y="score")
    assert np.isnan(f.ax.lines[0].get_ydata()[1])
    pd.testing.assert_frame_equal(frame, before)


def test_repeated_x_requires_explicit_aggregation():
    with pytest.raises(ValueError, match="duplicate"):
        rd.line(x=[0, 0], y=[1, 2])


def test_invalid_inputs_are_rejected():
    for kw in [
        {"x": [], "y": []},
        {"x": [1], "y": [1, 2]},
        {"x": [1, 2], "y": [1, np.inf]},
        {"x": [1, 2], "y": [np.nan, np.nan]},
    ]:
        with pytest.raises(ValueError):
            rd.line(**kw)
    with pytest.raises(ValueError, match="positive"):
        rd.line(x=[0, 1], y=[1, 2], xscale="log")


def test_bar_preserves_zero_negative_values_and_group_order():
    frame = pd.DataFrame(
        {
            "task": ["Z", "A", "Z", "A"],
            "value": [2, -1, 3, 4],
            "method": ["Ours", "Ours", "Baseline", "Baseline"],
        }
    )
    f = rd.bar(frame, x="task", y="value", hue="method")
    assert [t.get_text() for t in f.ax.get_xticklabels()] == ["Z", "A"]
    assert f.ax.get_ylim()[0] <= 0 <= f.ax.get_ylim()[1]
    assert sorted(p.get_height() for p in f.ax.patches) == [-1, 2, 3, 4]


def test_stacked_bars_use_separate_positive_and_negative_baselines():
    f = rd.bar(
        pd.DataFrame({"x": ["A"] * 3, "y": [2, -1, 3], "g": ["a", "b", "c"]}),
        x="x",
        y="y",
        hue="g",
        stacked=True,
    )
    assert [(p.get_y(), p.get_height()) for p in f.ax.patches] == [(0, 2), (0, -1), (2, 3)]


def test_mosaic_ownership_shared_legend_and_config_snapshot():
    f = rd.figure(layout=[["a", "a"], ["b", "c"]], width="double", panel_labels=True)
    f["a"].line(x=[0, 1], y=[2, 3], label="Ours", xlabel="Step", ylabel="Score")
    f["b"].bar(x=["A", "B"], y=[2, 3], label="Ours", xlabel="Task", ylabel="Score")
    f["c"].scatter(x=[1, 2], y=[3, 4], label="Baseline", xlabel="Cost", ylabel="Score")
    assert len(f.mpl.axes) == 3
    assert f["b"].ax.figure is f.mpl
    f.legend()
    assert [t.get_text() for t in f.mpl.legends[0].get_texts()] == ["Ours", "Baseline"]
    previous = f.config
    rd.configure(theme="slides")
    assert f.config == previous
    json.dumps(f.provenance())


def test_existing_axes_is_used_without_creating_another_figure():
    from matplotlib.figure import Figure

    native = Figure()
    ax = native.subplots()
    f = rd.line(x=[1, 2], y=[3, 4], ax=ax, xlabel="X", ylabel="Y")
    assert f.ax is ax and f.mpl is native
