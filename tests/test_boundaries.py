"""Cross-panel and export boundaries that can silently misrepresent a figure."""

import json

import matplotlib as mpl
import numpy as np
import pytest

import rendeer as rd


def test_shared_colorbar_rejects_same_limits_with_different_centers():
    f = rd.figure(layout=[["a", "b"]], width="double")
    for name, center in [("a", 0), ("b", 0.5)]:
        f[name].heatmap([[-1, 1]], center=center, vmin=-1, vmax=1, colorbar=False)
    with pytest.raises(ValueError, match="normalization"):
        f.colorbar(panels=["a", "b"], label="Value")


def test_validation_detects_semantic_color_changed_through_native_axes():
    f = rd.line(x=[0, 1], y=[1, 2], label="Ours", xlabel="Step", ylabel="Score")
    f.ax.lines[0].set_color("red")
    assert "inconsistent_color" in {i.code for i in f.validate().errors}


def test_provenance_captures_effective_axis_changes():
    f = rd.line(x=[1, 10], y=[1, 2], xlabel="Step", ylabel="Score")
    f.ax.set_xscale("log")
    f.ax.set_ylabel("Updated score")
    record = f.provenance()["axes"]["a"]
    assert record["xscale"] == "log"
    assert record["ylabel"] == "Updated score"
    json.dumps(f.provenance())


def test_unsupported_numeric_axis_conversion_does_not_drop_values():
    with pytest.raises(ValueError, match="positive"):
        rd.line(x=[1, 2], y=[1, 2], lower=[0, 1], upper=[2, 3], uncertainty="range", yscale="log")
    f = rd.area(x=[0, 1, 2], y=[1, np.nan, -2], xlabel="Time", ylabel="Change")
    assert f.ax.get_ylim()[0] < 0 < f.ax.get_ylim()[1]
    assert np.isnan(f.ax.lines[0].get_ydata()[1])


def test_shared_limits_require_compatible_scales():
    f = rd.figure(layout=[["a", "b"]], width="double")
    f["a"].line(x=[1, 2], y=[1, 2], xlabel="X", ylabel="Y")
    f["b"].line(x=[1, 2], y=[3, 4], xlabel="X", ylabel="Y")
    f.share_limits()
    assert f["a"].ax.get_ylim() == f["b"].ax.get_ylim()
    f["b"].ax.set_yscale("log")
    with pytest.raises(ValueError, match="scales"):
        f.share_limits()


def test_preview_does_not_inherit_global_tight_crop():
    from io import BytesIO

    from PIL import Image

    with mpl.rc_context({"savefig.bbox": "tight"}):
        f = rd.line(x=[1, 2], y=[3, 4], xlabel="X", ylabel="Y", width=4, height=3)
        with Image.open(BytesIO(f._repr_png_())) as image:
            assert image.size == (600, 450)


def test_each_preset_has_legible_fixed_size_render():
    for theme in ["paper", "nature", "ml", "ieee", "slides", "report", "grayscale"]:
        with rd.context(theme=theme):
            f = rd.line(x=[1, 2, 3], y=[2, 4, 3], xlabel="Time (s)", ylabel="Value")
            assert f.validate().passed
