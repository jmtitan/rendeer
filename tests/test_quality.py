"""Rendered-canvas checks and safe exports; no mocked Matplotlib behavior."""

import json

import pytest
from PIL import Image

import rendeer as rd


def test_clean_figure_passes_and_labels_outside_canvas_fail():
    f = rd.line(x=[0, 1], y=[1, 2], xlabel="Step", ylabel="Score")
    assert f.validate().passed
    f.mpl.text(1.1, 0.5, "Outside")
    assert "clipped_text" in {i.code for i in f.validate().errors}


def test_missing_labels_and_small_text_are_errors():
    f = rd.line(x=[0, 1], y=[1, 2])
    assert "missing_label" in {i.code for i in f.validate().errors}
    f.ax.text(0.5, 1.5, "Too small", fontsize=2)
    assert "small_text" in {i.code for i in f.validate().errors}


def test_text_overlap_is_detected_without_claiming_artist_coverage():
    f = rd.scatter(x=[0, 1], y=[0, 1], xlabel="X", ylabel="Y")
    f.ax.text(0.5, 0.5, "First label")
    f.ax.text(0.5, 0.5, "Second label")
    report = f.validate()
    assert "text_overlap" in {i.code for i in report.errors}
    assert report.coverage
    json.dumps(report.to_dict())


def test_strict_save_refuses_defective_figure_before_writing(output_dir):
    with rd.context(strict=True):
        f = rd.line(x=[0, 1], y=[1, 2])
    destination = output_dir / "should-not-exist.png"
    with pytest.raises(rd.QualityError, match="missing_label"):
        f.save(destination, overwrite=True)
    assert not destination.exists()


def test_truncated_bar_axis_is_flagged():
    f = rd.bar(x=["A", "B"], y=[10, 11], xlabel="Method", ylabel="Score")
    f.ax.set_ylim(9, 12)
    assert "zero_baseline" in {i.code for i in f.validate().errors}


def test_annotation_placement_is_deterministic_and_keeps_labels():
    results = []
    for _ in range(2):
        f = rd.scatter(x=[0, 1, 2], y=[0, 1, 2], xlabel="X", ylabel="Y")
        f["a"].annotate(["First", "Second"], x=[1, 1], y=[1, 1])
        f.render()
        results.append([tuple(t.get_position()) for t in f.ax.texts])
        assert len(f.ax.texts) == 2
        assert "text_overlap" not in {i.code for i in f.validate().errors}
    assert results[0] == results[1]


def test_png_dimensions_manifest_overwrite_and_format_restrictions(output_dir):
    f = rd.line(x=[0, 1], y=[1, 2], xlabel="Step", ylabel="Score", width=4, height=3)
    path = output_dir / "size.png"
    f.save(path, dpi=150, overwrite=True)
    with Image.open(path) as im:
        assert im.size == (600, 450)
        assert im.info["dpi"][0] == pytest.approx(150, abs=0.02)
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["size_inches"] == [4, 3]
    assert manifest["exports"][0]["sha256"]
    with pytest.raises(FileExistsError):
        f.save(path)
    with pytest.raises(ValueError, match="PNG and PDF"):
        f.save(output_dir / "forbidden.svg")
    with pytest.raises(ValueError, match="PNG and PDF"):
        f.save(output_dir / "mixed", formats=["png", "svg"])
    assert not (output_dir / "mixed.png").exists()


def test_repeated_png_render_is_identical():
    f = rd.line(x=[0, 1, 2], y=[0, 1, 0], xlabel="Step", ylabel="Value")
    first = f._repr_png_()
    second = f._repr_png_()
    assert first == second
    assert first.startswith(b"\x89PNG")


def test_long_categorical_ticks_are_rotated_without_shrinking():
    f = rd.bar(
        x=["Long baseline method", "Long proposed method", "Long ablation method"],
        y=[1, 2, 3],
        xlabel="Method",
        ylabel="Score",
    )
    before = f.ax.get_xticklabels()[0].get_fontsize()
    f.render()
    assert f.ax.get_xticklabels()[0].get_rotation() != 0
    assert f.ax.get_xticklabels()[0].get_fontsize() == before
    assert f.validate().passed
