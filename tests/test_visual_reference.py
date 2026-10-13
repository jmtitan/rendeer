"""Compare against a human-inspected, curated PNG reference at physical size."""

from pathlib import Path

import matplotlib as mpl
from gallery import synthetic_data
from matplotlib.testing.compare import compare_images

import rendeer as rd


def test_learning_curve_matches_reviewed_reference(output_dir):
    modern = tuple(int(x) for x in mpl.__version__.split(".")[:2]) >= (3, 11)
    reference = (
        Path(__file__).parent / "reference" / ("learning-mpl311.png" if modern else "learning.png")
    )
    assert reference.is_file(), "The reviewed reference fixture must be shipped"
    data = synthetic_data()
    figure = rd.line(
        data["curves"],
        x="step",
        y="accuracy",
        hue="method",
        unit="seed",
        error="sd",
        xlabel="Step",
        ylabel="Accuracy",
    )
    actual = output_dir / "visual-actual.png"
    figure.save(actual, overwrite=True, manifest=False)
    # A small RMS allowance handles rasterizer differences between supported Matplotlib versions.
    assert compare_images(str(reference), str(actual), tol=1.5) is None
