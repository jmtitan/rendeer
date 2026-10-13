"""Hand-calculated cases catch silent aggregation and incorrect uncertainty."""

import numpy as np
import pandas as pd
import pytest

import rendeer as rd


def test_summary_requires_independent_unit_and_rejects_duplicate_unit():
    data = pd.DataFrame({"step": [0, 0], "score": [1.0, 3.0], "seed": [1, 1]})
    with pytest.raises(ValueError, match="unit"):
        rd.summarize(data, x="step", y="score", error="ci95")
    with pytest.raises(ValueError, match="duplicate"):
        rd.summarize(data, x="step", y="score", unit="seed", error="sd")


def test_summary_sd_se_and_t_interval_are_distinct():
    data = pd.DataFrame({"step": [0, 0, 0], "score": [1.0, 2.0, 3.0], "seed": [0, 1, 2]})
    sd = rd.summarize(data, x="step", y="score", unit="seed", error="sd")
    se = rd.summarize(data, x="step", y="score", unit="seed", error="se")
    ci = rd.summarize(data, x="step", y="score", unit="seed", error="ci95")
    assert sd.loc[0, "estimate"] == 2
    assert sd.loc[0, "lower"] == 1
    assert se.loc[0, "upper"] == pytest.approx(2 + 1 / np.sqrt(3))
    assert ci.loc[0, "lower"] == pytest.approx(-0.4841377117)
    assert ci.loc[0, "n"] == 3


def test_singleton_interval_is_unavailable_instead_of_zero():
    data = pd.DataFrame({"step": [0], "score": [1.0], "seed": [1]})
    with pytest.warns(UserWarning, match="fewer than two"):
        result = rd.summarize(data, x="step", y="score", unit="seed", error="sd")
    assert np.isnan(result.loc[0, "lower"])


def test_explicit_bounds_are_validated():
    with pytest.raises(ValueError, match="bounds"):
        rd.interval(x=[0, 1], y=[2, 3], lower=[4, 1], upper=[5, 4], uncertainty="range")
    with pytest.raises(ValueError, match="uncertainty"):
        rd.interval(x=[0], y=[2], lower=[1], upper=[3])


def test_line_band_has_explicit_definition_in_provenance():
    data = pd.DataFrame({"step": [0, 0, 1, 1], "score": [1, 3, 2, 4], "seed": [0, 1, 0, 1]})
    f = rd.line(data, x="step", y="score", unit="seed", error="sd")
    np.testing.assert_allclose(f.ax.lines[0].get_ydata(), [2, 3])
    assert len(f.ax.collections) == 1
    assert f.provenance()["plots"][0]["uncertainty"] == "sd"
