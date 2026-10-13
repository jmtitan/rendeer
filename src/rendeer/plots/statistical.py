"""Distribution grammars with explicit exclusion and smoothing policies."""

from inspect import signature
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..data import vector
from ..figure import Panel, plot
from .comparison import _axis_label


def _samples(data: Any, x: Any, hue: Any, missing: str) -> tuple[list[tuple[str, np.ndarray]], int]:
    values = vector(data, x, "x", numeric=True)
    labels = vector(data, hue, "hue") if hue is not None else np.full(len(values), "Observed")
    if len(values) != len(labels) or pd.isna(labels).any():
        raise ValueError("hue must match x and cannot contain missing labels")
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be raise or drop")
    dropped = int(np.isnan(values).sum())
    if dropped and missing == "raise":
        raise ValueError("Distribution has missing observations; choose missing='drop' explicitly")
    result = []
    for label in pd.unique(labels):
        subset = values[labels == label]
        subset = subset[np.isfinite(subset)]
        if not len(subset):
            raise ValueError(f"No observed values for {label!r}")
        result.append((str(label), subset))
    return result, dropped


@plot
def hist(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    hue: Any = None,
    bins: Any = "auto",
    density: bool = False,
    missing: str = "raise",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
) -> None:
    """Histogram with common bin edges across comparison groups (no range exclusion)."""
    samples, dropped = _samples(data, x, hue, missing)
    all_values = np.concatenate([v for _, v in samples])
    edges = np.histogram_bin_edges(all_values, bins=bins)
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be finite and strictly increasing")
    if all_values.min() < edges[0] or all_values.max() > edges[-1]:
        raise ValueError("bins must include all observations")
    for name, values in samples:
        style = panel.style(name)
        panel.ax.hist(
            values,
            bins=edges.tolist(),
            density=density,
            color=style["color"],
            histtype="bar" if len(samples) == 1 else "step",
            linewidth=1.2,
            edgecolor="#252525" if len(samples) == 1 else style["color"],
            alpha=0.85,
            label=name,
            linestyle=style["linestyle"],
        )
    panel.finish(
        "hist",
        xlabel=_axis_label(x, xlabel),
        ylabel=ylabel or ("Density" if density else "Count"),
        title=title,
        legend=legend and hue is not None,
        bin_edges=edges.tolist(),
        density=density,
        missing_dropped=dropped,
        sample_sizes={k: len(v) for k, v in samples},
    )


@plot
def kde(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    hue: Any = None,
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    missing: str = "raise",
    xlabel: str | None = None,
    ylabel: str = "Density",
    title: str | None = None,
    legend: bool = True,
) -> None:
    """Gaussian KDE using SciPy's bandwidth factor, clipped to observed support."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise ImportError("kde requires rendeer[stats]") from exc
    samples, dropped = _samples(data, x, hue, missing)
    factors = {}
    for name, values in samples:
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError("KDE requires at least two distinct observations")
        estimate = gaussian_kde(values, bw_method=bandwidth)
        grid = np.linspace(values.min(), values.max(), 256)
        style = panel.style(name)
        style.pop("marker")
        panel.ax.plot(grid, estimate(grid), label=name, **style)
        factors[name] = float(estimate.factor)
    panel.finish(
        "kde",
        xlabel=_axis_label(x, xlabel),
        ylabel=ylabel,
        title=title,
        legend=legend and hue is not None,
        bandwidth=bandwidth,
        factors=factors,
        missing_dropped=dropped,
        support="observed range; clipped tails not renormalized",
    )


@plot
def ecdf(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    hue: Any = None,
    missing: str = "raise",
    xlabel: str | None = None,
    ylabel: str = "Cumulative probability",
    title: str | None = None,
    legend: bool = True,
) -> None:
    """Exact empirical CDF: ties share a jump, all observations contribute."""
    samples, dropped = _samples(data, x, hue, missing)
    for name, values in samples:
        unique, counts = np.unique(values, return_counts=True)
        style = panel.style(name)
        style.pop("marker")
        panel.ax.step(
            np.r_[unique[0], unique],
            np.r_[0, np.cumsum(counts) / len(values)],
            where="post",
            label=name,
            **style,
        )
    panel.ax.set_ylim(0, 1.02)
    panel.finish(
        "ecdf",
        xlabel=_axis_label(x, xlabel),
        ylabel=ylabel,
        title=title,
        legend=legend and hue is not None,
        missing_dropped=dropped,
    )


@plot
def box(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    hue: Any = None,
    missing: str = "raise",
    whis: float = 1.5,
    xlabel: str | None = None,
    ylabel: str = "Group",
    title: str | None = None,
) -> None:
    """Horizontal Tukey boxes; median, quartiles and all outliers remain visible."""
    samples, dropped = _samples(data, x, hue, missing)
    if not np.isfinite(whis) or whis <= 0:
        raise ValueError("whis must be positive")
    for index, (name, values) in enumerate(samples):
        style = panel.style(name)
        orientation: dict[str, Any] = (
            {"orientation": "horizontal"}
            if "orientation" in signature(panel.ax.boxplot).parameters
            else {"vert": False}
        )
        panel.ax.boxplot(
            values,
            positions=[index],
            **orientation,
            widths=0.55,
            whis=whis,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
            boxprops={"facecolor": style["color"], "alpha": 0.55, "linewidth": 0.7},
            medianprops={"color": "#252525", "linewidth": 1.2},
            flierprops={"marker": style["marker"], "markersize": 3, "markeredgecolor": "#252525"},
        )
    panel.ax.set_yticks(range(len(samples)), [k for k, _ in samples])
    panel.finish(
        "box",
        xlabel=_axis_label(x, xlabel),
        ylabel=ylabel,
        title=title,
        whis=whis,
        missing_dropped=dropped,
        outliers="shown",
    )


@plot
def violin(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    hue: Any = None,
    missing: str = "raise",
    bandwidth: Literal["scott", "silverman"] | float = "scott",
    xlabel: str | None = None,
    ylabel: str = "Group",
    title: str | None = None,
) -> None:
    """Horizontal density silhouettes, medians and extrema; width is normalized per group."""
    samples, dropped = _samples(data, x, hue, missing)
    for index, (name, values) in enumerate(samples):
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError("Violin density requires at least two distinct observations; use box")
        style = panel.style(name)
        orientation: dict[str, Any] = (
            {"orientation": "horizontal"}
            if "orientation" in signature(panel.ax.violinplot).parameters
            else {"vert": False}
        )
        artists: Any = panel.ax.violinplot(
            values,
            positions=[index],
            **orientation,
            widths=0.7,
            showmedians=True,
            showextrema=True,
            bw_method=bandwidth,
        )
        for body in artists["bodies"]:
            body.set_facecolor(style["color"])
            body.set_edgecolor("#252525")
            body.set_alpha(0.6)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            artists[key].set_color("#252525")
            artists[key].set_linewidth(0.8)
    panel.ax.set_yticks(range(len(samples)), [k for k, _ in samples])
    panel.finish(
        "violin",
        xlabel=_axis_label(x, xlabel),
        ylabel=ylabel,
        title=title,
        bandwidth=bandwidth,
        width_normalization="per group",
        missing_dropped=dropped,
    )
