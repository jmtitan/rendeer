"""Comparative plots preserve observation order and require explicit summaries."""

from typing import Any

import numpy as np
import pandas as pd

from ..data import bounds, groups, summarize, xy_frame
from ..figure import Panel, plot


def _axis_label(value: Any, override: str | None) -> str | None:
    return override if override is not None else (value if isinstance(value, str) else None)


def _scales(panel: Panel, frame: pd.DataFrame, xscale: str, yscale: str) -> None:
    for axis, scale in [("x", xscale), ("y", yscale)]:
        if scale not in {"linear", "log", "symlog"}:
            raise ValueError("scale must be linear, log or symlog")
        if scale == "log" and (frame[axis].dropna() <= 0).any():
            raise ValueError("Log scales require strictly positive observations")
        getattr(panel.ax, f"set_{axis}scale")(scale)


@plot
def line(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    error: str | None = None,
    unit: str | None = None,
    lower: Any = None,
    upper: Any = None,
    uncertainty: str | None = None,
    label: str | None = None,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    legend: bool = True,
    sort: bool = False,
    **kwargs: Any,
) -> None:
    """Observed line (no smoothing); optional mean and explicitly defined band."""
    original_x, original_y = x, y
    if error:
        if lower is not None or upper is not None:
            raise ValueError("Use computed or supplied bounds, not both")
        data = summarize(data, x=x, y=y, hue=hue, unit=unit, error=error)
        y, lower, upper, uncertainty = "estimate", "lower", "upper", error
    frame = xy_frame(data, x, y, hue)
    if frame.duplicated(["x", "hue"]).any():
        raise ValueError(
            "duplicate x within a series; use error with an independent unit or pre-aggregate"
        )
    if lower is not None or upper is not None:
        if error:
            lo, hi = data["lower"].to_numpy(), data["upper"].to_numpy()
        else:
            lo, hi = bounds(data, lower, upper, frame.y.to_numpy(), uncertainty)
        frame["lower"], frame["upper"] = lo, hi
    _scales(panel, frame, xscale, yscale)
    for name, group in groups(frame):
        if sort:
            group = group.sort_values("x", kind="stable")
        name = label or name
        style = panel.style(name, color)
        style.update(kwargs)
        panel.ax.plot(
            group.x, group.y, label=name if hue is not None or label else "_nolegend_", **style
        )
        if "lower" in group:
            if yscale == "log" and (group.lower.dropna() <= 0).any():
                raise ValueError("Log scales require positive interval bounds")
            panel.ax.fill_between(
                group.x, group.lower, group.upper, color=style["color"], alpha=0.16, linewidth=0
            )
    panel.finish(
        "line",
        xlabel=_axis_label(original_x, xlabel),
        ylabel=_axis_label(original_y, ylabel),
        title=title,
        legend=legend and (hue is not None or label is not None),
        uncertainty=uncertainty,
        unit=unit,
        estimator="mean" if error else None,
        sort=sort,
        xscale=xscale,
        yscale=yscale,
        missing="gaps",
    )


@plot
def scatter(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    label: str | None = None,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    legend: bool = True,
    rasterized: bool = False,
    **kwargs: Any,
) -> None:
    """Raw paired observations, with redundant semantic marker shapes."""
    frame = xy_frame(data, x, y, hue)
    _scales(panel, frame, xscale, yscale)
    for name, group in groups(frame):
        style = panel.style(label or name, color)
        style.pop("linestyle")
        style.update(s=24, edgecolors="#252525", linewidths=0.35, alpha=0.85)
        style.update(kwargs)
        panel.ax.scatter(
            group.x,
            group.y,
            label=label or name if hue is not None or label else "_nolegend_",
            rasterized=rasterized,
            **style,
        )
    panel.finish(
        "scatter",
        xlabel=_axis_label(x, xlabel),
        ylabel=_axis_label(y, ylabel),
        title=title,
        legend=legend and (hue is not None or label is not None),
        xscale=xscale,
        yscale=yscale,
        missing=int(frame.y.isna().sum()),
        rasterized=rasterized,
    )


@plot
def bar(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    stacked: bool = False,
    orient: str = "vertical",
    highlight: str | None = None,
    order: list[Any] | None = None,
    label: str | None = None,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> None:
    """One value per category/series, grouped or signed-stacked, with a zero baseline."""
    frame = xy_frame(data, x, y, hue)
    if orient not in {"vertical", "horizontal"}:
        raise ValueError("orient must be vertical or horizontal")
    if frame.duplicated(["x", "hue"]).any():
        raise ValueError("duplicate bar category; explicitly summarize independent units first")
    categories = list(pd.unique(frame.x)) if order is None else list(order)
    if len(set(categories)) != len(categories) or set(categories) != set(frame.x):
        raise ValueError("order must contain every category exactly once")
    if stacked and (frame.y.isna().any() or len(frame) != len(categories) * frame.hue.nunique()):
        raise ValueError("Stacked bars require complete components; missing is not zero")
    series = list(groups(frame))
    positive, negative = np.zeros(len(categories)), np.zeros(len(categories))
    width = 0.72 / (1 if stacked else len(series))
    for index, (name, group) in enumerate(series):
        values = group.set_index("x").y.reindex(categories).to_numpy()
        positions = np.arange(len(categories), dtype=float)
        if not stacked:
            positions += (index - (len(series) - 1) / 2) * width
        bottom = np.where(values >= 0, positive, negative) if stacked else np.zeros(len(categories))
        identity = label or name
        style = {
            "color": panel.style(identity, color)["color"],
            "edgecolor": "#252525",
            "linewidth": 0.45,
        }
        if highlight:
            if hue is not None:
                style["alpha"] = 1 if name == highlight else 0.65
            else:
                style["color"] = [
                    panel.style(str(c), "primary" if str(c) == highlight else "neutral")["color"]
                    for c in categories
                ]
        if panel.owner.config.theme == "grayscale" or panel.owner.config.palette == "grayscale":
            style["hatch"] = ("", "//", "..", "xx")[index % 4]
        style.update(kwargs)
        drawing = panel.ax.bar if orient == "vertical" else panel.ax.barh
        size: dict[str, Any] = (
            {"width": width, "bottom": bottom}
            if orient == "vertical"
            else {"height": width, "left": bottom}
        )
        drawing(
            positions,
            values,
            **size,
            label=identity if hue is not None or label else "_nolegend_",
            **style,
        )
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    ticks = panel.ax.set_xticks if orient == "vertical" else panel.ax.set_yticks
    ticks(np.arange(len(categories)), [str(c) for c in categories])
    if orient == "horizontal":
        panel.ax.invert_yaxis()
    panel.finish(
        "bar",
        xlabel=_axis_label(x if orient == "vertical" else y, xlabel),
        ylabel=_axis_label(y if orient == "vertical" else x, ylabel),
        title=title,
        legend=legend and (hue is not None or label is not None),
        stacked=stacked,
        orient=orient,
        order=[str(c) for c in categories],
        missing=int(frame.y.isna().sum()),
    )


@plot
def interval(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    lower: Any,
    upper: Any,
    uncertainty: str | None = None,
    hue: Any = None,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> None:
    """Point-range plot with supplied absolute bounds and a required definition."""
    frame = xy_frame(data, x, y, hue)
    frame["lower"], frame["upper"] = bounds(data, lower, upper, frame.y.to_numpy(), uncertainty)
    for name, group in groups(frame):
        style = panel.style(label or name)
        style.update(linestyle="none", capsize=2.5, elinewidth=1)
        style.update(kwargs)
        panel.ax.errorbar(
            group.x,
            group.y,
            yerr=np.vstack([group.y - group.lower, group.upper - group.y]),
            label=label or name if hue is not None or label else "_nolegend_",
            **style,
        )
    panel.finish(
        "interval",
        xlabel=_axis_label(x, xlabel),
        ylabel=_axis_label(y, ylabel),
        title=title,
        uncertainty=uncertainty,
        legend=legend and (hue is not None or label is not None),
    )


@plot
def area(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: str | None = None,
    color: str | None = None,
    **kwargs: Any,
) -> None:
    """A single observed series filled to zero, with missing-value gaps."""
    frame = xy_frame(data, x, y)
    if frame.x.duplicated().any():
        raise ValueError("duplicate x in area plot")
    style = panel.style(label or "Observed", color)
    panel.ax.fill_between(frame.x, 0, frame.y, color=style["color"], alpha=0.2, **kwargs)
    panel.ax.plot(frame.x, frame.y, **style, label=label or "_nolegend_")
    panel.finish(
        "area",
        xlabel=_axis_label(x, xlabel),
        ylabel=_axis_label(y, ylabel),
        legend=label is not None,
        missing="gaps",
    )
