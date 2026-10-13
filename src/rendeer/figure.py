"""Figure ownership and one plot registry shared by panels and standalone calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import partial, wraps
from io import BytesIO
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure as NativeFigure

from .colors import LINESTYLES, MARKERS, color_for, identity_index
from .config import Config, get_config
from .theme import THEMES, rc_params

PLOTS: dict[str, Callable[..., Any]] = {}


def plot(function: Callable[..., Any]) -> Callable[..., Figure]:
    """Register an axes-level implementation with a standalone Figure wrapper."""
    PLOTS[function.__name__] = function

    @wraps(function)
    def standalone(
        *args: Any,
        ax: Axes | None = None,
        width: str | float = "single",
        height: float | None = None,
        **kwargs: Any,
    ) -> Figure:
        owner = Figure.from_axes(ax) if ax is not None else figure(width=width, height=height)
        owner["a"]._draw(function, *args, **kwargs)
        return owner

    return standalone


class Panel:
    """A named native Axes. All registered rd plot functions are also methods."""

    def __init__(self, owner: Figure, name: str, ax: Axes) -> None:
        self.owner, self.name, self.ax = owner, name, ax
        self.kinds: set[str] = set()
        self.required_labels = (True, True)
        self.semantic: dict[str, str] = {}
        self.mappable: Any = None
        self.annotations: list[Any] = []
        self.semantic_artists: list[tuple[Any, str, str]] = []

    def __getattr__(self, name: str) -> Any:
        if name in PLOTS:
            return partial(self._draw, PLOTS[name])
        raise AttributeError(name)

    def _draw(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Panel:
        previous = set(self.ax.get_children())
        with mpl.rc_context(rc_params(self.owner.config)):
            function(self, *args, **kwargs)
        for artist in self.ax.get_children():
            label = artist.get_label()
            if artist not in previous and label in self.semantic:
                self.semantic_artists.append((artist, label, self.semantic[label]))
        return self

    def style(self, label: str, color: str | None = None) -> dict[str, Any]:
        from .colors import resolve_color

        index = identity_index(label)
        resolved = resolve_color(color) if color else color_for(label, self.owner.config)
        self.semantic[label] = resolved
        return {
            "color": resolved,
            "marker": MARKERS[index % len(MARKERS)],
            "linestyle": LINESTYLES[index % len(LINESTYLES)],
        }

    def finish(
        self,
        kind: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        legend: bool = False,
        **record: Any,
    ) -> None:
        self.kinds.add(kind)
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if title is not None:
            self.ax.set_title(title)
        if legend:
            handles, labels = self.ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles, strict=True))
            if unique:
                self.ax.legend(
                    unique.values(),
                    unique.keys(),
                    loc="lower left",
                    bbox_to_anchor=(0, 1.01),
                    ncols=min(3, len(unique)),
                    borderaxespad=0,
                    handlelength=1.6,
                    columnspacing=1.0,
                )
        self.owner.records.append({"panel": self.name, "kind": kind, **record})

    def annotate(self, labels: Sequence[str], *, x: Any, y: Any) -> Panel:
        from .data import vector

        xx, yy = vector(None, x, "x", numeric=True), vector(None, y, "y", numeric=True)
        if len(labels) != len(xx) or len(xx) != len(yy):
            raise ValueError("labels, x and y must have equal lengths")
        with mpl.rc_context(rc_params(self.owner.config)):
            for label, px, py in zip(labels, xx, yy, strict=True):
                text = self.ax.annotate(
                    str(label),
                    xy=(px, py),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=THEMES[self.owner.config.theme].font_size - 1,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#666666"},
                )
                self.annotations.append(text)
        return self

    def dual_axis(self, *, justification: str, ylabel: str) -> Panel:
        """Explicitly justified second y-axis; scales still require human review."""
        if not justification.strip() or not ylabel.strip():
            raise ValueError("dual_axis requires justification and a unit-bearing ylabel")
        with mpl.rc_context(rc_params(self.owner.config)):
            ax = self.ax.twinx()
            ax.spines["right"].set_visible(True)
            ax.set_ylabel(ylabel)
        panel = Panel(self.owner, f"{self.name}_right", ax)
        panel.required_labels = (False, True)
        self.owner.panels[panel.name] = panel
        self.owner.records.append(
            {"panel": panel.name, "kind": "dual_axis", "justification": justification}
        )
        return panel


class Figure:
    def __init__(self, native: NativeFigure, axes: Mapping[str, Axes], config: Config) -> None:
        self.mpl, self.config = native, config
        self.panels = {str(k): Panel(self, str(k), ax) for k, ax in axes.items()}
        self.records: list[dict[str, Any]] = []

    @classmethod
    def from_axes(cls, ax: Axes) -> Figure:
        return cls(ax.figure, {"a": ax}, get_config())  # type: ignore[arg-type]

    def __getitem__(self, name: str) -> Panel:
        return self.panels[name]

    @property
    def ax(self) -> Axes:
        if len(self.panels) != 1:
            raise ValueError("Use fig['panel'].ax for multi-panel figures")
        return next(iter(self.panels.values())).ax

    def render(self) -> Figure:
        from .layout import refine

        with mpl.rc_context(rc_params(self.config)):
            refine(self)
        return self

    def validate(self) -> Any:
        from .validation import inspect

        with mpl.rc_context(rc_params(self.config)):
            self.render()
            return inspect(self)

    def save(self, path: Any, **kwargs: Any) -> list[Any]:
        from .export import save

        return save(self, path, **kwargs)

    def _repr_png_(self) -> bytes:
        from .validation import QualityError

        report = self.validate()
        if self.config.strict and not report.passed:
            raise QualityError(report)
        buffer = BytesIO()
        with mpl.rc_context(rc_params(self.config)):
            self.mpl.savefig(buffer, format="png", dpi=150, bbox_inches=None)
        return buffer.getvalue()

    def provenance(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "size_inches": self.mpl.get_size_inches().tolist(),
            "semantic_colors": {k: v for p in self.panels.values() for k, v in p.semantic.items()},
            "plots": list(self.records),
            "axes": {
                name: {
                    "xlabel": p.ax.get_xlabel(),
                    "ylabel": p.ax.get_ylabel(),
                    "title": p.ax.get_title(),
                    "xlim": list(p.ax.get_xlim()),
                    "ylim": list(p.ax.get_ylim()),
                    "xscale": p.ax.get_xscale(),
                    "yscale": p.ax.get_yscale(),
                }
                for name, p in self.panels.items()
            },
            "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__},
        }

    def legend(self, *, ncols: int | None = None) -> Figure:
        unique: dict[str, Any] = {}
        for panel in self.panels.values():
            handles, labels = panel.ax.get_legend_handles_labels()
            unique.update(zip(labels, handles, strict=True))
            existing = panel.ax.get_legend()
            if existing is not None:
                existing.remove()
        for previous in list(self.mpl.legends):
            previous.remove()
        if unique:
            with mpl.rc_context(rc_params(self.config)):
                self.mpl.legend(
                    unique.values(),
                    unique.keys(),
                    loc="outside upper center",
                    ncols=ncols or min(4, len(unique)),
                )
        return self

    def colorbar(self, *, panels: Sequence[str], label: str) -> Any:
        selected = [self[name] for name in panels]
        maps = [p.mappable for p in selected]
        if not maps or any(m is None for m in maps):
            raise ValueError("Each selected panel must contain a heatmap")
        signatures = [
            (m.get_clim(), m.get_cmap().name, type(m.norm), getattr(m.norm, "vcenter", None))
            for m in maps
        ]
        if any(s != signatures[0] for s in signatures):
            raise ValueError("Shared colorbar requires identical normalization and colormap")
        reference = maps[0].cmap(np.linspace(0, 1, 256))
        if any(not np.array_equal(m.cmap(np.linspace(0, 1, 256)), reference) for m in maps[1:]):
            raise ValueError("Shared colorbar requires identical normalization and colormap")
        with mpl.rc_context(rc_params(self.config)):
            return self.mpl.colorbar(maps[0], ax=[p.ax for p in selected], label=label, shrink=0.9)

    def share_limits(self, *, panels: Sequence[str] | None = None, axis: str = "y") -> Figure:
        if axis not in {"x", "y"}:
            raise ValueError("axis must be x or y")
        selected = [self[n].ax for n in (panels or list(self.panels))]
        scales = {getattr(ax, f"get_{axis}scale")() for ax in selected}
        if len(scales) != 1:
            raise ValueError("Cannot share limits across different scales")
        limits = [getattr(ax, f"get_{axis}lim")() for ax in selected]
        if any(a > b for a, b in limits):
            raise ValueError("Inverted axes require explicit limits")
        low, high = min(a for a, _ in limits), max(b for _, b in limits)
        for ax in selected:
            getattr(ax, f"set_{axis}lim")(low, high)
        return self


def figure(
    *,
    layout: Sequence[Sequence[str]] | None = None,
    width: str | float = "single",
    height: float | None = None,
    sharex: bool = False,
    sharey: bool = False,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    panel_labels: bool = False,
) -> Figure:
    config = get_config()
    theme = THEMES[config.theme]
    if isinstance(width, str):
        if width not in {"single", "double"}:
            raise ValueError("width must be single, double, or inches")
        width = float(getattr(theme, width))
    width_inches = float(width)
    mosaic: Any = [list(row) for row in layout] if layout is not None else [["a"]]
    nrows, ncols = len(mosaic), len(mosaic[0]) if len(mosaic) else 0
    if nrows == 0 or ncols == 0:
        raise ValueError("layout must not be empty")
    height = height if height is not None else max(2.35, width_inches / ncols * 0.72) * nrows
    if not np.isfinite([width_inches, height]).all() or width_inches <= 0 or height <= 0:
        raise ValueError("Figure dimensions must be finite and positive")
    with mpl.rc_context(rc_params(config)):
        native = NativeFigure(figsize=(width_inches, height), layout="constrained", dpi=100)
        FigureCanvasAgg(native)
        axes = native.subplot_mosaic(
            mosaic,
            sharex=sharex,
            sharey=sharey,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        result = Figure(native, axes, config)
        if panel_labels:
            for i, panel in enumerate(result.panels.values()):
                text = panel.ax.annotate(
                    f"({chr(97 + i)})",
                    (0, 1),
                    xycoords="axes fraction",
                    xytext=(-30, 8),
                    textcoords="offset points",
                    fontsize=8 if config.theme == "nature" else theme.font_size,
                    fontweight="bold",
                    annotation_clip=False,
                )
                text.set_gid("rendeer-panel-label")
    return result
