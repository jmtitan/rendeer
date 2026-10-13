"""Bounded deterministic refinement; never shrink text or delete observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from matplotlib.ticker import AutoLocator, MaxNLocator

if TYPE_CHECKING:
    from .figure import Figure


def text_box(text: Text, renderer: Any) -> Any:
    # Annotation.get_window_extent includes its leader line. Measure the label only.
    return Text.get_window_extent(text, renderer=renderer)


def overlap(a: Any, b: Any, *, tolerance: float = 1.0) -> bool:
    return (
        min(a.x1, b.x1) - max(a.x0, b.x0) > tolerance
        and min(a.y1, b.y1) - max(a.y0, b.y0) > tolerance
    )


def _tick_collision(labels: list[Any], renderer: Any) -> bool:
    boxes = [text_box(t, renderer) for t in labels if t.get_visible() and t.get_text()]
    return any(overlap(a, b) for a, b in zip(boxes, boxes[1:], strict=False))


def refine(figure: Figure) -> None:
    native = figure.mpl
    if not hasattr(native.canvas, "get_renderer"):
        FigureCanvasAgg(native)
    # Three canvas/layout passes bound the work independently of machine speed.
    for _ in range(3):
        native.canvas.draw()
        renderer = cast(Any, native.canvas).get_renderer()
        for panel in figure.panels.values():
            ax = panel.ax
            for axis, labels in [
                (ax.xaxis, ax.get_xticklabels()),
                (ax.yaxis, ax.get_yticklabels()),
            ]:
                if not _tick_collision(labels, renderer):
                    continue
                if isinstance(axis.get_major_locator(), (AutoLocator, MaxNLocator)):
                    axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
                elif axis is ax.xaxis:
                    angle = min(90, max(30, max(t.get_rotation() for t in labels)) + 15)
                    for tick in labels:
                        tick.set_rotation(angle)
                        tick.set_horizontalalignment("right")
            legend = ax.get_legend()
            if legend is not None:
                box = legend.get_window_extent(renderer)
                if box.width > ax.bbox.width and getattr(legend, "_ncols", 1) > 1:
                    legend.set_ncols(1)
    renderer = cast(Any, native.canvas).get_renderer()
    for panel in figure.panels.values():
        occupied = [
            text_box(t, renderer)
            for t in panel.ax.texts
            if t not in panel.annotations and t.get_visible() and t.get_text()
        ]
        for annotation in panel.annotations:
            # Fixed candidate order is reproducible and does not alter label text.
            candidates = [
                (5, 5),
                (5, 18),
                (5, -18),
                (-45, 5),
                (-45, 18),
                (-45, -18),
                (5, 32),
                (-45, 32),
                (20, -32),
                (-60, -32),
            ]
            best, best_score = candidates[0], float("inf")
            for offset in candidates:
                annotation.set_position(offset)
                annotation.update_positions(renderer)
                box = text_box(annotation, renderer)
                outside = not panel.ax.bbox.contains(box.x0, box.y0) or not panel.ax.bbox.contains(
                    box.x1, box.y1
                )
                score = 10 * int(outside) + sum(overlap(box, b) for b in occupied)
                if score < best_score:
                    best, best_score = offset, score
                if score == 0:
                    break
            annotation.set_position(best)
            annotation.update_positions(renderer)
            box = text_box(annotation, renderer)
            if np.isfinite(box.extents).all():
                occupied.append(box)
    native.canvas.draw()
