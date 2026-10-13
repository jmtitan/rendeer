"""Measured defects are errors; perceptual/design heuristics are warnings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib.colors import to_hex

from .colors import contrast_ratio
from .layout import overlap, text_box
from .theme import THEMES

if TYPE_CHECKING:
    from .figure import Figure


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    panel: str | None = None


@dataclass(frozen=True)
class QualityReport:
    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]
    coverage: str = (
        "Rendered text bounding boxes, axes labels, font sizes, limits, zero baselines, "
        "registered semantic colors, and legend bounds. Arbitrary artist occlusion, "
        "glyph availability, color-vision accessibility and scientific validity need human review."
    )

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, **asdict(self)}


class QualityError(ValueError):
    def __init__(self, report: QualityReport) -> None:
        self.report = report
        super().__init__(
            "Figure quality check failed: "
            + "; ".join(f"{i.code}: {i.message}" for i in report.errors)
        )


def inspect(figure: Figure) -> QualityReport:
    errors, recommendations = [], []
    native = figure.mpl
    renderer = cast(Any, native.canvas).get_renderer()
    canvas = native.bbox
    minimum = THEMES[figure.config.theme].minimum_font
    texts: list[tuple[Any, str | None]] = [(t, None) for t in native.texts]
    semantics: dict[str, str] = {}
    for name, panel in figure.panels.items():
        ax = panel.ax
        if not panel.kinds:
            errors.append(Issue("empty_panel", "Panel contains no registered data plot", name))
        for required, label, axis_name in zip(
            panel.required_labels, [ax.get_xlabel(), ax.get_ylabel()], ["x", "y"], strict=True
        ):
            if required and not label.strip():
                errors.append(
                    Issue(
                        "missing_label",
                        f"Supply {axis_name}label including units where applicable",
                        name,
                    )
                )
        for limits in [ax.get_xlim(), ax.get_ylim()]:
            if not np.isfinite(limits).all() or limits[0] == limits[1]:
                errors.append(Issue("axis_range", "Axis limits must be finite and distinct", name))
        if panel.kinds & {"bar", "area", "hist"}:
            horizontal = any(
                r.get("orient") == "horizontal" for r in figure.records if r["panel"] == name
            )
            limits = ax.get_xlim() if horizontal else ax.get_ylim()
            if not min(limits) <= 0 <= max(limits):
                errors.append(
                    Issue("zero_baseline", "Length/area encoding must include zero", name)
                )
        for label, color in panel.semantic.items():
            if label in semantics and semantics[label] != color:
                errors.append(
                    Issue(
                        "inconsistent_color", f"{label!r} has different colors across panels", name
                    )
                )
            semantics[label] = color
            if contrast_ratio(color, to_hex(ax.get_facecolor())) < 3:
                recommendations.append(
                    Issue(
                        "mark_contrast",
                        f"{label!r} has contrast below 3:1; retain markers/edges",
                        name,
                    )
                )
        for artist, label, expected in panel.semantic_artists:
            actual = None
            if hasattr(artist, "get_color"):
                actual = to_hex(artist.get_color())
            elif hasattr(artist, "get_facecolor"):
                colors = np.asarray(artist.get_facecolor())
                if colors.size:
                    actual = to_hex(tuple(colors[0] if colors.ndim == 2 else colors))
            if actual is not None and actual != expected:
                errors.append(
                    Issue("inconsistent_color", f"Native artist for {label!r} changed color", name)
                )
        by_color: dict[str, list[str]] = {}
        for label, color in panel.semantic.items():
            by_color.setdefault(color, []).append(label)
        for labels in by_color.values():
            if len(labels) > 1:
                recommendations.append(
                    Issue(
                        "ambiguous_colors",
                        f"Shared color for {labels}; assign explicit colors",
                        name,
                    )
                )
        legend = ax.get_legend()
        if legend is not None:
            texts.extend((t, name) for t in legend.get_texts())
            box = legend.get_window_extent(renderer)
            if overlap(box, ax.bbox):
                recommendations.append(
                    Issue(
                        "legend_over_data", "Legend occupies data region; inspect occlusion", name
                    )
                )
        if any(r.get("kind") == "dual_axis" and r["panel"] == name for r in figure.records):
            recommendations.append(
                Issue("dual_axis", "Independent axis scaling requires scientific review", name)
            )
    # Include colorbar axes too. Filter numeric ticks that the renderer places beyond visible limits.
    for ax in native.axes:
        panel_name = next((n for n, p in figure.panels.items() if p.ax is ax), None)
        texts.extend(
            (t, panel_name)
            for t in [
                ax.xaxis.label,
                ax.yaxis.label,
                ax.title,
                ax.xaxis.offsetText,
                ax.yaxis.offsetText,
                *ax.texts,
            ]
        )
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if low - 1e-9 <= tick.get_loc() <= high + 1e-9:
                    texts.extend((t, panel_name) for t in [tick.label1, tick.label2])
    for legend in native.legends:
        texts.extend((t, None) for t in legend.get_texts())
    measured, seen = [], set()
    for text, text_panel in texts:
        if id(text) in seen or not text.get_visible() or not text.get_text().strip():
            continue
        seen.add(id(text))
        box = text_box(text, renderer)
        if not np.isfinite(box.extents).all() or box.width <= 0 or box.height <= 0:
            recommendations.append(Issue("unmeasured_text", "Text extent unavailable", text_panel))
            continue
        content = text.get_text()[:70]
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.width + 1 or box.y1 > canvas.height + 1:
            errors.append(Issue("clipped_text", f"Outside canvas: {content!r}", text_panel))
        if text.get_fontsize() < minimum:
            errors.append(Issue("small_text", f"{content!r} is below {minimum:g} pt", text_panel))
        if text.get_gid() != "rendeer-cell":
            if contrast_ratio(to_hex(text.get_color()), to_hex(native.get_facecolor())) < 4.5:
                recommendations.append(
                    Issue("text_contrast", f"{content!r} has contrast below 4.5:1", text_panel)
                )
        measured.append((text, box, text_panel))
    for i, (a, box, overlap_panel) in enumerate(measured):
        for b, other, _ in measured[i + 1 :]:
            if overlap(box, other):
                errors.append(
                    Issue(
                        "text_overlap",
                        f"{a.get_text()[:40]!r} overlaps {b.get_text()[:40]!r}",
                        overlap_panel,
                    )
                )
    return QualityReport(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(recommendations)))
