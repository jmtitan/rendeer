"""Physical-size presets. Presets are starting points, not compliance certificates."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .config import Config


@dataclass(frozen=True)
class Theme:
    single: float
    double: float
    font_size: float
    minimum_font: float
    line_width: float = 1.4
    marker_size: float = 4.0


THEMES = MappingProxyType(
    {
        "paper": Theme(3.5, 7.2, 9, 7),
        "nature": Theme(89 / 25.4, 183 / 25.4, 7, 5, 1.0, 3.0),
        "ml": Theme(3.25, 6.75, 9, 7),
        "ieee": Theme(3.5, 7.16, 8, 6),
        "slides": Theme(6.4, 12.8, 16, 12, 2.2, 6),
        "report": Theme(4.0, 8.0, 10, 8),
        "grayscale": Theme(3.5, 7.2, 9, 7),
    }
)


def rc_params(config: Config) -> dict[str, Any]:
    theme = THEMES[config.theme]
    result = {
        "font.family": "sans-serif",
        "font.sans-serif": [config.font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": theme.font_size,
        "axes.labelsize": theme.font_size,
        "xtick.labelsize": theme.font_size - 1,
        "ytick.labelsize": theme.font_size - 1,
        "legend.fontsize": theme.font_size - 1,
        "axes.titlesize": theme.font_size,
        "text.color": "#252525",
        "axes.labelcolor": "#252525",
        "axes.edgecolor": "#666666",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": theme.line_width,
        "lines.markersize": theme.marker_size,
        "legend.frameon": False,
        "legend.borderaxespad": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": config.dpi,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "text.usetex": config.usetex,
        "axes.formatter.useoffset": False,
    }
    result.update(dict(config.rc))
    return result
