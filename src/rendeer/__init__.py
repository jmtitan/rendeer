"""Composable scientific figures."""

from .colors import COLORMAPS, PALETTES, TOKENS, color_for, contrast_ratio
from .composition import facet, pairplot
from .config import Config, configure, context, get_config, set_colors
from .data import summarize
from .figure import Figure, Panel, figure
from .plots.comparison import area, bar, interval, line, scatter
from .plots.matrix import confusion, correlation, heatmap
from .plots.ml import calibration, pareto, precision_recall, roc
from .plots.statistical import box, ecdf, hist, kde, violin
from .validation import Issue, QualityError, QualityReport

__version__ = "0.1.0"
__all__ = [
    "COLORMAPS",
    "PALETTES",
    "TOKENS",
    "Config",
    "Figure",
    "Panel",
    "Issue",
    "QualityError",
    "QualityReport",
    "area",
    "bar",
    "color_for",
    "configure",
    "context",
    "contrast_ratio",
    "figure",
    "get_config",
    "interval",
    "line",
    "scatter",
    "set_colors",
    "summarize",
    "box",
    "ecdf",
    "hist",
    "kde",
    "violin",
    "confusion",
    "correlation",
    "heatmap",
    "calibration",
    "pareto",
    "precision_recall",
    "roc",
    "facet",
    "pairplot",
]
